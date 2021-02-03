"""
``rls-dump-proto`` command for inspecting contents of RL-Scope
trace files (binary protobuf files).
"""
from rlscope.profiler.rlscope_logging import logger
import argparse
import textwrap
import sys
import csv

from rlscope.protobuf.pyprof_pb2 import CategoryEventsProto, ProcessMetadata, IncrementalTrainingProgress
from rlscope.protobuf.rlscope_prof_pb2 import CUDAAPIPhaseStatsProto, MachineDevsEventsProto, OpStackProto
# from rlscope.protobuf.rlscope_prof_pb2 import CUDAAPIPhaseStatsProto, MachineDevsEventsProto, OpStackProto
# src/libs/range_sampling/range_sampling.proto
try:
    from range_sampling.range_sampling_pb2 import GPUHwCounterSampleProto
except ImportError:
    GPUHwCounterSampleProto = None

from rlscope.protobuf.unit_test_pb2 import \
    IMLUnitTestOnce, \
    IMLUnitTestMultiple

from rlscope.parser.common import *

def dump_proto_txt(proto, stream):
    print(proto, file=stream)

def load_proto(path, ProtoKlass):
    logger.info("> LOAD: {name} @ {path}".format(
        name=ProtoKlass.__name__,
        path=path))
    with open(path, 'rb') as f:
        proto = ProtoKlass()
        proto.ParseFromString(f.read())
    return proto
    # print(proto, file=stream)

def each_row(proto):
    if type(proto) == CategoryEventsProto:
        row = dict()
        row["process_name"] = proto.process_name
        row["phase"] = proto.phase
        row["machine_name"] = proto.machine_name
        for category_name, event_list in proto.category_events.items():
            row["category_name"] = category_name
            for event in event_list.events:
                row["thread_id"] = event.thread_id
                row["start_time_us"] = event.start_time_us
                row["duration_us"] = event.duration_us
                row["name"] = event.name
                yield dict(row)
    else:
        raise NotImplementedError("Not sure how to output {klass} protobuf as csv.".format(
            klass=type(proto).__name__))

def dump_csv(proto, stream):
    header = None
    writer = None
    for row in each_row(proto):
        if header is None:
            header = sorted(row.keys())
            # writer = csv.writer(stream)
            writer = csv.DictWriter(stream, fieldnames=header)
            writer.writeheader()
        writer.writerow(row)

from rlscope.profiler.rlscope_logging import logger
def main():
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__.lstrip().rstrip()),
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--proto",
                        required=True,
                        help=textwrap.dedent("""\
                        Protofile.
                        """))
    parser.add_argument("--csv",
                        action='store_true',
                        help=textwrap.dedent("""\
                        Output CSV (if supported).
                        """))
    args = parser.parse_args()

    def do_dump(path, ProtoKlass):
        proto = load_proto(path, ProtoKlass)
        if args.csv:
            dump_csv(proto, sys.stdout)
        else:
            dump_proto_txt(proto, sys.stdout)

    # if is_tfprof_file(args.proto):
    #     from tensorflow.core.profiler.tfprof_log_pb2 import ProfileProto
    #     do_dump(args.proto, ProfileProto)
    if is_pyprof_file(args.proto) or is_dump_event_file(args.proto):
        do_dump(args.proto, CategoryEventsProto)
    elif is_unit_test_once_file(args.proto):
        do_dump(args.proto, IMLUnitTestOnce)
    elif is_unit_test_multiple_file(args.proto):
        do_dump(args.proto, IMLUnitTestMultiple)
    elif is_machine_util_file(args.proto):
        do_dump(args.proto, MachineUtilization)
    elif is_process_metadata_file(args.proto):
        do_dump(args.proto, ProcessMetadata)
    elif is_training_progress_file(args.proto):
        do_dump(args.proto, IncrementalTrainingProgress)
    elif is_cuda_api_stats_file(args.proto) or is_fuzz_cuda_api_stats_file(args.proto):
        do_dump(args.proto, CUDAAPIPhaseStatsProto)
    elif is_cuda_device_events_file(args.proto):
        do_dump(args.proto, MachineDevsEventsProto)
    elif is_op_stack_file(args.proto):
        do_dump(args.proto, OpStackProto)
    elif GPUHwCounterSampleProto is not None and is_gpu_hw_file(args.proto):
        do_dump(args.proto, GPUHwCounterSampleProto)
    elif is_pyprof_call_times_file(args.proto):
        call_times_data = read_pyprof_call_times_file(args.proto)
        pprint.pprint(call_times_data)
    else:
        logger.error("Not sure what protobuf class to use for files like \"{path}\"".format(
            path=args.proto))
        sys.exit(1)

if __name__ == '__main__':
    main()
