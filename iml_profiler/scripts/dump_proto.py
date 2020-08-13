from iml_profiler.profiler.iml_logging import logger
import argparse
import textwrap
import sys

# from tensorflow.core.profiler.tfprof_log_pb2 import ProfileProto
from iml_profiler.protobuf.pyprof_pb2 import CategoryEventsProto, ProcessMetadata, IncrementalTrainingProgress
from iml_profiler.protobuf.iml_prof_pb2 import CUDAAPIPhaseStatsProto, MachineDevsEventsProto, OpStackProto
# from iml_profiler.protobuf.iml_prof_pb2 import CUDAAPIPhaseStatsProto, MachineDevsEventsProto, OpStackProto
# src/libs/range_sampling/range_sampling.proto
from range_sampling.range_sampling_pb2 import GPUHwCounterSampleProto

from iml_profiler.protobuf.unit_test_pb2 import \
    IMLUnitTestOnce, \
    IMLUnitTestMultiple

from iml_profiler.parser.common import *

def dump_proto_txt(path, ProtoKlass, stream):
    logger.info("> DUMP: {name} @ {path}".format(
        name=ProtoKlass.__name__,
        path=path))
    with open(path, 'rb') as f:
        proto = ProtoKlass()
        proto.ParseFromString(f.read())
    print(proto, file=stream)

from iml_profiler.profiler.iml_logging import logger
def main():
    parser = argparse.ArgumentParser("Dump protobuf files to txt")
    parser.add_argument("--proto",
                        required=True,
                        help=textwrap.dedent("""
                        Protofile.
                        """))
    args = parser.parse_args()

    if is_tfprof_file(args.proto):
        dump_proto_txt(args.proto, ProfileProto, sys.stdout)
    elif is_pyprof_file(args.proto) or is_dump_event_file(args.proto):
        dump_proto_txt(args.proto, CategoryEventsProto, sys.stdout)
    elif is_unit_test_once_file(args.proto):
        dump_proto_txt(args.proto, IMLUnitTestOnce, sys.stdout)
    elif is_unit_test_multiple_file(args.proto):
        dump_proto_txt(args.proto, IMLUnitTestMultiple, sys.stdout)
    elif is_machine_util_file(args.proto):
        dump_proto_txt(args.proto, MachineUtilization, sys.stdout)
    elif is_process_metadata_file(args.proto):
        dump_proto_txt(args.proto, ProcessMetadata, sys.stdout)
    elif is_training_progress_file(args.proto):
        dump_proto_txt(args.proto, IncrementalTrainingProgress, sys.stdout)
    elif is_cuda_api_stats_file(args.proto) or is_fuzz_cuda_api_stats_file(args.proto):
        dump_proto_txt(args.proto, CUDAAPIPhaseStatsProto, sys.stdout)
    elif is_cuda_device_events_file(args.proto):
        dump_proto_txt(args.proto, MachineDevsEventsProto, sys.stdout)
    elif is_op_stack_file(args.proto):
        dump_proto_txt(args.proto, OpStackProto, sys.stdout)
    elif is_gpu_hw_file(args.proto):
        dump_proto_txt(args.proto, GPUHwCounterSampleProto, sys.stdout)
    elif is_pyprof_call_times_file(args.proto):
        call_times_data = read_pyprof_call_times_file(args.proto)
        pprint.pprint(call_times_data)
    else:
        logger.error("Not sure what protobuf class to use for files like \"{path}\"".format(
            path=args.proto))
        sys.exit(1)

if __name__ == '__main__':
    main()
