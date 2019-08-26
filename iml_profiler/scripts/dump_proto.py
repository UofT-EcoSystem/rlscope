import logging
import argparse
import textwrap
import sys

from tensorflow.core.profiler.tfprof_log_pb2 import ProfileProto
from iml_profiler.protobuf.pyprof_pb2 import Pyprof, ProcessMetadata, IncrementalTrainingProgress
from iml_profiler.protobuf.iml_prof_pb2 import CUDAAPIPhaseStatsProto, MachineDevsEventsProto

from iml_profiler.protobuf.unit_test_pb2 import \
    IMLUnitTestOnce, \
    IMLUnitTestMultiple

from iml_profiler.parser.common import *

def dump_proto_txt(path, ProtoKlass, stream):
    logging.info("> DUMP: {name} @ {path}".format(
        name=ProtoKlass.__name__,
        path=path))
    with open(path, 'rb') as f:
        proto = ProtoKlass()
        proto.ParseFromString(f.read())
    print(proto, file=stream)

from iml_profiler.profiler import iml_logging
def main():
    iml_logging.setup_logging()
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
        dump_proto_txt(args.proto, Pyprof, sys.stdout)
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
    elif is_pyprof_call_times_file(args.proto):
        call_times_data = read_pyprof_call_times_file(args.proto)
        pprint.pprint(call_times_data)
    else:
        logging.info("ERROR: Not sure what protobuf class to use for files like \"{path}\"".format(
            path=args.proto))
        sys.exit(1)

if __name__ == '__main__':
    main()
