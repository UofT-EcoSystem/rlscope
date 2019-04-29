import argparse
import textwrap

from tensorflow.core.profiler.tfprof_log_pb2 import ProfileProto
from prof_protobuf.pyprof_pb2 import Pyprof

from parser.common import *

def dump_proto_txt(path, ProtoKlass, stream):
    print("> DUMP: {name} @ {path}".format(
        name=ProtoKlass.__name__,
        path=path))
    with open(path, 'rb') as f:
        proto = ProtoKlass()
        proto.ParseFromString(f.read())
    print(proto, file=stream)

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
        dump_proto_txt(args.proto, Pyprof, sys.stdout)
    elif is_machine_util_file(args.proto):
        dump_proto_txt(args.proto, MachineUtilization, sys.stdout)
    elif is_pyprof_call_times_file(args.proto):
        call_times_data = read_pyprof_call_times_file(args.proto)
        pprint.pprint(call_times_data)
    else:
        print("ERROR: Not sure what protobuf class to use for files like \"{path}\"".format(
            path=args.proto))
        sys.exit(1)

if __name__ == '__main__':
    main()
