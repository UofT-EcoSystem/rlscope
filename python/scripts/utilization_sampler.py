import signal
import time
import argparse
import textwrap
import psutil
import platform
import cpuinfo

import GPUtil

from tensorflow.core.profiler.tfprof_log_pb2 import ProfileProto
from proto.protobuf.pyprof_pb2 import Pyprof

from parser.common import *

class _SigTermWatcher:
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
        self.kill_now = False

    def exit_gracefully(self, signum, frame):
        self.kill_now = True

SigTermWatcher = _SigTermWatcher()

def get_machine_name():
    machine_name = platform.node()
    return machine_name

def sample_cpu_utilization():
    """
    Report a single [0..1] value representing current system-wide CPU utilization.

    psutil.cpu_percent() returns EXACTLY this.
    From psutil.cpu_percent docstring:
        "
        Return a float representing the current system-wide CPU
        utilization as a percentage.
        "

    NOTE: It's also possible to get INDIVIDUAL utilization for each CPU,
    if we choose to do that in the future.
    """
    cpu_util = psutil.cpu_percent()
    epoch_time_usec = now_us()
    # NOTE: This has all sorts of CPU architecture information (e.g. l2 cache size)
    cpu_info = cpuinfo.get_cpu_info()
    # 'Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz'
    device_name = cpu_info['brand']
    return {
        'util':cpu_util,
        'device_name':device_name,
        'epoch_time_usec':epoch_time_usec,
    }

def sample_gpu_utilization():
    """
    Report a single [0..1] value representing current GPU utilization.
    Report a separate value for EACH GPU in the system.

    We use GPUtil for querying nvidia GPU utilization.
    The module looks kinda meh, but it does the job of
    parsing nvidia-smi on both Windows/Linux for us.
    """
    # gpus[0].__dict__
    #
    # {'display_active': 'Enabled',
    #  'display_mode': 'Enabled',
    #  'driver': '418.43',
    #  'id': 0,
    #  'load': 0.0,
    #  'memoryFree': 7910.0,
    #  'memoryTotal': 7951.0,
    #  'memoryUsed': 41.0,
    #  'memoryUtil': 0.005156584077474532,
    #  'name': 'GeForce RTX 2070',
    #  'serial': '[Not Supported]',
    #  'temperature': 33.0,
    #  'uuid': 'GPU-e9c6b1d8-2b80-fee2-b750-08c5adcaac3f'}

    gpus = GPUtil.getGPUs()
    epoch_time_usec = now_us()
    return [{
        'device_name':gpu.name,
        'util':gpu.load,
        'epoch_time_usec':epoch_time_usec,
    } for gpu in gpus]

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
    elif is_pyprof_call_times_file(args.proto):
        call_times_data = read_pyprof_call_times_file(args.proto)
        pprint.pprint(call_times_data)
    else:
        print("ERROR: Not sure what protobuf class to use for files like \"{path}\"".format(
            path=args.proto))
        sys.exit(1)

    while True:
        time.sleep(1)
        print("doing something in a loop ...")
        if SigTermWatcher.kill_now:
            break
    print("End of the program. I was killed gracefully :)")



if __name__ == '__main__':
    main()
