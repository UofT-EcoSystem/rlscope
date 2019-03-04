import signal
import time
import subprocess
import argparse
import textwrap
import psutil
import platform
import cpuinfo
import concurrent.futures

from os.path import join as _j, abspath as _a, dirname as _d, exists as _e, basename as _b

import GPUtil

from proto.protobuf.pyprof_pb2 import Pyprof, MachineUtilization, DeviceUtilization, UtilizationSample

from parser.common import *

from profiler.profilers import trace_suffix

class UtilizationSampler:
    def __init__(self, directory, util_dump_frequency_sec, util_sample_frequency_sec,
                 debug=False, debug_single_thread=False):
        self.directory = directory
        self.util_dump_frequency_sec = util_dump_frequency_sec
        self.util_sample_frequency_sec = util_sample_frequency_sec
        self.trace_id = 0
        self.pool = concurrent.futures.ProcessPoolExecutor(max_workers=1)
        self.machine_util = mk_machine_util()
        self.debug = debug
        self.debug_single_thread = debug_single_thread

    def add_util(self, start_time_sec, util):
        """
        NOTE: we record all the utilizations at the same time, to make analysis more effective.

        :param start_time_sec
            Epoch seconds when sample was taken.
        """
        if util['device_name'] not in self.machine_util.device_util:
            device_util = self.machine_util.device_util[util['device_name']]
            device_util.device_name = util['device_name']
        else:
            device_util = self.machine_util.device_util[util['device_name']]

        start_time_usec = start_time_sec/MICROSECONDS_IN_SECOND
        sample = UtilizationSample(
            start_time_us=int(start_time_usec),
            util=util['util'],
        )
        device_util.samples.extend([sample])

    def add_utils(self, start_time_sec, utils):
        for util in utils:
            self.add_util(start_time_sec, util)

    def run(self):
        if self.debug:
            print("> {klass}: Start collecting CPU/GPU utilization samples".format(
                klass=self.__class__.__name__,
            ))
        self.last_dump_sec = time.time()
        n_samples = 0
        with self.pool:
            while True:
                before = time.time()
                time.sleep(self.util_sample_frequency_sec)
                after = time.time()
                one_ms = after - before
                if self.debug:
                    print("> {klass}: Slept for {ms} ms".format(
                        klass=self.__class__.__name__,
                        ms=one_ms*MILLISECONDS_IN_SECOND,
                    ))

                if SigTermWatcher.kill_now:
                    if self.debug:
                        print("> {klass}: Got SIGINT; stop collecting utilization samples and exit".format(
                            klass=self.__class__.__name__,
                        ))
                    break

                now_sec = time.time()
                if now_sec - self.last_dump_sec >= self.util_dump_frequency_sec:
                    machine_util = self.machine_util
                    trace_id = self.trace_id

                    if self.debug:
                        print("> {klass}: Dump CPU/GPU utilization after {sec} seconds (# samples = {n}, sampled every {every_ms} ms) @ {path}".format(
                            klass=self.__class__.__name__,
                            sec=self.util_dump_frequency_sec,
                            every_ms=self.util_sample_frequency_sec*MILLISECONDS_IN_SECOND,
                            n=n_samples,
                            path=get_trace_path(self.directory, trace_id),
                        ))

                    if self.debug_single_thread:
                        dump_machine_util(self.directory, trace_id, machine_util)
                    else:
                        # NOTE: No need to wait for previous dumps to complete.
                        dump_future = self.pool.submit(dump_machine_util, self.directory, trace_id, machine_util)
                    self.machine_util = mk_machine_util()
                    self.trace_id += 1
                    n_samples = 0
                    self.last_dump_sec = time.time()

                if self.debug:
                    print("> {klass}: # Samples = {n} @ {sec}".format(
                        klass=self.__class__.__name__,
                        sec=now_sec,
                        n=n_samples,
                    ))

                # This takes a second to run...why?
                cpu_util = sample_cpu_utilization()
                gpu_utils = sample_gpu_utilization()
                self.add_util(now_sec, cpu_util)
                self.add_utils(now_sec, gpu_utils)

                n_samples += 1

# 100 ms
MIN_SAMPLE_FREQUENCY_SEC = 100/MILLISECONDS_IN_SECOND
# 500 ms
DEFAULT_SAMPLE_FREQUENCY_SEC = 500/MILLISECONDS_IN_SECOND
def measure_samples_per_sec():
    """
    Results:
    > nvidia-smi: 0.04176193714141846 calls/sec, measured over 100 calls
    > sample_gpu_utilization: 0.03259437084197998 calls/sec, measured over 100 calls
    > sample_cpu_utilization: 0.00022827625274658204 calls/sec, measured over 100 calls

    So, the maximum sample frequency is:
    CPU util:
    - every 0.22827625 ms
    GPU util:
    - every 32.594371 ms

    Reasonable choices to sample every:
    - 100 ms
    - 500 ms
    - 1 sec
    """
    # Q: How quickly can we call nvidia-smi to get utilization info?

    def report_calls_per_sec(name, func, iterations):
        start_sec = time.time()
        for i in range(iterations):
            func()
        end_sec = time.time()
        calls_per_sec = (end_sec - start_sec)/float(iterations)
        print("> {name}: {calls_per_sec} calls/sec, measured over {iters} calls".format(
            name=name,
            calls_per_sec=calls_per_sec,
            iters=iterations))

    def run_nvidia_smi():
        return subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    report_calls_per_sec('nvidia-smi', run_nvidia_smi, iterations=100)

    def run_sample_gpu_utilization():
        return sample_gpu_utilization()
    report_calls_per_sec('sample_gpu_utilization', run_sample_gpu_utilization, iterations=100)

    def run_sample_cpu_utilization():
        return sample_cpu_utilization()
    report_calls_per_sec('sample_cpu_utilization', run_sample_cpu_utilization, iterations=100)


def dump_machine_util(directory, trace_id, machine_util):
    """
    NOTE: Run in a separate thread/process; should NOT perform state modifications.
    """
    trace_path = get_trace_path(directory, trace_id)
    with open(trace_path, 'wb') as f:
        f.write(machine_util.SerializeToString())

def get_trace_path(directory, trace_id):
    path = _j(directory, 'machine_util{trace}.proto'.format(
        trace=trace_suffix(trace_id),
    ))
    return path

class _SigTermWatcher:
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
        self.kill_now = False

    def exit_gracefully(self, signum, frame):
        self.kill_now = True

SigTermWatcher = _SigTermWatcher()

def get_machine_name():
    """
    Portable way of calling hostname shell-command.

    :return:
        Unique name for a node in the cluster
    """
    machine_name = platform.node()
    return machine_name

# Cache cpuinfo, since this call takes 1 second to run, and we need to sample at millisecond frequency.
# NOTE: This has all sorts of CPU architecture information (e.g. l2 cache size)
CPU_INFO = cpuinfo.get_cpu_info()
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
    device_name = CPU_INFO['brand']
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

def mk_machine_util():
    machine_name = get_machine_name()
    machine_util = MachineUtilization(
        machine_name=machine_name,
    )
    return machine_util

def main():
    parser = argparse.ArgumentParser("Sample GPU/CPU utilization over the course of training")
    parser.add_argument('--iml-util-sample-frequency-sec',
                        type=float,
                        default=DEFAULT_SAMPLE_FREQUENCY_SEC,
                        help=textwrap.dedent("""
    IML: How frequently (in seconds) should we sample GPU/CPU utilization?
    default: sample every 500 ms.
    """))
    parser.add_argument('--iml-util-dump-frequency-sec',
                        type=float,
                        default=10.,
                        help=textwrap.dedent("""
    IML: How frequently (in seconds) should we sample GPU/CPU utilization?
    default: dump every 10 seconds.
    """))
    parser.add_argument('--iml-directory',
                        required=True,
                        help=textwrap.dedent("""
    IML: profiling output directory.
    """))
    parser.add_argument('--iml-debug',
                        action='store_true',
                        help=textwrap.dedent("""
    IML: debug profiler.
    """))
    parser.add_argument('--iml-debug-single-thread',
                        action='store_true',
                        help=textwrap.dedent("""
    IML: debug with single thread.
    """))

    parser.add_argument('--measure-samples-per-sec',
                        action='store_true',
                        help=textwrap.dedent("""
    Determines reasonable values for --iml-util-sample-frequency-sec.
    
    How fast can we call nvidia-smi (to sample GPU utilization)?  
    How fast can we gather CPU utilization?
    """))
    args = parser.parse_args()

    if args.measure_samples_per_sec:
        measure_samples_per_sec()
        return

    if args.iml_util_sample_frequency_sec < MIN_SAMPLE_FREQUENCY_SEC:
        parser.error("Need --iml-util-sample-frequency-sec={val} to be larger than minimum sample frequency ({min} sec)".format(
            val=args.iml_util_sample_frequency_sec,
            min=MIN_SAMPLE_FREQUENCY_SEC,
        ))

    util_sampler = UtilizationSampler(
        directory=args.iml_directory,
        util_dump_frequency_sec=args.iml_util_dump_frequency_sec,
        util_sample_frequency_sec=args.iml_util_sample_frequency_sec,
        debug=args.iml_debug,
        debug_single_thread=args.iml_debug_single_thread,
    )
    util_sampler.run()

if __name__ == '__main__':
    main()
