import logging
import signal
import time
import subprocess
import argparse
import textwrap
import psutil
import platform
import cpuinfo
import concurrent.futures
import sys
import numpy as np

from os.path import join as _j, abspath as _a, dirname as _d, exists as _e, basename as _b

import GPUtil

from iml_profiler.protobuf.pyprof_pb2 import Pyprof, MachineUtilization, DeviceUtilization, UtilizationSample

from iml_profiler.parser.common import *

from iml_profiler.profiler.profilers import trace_suffix, get_util_sampler_parser, MIN_UTIL_SAMPLE_FREQUENCY_SEC

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
        self.pending_dump_futures = []

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

        start_time_usec = start_time_sec*MICROSECONDS_IN_SECOND
        # NOTE: When UtilizationSample.util is 0.0 and we print it, it just won't print
        # "util: 0.0" which you may confused with util being None.
        # assert util['util'] is not None
        sample = UtilizationSample(
            start_time_us=int(start_time_usec),
            util=util['util'],
        )
        device_util.samples.extend([sample])

    def add_utils(self, start_time_sec, utils):
        for util in utils:
            self.add_util(start_time_sec, util)

    def _maybe_dump(self, cur_time_sec, dump=False):
        # now_sec = time.time()
        if self.n_samples > 0 and ( dump or ( cur_time_sec - self.last_dump_sec >= self.util_dump_frequency_sec ) ):
            machine_util = self.machine_util
            trace_id = self.trace_id

            if self.debug:
                logging.info("> {klass}: Dump CPU/GPU utilization after {sec} seconds (# samples = {n}, sampled every {every_ms} ms) @ {path}".format(
                    klass=self.__class__.__name__,
                    sec=self.util_dump_frequency_sec,
                    every_ms=self.util_sample_frequency_sec*MILLISECONDS_IN_SECOND,
                    n=self.n_samples,
                    path=get_trace_path(self.directory, trace_id),
                ))

            machine_util_str = machine_util.SerializeToString()
            if self.debug_single_thread:
                dump_machine_util(self.directory, trace_id, machine_util_str, self.debug)
            else:
                # NOTE: No need to wait for previous dumps to complete.
                #
                # https://groups.google.com/d/msg/protobuf/VqWJ3BmQXVg/iwO_m6apVlkJ
                #
                # Protobuf is a C-extension class that cannot be pickled.
                # Multiprocessing uses pickling to send data to processes.
                # So, we must serialize/deserialize protobuf manually.
                dump_future = self.pool.submit(dump_machine_util, self.directory, trace_id, machine_util_str, self.debug)
                self.pending_dump_futures.append(dump_future)
            self.machine_util = mk_machine_util()
            self.trace_id += 1
            self.n_samples = 0
            self.last_dump_sec = time.time()

        # return now_sec

    def run(self):
        if self.debug:
            logging.info("> {klass}: Start collecting CPU/GPU utilization samples".format(
                klass=self.__class__.__name__,
            ))
        self.last_dump_sec = time.time()
        self.n_samples = 0
        with self.pool:
            while True:
                before = time.time()
                time.sleep(self.util_sample_frequency_sec)
                after = time.time()
                one_ms = after - before
                if self.debug:
                    logging.info("> {klass}: Slept for {ms} ms".format(
                        klass=self.__class__.__name__,
                        ms=one_ms*MILLISECONDS_IN_SECOND,
                    ))

                cur_time_sec = time.time()

                if SigTermWatcher.kill_now:
                    if self.debug:
                        logging.info("> {klass}: Got SIGINT; dumping remaining collected samples and exiting".format(
                            klass=self.__class__.__name__,
                        ))
                        # Dump any remaining samples we have not dumped yet.
                        self._maybe_dump(cur_time_sec, dump=True)
                        self.check_pending_dump_calls(wait=True)
                    break

                self._maybe_dump(cur_time_sec)

                if self.debug:
                    logging.info("> {klass}: # Samples = {n} @ {sec}".format(
                        klass=self.__class__.__name__,
                        sec=cur_time_sec,
                        n=self.n_samples,
                    ))

                self.check_pending_dump_calls()

                cpu_util = sample_cpu_utilization()
                gpu_utils = sample_gpu_utilization()
                if self.debug:
                    logging.info("> {klass}: utils = \n{utils}".format(
                        klass=self.__class__.__name__,
                        utils=textwrap.indent(
                            pprint.pformat({'cpu_util':cpu_util, 'gpu_utils':gpu_utils}),
                            prefix="  "),
                    ))
                self.add_util(cur_time_sec, cpu_util)
                self.add_utils(cur_time_sec, gpu_utils)

                self.n_samples += 1

    def check_pending_dump_calls(self, wait=False):
        del_indices = []
        for i, dump_future in enumerate(self.pending_dump_futures):
            if wait:
                dump_future.result()
            if dump_future.done():
                ex = dump_future.exception(timeout=0)
                if ex is not None:
                    raise ex
                del_indices.append(i)
        for i in del_indices:
            del self.pending_dump_futures[i]


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
        logging.info("> {name}: {calls_per_sec} calls/sec, measured over {iters} calls".format(
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


def dump_machine_util(directory, trace_id, machine_util, debug):
    """
    NOTE: Run in a separate thread/process; should NOT perform state modifications.
    """
    # Q: Do both windows and linux multiprocessing.Process inherit CPU affinity...?
    if type(machine_util) != MachineUtilization:
        machine_util_str = machine_util
        machine_util = MachineUtilization()
        machine_util.ParseFromString(machine_util_str)

    trace_path = get_trace_path(directory, trace_id)
    with open(trace_path, 'wb') as f:
        f.write(machine_util.SerializeToString())

    if debug:
        logging.info("> Dumped @ {path}".format(path=trace_path))

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
    cpu_util = psutil.cpu_percent()/100.
    # cpu_utils = [util/100. for util in psutil.cpu_percent(percpu=True)]
    # runnable_cpus = set(get_runnable_cpus())
    # runnable_utils = [util for cpu, util in enumerate(cpu_utils) if cpu in runnable_cpus]
    # # Q: Should we only collect utilization of the CPUs the process is assigned to?
    # cpu_util = np.mean(runnable_utils)
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

def disable_test_sample_cpu_util():
    tries = 10000
    for i in range(tries):
        cpu_util = sample_cpu_utilization()
        assert 0 <= cpu_util['util'] <= 1

def disable_test_sample_gpu_util():
    tries = 100
    import tensorflow as tf
    import multiprocessing

    class GPURunner:
        def __init__(self, should_stop):
            self.should_stop = should_stop
            # Allow multiple users to use the TensorFlow API.
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.name = 'GPURunner'

            self.N = 1000
            self.zeros = np.zeros((self.N, self.N))
            self.feed_dict = {
                'a':self.zeros,
                'b':self.zeros,
            }

            self.mk_graph()
            self.run()

        def mk_graph(self):
            with tf.name_scope(self.name):
                self.a = tf.placeholder(float)
                self.b = tf.placeholder(float)
                self.c = self.a * self.b

        def run(self):
            while True:
                with self.should_stop:
                    if self.should_stop.value:
                        break
                c = self.sess.run(self.c, feed_dict=self.feed_dict)
                assert np.equal(c, 0.).all()

    should_stop = multiprocessing.Value('i', 0)

    gpu_runner = multiprocessing.Process(target=GPURunner, args=(should_stop,))
    gpu_runner.start()
    # Wait for it to start using the GPU...
    time.sleep(2)

    try:
        for i in range(tries):
            gpu_utils = sample_gpu_utilization()
            for gpu_util in gpu_utils:
                pprint.pprint({'gpu_util':gpu_util})
                assert 0 <= gpu_util['util'] <= 1
    finally:
        with should_stop:
            should_stop.value = 1
        gpu_runner.join(timeout=2)
        gpu_runner.terminate()

from iml_profiler.profiler import glbl
def main():
    glbl.setup_logging()
    parser = get_util_sampler_parser()
    # To make it easy to launch utilization sampler manually in certain code bases,
    # allow ignoring all the --iml-* arguments:
    #
    # e.g. in minigo's loop_main.sh shell script we do
    #   python3 -m scripts.utilization_sampler "$@" --iml-directory $BASE_DIR &
    # where $@ contains all the --iml-* args.
    args, extra_argv = parser.parse_known_args()
    # args = parser.parse_args()

    if args.kill:
        for proc in psutil.process_iter():
            # if proc.name() == sys.argv[0]:
            # pinfo = proc.as_dict(attrs=['pid', 'name', 'username'])
            pinfo = proc.as_dict(attrs=['pid', 'username', 'cmdline'])
            pprint.pprint({'pinfo': pinfo})
            # cmdline = proc.cmdline()
            try:
                logging.info(pinfo['cmdline'])
                if re.search(r'iml-util-sampler', ' '.join(pinfo['cmdline'])) and pinfo['pid'] != os.getpid():
                    logging.info("> Kill iml-util-sampler: {proc}".format(
                        proc=proc))
                    proc.kill()
            except psutil.NoSuchProcess:
                pass
        sys.exit(0)

    if args.iml_directory is None:
        logging.info("--iml-directory is required: directory where trace-files are saved")
        parser.print_help()
        sys.exit(1)

    os.makedirs(args.iml_directory, exist_ok=True)

    # proc = psutil.Process(pid=os.getpid())
    # dump_cpus = get_dump_cpus()
    # proc.cpu_affinity(dump_cpus)
    # logging.info("> Set CPU affinity of iml-util-sampler to: {cpus}".format(
    #     cpus=dump_cpus,
    # ))

    if args.measure_samples_per_sec:
        measure_samples_per_sec()
        return

    if args.iml_util_sample_frequency_sec < MIN_UTIL_SAMPLE_FREQUENCY_SEC:
        parser.error("Need --iml-util-sample-frequency-sec={val} to be larger than minimum sample frequency ({min} sec)".format(
            val=args.iml_util_sample_frequency_sec,
            min=MIN_UTIL_SAMPLE_FREQUENCY_SEC,
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
