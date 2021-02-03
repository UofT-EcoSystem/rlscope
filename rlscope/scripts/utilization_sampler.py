"""
``rls-util-sampler`` script for collecting GPU (and CPU) utilization
using ``nvidia-smi`` every 0.5 second over the course of training.
"""
from rlscope.profiler.rlscope_logging import logger
import signal
import time
import subprocess
import argparse
import textwrap
import psutil
import platform
import threading
import cpuinfo
import concurrent.futures
import sys
import numpy as np

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from os.path import join as _j, abspath as _a, dirname as _d, exists as _e, basename as _b

from rlscope.profiler import nvidia_gpu_query

from rlscope import py_config

from rlscope.protobuf.pyprof_pb2 import CategoryEventsProto, MachineUtilization, DeviceUtilization, UtilizationSample

from rlscope.profiler.util import log_cmd, print_cmd, get_cpu_brand
from rlscope.parser.common import *
from rlscope.profiler import timer as rlscope_timer

from rlscope.parser import check_host
from rlscope.parser.exceptions import RLScopeConfigurationError

BYTES_IN_KB = 1 << 10
BYTES_IN_MB = 1 << 20

# 100 ms
MIN_UTIL_SAMPLE_FREQUENCY_SEC = 100/constants.MILLISECONDS_IN_SECOND
# 500 ms
DEFAULT_UTIL_SAMPLE_FREQUENCY_SEC = 500/constants.MILLISECONDS_IN_SECOND
def get_util_sampler_parser(add_rlscope_root_pid=True, only_fwd_arguments=False):
    # rlscope_root_pid=None,
    """
    :param fwd_arguments:
        Only add arguments that should be forwarded to utilization_sampler.py from ML scripts.
    :return:
    """
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__.lstrip().rstrip()),
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--rlscope-directory',
                        # required=True,
                        help=textwrap.dedent("""\
    RL-Scope: profiling output directory.
    """))
    parser.add_argument('--rlscope-debug',
                        action='store_true',
                        help=textwrap.dedent("""\
    RL-Scope: debug profiler.
    """))
    if only_fwd_arguments:
        return parser

    parser.add_argument('--rlscope-util-sample-frequency-sec',
                        type=float,
                        default=DEFAULT_UTIL_SAMPLE_FREQUENCY_SEC,
                        help=textwrap.dedent("""\
    RL-Scope: How frequently (in seconds) should we sample GPU/CPU utilization?
    default: sample every 500 ms.
    """))
    if add_rlscope_root_pid:
        parser.add_argument('--rlscope-root-pid',
                            required=True, type=int,
                            help=textwrap.dedent("""\
        RL-Scope: (internal use)
        The PID of the root training process on this machine.
        When sampling memory usage, we sample total memory usage of this 
        process and its entire process tree (i.e. to support multi-process training).
        """))
    parser.add_argument('--rlscope-util-dump-frequency-sec',
                        type=float,
                        default=10.,
                        help=textwrap.dedent("""\
    RL-Scope: How frequently (in seconds) should we sample GPU/CPU utilization?
    default: dump every 10 seconds.
    """))
    parser.add_argument('--rlscope-debug-single-thread',
                        action='store_true',
                        help=textwrap.dedent("""\
    RL-Scope: debug with single thread.
    """))

    # parser.add_argument('--measure-samples-per-sec',
    #                     action='store_true',
    #                     help=textwrap.dedent("""\
    # Determines reasonable values for --rlscope-util-sample-frequency-sec.
    #
    # How fast can we call nvidia-smi (to sample GPU utilization)?
    # How fast can we gather CPU utilization?
    # """))

    parser.add_argument('--skip-smi-check',
                        action='store_true',
                        help=textwrap.dedent("""\
    If NOT set, we will make sure nvidia-smi runs quickly (i.e. under 1 second); 
    this is needed for GPU sampling.
    """))

    parser.add_argument('--kill',
                        action='store_true',
                        help=textwrap.dedent("""\
    Kill any existing instances of this script that are still running, then exit.
    """))

    return parser


def CPU_as_util(info):
    assert type(info) in {MachineProcessCPUInfo}
    util = {
        'util':info.cpu_util,
        'device_name':info.device_name,
        'epoch_time_usec':info.epoch_time_usec,
        'total_resident_memory_bytes':info.total_resident_memory_bytes,
    }
    return util

def GPU_as_util(gpu, epoch_time_usec, total_resident_memory_bytes):
    util = {
        'util': gpu['utilization.gpu'],
        'device_name': gpu['name'],
        'epoch_time_usec': epoch_time_usec,
        'total_resident_memory_bytes': total_resident_memory_bytes,
    }
    return util

class UtilizationSampler:
    def __init__(self, directory, pid, util_dump_frequency_sec, util_sample_frequency_sec,
                 async_process=None,
                 debug=False, debug_single_thread=False):
        self.directory = directory
        self.pid = pid
        self.async_process = async_process
        if self.async_process is not None:
            assert self.pid == self.async_process.pid
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

        start_time_usec = start_time_sec*constants.MICROSECONDS_IN_SECOND
        # NOTE: When UtilizationSample.util is 0.0 and we print it, it just won't print
        # "util: 0.0" which you may confused with util being None.
        # assert util['util'] is not None
        sample = UtilizationSample(
            start_time_us=int(start_time_usec),
            util=util['util'],
            total_resident_memory_bytes=util['total_resident_memory_bytes'],
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
                logger.info("> {klass}: Dump CPU/GPU utilization after {sec} seconds (# samples = {n}, sampled every {every_ms} ms) @ {path}".format(
                    klass=self.__class__.__name__,
                    sec=self.util_dump_frequency_sec,
                    every_ms=self.util_sample_frequency_sec*constants.MILLISECONDS_IN_SECOND,
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

    def _prepare_to_stop(self, cur_time_sec):
        # Dump any remaining samples we have not dumped yet.
        self._maybe_dump(cur_time_sec, dump=True)
        self.check_pending_dump_calls(wait=True)

    def run(self):
        SigTermWatcher = _SigTermWatcher()
        if self.async_process is not None:
            proc_watcher = ProcessWatcher(self.async_process)
        else:
            proc_watcher = None

        if self.debug:
            logger.info("> {klass}: Start collecting CPU/GPU utilization samples".format(
                klass=self.__class__.__name__,
            ))
        self.last_dump_sec = time.time()
        self.n_samples = 0
        self.exit_status = 0
        should_stop = False
        with self.pool:
            while True:
                before = time.time()
                time.sleep(self.util_sample_frequency_sec)
                after = time.time()
                one_ms = after - before
                if self.debug:
                    logger.info("> {klass}: Slept for {ms} ms".format(
                        klass=self.__class__.__name__,
                        ms=one_ms*constants.MILLISECONDS_IN_SECOND,
                    ))

                cur_time_sec = time.time()

                if SigTermWatcher.kill_now:
                    if self.debug:
                        logger.info("> {klass}: Got SIGINT; dumping remaining collected samples and exiting".format(
                            klass=self.__class__.__name__,
                        ))
                    should_stop = True
                elif proc_watcher is not None and proc_watcher.finished.is_set():
                    if self.debug:
                        logger.info("> {klass}: cmd terminated with exit status {ret}".format(
                            klass=self.__class__.__name__,
                            ret=proc_watcher.retcode,
                        ))
                    assert proc_watcher.retcode is not None
                    should_stop = True
                    self.exit_status = proc_watcher.retcode

                if should_stop:
                    self._prepare_to_stop(cur_time_sec)
                    break

                self._maybe_dump(cur_time_sec)

                if self.debug:
                    logger.info("> {klass}: # Samples = {n} @ {sec}".format(
                        klass=self.__class__.__name__,
                        sec=cur_time_sec,
                        n=self.n_samples,
                    ))

                self.check_pending_dump_calls()

                cpu_util = None
                gpu_utils = None
                try:
                    machine_cpu_info = MachineProcessCPUInfo(self.pid)
                    cpu_util = sample_cpu_utilization(machine_cpu_info)
                    machine_gpu_info = nvidia_gpu_query.MachineGPUInfo(debug=self.debug)
                    gpu_utils = sample_gpu_utilization(machine_gpu_info, self.pid, debug=self.debug)
                except psutil.NoSuchProcess as e:
                    logger.info("Exiting rls-util-sampler since pid={pid} no longer exists".format(pid=e.pid))
                    should_stop = True
                    continue

                if should_stop:
                    self._prepare_to_stop(cur_time_sec)
                    break

                if self.debug:
                    logger.info("> {klass}: utils = \n{utils}".format(
                        klass=self.__class__.__name__,
                        utils=textwrap.indent(
                            pprint.pformat({'cpu_util':cpu_util, 'gpu_utils':gpu_utils}),
                            prefix="  "),
                    ))
                if cpu_util is not None:
                    self.add_util(cur_time_sec, cpu_util)
                if gpu_utils is not None:
                    self.add_utils(cur_time_sec, gpu_utils)
                if cpu_util is not None or gpu_utils is not None:
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
        # Delete reverse sorted order, otherwise indices will be messed up.
        for i in reversed(sorted(del_indices)):
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
        logger.info("> {name}: {calls_per_sec} calls/sec, measured over {iters} calls".format(
            name=name,
            calls_per_sec=calls_per_sec,
            iters=iterations))

    def run_nvidia_smi():
        return subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    report_calls_per_sec('nvidia-smi', run_nvidia_smi, iterations=100)

    def run_sample_gpu_utilization():
        machine_gpu_info = nvidia_gpu_query.MachineGPUInfo()
        return sample_gpu_utilization(machine_gpu_info, pid=os.getpid())
    report_calls_per_sec('sample_gpu_utilization', run_sample_gpu_utilization, iterations=100)

    def run_sample_cpu_utilization():
        machine_cpu_info = MachineProcessCPUInfo()
        return sample_cpu_utilization(machine_cpu_info)
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
        logger.info("> Dumped @ {path}".format(path=trace_path))

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

class ProcessWatcher:
    def __init__(self, proc):
        assert proc is not None
        self.proc = proc
        self.retcode = None
        self.finished = threading.Event()
        self.pool = ThreadPoolExecutor()
        self.pool.submit(self._watch_process, self)

    @staticmethod
    def _watch_process(self):
        self.retcode = self.proc.wait()
        self.finished.set()

# Cache cpuinfo, since this call takes 1 second to run, and we need to sample at millisecond frequency.
# NOTE: This has all sorts of CPU architecture information (e.g. l2 cache size)
CPU_INFO = cpuinfo.get_cpu_info()
def sample_cpu_utilization(machine_cpu_info):
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
    util = CPU_as_util(machine_cpu_info)
    return util

class MachineProcessCPUInfo:
    """
    Collect all possible information about a CPU process running on this machine from psutil.

    We use this to maintain consistency of information collected.
    """
    def __init__(self, pid, epoch_time_usec=None, debug=False):
        self.pid = pid
        if epoch_time_usec is None:
            epoch_time_usec = rlscope_timer.now_us()
        self.debug = debug
        self.epoch_time_usec = epoch_time_usec
        self.process_tree = get_process_tree(pid)
        self.total_resident_memory_bytes = self._sample_cpu_total_resident_memory_bytes(self.process_tree)
        self.cpu_util = self._sample_cpu_util()
        self.device_name = get_cpu_brand(CPU_INFO)

    def _sample_cpu_util(self):
        return psutil.cpu_percent()/100.

    def _sample_cpu_total_resident_memory_bytes(self, procs):
        mem_infos = []
        for proc in procs:
            try:
                mem_info = proc.memory_info()
                mem_infos.append(mem_info)
            except psutil.NoSuchProcess as e:
                if self.debug:
                    logger.info((
                        "Tried to sample resident memory from proc={proc}, "
                        "but it looks like it exited; skipping").format(
                        proc=proc))
        total_resident_memory_bytes = np.sum(m.rss for m in mem_infos)
        return total_resident_memory_bytes

def get_process_tree(pid):
    parent = psutil.Process(pid=pid)
    children = parent.children(recursive=True)
    procs = [parent] + children

    pids_set = set(proc.pid for proc in procs)
    pids_list = [proc.pid for proc in procs]
    assert len(pids_set) == len(pids_list)

    return procs

def sample_gpu_total_resident_memory_bytes(machine_gpu_info, gpu, pid, debug=False):
    procs = get_process_tree(pid)
    # if debug:
    #     logger.info(pprint_msg({'sample_gpu_bytes.procs': procs, 'pid': pid, 'gpu': gpu}))

    pids = set(proc.pid for proc in procs)


    gpu_procs = machine_gpu_info.processes(gpu)
    # if debug:
    #     logger.info(pprint_msg({'sample_gpu_bytes.gpu_procs': gpu_procs}))
    total_resident_memory_bytes = 0
    for gpu_proc in gpu_procs:
        if not py_config.IS_DOCKER:
            if gpu_proc['pid'] in pids:
                total_resident_memory_bytes += gpu_proc['used_gpu_memory'] * BYTES_IN_MB
        else:
            # nvidia-smi BUG: nvidia-smi doesn't respect pid-namespacing of docker container
            # and uses pids of processes outside the container.
            #
            # Instead, get total GPU memory used by summing across ALL GPU processes.
            total_resident_memory_bytes += gpu_proc['used_gpu_memory'] * BYTES_IN_MB

    return total_resident_memory_bytes

def sample_gpu_utilization(machine_gpu_info, pid, debug=False):
    """
    Report a single [0..1] value representing current GPU utilization.
    Report a separate value for EACH GPU in the system.
    """
    gpus = machine_gpu_info.gpus()
    epoch_time_usec = rlscope_timer.now_us()
    gpu_utils = []
    # if debug:
    #     logger.info(pprint_msg({'sample_gpu_bytes.gpus': gpus}))
    for gpu in gpus:
        total_resident_memory_bytes = sample_gpu_total_resident_memory_bytes(machine_gpu_info, gpu, pid, debug=debug)
        gpu_util = GPU_as_util(
            gpu,
            epoch_time_usec=epoch_time_usec,
            total_resident_memory_bytes=total_resident_memory_bytes)
        gpu_utils.append(gpu_util)
    return gpu_utils

    # gpu_procs looks like:
    #
    # [{'gpu_bus_id': '00000000:07:00.0',
    #   'gpu_name': 'GeForce RTX 2070',
    #   'gpu_serial': None,
    #   'gpu_uuid': 'GPU-e9c6b1d8-2b80-fee2-b750-08c5adcaac3f',
    #   'pid': 9994,
    #   'process_name': 'python',
    #   'timestamp': datetime.time(10, 58, 18, 239000),
    #   'used_gpu_memory': 649}]

    # gpus looks like:
    #
    # [{'accounting.buffer_size': 4000,
    #   'accounting.mode': 'Disabled',
    #   'clocks.applications.graphics': None,
    #   'clocks.applications.memory': None,
    #   'clocks.current.graphics': 300,
    #   'clocks.current.memory': 405,
    #   'clocks.current.sm': 300,
    #   'clocks.current.video': 540,
    #   'clocks.default_applications.graphics': None,
    #   'clocks.default_applications.memory': None,
    #   'clocks.max.graphics': 2100,
    #   'clocks.max.memory': 7001,
    #   'clocks.max.sm': 2100,
    #   'clocks_throttle_reasons.active': '0x0000000000000001',
    #   'clocks_throttle_reasons.applications_clocks_setting': 'Not Active',
    #   'clocks_throttle_reasons.gpu_idle': 'Active',
    #   'clocks_throttle_reasons.hw_power_brake_slowdown': 'Not Active',
    #   'clocks_throttle_reasons.hw_slowdown': 'Not Active',
    #   'clocks_throttle_reasons.hw_thermal_slowdown': 'Not Active',
    #   'clocks_throttle_reasons.supported': '0x00000000000001FF',
    #   'clocks_throttle_reasons.sw_power_cap': 'Not Active',
    #   'clocks_throttle_reasons.sw_thermal_slowdown': 'Not Active',
    #   'clocks_throttle_reasons.sync_boost': 'Not Active',
    #   'compute_mode': 'Default',
    #   'count': 1,
    #   'display_active': 'Enabled',
    #   'display_mode': 'Enabled',
    #   'driver_model.current': None,
    #   'driver_model.pending': None,
    #   'driver_version': '418.67',
    #   'ecc.errors.corrected.aggregate.device_memory': None,
    #   'ecc.errors.corrected.aggregate.l1_cache': None,
    #   'ecc.errors.corrected.aggregate.l2_cache': None,
    #   'ecc.errors.corrected.aggregate.register_file': None,
    #   'ecc.errors.corrected.aggregate.texture_memory': None,
    #   'ecc.errors.corrected.aggregate.total': None,
    #   'ecc.errors.corrected.volatile.device_memory': None,
    #   'ecc.errors.corrected.volatile.l1_cache': None,
    #   'ecc.errors.corrected.volatile.l2_cache': None,
    #   'ecc.errors.corrected.volatile.register_file': None,
    #   'ecc.errors.corrected.volatile.texture_memory': None,
    #   'ecc.errors.corrected.volatile.total': None,
    #   'ecc.errors.uncorrected.aggregate.device_memory': None,
    #   'ecc.errors.uncorrected.aggregate.l1_cache': None,
    #   'ecc.errors.uncorrected.aggregate.l2_cache': None,
    #   'ecc.errors.uncorrected.aggregate.register_file': None,
    #   'ecc.errors.uncorrected.aggregate.texture_memory': None,
    #   'ecc.errors.uncorrected.aggregate.total': None,
    #   'ecc.errors.uncorrected.volatile.device_memory': None,
    #   'ecc.errors.uncorrected.volatile.l1_cache': None,
    #   'ecc.errors.uncorrected.volatile.l2_cache': None,
    #   'ecc.errors.uncorrected.volatile.register_file': None,
    #   'ecc.errors.uncorrected.volatile.texture_memory': None,
    #   'ecc.errors.uncorrected.volatile.total': None,
    #   'ecc.mode.current': None,
    #   'ecc.mode.pending': None,
    #   'encoder.stats.averageFps': 0,
    #   'encoder.stats.averageLatency': 0,
    #   'encoder.stats.sessionCount': 0,
    #   'enforced.power.limit': 175.0,
    #   'fan.speed': 30,
    #   'gom.current': None,
    #   'gom.pending': None,
    #   'index': 0,
    #   'inforom.ecc': None,
    #   'inforom.img': 'G001.0000.02.04',
    #   'inforom.oem': 1.1,
    #   'inforom.pwr': None,
    #   'memory.free': 7252,
    #   'memory.total': 7952,
    #   'memory.used': 700,
    #   'name': 'GeForce RTX 2070',
    #   'pci.bus': '0x07',
    #   'pci.bus_id': '00000000:07:00.0',
    #   'pci.device': '0x00',
    #   'pci.device_id': '0x1F0210DE',
    #   'pci.domain': '0x0000',
    #   'pci.sub_device_id': '0x251619DA',
    #   'pcie.link.gen.current': 1,
    #   'pcie.link.gen.max': 3,
    #   'pcie.link.width.current': 16,
    #   'pcie.link.width.max': 16,
    #   'persistence_mode': 'Disabled',
    #   'power.default_limit': 175.0,
    #   'power.draw': 9.48,
    #   'power.limit': 175.0,
    #   'power.management': 'Enabled',
    #   'power.max_limit': 200.0,
    #   'power.min_limit': 125.0,
    #   'pstate': 'P8',
    #   'retired_pages.double_bit.count': None,
    #   'retired_pages.pending': None,
    #   'retired_pages.single_bit_ecc.count': None,
    #   'serial': None,
    #   'temperature.gpu': 33,
    #   'temperature.memory': 'N/A',
    #   'timestamp': datetime.time(10, 58, 7, 443000),
    #   'utilization.gpu': 0.0,
    #   'utilization.memory': 0.02,
    #   'uuid': 'GPU-e9c6b1d8-2b80-fee2-b750-08c5adcaac3f',
    #   'vbios_version': '90.06.18.00.8D'}]

def mk_machine_util():
    machine_name = get_machine_name()
    machine_util = MachineUtilization(
        machine_name=machine_name,
    )
    return machine_util

def disable_test_sample_cpu_util():
    tries = 10000
    for i in range(tries):
        machine_cpu_info = MachineProcessCPUInfo()
        cpu_util = sample_cpu_utilization(machine_cpu_info)
        assert 0 <= cpu_util['util'] <= 1

def disable_test_sample_gpu_util():
    tries = 100
    import tensorflow as tf
    import multiprocessing

    class GPURunner:
        def __init__(self, should_stop):
            self.should_stop = should_stop
            # Allow multiple users to use the TensorFlow API.
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.compat.v1.Session(config=config)
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
            machine_gpu_info = nvidia_gpu_query.MachineGPUInfo()
            gpu_utils = sample_gpu_utilization(machine_gpu_info, pid=os.getpid())
            for gpu_util in gpu_utils:
                pprint.pprint({'gpu_util':gpu_util})
                assert 0 <= gpu_util['util'] <= 1
    finally:
        with should_stop:
            should_stop.value = 1
        gpu_runner.join(timeout=2)
        gpu_runner.terminate()

from rlscope.profiler.rlscope_logging import logger
def main():

    try:
        check_host.check_config()
    except RLScopeConfigurationError as e:
        logger.error(e)
        sys.exit(1)

    rlscope_util_argv, cmd_argv = split_argv_on(sys.argv[1:])
    parser = get_util_sampler_parser(add_rlscope_root_pid=len(cmd_argv) == 0)
    args = parser.parse_args(rlscope_util_argv)

    # To make it easy to launch utilization sampler manually in certain code bases,
    # allow ignoring all the --rlscope-* arguments:
    #
    # e.g. in minigo's loop_main.sh shell script we do
    #   python3 -m scripts.utilization_sampler "$@" --rlscope-directory $BASE_DIR &
    # where $@ contains all the --rlscope-* args.
    args, extra_argv = parser.parse_known_args()
    # args = parser.parse_args()

    # NOTE: During profiling, we depend on this being called from the root training script.
    if not args.skip_smi_check:
        nvidia_gpu_query.check_nvidia_smi()

    if args.kill:
        for proc in psutil.process_iter():
            # if proc.name() == sys.argv[0]:
            # pinfo = proc.as_dict(attrs=['pid', 'name', 'username'])
            pinfo = proc.as_dict(attrs=['pid', 'username', 'cmdline'])
            pprint.pprint({'pinfo': pinfo})
            # cmdline = proc.cmdline()
            try:
                logger.info(pinfo['cmdline'])
                if re.search(r'rls-util-sampler', ' '.join(pinfo['cmdline'])) and pinfo['pid'] != os.getpid():
                    logger.info("> Kill rls-util-sampler: {proc}".format(
                        proc=proc))
                    proc.kill()
            except psutil.NoSuchProcess:
                pass
        sys.exit(0)

    if args.rlscope_directory is None:
        logger.info("--rlscope-directory is required: directory where trace-files are saved")
        parser.print_help()
        sys.exit(1)

    os.makedirs(args.rlscope_directory, exist_ok=True)

    # if args.measure_samples_per_sec:
    #     measure_samples_per_sec()
    #     return

    if args.rlscope_util_sample_frequency_sec < MIN_UTIL_SAMPLE_FREQUENCY_SEC:
        parser.error("Need --rlscope-util-sample-frequency-sec={val} to be larger than minimum sample frequency ({min} sec)".format(
            val=args.rlscope_util_sample_frequency_sec,
            min=MIN_UTIL_SAMPLE_FREQUENCY_SEC,
        ))


    rlscope_root_pid = None
    cmd_proc = None
    if len(cmd_argv) != 0:
        exe_path = shutil.which(cmd_argv[0])
        if exe_path is None:
            print("RL-Scope ERROR: couldn't locate {exe} on $PATH; try giving a full path to {exe} perhaps?".format(
                exe=cmd_argv[0],
            ))
            sys.exit(1)
        cmd = [exe_path] + cmd_argv[1:]
        print_cmd(cmd)

        sys.stdout.flush()
        sys.stderr.flush()
        cmd_proc = subprocess.Popen(cmd)
        rlscope_root_pid = cmd_proc.pid
    else:
        rlscope_root_pid = args.rlscope_root_pid

    # NOTE: usually, we have rls-prof program signal us to terminate.
    # However if they provide a cmd, we would like to terminate sampler when cmd finishes, and return cmd's exit status.
    util_sampler = UtilizationSampler(
        directory=args.rlscope_directory,
        pid=rlscope_root_pid,
        async_process=cmd_proc,
        util_dump_frequency_sec=args.rlscope_util_dump_frequency_sec,
        util_sample_frequency_sec=args.rlscope_util_sample_frequency_sec,
        debug=args.rlscope_debug,
        debug_single_thread=args.rlscope_debug_single_thread,
    )
    util_sampler.run()
    sys.exit(util_sampler.exit_status)

class UtilSamplerProcess:
    def __init__(self, rlscope_directory, debug=False):
        self.rlscope_directory = rlscope_directory
        self.debug = debug
        self.proc = None
        self.proc_pid = None

    def _launch_utilization_sampler(self):
        util_cmdline = ['rls-util-sampler']
        util_cmdline.extend(['--rlscope-directory', self.rlscope_directory])
        # Sample memory-usage of the entire process tree rooted at ths process.
        util_cmdline.extend(['--rlscope-root-pid', str(os.getpid())])
        if self.debug:
            util_cmdline.append('--rlscope-debug')
        # We make sure nvidia-smi runs fast at the VERY START of training
        # (to avoid false alarms when training is busy with the CPU/GPU).
        # util_cmdline.append('--skip-smi-check')
        if self.debug:
            log_cmd(util_cmdline)
        self.proc = subprocess.Popen(util_cmdline)
        self.proc_pid = self.proc.pid
        logger.info("RL-Scope: CPU/GPU utilization sampler running @ pid={pid}".format(pid=self.proc_pid))

    def _terminate_utilization_sampler(self, warn_terminated=True):
        assert self.proc_pid is not None
        logger.info("RL-Scope: terminating CPU/GPU utilization sampler @ pid={pid}".format(pid=self.proc_pid))

        try:
            proc = psutil.Process(self.proc_pid)
        except psutil.NoSuchProcess as e:
            if warn_terminated:
                logger.info("RL-Scope: Warning; tried to terminate utilization sampler @ pid={pid} but it wasn't running".format(pid=self.proc_pid))
            return

        proc.terminate()
        self.proc = None
        self.proc_pid = None

    @property
    def running(self):
        return self.proc is not None

    def start(self):
        if not self.running:
            self._launch_utilization_sampler()

    def stop(self):
        if self.running:
            self._terminate_utilization_sampler()

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

def util_sampler(*args, **kwargs):
    """
    Sample machine CPU/GPU utilization, and output trace-files to <rlscope_directory>.

    NOTE:
    - This is meant to be used WITHOUT rlscope profiling active
    - You should be running vanilla tensorflow (not RL-Scope modified tensorflow)

    :param rlscope_directory:
        Where to store trace-files containing CPU/GPU utilization info.
    :param debug:
        Extra logger.
    """
    import rlscope.api
    if rlscope.api.prof is not None:
        assert rlscope.api.prof.disabled
    # import rlscope.api
    # if rlscope.api.prof is not None:
    #     raise RuntimeError(textwrap.dedent("""\
    #         RL-Scope ERROR:
    #         When using rlscope.util_sampler(...) to collect CPU/GPU utilization information,
    #         you should not be doing any other profiling with IML.
    #
    #         In particular, you should NOT call rlscope.handle_rlscope_args(...).
    #         """))
    return UtilSamplerProcess(*args, **kwargs)

if __name__ == '__main__':
    main()
