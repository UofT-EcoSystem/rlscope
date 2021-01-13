"""
Collect GPU metrics from ``nvidia-smi``.

Uses the ``--csv`` option to provide additional info not provided
by the plain-old ``nvidia-smi`` command.
Used to collect GPU utilization info.

See also
---------
rlscope.parser.cpu_gpu_util : Plotting GPU utilization.
"""
from rlscope.profiler.util import pprint_msg
from rlscope.profiler.util import print_cmd

from rlscope.profiler.rlscope_logging import logger
import textwrap
import sys
import time
import datetime
import re
import subprocess
from distutils import spawn
import os
import platform

# nvidia-smi --help--query-accounted-apps
#
# NOTE: doesn't work for compute apps (not sure about graphics).
query_accounted_apps_opts = [
    "timestamp",
    "gpu_name",
    "gpu_bus_id",
    "gpu_serial",
    "gpu_uuid",
    "vgpu_instance",
    "pid",
    "gpu_utilization",
    "mem_utilization",
    "max_memory_usage",
    "time",
]

# nvidia-smi --help-query-compute-apps
#
# NOTE: works for compute apps, not sure about graphics.
query_compute_apps_opts = [
    "timestamp",
    "gpu_name",
    "gpu_bus_id",
    "gpu_serial",
    "gpu_uuid",
    "pid",
    "process_name",
    "used_gpu_memory",
]

# $ nvidia-smi --help-query-gpu
#
# Query all GPU info.
# NOTE: Different versions of nvidia-smi will have different options available;
# doesn't look like there's a good way to query available fields in a structured (e.g. csv) format.
# So instead, lets JUST select fields we expect to find for any nvidia-smi.
query_gpu_opts = [
    "timestamp",
    # "driver_version",
    # "count",
    "name",
    # "serial",
    "uuid",
    # "pci.bus_id",
    # "pci.domain",
    # "pci.bus",
    # "pci.device",
    # "pci.device_id",
    # "pci.sub_device_id",
    # "pcie.link.gen.current",
    # "pcie.link.gen.max",
    # "pcie.link.width.current",
    # "pcie.link.width.max",
    # "index",
    # "display_mode",
    # "display_active",
    # "persistence_mode",
    # "accounting.mode",
    # "accounting.buffer_size",
    # "driver_model.current",
    # "driver_model.pending",
    # "vbios_version",
    # "inforom.img",
    # "inforom.oem",
    # "inforom.ecc",
    # "inforom.pwr",
    # "gom.current",
    # "gom.pending",
    # "fan.speed",
    # "pstate",
    # "clocks_throttle_reasons.supported",
    # "clocks_throttle_reasons.active",
    # "clocks_throttle_reasons.gpu_idle",
    # "clocks_throttle_reasons.applications_clocks_setting",
    # "clocks_throttle_reasons.sw_power_cap",
    # "clocks_throttle_reasons.hw_slowdown",
    # "clocks_throttle_reasons.hw_thermal_slowdown",
    # "clocks_throttle_reasons.hw_power_brake_slowdown",
    # "clocks_throttle_reasons.sw_thermal_slowdown",
    # "clocks_throttle_reasons.sync_boost",
    "memory.total",
    "memory.used",
    "memory.free",
    # "compute_mode",
    "utilization.gpu",
    "utilization.memory",
    # "encoder.stats.sessionCount",
    # "encoder.stats.averageFps",
    # "encoder.stats.averageLatency",
    # "ecc.mode.current",
    # "ecc.mode.pending",
    # "ecc.errors.corrected.volatile.device_memory",
    # "ecc.errors.corrected.volatile.register_file",
    # "ecc.errors.corrected.volatile.l1_cache",
    # "ecc.errors.corrected.volatile.l2_cache",
    # "ecc.errors.corrected.volatile.texture_memory",
    # "ecc.errors.corrected.volatile.total",
    # "ecc.errors.corrected.aggregate.device_memory",
    # "ecc.errors.corrected.aggregate.register_file",
    # "ecc.errors.corrected.aggregate.l1_cache",
    # "ecc.errors.corrected.aggregate.l2_cache",
    # "ecc.errors.corrected.aggregate.texture_memory",
    # "ecc.errors.corrected.aggregate.total",
    # "ecc.errors.uncorrected.volatile.device_memory",
    # "ecc.errors.uncorrected.volatile.register_file",
    # "ecc.errors.uncorrected.volatile.l1_cache",
    # "ecc.errors.uncorrected.volatile.l2_cache",
    # "ecc.errors.uncorrected.volatile.texture_memory",
    # "ecc.errors.uncorrected.volatile.total",
    # "ecc.errors.uncorrected.aggregate.device_memory",
    # "ecc.errors.uncorrected.aggregate.register_file",
    # "ecc.errors.uncorrected.aggregate.l1_cache",
    # "ecc.errors.uncorrected.aggregate.l2_cache",
    # "ecc.errors.uncorrected.aggregate.texture_memory",
    # "ecc.errors.uncorrected.aggregate.total",
    # "retired_pages.single_bit_ecc.count",
    # "retired_pages.double_bit.count",
    # "retired_pages.pending",
    # "temperature.gpu",
    # "temperature.memory",
    # "power.management",
    # "power.draw",
    # "power.limit",
    # "enforced.power.limit",
    # "power.default_limit",
    # "power.min_limit",
    # "power.max_limit",
    # "clocks.current.graphics",
    # "clocks.current.sm",
    # "clocks.current.memory",
    # "clocks.current.video",
    # "clocks.applications.graphics",
    # "clocks.applications.memory",
    # "clocks.default_applications.graphics",
    # "clocks.default_applications.memory",
    # "clocks.max.graphics",
    # "clocks.max.sm",
    # "clocks.max.memory",
]

def _get_nvidia_smi_path():
    if platform.system() == "Windows":
        # If the platform is Windows and nvidia-smi
        # could not be found from the environment path,
        # try to find it from system drive with default installation path
        nvidia_smi = spawn.find_executable('nvidia-smi')
        if nvidia_smi is None:
            nvidia_smi = "%s\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe" % os.environ['systemdrive']
    else:
        nvidia_smi = "nvidia-smi"
    return nvidia_smi
NVIDIA_SMI_EXEC = _get_nvidia_smi_path()

NVIDIA_TIMESTAMP_FMT = re.compile(r'(?P<year>\d+)/(?P<month>\d+)/(?P<day>\d+) (?P<hour>\d+):(?P<minute>\d+):(?P<second>\d+)\.(?P<msec>\d+)')
def parse_value(name, string):
    if name == 'driver_version':
        return string

    if string == '[Not Supported]':
        return None

    if name == 'timestamp':
        # 'timestamp': '2019/07/12 10:09:26.160',
        # The timestamp of where the query was made in format "YYYY/MM/DD HH:MM:SS.msec".
        #
        # NOTE: datetime doesn't support parsing milliseconds (%f is microseconds).
        # date = datetime.datetime.strptime(string, '%Y/%m/%d %I:%M:%S.%f')
        m = re.search(NVIDIA_TIMESTAMP_FMT, string)
        if m is None:
            # If we fail to parse a date, just return the raw string.
            return string
        date = datetime.time(
            hour=int(m.group('hour')),
            minute=int(m.group('minute')),
            second=int(m.group('second')),
            microsecond=int(m.group('msec')) * int(1e3),
        )
        return date

    try:
        number = int(string)
        return number
    except ValueError:
        pass

    try:
        number = float(string)
        return number
    except ValueError:
        pass

    return string

class MachineGPUInfo:
    """
    Collect all possible information about GPUs running on this machine from nvidia-smi.

    Using this class is good for minimizing the number of nvidia-smi calls.
    For Some CUDA installation, nvidia-smi can take nearly 0.5 seconds to run
    (not sure why!).

    :return:
    """
    def __init__(self, debug=False):
        self._debug = debug
        self._all_procs = processes(debug=debug)
        self._gpus = gpus(debug=debug)

    def gpus(self):
        return self._gpus

    def processes(self, gpu=None):
        if gpu is None:
            return self._all_procs
        return processes_filter_by_gpu(gpu, self._all_procs)

def run_nvidia_smi(cmd_opts=[], debug=False):
    cmd = [
              NVIDIA_SMI_EXEC,
          ] + cmd_opts
    if debug:
        print_cmd(cmd)
    # output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode('utf-8')
    lines = run_cmd(cmd)
    return lines

def run_cmd(cmd, **kwargs):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, **kwargs)
    with p:
        # NOTE: if this blocks it may be because there's a zombie utilization_sampler.py still running
        # (that was created by the training script) that hasn't been terminated!
        lines = []
        for line in p.stdout:
            # b'\n'-separated lines
            line = line.decode("utf-8")
            line = line.rstrip()
            lines.append(line)
    return lines

# I expect nvidia-smi to take ~ 0.08 seconds.
# I've seen it take ~ 2 seconds when it misbehaves.
MAX_NVIDIA_SMI_TIME_SEC = 1
def check_nvidia_smi(exit_if_fail=False, debug=False):
    """
    Make sure nvidia-smi runs fast enough to perform GPU utilization sampling.
    :return:
    """
    start_t = time.time()
    # $ nvidia-smi
    smi_output = run_nvidia_smi(debug=debug)
    end_t = time.time()
    nvidia_smi_sec = end_t - start_t
    if nvidia_smi_sec > MAX_NVIDIA_SMI_TIME_SEC:
        # $ sudo service nvidia-persistenced start
        errmsg = textwrap.dedent("""
        RL-Scope WARNING: nvidia-smi takes a long time to run on your system.
        In particular, it took {sec} sec to run nvidia-smi (we would prefer < {limit_sec}).
        This will interfere with sampling GPU utilization.
        You can fix this by running the following command:
        
        # Start systemd nvidia-persistenced service (if it's not already running).
        $ sudo nvidia-persistenced --persistence-mode
        
        For more details see:
        https://devtalk.nvidia.com/default/topic/1011192/nvidia-smi-is-slow-on-ubuntu-16-04-/
        """).format(
            sec=nvidia_smi_sec,
            limit_sec=MAX_NVIDIA_SMI_TIME_SEC,
        )
        if exit_if_fail:
            logger.error(errmsg)
            sys.exit(1)
        else:
            logger.warning(errmsg)


def _parse_entity(cmd_opts, csv_fields, post_process_entity=None, debug=False):
    format_opts = [
        'csv',
        'noheader',
        'nounits',
    ]

    cmd = [
              NVIDIA_SMI_EXEC,
              '--format={opts}'.format(opts=','.join(format_opts)),
          ] + cmd_opts

    if debug:
        print_cmd(cmd)
    # output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode('utf-8')
    # lines = output.split(os.linesep)
    lines = run_cmd(cmd)
    entities = []
    for line in lines:
        if re.search(r'^\s*$', line):
            continue
        fields = line.split(', ')
        if len(fields) != len(csv_fields):
            if debug:
                logger.info(pprint_msg({
                    'fields': fields,
                    'csv_fields': csv_fields,
                }))
            assert len(fields) == len(csv_fields)
        dic = dict()
        for k, v in zip(csv_fields, fields):
            value = parse_value(k, v)
            dic[k] = value
        entities.append(dic)
    if post_process_entity is not None:
        for entity in entities:
            post_process_entity(entity)
    return entities

def processes(gpu=None, debug=False):
    """
    Parse GPU process information.

    See query_compute_apps_opts for what gets parsed.

    :param gpu
        If provided, limit reported processes to this gpu.

        One of the GPU's returned by nvidia_gpu_query.gpus().
        (not this is not a gpu-name-string).
    """
    cmd_opts = [
        '--query-compute-apps={opts}'.format(opts=','.join(query_compute_apps_opts)),
    ]
    all_procs = _parse_entity(cmd_opts, query_compute_apps_opts, debug=debug)
    if gpu is None:
        return all_procs

    gpu_procs = processes_filter_by_gpu(gpu, all_procs)
    return gpu_procs

def processes_filter_by_gpu(gpu, all_procs):
    """
    Filter processes across all gpus (from nvidia_gpu_query.gpus())
    to processes belonging to just <gpu>.
    """
    # Limit processes to the provided gpu.
    gpu_procs = []
    for gpu_proc in all_procs:
        # NOTE: gpu_uuid is a unique identifier for the physical card.
        # We can use this to differentiate between the same model of card.
        if gpu_proc['gpu_uuid'] == gpu['uuid']:
            gpu_procs.append(gpu_proc)
    return gpu_procs

def gpus(debug=False):
    """
    Parse GPU device information.

    See query_gpu_opts for what gets parsed.
    """
    cmd_opts = [
        '--query-gpu={opts}'.format(opts=','.join(query_gpu_opts)),
    ]
    def post_process_entity(proc):
        if 'utilization.gpu' in proc:
            proc['utilization.gpu'] = proc['utilization.gpu']/100.
        if 'utilization.memory' in proc:
            proc['utilization.memory'] = proc['utilization.memory']/100.
    return _parse_entity(cmd_opts, query_gpu_opts, post_process_entity=post_process_entity, debug=debug)

RUN_STRESS_TEST = False

if RUN_STRESS_TEST:
    def test_stress_nvidia_smi():
        # Seems fine.
        def _test_stress_nvidia_smi():
            check_nvidia_smi()

            num_samples = 10000
            from progressbar import progressbar
            for i in progressbar(range(num_samples)):
                gpu_info = MachineGPUInfo()
        _test_stress_nvidia_smi()

