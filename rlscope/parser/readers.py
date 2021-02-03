"""
Read (old) RL-Scope trace files.

.. deprecated:: 1.0.0
    Old TensorFlow-specific profiling data.
"""
from rlscope.profiler.rlscope_logging import logger
# from tensorflow.core.profiler.tfprof_log_pb2 import ProfileProto
from rlscope.protobuf.pyprof_pb2 import CategoryEventsProto

from rlscope.parser.common import *
from rlscope.parser import constants
from rlscope.parser.stats import KernelTime, category_times_add_time

from rlscope.protobuf import rlscope_prof_pb2

DEFAULT_group_by_device = False
DEFAULT_ignore_categories = {constants.CATEGORY_DUMMY_EVENT, constants.CATEGORY_UNKNOWN}
DEFAULT_debug = False

class TFProfCategoryTimesReader:
    """
    Reads a tfprof protobuf file, outputs "category times" format.

    Category times format:

        A dict, where the keys are a single category name (not a combination),
        and the values are a list of raw start/end times for that category:
        {
            'Python': [(start[0][0], end[0][0]), (start[0][1], end[0][1]), ...],
            'GPU': [(start[1][0], end[1][0]), (start[1][1], end[1][1]), ...],
            ...
        }

        NOTE: start/end tuples are in sorted order (by their start time)
    """
    def __init__(self, profile_path):
        self.profile_path = profile_path
        with open(self.profile_path, 'rb') as f:
            self.proto = ProfileProto()
            self.proto.ParseFromString(f.read())

            # with open(self._tfprof_txt_path, 'w') as f:
            #     print(tfprof_reader.proto, file=f)
            #
            # with open(self._pyprof_txt_path, 'w') as f:
            #     print(pyprof_reader.proto, file=f)

    def print(self, f):
        print(self.proto, file=f)

    @property
    def process_name(self):
        return self.proto.process_name

    @property
    def machine_name(self):
        return self.proto.machine_name

    @property
    def phase(self):
        return self.proto.phase
    @property
    def phase_name(self):
        return self.phase

    @property
    def steps(self):
        return sorted(self.proto.steps)

    def parse(self, step, bench_name,
              group_by_device=DEFAULT_group_by_device,
              ignore_categories=DEFAULT_ignore_categories,
              debug=DEFAULT_debug):

        # PSEUDOCODE:
        # Compute [CUDA CPU time | GPU time | CUDA CPU/GPU time] overlap/non-overlap for a single step.
        #
        # We want to "group together" the 3 CPU lane thread times.
        # More when GPU time overlaps with nothing in CPU time,
        # and CPU time overlaps with nothing in GPU time.
        #
        # To get the average stats, we average ACROSS the steps measured.
        # Would be nice to know CPU is every executing anything else...
        # thread-pool threads are just blocked, but is the "main" thread doing anything?

        # self.kernel_times()
        # self.api_times()
        # logger.info("HELLO PROTO 1")
        # logger.info(self.proto)
        # PSEUDOCODE:
        # for step in steps:
        #
        # logger.info("HELLO PROTO 2")

        # for step in self.proto.steps:
        # Categories:
        # - [CUDA API CPU]
        # - [GPU]
        # - Future: [Framework CPU]
        category_times = dict()

        for node_id, node in self.proto.nodes.items():
            if step not in node.execs.keys():
                continue
            self._add_all_times(category_times, step, node, self.get_accelerator_execs, group_by_device=group_by_device)
            self._add_all_times(category_times, step, node, self.get_cpu_execs, group_by_device=group_by_device)

        return category_times

    def _add_time(self, category_times, device, ktime, group_by_device):
        def _add_time(category, group_by_device):
            if category not in category_times:
                if group_by_device:
                    category_times[category] = dict()
                else:
                    category_times[category] = []

            if group_by_device and device not in category_times[category]:
                category_times[category][device] = []

            if group_by_device:
                add_to = category_times[category][device]
            else:
                add_to = category_times[category]
            add_to.append(ktime)

        if IsGPUTime(device):
            _add_time(constants.CATEGORY_GPU, False)
        elif IsCPUTime(device):
            _add_time(constants.CATEGORY_CUDA_API_CPU, group_by_device)
        else:
            raise NotImplementedError("Not sure what category device={dev} falls under.".format(dev=device))

    def _add_all_times(self, category_times, step, node, get_execs, group_by_device):
        """
        :param get_execs:
            ExecProfile -> map<string, ExecTime>
            Either reteurn accelerator_execs or cpu_execs
        """
        exec_profile = node.execs[step]
        execs = get_execs(exec_profile)
        for device in execs.keys():
            for tupl in execs[device].times:
                start_us, duration_us = tupl.int64_values
                ktime = KernelTime(start_usec=start_us, time_usec=duration_us, name=node.name)
                category_times_add_time(category_times, device, ktime, group_by_device)

    def get_accelerator_execs(self, exec_profile):
        return exec_profile.accelerator_execs

    def each_device(self, step, node, get_execs):
        exec_profile = node.execs[step]
        execs = get_execs(exec_profile)
        return execs.keys()
        # return list(execs.keys())

    def each_event(self, device, step, node, get_execs):
        exec_profile = node.execs[step]
        execs = get_execs(exec_profile)
        for tupl in execs[device].times:
            start_us, duration_us = tupl.int64_values
            name = node.name
            category = get_category_from_device(device)
            yield category, start_us, duration_us, name

    def num_all_events(self):
        return _num_all_events(self)

    def _events_for(self, step, node, get_execs):
        devices = self.each_device(step, node, get_execs)
        for device in devices:
            for event in self.each_event(device, step, node, get_execs):
                yield device, event

    def get_device_names(self):
        device_names = set()
        # for step in self.steps:
        #     for node_id, node in self.proto.nodes.items():
        for node_id, node in self.proto.nodes.items():
            for step in self.steps:
                # if step not in node.execs.keys():
                if step not in node.execs:
                    continue

                devices = self.each_device(step, node, self.get_accelerator_execs)
                device_names.update(devices)

                devices = self.each_device(step, node, self.get_cpu_execs)
                device_names.update(devices)
        return device_names

    def all_events_for_step(self, step):
        for node_id, node in self.proto.nodes.items():
            # if step not in node.execs.keys():
            if step not in node.execs:
                continue

            # TODO: Some of the events in the tfprof trace are DEFINITELY duplicates.
            # e.g. memcpy / stream:all have overlaps.
            # I think my compute overlap code handles duplicates,
            # as long as they belong to the same category.
            for device, event in self._events_for(step, node, self.get_accelerator_execs):
                yield device, event
            for device, event in self._events_for(step, node, self.get_cpu_execs):
                yield device, event

    def all_events(self, debug=False):
        for step in self.steps:
            for item in self.all_events_for_step(step):
                yield item

    def get_cpu_execs(self, exec_profile):
        return exec_profile.cpu_execs

class PyprofCategoryTimesReader:
    def __init__(self, profile_path):
        self.profile_path = profile_path
        with open(self.profile_path, 'rb') as f:
            self.proto = CategoryEventsProto()
            self.proto.ParseFromString(f.read())

    def print(self, f):
        print(self.proto, file=f)

    @property
    def steps(self):
        return sorted(self.proto.steps)

    def parse(self, step, bench_name,
              group_by_device=DEFAULT_group_by_device,
              ignore_categories=DEFAULT_ignore_categories,
              debug=DEFAULT_debug):
        category_times = dict()

        if step not in self.proto.steps:
            logger.info("> WARNING: didn't find step in pyprof @ {path}".format(
                path=self.profile_path))
            return category_times

        category_times[constants.CATEGORY_PYTHON] = []
        self.add_event_times_to(category_times[constants.CATEGORY_PYTHON], self.proto.python_events[step].events)

        # clib_times = dict()
        for category, clib_events in self.proto.clibs[step].clibs.items():
            if category in [constants.CATEGORY_DUMMY_EVENT]:
                continue
            assert category not in category_times
            category_times[category] = []
            # TODO: add C API function name to proto / KernelTime
            self.add_event_times_to(category_times[category], clib_events.events)

        return category_times

    def add_event_times_to(self, ktimes, events):
        for event in events:
            if hasattr(event, 'name'):
                name = event.name
            else:
                name = None
            ktime = KernelTime(start_usec=event.start_time_us, time_usec=event.duration_us, name=name)
            ktimes.append(ktime)

class CUDADeviceEventsReader:
    """
    Read GPU-side events (kernel, memcpy, memset).
    """
    def __init__(self, profile_path):
        self.profile_path = profile_path
        self.proto = read_cuda_device_events_file(self.profile_path)

    def print(self, f):
        print(self.proto, file=f)

    @property
    def process_name(self):
        return self.proto.process_name

    @property
    def machine_name(self):
        return self.proto.machine_name

    @property
    def phase(self):
        return self.proto.phase
    @property
    def phase_name(self):
        return self.phase

    def num_all_events(self):
        return _num_all_events(self)

    def get_device_names(self):
        device_names = set()

        for device_name, dev_events_proto in self.proto.dev_events.items():
            device_names.add(device_name)

        return device_names

    def all_events(self, debug=False):

        def get_event_name(event):
            if event.name != "" and event.name is not None:
                return event.name

            if event.cuda_event_type == rlscope_prof_pb2.UNKNOWN:
                return "Unknown"
            elif event.cuda_event_type == rlscope_prof_pb2.KERNEL:
                return "Kernel"
            elif event.cuda_event_type == rlscope_prof_pb2.MEMCPY:
                return "Memcpy"
            elif event.cuda_event_type == rlscope_prof_pb2.MEMSET:
                return "Memset"
            else:
                raise NotImplementedError("Not sure what Event.name to use for event.cuda_event_type == {code}".format(
                    code=event.cuda_event_type,
                ))

        category = constants.CATEGORY_GPU
        for device_name, dev_events_proto in self.proto.dev_events.items():
            for event in dev_events_proto.events:
                name = get_event_name(event)
                yield device_name, category, event.start_time_us, event.duration_us, name

class CUDAAPIStatsReader:
    """
    Read CUDA API events (cudaLaunchKernel, cudaMemcpyAsync).
    """
    def __init__(self, profile_path):
        self.profile_path = profile_path
        self.proto = read_cuda_api_stats_file(self.profile_path)

    def print(self, f):
        print(self.proto, file=f)

    @property
    def process_name(self):
        return self.proto.process_name

    @property
    def machine_name(self):
        return self.proto.machine_name

    @property
    def phase(self):
        return self.proto.phase
    @property
    def phase_name(self):
        return self.phase

    def num_all_events(self):
        return _num_all_events(self)

    def all_events(self, debug=False):
        category = constants.CATEGORY_CUDA_API_CPU
        for event in self.proto.events:
            name = event.api_name
            yield category, event.start_time_us, event.duration_us, name

    def cuda_api_call_events(self, debug=False):
        # category = constants.CATEGORY_CUDA_API_CPU
        # name = event.api_name
        # yield category, event.start_time_us, event.duration_us, name
        for event in self.proto.events:
            yield event

class CategoryEventsReader:
    """
    Read Category events (e.g. python events).
    """
    def __init__(self, profile_path):
        self.profile_path = profile_path
        self.proto = read_category_events_file(self.profile_path)

    def print(self, f):
        print(self.proto, file=f)

    @property
    def process_name(self):
        return self.proto.process_name

    @property
    def machine_name(self):
        return self.proto.machine_name

    @property
    def phase(self):
        return self.proto.phase
    @property
    def phase_name(self):
        return self.phase

    def num_all_events(self):
        return _num_all_events(self)

    def all_events(self, debug=False):
        for category, event_list in self.proto.category_events.items():
            for event in event_list.events:
                name = event.name
                yield category, event.start_time_us, event.duration_us, name

class OpStackReader:
    """
    Read OpStack overhead events (e.g. number of pyprof_annotation's).
    """
    def __init__(self, profile_path):
        self.profile_path = profile_path
        self.proto = read_op_stack_file(self.profile_path)

    def print(self, f):
        print(self.proto, file=f)

    @property
    def process_name(self):
        return self.proto.process_name

    @property
    def machine_name(self):
        return self.proto.machine_name

    @property
    def phase(self):
        return self.proto.phase
    @property
    def phase_name(self):
        return self.phase

    def num_all_events(self):
        return _num_all_events(self)

    def all_events(self, debug=False):
        for overhead_type, phase_overhead_events in self.proto.overhead_events.items():
            for overhead_type, phase_overhead_events in self.proto.overhead_events.items():
                for phase, operation_overhead_events in phase_overhead_events.items():
                    operation_name = operation_overhead_events.operation_name
                    num_overhead_events = operation_overhead_events.num_overhead_events
                    yield overhead_type, phase, operation_name, num_overhead_events

def _num_all_events(self):
    n = 0
    for _ in self.all_events():
        n += 1
    return n
