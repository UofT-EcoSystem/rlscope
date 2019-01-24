# from proto.tensorflow.core.profiler.tfprof_log_pb2 import ProfileProto
from tensorflow.core.profiler.tfprof_log_pb2 import ProfileProto
from proto.protobuf.pyprof_pb2 import Pyprof

from parser.common import *
from parser.stats import KernelTime, category_times_add_time

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
    def phase(self):
        return self.proto.phase

    @property
    def steps(self):
        return sorted(self.proto.steps)

    def parse(self, step, bench_name, group_by_device=False, include_dummy=False):

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
        # print("HELLO PROTO 1")
        # print(self.proto)
        # PSEUDOCODE:
        # for step in steps:
        #
        # import ipdb; ipdb.set_trace()
        # print("HELLO PROTO 2")

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
            _add_time(CATEGORY_GPU, False)
        elif IsCPUTime(device):
            _add_time(CATEGORY_CUDA_API_CPU, group_by_device)
        else:
            raise NotImplementedError("Not sure what category device={dev} falls under.".format(dev=device))

    def get_category(self, device):
        if IsGPUTime(device):
            return CATEGORY_GPU
        elif IsCPUTime(device):
            return CATEGORY_CUDA_API_CPU
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
        return list(execs.keys())

    def each_event(self, device, step, node, get_execs):
        exec_profile = node.execs[step]
        execs = get_execs(exec_profile)
        for tupl in execs[device].times:
            start_us, duration_us = tupl.int64_values
            name = node.name
            category = self.get_category(device)
            yield category, start_us, duration_us, name

    def num_all_events(self):
        n = 0
        for _ in self.all_events():
            n += 1
        return n

    def all_events(self):
        def events_for(step, node, get_execs):
            devices = self.each_device(step, node, get_execs)
            for device in devices:
                for event in self.each_event(device, step, node, get_execs):
                    yield device, event

        for step in self.steps:
            for node_id, node in self.proto.nodes.items():
                if step not in node.execs.keys():
                    continue

                # TODO: Some of the events in the tfprof trace are DEFINITELY duplicates.
                # e.g. memcpy / stream:all have overlaps.
                # I think my compute overlap code handles duplicates,
                # as long as they belong to the same category.
                for device, event in events_for(step, node, self.get_accelerator_execs):
                    yield device, event
                for device, event in events_for(step, node, self.get_cpu_execs):
                    yield device, event

    def get_cpu_execs(self, exec_profile):
        return exec_profile.cpu_execs

class PyprofCategoryTimesReader:
    def __init__(self, profile_path):
        self.profile_path = profile_path
        with open(self.profile_path, 'rb') as f:
            self.proto = Pyprof()
            self.proto.ParseFromString(f.read())

    def print(self, f):
        print(self.proto, file=f)

    @property
    def steps(self):
        return sorted(self.proto.steps)

    def parse(self, step, bench_name, group_by_device=False, include_dummy=False):
        category_times = dict()

        if step not in self.proto.steps:
            print("> WARNING: didn't find step in pyprof @ {path}".format(
                path=self.profile_path))
            return category_times

        category_times[CATEGORY_PYTHON] = []
        self.add_event_times_to(category_times[CATEGORY_PYTHON], self.proto.python_events[step].events)

        # clib_times = dict()
        for category, clib_events in self.proto.clibs[step].clibs.items():
            if category in [CATEGORY_DUMMY_EVENT]:
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

