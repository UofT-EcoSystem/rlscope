from rlscope.profiler.rlscope_logging import logger
import decimal

from rlscope.parser.common import *
from rlscope.parser import constants

def dump_category_times(category_times, json_path, print_log=True, category_as_str=None):
    trace_events_dumper = TraceEventsDumper(category_times, json_path, category_as_str=category_as_str)
    trace_events_dumper.dump(print_log)

class TraceEventsDumper:
    def __init__(self, category_times, json_path,
                 category_as_str=None,
                 debug=False):
        self.category_times = category_times
        self.json_path = json_path
        self.reproduce_tfprof = False
        self.category_as_str = category_as_str
        self.debug = debug
        self.reset()

    def dump(self, print_log=True):
        self.reset()
        self.js_add_category_times(self.category_times)
        if print_log:
            logger.info("> Write traceEvents to: {path}".format(path=self.json_path))

        do_dump_json(self.js, self.json_path, cls=DecimalEncoder)

    def _cat(self, category):
        if self.category_as_str is None:
            return category
        return self.category_as_str(category)

    def js_add_category_times(self, category_times):
        """
        self.js = {
            "traceEvents": [
                {
                    "args": {
                        "name": "Op scheduling threads: /job:localhost/replica:0/task:0/device:gpu:0"
                    },
                    "name": "process_name",
                    "ph": "M",
                    "pid": 0
                },
                {
                    "args": {
                        "name": "Op scheduling threads: /job:localhost/replica:0/task:0/device:cpu:0"
                    },
                    "name": "process_name",
                    "ph": "M",
                    "pid": 1
                },
                {
                    "args": {
                        "name": "Op execution threads: /device:gpu:0/stream:all"
                    },
                    "name": "process_name",
                    "ph": "M",
                    "pid": 2
                },
                {
                    "args": {
                        "name": "deepq/target_q_func/action_value/fully_connected_1/biases"
                    },
                    "cat": "Op",
                    "dur": 138,
                    "name": "deepq/target_q_func/action_value/fully_connected_1/biases",
                    "ph": "X",
                    "pid": 0,
                    "tid": 0,
                    "ts": 1546471006818165
                },
            ...
        }
        """
        self.js = {
            "traceEvents": [
            ],
        }

        for category, times in category_times.items():
            # if category == constants.CATEGORY_CUDA_API_CPU and type(category_times[constants.CATEGORY_CUDA_API_CPU]) == dict:
            cat = self._cat(category)
            if type(category_times[category]) == dict:
                # category_times[constants.CATEGORY_CUDA_API_CPU] : device -> (tid -> times)
                # assert type(category_times[constants.CATEGORY_CUDA_API_CPU]) == dict
                for device, all_times in category_times[category].items():
                    category_name = "{cat}: {dev}".format(cat=cat, dev=device)
                    self.js_add_section(category_name)
                    tid_to_times = self.js_split_into_tids(all_times)
                    for tid, times in tid_to_times.items():
                        for ktime in times:
                            name = self._ktime_name(ktime)
                            self.js_add_time(name, category_name, ktime, tid)
            # elif category == constants.CATEGORY_GPU:
            #     ...
            elif self.reproduce_tfprof and category in [constants.CATEGORY_PYTHON, constants.CATEGORY_TF_API]:
                # Ignore pyprof for now.
                # We just want to test that we are properly reproducing the tfprof traceEvents json.
                pass
            else:
                assert type(category_times[category]) == list
                category_name = self._cat(category)
                # category_name = category
                self.js_add_section(category_name)
                for ktime in times:
                    name = self._ktime_name(ktime)
                    self.js_add_time(name, category_name, ktime, tid=0)
            # else:
            #     raise NotImplementedError("Not sure how to handle category={c}".format(
            #         c=category))

    def _ktime_name(self, ktime):
        if hasattr(ktime, 'event_id') and ktime.name is not None:
            return "{id}: {name}".format(
                id=ktime.event_id,
                name=ktime.name)
        elif ktime.name is not None:
            return ktime.name
        return 'unknown'

    def js_split_into_tids(self, times):
        times = sorted(times, key=lambda ktime: ktime.start_time_usec)
        # tid -> times
        next_tid = 0
        allocated_tids = []
        tid_times = dict()

        for ktime in times:
            tid = -1
            # c_tid = candidate tid
            for c_tid in reversed_iter(allocated_tids):
                if ktime.start_time_usec < tid_times[c_tid][-1].end_time_usec:
                    # Case (1): kernel start-time < lane's last end-time
                    #   Kernel starts earlier in this lane.
                    #   Either try the next lane, or allocate a new lane
                    pass
                elif ktime.start_time_usec >= tid_times[c_tid][-1].end_time_usec:
                    # Case (2): kernel start-time > lane's last end-time
                    #   Kernel starts later in this lane.
                    #   Place the kernel in this lane.
                    tid = c_tid
                    break
                # else:
                #     # Case (3): lane's last end-time == kernel's start-time.
                #     #   The current kernel starts executing right as the last kernel in this lane ends.
                #     #   Look at the kernel from this lane that execute before the current one.
                #     #   Q: This seems pointless, seems like we're going to fall into case 2.

            if tid == -1:
                tid = next_tid
                allocated_tids.append(tid)
                next_tid += 1
                tid_times[tid] = []

            tid_times[tid].append(ktime)

        return tid_times

    def js_add_section(self, name):
        if name in self.category_to_pid:
            return
        pid = self._js_allocate_pid()
        section = {
            'args': {
                'name': name,
            },
            'name': 'process_name',
            'ph': 'M',
            'pid': pid,
        }
        assert name not in self.category_to_pid
        self.category_to_pid[name] = pid
        self.js['traceEvents'].append(section)

    def js_add_time(self, name, category, ktime, tid):
        pid = self.category_to_pid[category]
        jtime = {
            'args': {
                'name': name,
            },
            'cat': 'Op',
            'dur': ktime.total_time_usec,
            'name': name,
            'ph': 'X',
            'pid': pid,
            'tid': tid,
            'ts': ktime.start_time_usec,
        }
        self.js['traceEvents'].append(jtime)

    def _js_allocate_pid(self):
        pid = self._next_pid
        self._next_pid += 1
        return pid

    def reset(self):
        self._next_pid = 0
        self.category_to_pid = dict()
