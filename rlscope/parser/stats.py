"""
Plain-old-data classes for representing start/end timestamps.
"""
# pip install progressbar2
from rlscope.profiler.rlscope_logging import logger
import progressbar

from rlscope.parser.common import *
from rlscope.parser import constants

VARIABLE_HEADER = ['Type', 'Time(%)',
                   'Avg', 'Std', 'Std/Avg(%)',
                   # The n-th call to this function during a repetition.
                   'Call#', 'Name',
                   'Sample#',
                   'Time',
                   # 'Time', 'Calls',
                   # 'Min', 'Max',
                   ]

SEPARATE_CALLS_HEADER = ['Type', 'Time(%)',
                         'Avg', 'Std', 'Std/Avg(%)',
                         # The n-th call to this function during a repetition.
                         'Call#', 'Name',
                         'Time', 'Calls',
                         'Min', 'Max',
                         ]

KERNEL_TIME_OVERWRITE_NONE = {'process_name'}

class KernelTime:
    """
    KernelTime(time_usec=..., start_usec=...)
    KernelTime(start_usec=..., end_usec=...)
    """
    def __init__(self, time_usec=None, start_usec=None, end_usec=None, name=None, process_name=None, create_from=None,
                 # Swallow any excess arguments
                 **kwargs):
        self.name = name
        self.process_name = process_name
        if time_usec is None:
            # KernelTime(start_usec=..., end_usec=...)
            self.start_usec = start_usec
            self.end_usec = end_usec
            self.time_usec = self.end_usec - self.start_usec
        elif end_usec is None:
            # KernelTime(time_usec=..., start_usec=...)
            self.start_usec = start_usec
            self.time_usec = time_usec
            self.end_usec = self.start_usec + self.time_usec
        else:
            self.start_usec = start_usec
            self.end_usec = end_usec
            self.time_usec = time_usec
            assert self.time_usec == self.end_usec - self.start_usec

        # check_no_decimal(self.start_usec)
        # check_no_decimal(self.end_usec)
        # check_no_decimal(self.time_usec)

        if create_from is not None:
            self._get_attrs(create_from, overwrite_none=KERNEL_TIME_OVERWRITE_NONE)

    def _get_attrs(self, ktime, overwrite_none):
        """
        OpStack.get_absorbed_ops/split_op_stacks creates new events
        by "splitting" an event into multiple smaller events.

        Inherit attributes from the original event when splitting;
        e.g. process_name, phase_name

        :param ktime:
            The event we are "splitting from"
        :return:
        """
        for key, value in ktime.__dict__.items():
            if not hasattr(self, key) or ( key in overwrite_none and getattr(self, key) is None ):
                setattr(self, key, value)

    def copy(self):
        ktime = KernelTime(start_usec=self.start_usec, end_usec=self.end_usec, name=self.name, create_from=self)
        return ktime

    def subsumes(self, op2):
        op1 = self
        # return op1.start_time_usec < op2.start_time_usec < op2.end_time_usec < op1.end_time_usec
        return op1.start_time_usec <= op2.start_time_usec <= op2.end_time_usec <= op1.end_time_usec

    def equals(self, op2):
        op1 = self
        return op1.start_time_usec == op2.start_time_usec and \
               op1.end_time_usec == op2.end_time_usec

    def set_start_end(self, start_usec, end_usec):
        self.start_usec = start_usec
        self.end_usec = end_usec
        self.time_usec = self.end_usec - self.start_usec
        # check_no_decimal(self.start_usec)
        # check_no_decimal(self.end_usec)
        # check_no_decimal(self.time_usec)

    def set_end(self, end_usec):
        self.end_usec = end_usec
        self.time_usec = self.end_usec - self.start_usec
        # check_no_decimal(self.end_usec)
        # check_no_decimal(self.time_usec)

    @property
    def start_time_usec(self):
        assert self.start_usec is not None
        return self.start_usec

    @property
    def end_time_usec(self):
        assert self.end_usec is not None
        return self.end_usec

    @property
    def total_time_usec(self):
        return self.end_time_usec - self.start_time_usec

    @property
    def total_time_sec(self):
        diff = self.end_time_usec - self.start_time_usec
        NumberType = type(diff)
        time_sec = diff / NumberType(1e6)
        return time_sec

    def overlaps(self, ktime_b):
        ktime_a = self
        assert ktime_a.start_usec <= ktime_b.start_usec
        # return ktime_a.end_usec > ktime_b.start_usec
        return ktime_a.end_usec >= ktime_b.start_usec

    def overlap(self, ktime_b):
        assert self.overlaps(ktime_b)
        ktime_a = self
        return ktime_b.start_usec - ktime_a.end_usec

    def is_before(self, ktime_b):
        return self.start_time_usec <= ktime_b.start_time_usec
        # return self.end_time_usec <= ktime_b.start_time_usec

    def is_after(self, ktime_b):
        return ktime_b.is_before(self)

    def __eq__(self, ktime_b):
        ktime_a = self
        return ktime_a.start_usec == ktime_b.start_usec and \
               ktime_a.end_usec == ktime_b.end_usec

    def merge(self, ktime_b, name=None):
        """
        a.k.a. Union
        """
        assert self.overlaps(ktime_b)
        assert self.is_before(ktime_b)
        start_usec = self.start_time_usec
        end_usec = ktime_b.end_time_usec
        if name is None:
            # Q: Should we assert that self.name == ktime_b.name?
            name = self.name
        return KernelTime(end_usec - start_usec, start_usec, end_usec, name=name)

    def __str__(self):
        # return str(self.__dict__)
        if self.name is not None:
            return "(name={name}, start={start} us, dur={dur} us)".format(
                name=self.name, start=self.start_usec, dur=self.time_usec)
        # return "(start={start} us, dur={dur} us)".format(
        #     start=self.start_usec, dur=self.time_usec)
        return "(start={start} us, end={end} us)".format(
            start=self.start_usec, end=self.end_usec)

    def __repr__(self):
        return str(self)

    def intersect(self, ktime_b, name=None):
        assert self.overlaps(ktime_b)
        assert self.is_before(ktime_b)
        start_usec = ktime_b.start_time_usec
        end_usec = min(self.end_time_usec, ktime_b.end_time_usec)
        if name is None:
            name = self.name
        return KernelTime(end_usec - start_usec, start_usec, end_usec, name=name)

    @staticmethod
    def from_row(row):
        return obj_from_row(KernelTime, row)

#
# Category time format.
#

def category_times_add_time(category_times, device, ktime, group_by_device, category=None):
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

    if category is None and device is not None:
        category = get_category_from_device(device)
    assert category is not None

    if category in [constants.CATEGORY_CUDA_API_CPU]:
        group_by_device = group_by_device
    else:
        group_by_device = False

    _add_time(category, group_by_device)

class Stat:
    """
    Compute either the sum/min/avg/stdev of calls to a function.
    """
    def __init__(self, name, discard_first_sample, debug=False):
        self.kernel_times = []
        # self.num_calls = num_calls
        # self._call_num = 0
        self.num_calls = None
        self.debug = debug
        self.name = name
        self.discard_first_sample = discard_first_sample
        self._has_split = False

    def add(self, time_usec=None, start_usec=None, end_usec=None):
        assert self.num_calls is None
        self.kernel_times.append(KernelTime(time_usec, start_usec, end_usec, name=self.name))

    def add_ktime(self, ktime):
        assert type(ktime) == KernelTime
        assert ktime.name == self.name
        self.kernel_times.append(ktime)

    def add_time_sec(self, time_sec):
        time_usec = sec_to_us(time_sec)
        self.kernel_times.append(KernelTime(time_usec, name=self.name))

    def add_times_sec(self, times_sec):
        for time_sec in times_sec:
            self.add_time_sec(time_sec)

    def iteration_times_sec(self, num_calls):
        """
        :param num_calls:
            Total number of iterations (iters*reps)
        :return:
        """

        total_iterations = self.get_total_iterations(num_calls)
        # n_calls = len(self.kernel_times)
        # if n_calls % num_calls != 0:
        # if self.num_diff_calls == 1:
        if self.num_calls == 1:
            # Evenly divide the total time spent calling this function across all num_calls iterations.
            ret = np.empty(total_iterations)
            fill = (self.sum()/len(ret))/constants.MICROSECONDS_IN_SECOND
            ret.fill(fill)
            return ret

        # Say we have num_calls=1000 calls to Forward, but we have 22000 calls to this function.
        # Then each Forward has num_diff_calls=22 calls to this function.
        # We want to sum up those 22 calls (for each call to Forward) to get an array of 1000.
        ret = np.zeros(total_iterations)
        for call_idx in range(self.num_diff_calls):
            times_sec = self.times_sec(call_idx)
            if len(times_sec) != len(ret):
                import pdb; pdb.set_trace()
            ret += times_sec

        return ret

    def all_calls_during(self, call_num):
        """
        Return all the (e.g. 22) times this API call was made during the <call_num>-th call to Forward.

        :param call_num:
            call_num = One of the times Forward was called.
            call_num = 1...1000 if iterations*repetitions = 1000
        """
        assert 0 <= call_num < self.num_calls
        times = []
        for call_idx in range(self.num_diff_calls):
            times.append(self.kernel_times[call_idx][call_num])
        return times

    def split(self, num_calls):
        """
        Q: Am I dividing up calls correctly?
        If we call cudaLaunch 5 times for each iteration, and the number of iterations/calls is 1000, calls will look like:
        INITIALLY, kernel_times contains every time the function was ever called.

            kernel_times[0..4999] = [
                # First iteration…
                cudaLaunch[i=0]
                cudaLaunch[i=1]
                cudaLaunch[i=2]
                cudaLaunch[i=3]
                cudaLaunch[i=4]
                # Next iteration…
                cudaLaunch[i=5]
                cudaLaunch[i=6]
                cudaLaunch[i=7]
                cudaLaunch[i=8]
                cudaLaunch[i=9]
                ...
                # 5 * 1000 = 5000 calls in total
            ]

        Once we find out the number of iterations (num_calls) is 1000, we discover num_diff_calls=5000/1000 = 5.
        We use this to split up the times into calls with the same arguments:

            kernel_times[0..4] = [
                [
                    # Take the 1st call from each iteration-group
                    cudaLaunch[i=0],
                    cudaLaunch[i=5],
                    ...
                    # 1000 calls to cudaLaunch with the same arguments
                ]
                [
                    # Take the 2nd call from each iteration-group
                    cudaLaunch[i=1],
                    cudaLaunch[i=6],
                    ...
                ]
                ...
                [
                    # Take the 5th call from each iteration-group
                    cudaLaunch[i=4],
                    cudaLaunch[i=9],
                    ...
                ]
                # 5 "different" calls to cudaLaunch during a single call to Forward.
            ]

        Converts kernel_times from:
            List[KernelTime] -> List[List[KernelTime]]

        :param num_calls:
        The number of times (e.g.) Forward was called.
        num_calls = iterations * repetitions
        :return:
        """
        # logger.info("> Split func={name}".format(
        #     name=self.name))
        assert not self._has_split

        # Converts kernel_times from:
        #     List[KernelTime] -> List[List[KernelTime]]
        assert type(self.kernel_times) == list and \
               type(self.kernel_times[0]) == KernelTime

        # n_calls = the number of times this function was called.
        n_calls = len(self.kernel_times)
        # if self.discard_first_sample:
        #     n_calls = n_calls - 1

        if n_calls % num_calls != 0:
            # Number of calls to this function isn't divisible by the number
            # of times we expect it to have been called (num_calls = iterations*repetitions);
            # instead, just make num_calls = 1.
            if self.debug:
                logger.info("[n_calls={n_calls}, num_calls={num_calls}] Use num_calls=1 for function={name}".format(
                    n_calls=n_calls,
                    num_calls=num_calls,
                    name=self.name))
            kernel_times = [self.kernel_times]
            self.kernel_times = kernel_times
            self.num_calls = 1
            self.num_diff_calls = 1
            self.not_divisible = True

        else:

            # num_diff_calls = # of times this function is called (likely with different arguments)
            #                  during a single call to Forward
            self.num_diff_calls = int(n_calls / num_calls)

            if self.debug:
                logger.info("[n_calls={n_calls}, num_calls={num_calls}] Use num_calls={num_calls} for function={name}".format(
                    n_calls=n_calls,
                    num_calls=num_calls,
                    name=self.name))
            self.num_calls = num_calls
            self.not_divisible = False
            kernel_times = [[] for i in range(self.num_diff_calls)]
            for i, kt in enumerate(self.kernel_times):
                # Q: Initially, kernel_times[0...num_diff_calls] are different calls to the same function,
                # for one call to Forward.
                call_idx = i % self.num_diff_calls
                kernel_times[call_idx].append(kt)
            self.kernel_times = kernel_times

        self._has_split = True

        # Converts kernel_times from:
        #     List[KernelTime] -> List[List[KernelTime]]
        assert type(self.kernel_times) == list and \
               type(self.kernel_times[0]) == list and \
               type(self.kernel_times[0][0]) == KernelTime

    def _check_call_idx(self, call_idx):
        # if call_idx is None:
        #     assert self.num_calls is None
        # else:
        if call_idx is not None:
            assert call_idx < self.num_diff_calls

    def _maybe_discard(self, times):
        if self.discard_first_sample and len(times) > 1:
            return times[1:]
        return times

    def times_sec(self, call_idx=None):
        # JAMES TODO: This looks wrong with clock_monotonic...when did this make any sense?
        time_usecs = self._times_usec(call_idx)
        return [usec/constants.MICROSECONDS_IN_SECOND for usec in time_usecs]

    def _times_usec(self, call_idx=None):
        self._check_call_idx(call_idx)

        """
        self.kernel_times:
        
        num_calls = the number of times Forward was called (iterations*repetitions)
        
        Before calling self.split(num_calls):
            self.num_calls = None
            kernel_times = [
                all the times this API call was ever made, across all 1000 iterations*repetitions calls to Forward
            ].
        
        After call self.split(num_calls):
            self.num_calls = num_calls
            # kernel_times = List[List[KernelTime]]?
            # But I see kernel_times = List[List[List[KernelTime]]]?
            kernel_times = [
                [the 1st time the API call was made, across all 1000 iterations*repetitions calls to Forward],
                [the 2nd time the API call was made, across all 1000 iterations*repetitions calls to Forward],
                ...
                [the <num_diff_calls>th time the API call was made ....],
            ]
        NOTE: 
        """
        if self.num_calls is None:
            assert call_idx is None
            times = self._maybe_discard([kt.time_usec for kt in self.kernel_times])
        elif call_idx is not None:
            times = self._maybe_discard([kt.time_usec for kt in self.kernel_times[call_idx]])
        else:
            # Return min/max/avg/etc over ALL calls.
            # Useful when we want to do kt.sum().
            times = []
            for kts in self.kernel_times:
                for kt in self._maybe_discard(kts):
                    times.append(kt.time_usec)

        return times

    @property
    def total_iterations(self):
        if self.discard_first_sample:
            return self.num_calls - 1
        return self.num_calls

    def get_total_iterations(self, num_calls):
        if num_calls == 1:
            return num_calls
        if self.discard_first_sample:
            return num_calls - 1
        return num_calls

    def n_calls(self, call_idx=None):
        """
        Total number of times this function was called.
        :return:
        """
        self._check_call_idx(call_idx)

        n_calls = 0
        if self.num_calls is None:
            assert call_idx is None
            n_calls = len(self._maybe_discard(self.kernel_times))
        elif call_idx is not None:
            n_calls = len(self._maybe_discard(self.kernel_times[call_idx]))
        else:
            # Return min/max/avg/etc over ALL calls.
            # Useful when we want to do kt.sum().
            for kts in self.kernel_times:
                n_calls += len(self._maybe_discard(kts))

        return n_calls

    def avg(self, call_idx=None):
        self._check_call_idx(call_idx)
        mean_usec = np.mean(self._times_usec(call_idx))
        return mean_usec

    def std(self, call_idx=None):
        self._check_call_idx(call_idx)
        std_usec = np.std(self._times_usec(call_idx))
        return std_usec

    def sum(self, call_idx=None):
        self._check_call_idx(call_idx)
        sum_usec = sum(self._times_usec(call_idx))
        return sum_usec

    def min(self, call_idx=None):
        self._check_call_idx(call_idx)
        min_usec = min(self._times_usec(call_idx))
        return min_usec

    def max(self, call_idx=None):
        self._check_call_idx(call_idx)
        max_usec = max(self._times_usec(call_idx))
        return max_usec

    def calls(self, call_idx=None):
        self._check_call_idx(call_idx)
        return len(self._times_usec(call_idx))

    def dump(self, writer, summary_type, header, total_time, profile_data_type):
        if summary_type == 'nvprof':
            self.dump_nvprof(writer, header, total_time, profile_data_type)
        elif summary_type == 'separate_calls':
            self.dump_separate_calls(writer, header, total_time, profile_data_type)
        else:
            raise NotImplementedError

    def _ptime(self, usec):
        return pretty_time(time_sec=us_to_sec(usec), use_space=False)

    def _percent(self, percent):
        return "{percent:.2f}%".format(percent=100.*percent)

    def dump_variable(self, variable_writer, total_time, profile_data_type):
        """
        If Time(%) >= 1% and Std/Avg > 50%:
          Then report individual timings to nvprof.pretty.variable.txt:
          Same columns as before, but add Sample# column that goes from 1..1000
        """
        for call_idx in range(self.num_diff_calls):
            avg = self.avg(call_idx)
            std = self.std(call_idx)
            sm = self.sum(call_idx)
            time_percent = 100.*sm/total_time
            std_avg_percent = 100.*std/avg
            if time_percent >= 1. and std_avg_percent >= 50.:
                for sample_idx, time_usec in enumerate(self._times_usec(call_idx)):
                    row = {
                        'Type':profile_data_type,
                        'Time(%)':self._percent(sm/total_time),
                        'Std/Avg(%)':self._percent(std/avg),
                        'Time/Avg(%)':self._percent(time_usec/avg),
                        # 'Time':self._ptime(sm),
                        'Call#':call_idx,
                        'Name':self.name,
                        'Sample#':sample_idx,
                        'Time':self._ptime(time_usec),
                        'Avg':self._ptime(avg),
                        'Std':self._ptime(std),
                        # 'Calls':self.calls(call_idx),
                        # 'Avg':self._ptime(self.avg(call_idx)),
                        # 'Std':self._ptime(self.std(call_idx)),
                        # 'Std/Avg(%)':self._percent(self.std(call_idx)/self.avg(call_idx)),
                        # 'Min':self._ptime(self.min(call_idx)),
                        # 'Max':self._ptime(self.max(call_idx)),
                    }
                    variable_writer.writerow([row[k] for k in VARIABLE_HEADER])

    def dump_separate_calls(self, writer, header, total_time, profile_data_type):
        # Q: How to handle diff calls?
        # A:
        # # (a) create a Stat object for each Call#, and output a row for it.
        # # (b) split self.kernel_times for each Call#;
        #       adjust avg/sum/etc. to take a Call# index that determines over which calls to compute the statistic.
        #       In this case, the regular nvprof output is a special-case where the Call# is always 1.
        assert self.num_diff_calls is not None
        for call_idx in range(self.num_diff_calls):
            row = {
                'Type':profile_data_type,
                'Time(%)':self._percent(self.sum(call_idx)/total_time),
                'Time':self._ptime(self.sum(call_idx)),
                'Calls':self.calls(call_idx),
                'Avg':self._ptime(self.avg(call_idx)),
                'Std':self._ptime(self.std(call_idx)),
                'Std/Avg(%)':self._percent(self.std(call_idx)/self.avg(call_idx)),
                'Min':self._ptime(self.min(call_idx)),
                'Max':self._ptime(self.max(call_idx)),
                'Call#':call_idx,
                'Name':self.name,
            }
            writer.writerow([row[k] for k in header])

    def dump_nvprof(self, writer, header, total_time, profile_data_type):
        row = {
            'Type':profile_data_type,
            'Time(%)':"{percent:.2f}%".format(percent=100.*(self.sum()/total_time)),
            'Time':self._ptime(self.sum()),
            'Calls':self.calls(),
            'Avg':self._ptime(self.avg()),
            'Min':self._ptime(self.min()),
            'Max':self._ptime(self.max()),
            'Name':self.name,
        }
        writer.writerow([row[k] for k in header])

class KernelStat(Stat):
    def __init__(self, name, discard_first_sample, debug=False):
        super().__init__(name, discard_first_sample, debug=debug)

class Stats:
    def __init__(self, discard_first_sample, debug=False, name=None, has_overlap=True):
        self.discard_first_sample = discard_first_sample
        self.name_to_stat = dict()
        self.num_calls = None
        self.debug = debug
        self.name = name
        self.has_overlap = has_overlap

    def sum_calls_sec(self):
        """
        Returns total_times[call_num]

        for call_num = 1...1000 if Forward is called iterations*repetitions=1000 times
        """
        assert self.num_calls is not None
        total_times = np.zeros(self.num_calls)

        equally_divided_times = np.zeros(self.num_calls)
        api_calls_not_divisible = 0
        api_calls_divisible = 0
        for stat in self.stats:
            # If this is a call that isn't divisible by num_calls, then instead,
            # we'd like to evenly divide its time across each iteration.
            #
            # TODO: we should really make sure this isn't a large contributor...
            if stat.not_divisible:
                api_calls_not_divisible += 1
                total_time_sec = np.sum(stat.times_sec())
                equally_divided_times += total_time_sec/self.num_calls

        for call_num in range(self.num_calls):
            ktimes = []
            for stat in self.stats:
                if stat.not_divisible:
                    continue
                ktimes.extend(stat.all_calls_during(call_num))
            api_calls_divisible = len(ktimes)
            ktimes.sort(key=lambda k: k.start_usec)
            total_time = 0
            for ktime_a, ktime_b in zip(ktimes, ktimes[1:]):
                if ktime_a.overlaps(ktime_b):
                    # This warning goes off a LOT for CUDA API stuff.
                    #
                    # if not self.has_overlap:
                    #     logger.info(textwrap.dedent("""
                    #     WARNING: ktime_a={a} overlaps ktime_b={b}
                    #     > {a}
                    #       start = {a_start}
                    #       end = {a_end}
                    #     > {b}
                    #       start = {b_start}
                    #       end = {b_end}
                    #     """.format(
                    #         a=ktime_a.name,
                    #         a_start=ktime_a.start_usec,
                    #         a_end=ktime_a.end_usec,
                    #         b=ktime_b.name,
                    #         b_start=ktime_b.start_usec,
                    #         b_end=ktime_b.end_usec,
                    #     )))
                    overlap = ktime_a.overlap(ktime_b)
                    total_time += ktime_a.time_usec - overlap
                else:
                    total_time += ktime_a.time_usec
            if len(ktimes) > 0:
                total_time += ktimes[-1].time_usec

            total_times[call_num] = total_time

        if self.debug:
            logger.info(textwrap.dedent("""
            > {name} stats:
              num_calls = {num_calls}
              api_calls.not_divisible = {not_divisible}
              api_calls.divisible = {divisible}
            """.format(
                name=self.name,
                num_calls=self.num_calls,
                not_divisible=api_calls_not_divisible,
                divisible=api_calls_divisible,
            )))


        return total_times/constants.MICROSECONDS_IN_SECOND

    def sum_calls_sec_no_overlap(self, check_overlap=False):
        """
        Returns total_times[call_num]

        for call_num = 1...1000 if Forward is called iterations*repetitions=1000 times
        """

        if check_overlap:
            self.check_overlap()

        total_times = np.zeros(self.num_calls)
        for stat in self.stats:
            times_sec = stat.iteration_times_sec(self.num_calls)
            total_times += times_sec
        return total_times

    def check_overlap(self):
        """
        Make sure the times that get summed together for a given call_num do NOT overlap with each other.
        This could happen if, for example, a CUDA kernel and cuda Memcpy run at the same time!
        :return:
        """

        # Looks like overlap CAN happen, so we must account for it:
        #
        # ipdb> pp ktime_a.__dict__
        # {'end_usec': 1542751119289568.2,
        #  'name': 'void cudnn::detail::implicit_convolve_sgemm<float, float, 128, 5, 5, '
        #          '3, 3, 3, 1, true, false, true>(int, int, int, float const*, int, '
        #          'float*, float*, kernel_conv_params, int, float, float, int, float*, '
        #          'float*, int, int)',
        #  'start_usec': 1542751119289384.0,
        #  'time_usec': 184.126}
        # ipdb> pp ktime_b.__dict__
        # {'end_usec': 1542751119289412.8,
        #  'name': '[CUDA memcpy HtoD]',
        #  'start_usec': 1542751119289411.8,
        #  'time_usec': 0.992}

        def overlaps(ktime_a, ktime_b):
            # ktime_a.end_usec and ktime_b.start_usec CAN be equal:
            #
            # e.g.
            # ipdb> pp ktime_a.__dict__
            # {'end_usec': 1542751117249387.2,
            #  'name': 'void '
            #          'Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long '
            #          'long, 1, 1, int>, 16, Eigen::MakePointer>, '
            #          'Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, '
            #          '1, int>, 16, Eigen::MakePointer> const, '
            #          'Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, int>, 16, '
            #          'Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long '
            #          'const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, '
            #          'Eigen::GpuDevice>, '
            #          'int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long '
            #          'long, 1, 1, int>, 16, Eigen::MakePointer>, '
            #          'Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, '
            #          '1, int>, 16, Eigen::MakePointer> const, '
            #          'Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, int>, 16, '
            #          'Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long '
            #          'const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, '
            #          'Eigen::GpuDevice>, int)',
            #  'start_usec': 1542751117249385.0,
            #  'time_usec': 2.176}
            # ipdb> pp ktime_b.__dict__
            # {'end_usec': 1542751117249388.5,
            #  'name': '[CUDA memcpy DtoH]',
            #  'start_usec': 1542751117249387.2,
            #  'time_usec': 1.312}
            assert ktime_a.start_usec <= ktime_b.start_usec
            return ktime_a.overlaps(ktime_b)

        def _check_overlap(kernel_times):
            for k in kernel_times:
                assert k.start_usec is not None and k.end_usec is not None

            sorted_ktimes = sorted(kernel_times, key=lambda k: k.start_usec)
            for ktime_a, ktime_b in zip(sorted_ktimes, sorted_ktimes[1:]):
                assert not overlaps(ktime_a, ktime_b)

        assert self.num_calls is not None
        for call_num in range(self.num_calls):
            all_times = []
            for stat in self.stats:
                all_times.extend(stat.all_calls_during(call_num))
            _check_overlap(all_times)

    @property
    def total_iterations(self):
        if self.discard_first_sample:
            return self.num_calls - 1
        return self.num_calls

    def split(self, num_calls):
        self.num_calls = num_calls
        with progressbar.ProgressBar(max_value=len(self.name_to_stat.values())) as bar:
            for i, kt in enumerate(self.name_to_stat.values()):
                kt.split(num_calls)
                bar.update(i)

    def add(self, name,
            time_usec=None, start_usec=None, end_usec=None):
        # assert (start_usec is None and end_usec is None) or \
        #        (start_usec is not None and end_usec is not None)
        kt = self._get_stat(name)
        kt.add(time_usec=time_usec, start_usec=start_usec, end_usec=end_usec)

    def add_ktime(self, ktime):
        assert type(ktime) == KernelTime
        kt = self._get_stat(ktime.name)
        kt.add_ktime(ktime)

    def _get_stat(self, name):
        if name not in self.name_to_stat:
            self.name_to_stat[name] = KernelStat(name, self.discard_first_sample, debug=self.debug)
        return self.name_to_stat[name]

    def add_times_sec(self, name, times_sec):
        stat = self._get_stat(name)
        stat.add_times_sec(times_sec)

    def add_time_sec(self, name, time_sec):
        stat = self._get_stat(name)
        stat.add_time_sec(time_sec)

    def dump(self, f, profile_data_type, skip_header=False, summary_type='nvprof'):
        if summary_type == 'nvprof':
            self.dump_nvprof(f, profile_data_type, skip_header)
        elif summary_type == 'separate_calls':
            self.dump_separate_calls(f, profile_data_type, skip_header)
        else:
            raise NotImplementedError

    def dump_nvprof(self, f, profile_data_type, skip_header=False):
        """
        Dump the same "summary" that nvprof outputs.
        """
        writer = csv.writer(f, delimiter='|')
        stats = sorted(self.name_to_stat.values(), key=lambda kt: -1*kt.sum())
        header = ['Type', 'Time(%)', 'Time', 'Calls', 'Avg', 'Min', 'Max', 'Name']
        if not skip_header:
            writer.writerow(header)
        total_time = self.total_time()
        for kt in stats:
            kt.dump(writer, 'nvprof', header, total_time, profile_data_type)

    def dump_variable(self, f, profile_data_type, skip_header=False):
        variable_writer = csv.writer(f, delimiter='|')
        stats = sorted(self.name_to_stat.values(), key=lambda kt: -1*kt.sum())
        # Q: What do we really want to know?
        # A: What's the mean/stdev time spent in each function called during a single Forward call?
        # To answer this, we need to separate the profile output into the n-th calls to the function.
        # If 'Calls' isn't divisible by num_calls, either:
        # (a) remove it entirely from the profile output, or [ PROBLEM: it might take up a LOT of time ]
        # (b) treat it as a function that gets called once
        #     Call# = 0 or N/A?
        #     Stdev = stdev of all n-calls (stdev will be big)
        #     ^^^ I would like to show this, since if the Stdev is tiny, we don't care; if it's big, we care.
        # (c) let Call# = N/A
        #         Stdev = N/A
        #     ^^^ This is more accuracte.
        #
        # PSEUDOCODE:
        # num_calls = the total number of times Forward was called.
        if not skip_header:
            variable_writer.writerow(VARIABLE_HEADER)
        total_time = self.total_time()
        for kt in stats:
            kt.dump_variable(variable_writer, total_time, profile_data_type)

    @property
    def stats(self):
        return self.name_to_stat.values()

    @property
    def ordered_stats(self):
        stats = sorted(self.name_to_stat.values(), key=lambda kt: -1*kt.sum())
        return stats

    def total_time(self):
        total_time = 0
        for kt in self.name_to_stat.values():
            total_time += kt.sum()
        return total_time

    def dump_separate_calls(self, f, profile_data_type, skip_header=False):
        """
        Dump the same "summary" that nvprof outputs.
        """
        writer = csv.writer(f, delimiter='|')
        stats = sorted(self.name_to_stat.values(), key=lambda kt: -1*kt.sum())
        # Q: What do we really want to know?
        # A: What's the mean/stdev time spent in each function called during a single Forward call?
        # To answer this, we need to separate the profile output into the n-th calls to the function.
        # If 'Calls' isn't divisible by num_calls, either:
        # (a) remove it entirely from the profile output, or [ PROBLEM: it might take up a LOT of time ]
        # (b) treat it as a function that gets called once
        #     Call# = 0 or N/A?
        #     Stdev = stdev of all n-calls (stdev will be big)
        #     ^^^ I would like to show this, since if the Stdev is tiny, we don't care; if it's big, we care.
        # (c) let Call# = N/A
        #         Stdev = N/A
        #     ^^^ This is more accuracte.
        #
        # PSEUDOCODE:
        # num_calls = the total number of times Forward was called.
        if not skip_header:
            writer.writerow(SEPARATE_CALLS_HEADER)
        total_time = 0
        for kt in stats:
            total_time += kt.sum()
        for kt in stats:
            kt.dump(writer, 'separate_calls', SEPARATE_CALLS_HEADER, total_time, profile_data_type)


