"""
Utilities for recording data for unit-tests, and for reading in data for checking unit-test success/failure.

.. deprecated:: 1.0.0
    We never used this; instead we use pytest for unit testing.
"""
#
#

from rlscope.profiler.rlscope_logging import logger
from os.path import join as _j

from rlscope.protobuf.unit_test_pb2 import IMLUnitTestOnce, IMLUnitTestMultiple

from rlscope.parser.common import *
from rlscope.profiler import timer as rlscope_timer
# from rlscope.profiler.clib_wrap import MICROSECONDS_IN_SECOND

from rlscope.profiler import proto_util


#
# Reading in and checking unit-test success/failure.
# Used offline (NOT while running ML script).
#

class TestData:
    def __init__(self, debug=False):
        self.process_data = dict()
        self.test_name = None
        self.directory = None
        self.debug = debug

    def merge_proto(self, proto):
        if proto.process_name not in self.process_data:
            self.process_data[proto.process_name] = ProcessTestData(debug=self.debug)

        if isinstance(proto, IMLUnitTestOnce):
            self.process_data[proto.process_name].merge_proto_once(proto)
        elif isinstance(proto, IMLUnitTestMultiple):
            self.process_data[proto.process_name].merge_proto_multiple(proto)
        else:
            raise NotImplementedError

    def read_directory(self, directory):
        assert self.directory is None
        self.directory = directory
        if self.debug:
            logger.info("> Read {klass} rooted at {dir}".format(
                klass=self.__class__.__name__,
                dir=self.directory))
        for dirpath, dirnames, filenames in os.walk(self.directory):
            for base in filenames:
                path = _j(dirpath, base)
                if is_unit_test_multiple_file(path):
                    if self.debug:
                        logger.info("  Read {path}".format(path=path))
                    proto = read_unit_test_multiple_file(path)
                    self.merge_proto(proto)
                elif is_unit_test_once_file(path):
                    if self.debug:
                        logger.info("  Read {path}".format(path=path))
                    proto = read_unit_test_once_file(path)
                    self.merge_proto(proto)

    @property
    def processes(self):
        """
        Return processes, sorted by process.start_time_us.
        :return:
        """
        procs = sorted(self.process_data.values(), key=lambda ptd: ptd.process_start_time_sec)
        return procs

    def get_process(self, process_name):
        return self.process_data[process_name]

    def as_printable(self):
        """
        TestData:
            test_name = <test>

            (sorted by process_start_time_sec)
            processes (n):
                ProcessTestData:
                    ...
                ProcessTestData:
                    ...
        :return:
        """
        return {
            'TestData': {
                'test_name': self.test_name,
                'processes ({len})'.format(len=len(self.process_data)): [
                    proc.as_printable() for proc in self.processes
                ],
            },
        }

class ProcessTestData:
    def __init__(self, debug):
        self.mult = None
        self.once = None
        self.process_name = None
        self.test_name = None
        self.debug = debug

    def merge_proto_once(self, proto : IMLUnitTestOnce):
        assert self.once is None
        self.once = IMLUnitTestOnce()
        self._init_proto(self.once, from_proto=proto)

        if self.mult is not None:
            self._check_protos()

        self.once.MergeFrom(proto)
        # self.once.prof_event = proto.prof_event

    def merge_proto_multiple(self, proto : IMLUnitTestMultiple):
        if self.mult is None:
            self.mult = IMLUnitTestMultiple()
            self._init_proto(self.mult, from_proto=proto)

        if self.once is not None:
            self._check_protos()

        for phase, phase_events in proto.phase_events.items():
            self.mult.phase_events[phase].events.extend(phase_events.events)

    def events(self, phase):
        return self.mult.phase_events[phase].events

    @property
    def prof_event(self):
        return self.once.prof_event

    @property
    def first_phase(self):
        # 1. Get phase name that has minimum phase start_time_us.
        # 2. Get phase_event
        first_phase_name = min(self.phases, key=lambda phase: self._phase_event(phase).start_time_us)
        first_phase = self._phase_event(first_phase_name)
        return first_phase

    @property
    def last_phase(self):
        last_phase_name = max(self.phases, key=lambda phase: proto_util.event_end_us(self._phase_event(phase)))
        last_phase = self._phase_event(last_phase_name)
        return last_phase

    @property
    def process_duration_us(self):
        first_phase = self.first_phase
        last_phase = self.last_phase
        duration_us = proto_util.event_end_us(last_phase) - first_phase.start_time_us
        return duration_us
    @property
    def process_start_time_us(self):
        first_phase = self.first_phase
        return first_phase.start_time_us
    @property
    def process_end_time_us(self):
        last_phase = self.last_phase
        return proto_util.event_end_us(last_phase)

    @property
    def process_start_time_sec(self):
        return self.process_start_time_us/constants.MICROSECONDS_IN_SECOND
    @property
    def process_end_time_sec(self):
        return self.process_end_time_us/constants.MICROSECONDS_IN_SECOND
    @property
    def process_duration_sec(self):
        return self.process_duration_sec/constants.MICROSECONDS_IN_SECOND

    def _phase_event(self, phase):
        phase_events = self.mult.phase_events[phase]
        # Q: Can we have multiple of the same phases within a single process...?
        assert len(phase_events.events) == 1
        phase_event = phase_events.events[0]
        return phase_event
    # def phase_duration_sec(self, phase):
    #     assert phase in self.mult.phase_events
    #     phase_event = self.mult.phase_events[phase]
    #     return phase_event.duration_us/constants.MICROSECONDS_IN_SECOND

    def phase_start_time_us(self, phase):
        phase_event = self._phase_event(phase)
        return phase_event.start_time_us
    def phase_end_time_us(self, phase):
        phase_event = self._phase_event(phase)
        return proto_util.event_end_us(phase_event)
    def phase_duration_us(self, phase):
        phase_event = self._phase_event(phase)
        return phase_event.duration_us

    def phase_start_time_sec(self, phase):
        return self.phase_start_time_us(phase)/constants.MICROSECONDS_IN_SECOND
    def phase_end_time_sec(self, phase):
        return self.phase_end_time_us(phase)/constants.MICROSECONDS_IN_SECOND
    def phase_duration_sec(self, phase):
        return self.phase_duration_sec(phase)/constants.MICROSECONDS_IN_SECOND

    @property
    def phases(self):
        return sorted(self.mult.phase_events.keys())

    def _phase_as_printable(self, phase):
        events = sorted(self.mult.phase_events[phase].events, key=lambda event: event.start_time_us)
        return [self._event_as_printable(event) for event in events]

    def _event_as_printable(self, event):
        return "(start_sec={start_sec}, duration_sec={dur_sec})".format(
            start_sec=event.start_time_us/constants.MICROSECONDS_IN_SECOND,
            dur_sec=event.duration_us/constants.MICROSECONDS_IN_SECOND,
        )

    def as_printable(self):
        """
        ProcessTestData:
          process_name = <proc>
          test_name = <test>

          Profiler start/stop: (start_sec=..., duration_sec=...)

          Phases:
            Phase 1: bootstrap: (start_sec=..., duration_sec=...)
            Phase 2: selfplay_worker_0: (start_sec=..., duration_sec=...)
                ...
        :return:
        """
        return {
            'ProcessTestData': {
                'process_name': self.process_name,
                'test_name': self.test_name,
                'phases': [
                    (
                        'Phase {i}: {phase}'.format(phase=phase, i=i),
                        self._phase_as_printable(phase),
                    ) for i, phase in enumerate(self.phases)]
            },
        }

    def _check_protos(self):
        assert self.mult.process_name == self.once.process_name
        assert self.mult.test_name == self.once.test_name

    def _init_proto(self, proto, from_proto):
        proto.process_name = from_proto.process_name
        proto.test_name = from_proto.test_name
        self.process_name = proto.process_name
        self.test_name = proto.test_name

#
# Record data from unit-tests.
# Used by profiler at run-time.
#

class UnitTestDataDumper:
    def __init__(self, debug=False):
        self.process_name = None
        self.test_name = None
        self.cur_phase = None
        self.start_t = None
        self.stop_t = None
        self.phase_end = dict()
        self.phase_start = dict()
        # Just for debugging.
        self.stopped = False
        self.debug = debug

    def clear_dump(self):
        """
        Assuming we JUST took a IMLUnitTest proto file with the current state,
        clear all state that was recorded in that file, but KEEP any state that
        WASN'T and will be recorded in a future dump.

        State that WAS recorded:
        -

        In particular:

        We might be in the middle of a phase...
        i.e.
        - phase_start[cur_phase] is recorded, but NOT phase_end[cur_phase]
        - The dump we are performing right now WON'T include cur_phase; but a future dump will
        => clear should NOT forget what the current phase is.

        Clear state associated with the current
        :return:
        """

        # IMLUnitTestMultiple
        phases_to_remove = self._phases_recorded
        for phase in phases_to_remove:
            del self.phase_start[phase]
            del self.phase_end[phase]

        # IMLUnitTestOnce
        if self.start_t is not None and self.stop_t is not None:
            assert self.stopped
            self.start_t = None
            self.stop_t = None

        # NOTE: we leave self.cur_phase AS-IS; we may be dumping data in the middle of a phase.

    @property
    def _IMLUnitTestMultiple_is_dumpable(self):
        return len(self._phases_recorded) > 0
    @property
    def _IMLUnitTestOnce_is_dumpable(self):
        return self.start_t is not None and self.stop_t is not None

    @property
    def _IMLUnitTestMultiple_is_empty(self):
        return len(self.phase_end) == 0 and len(self.phase_start) == 0
    @property
    def _IMLUnitTestOnce_is_empty(self):
        return self.start_t is None and self.stop_t is None
    @property
    def is_empty(self):
        return self._IMLUnitTestMultiple_is_empty and self._IMLUnitTestOnce_is_empty

    def debug_empty(self):
        # IMLUnitTestMultiple
        pprint.pprint({
            'name':'IMLUnitTestMultiple',
            'phase_end':self.phase_end,
            'phase_start':self.phase_start,
        })
        # IMLUnitTestOnce
        pprint.pprint({
            'name':'IMLUnitTestOnce',
            'start_t':self.start_t,
            'stop_t':self.stop_t,
        })

    def start(self):
        assert not self.stopped
        self.start_t = rlscope_timer.now_us()
        # self.unwrapped_prof.start()
        # self.old_start()

    def stop(self):
        if self.stopped:
            logger.info("> Already ran UnitTestDataDumper.stop; ignoring.")
            return
        assert not self.stopped
        # self.unwrapped_prof.stop()
        time_t = rlscope_timer.now_us()
        # self.old_stop()
        self.stop_t = time_t
        self._add_time(self.phase_end, self.cur_phase, time_t)
        self.stopped = True

    def set_phase(self, phase):
        assert not self.stopped
        time_t = rlscope_timer.now_us()

        if self.cur_phase is not None:
            # End the previous phase
            self._add_time(self.phase_end, self.cur_phase, time_t)
        # Start the next phase
        self._add_time(self.phase_start, phase, time_t)

        self.cur_phase = phase

    def dump(self,
             # get_unit_test_*_path(...) args.
             directory, trace_id, bench_name,
             # IMLUnitTest* unit-test identifiers.
             process_name, test_name):

        def print_debug(proto_name, proto, path):
            logger.info("> Dump {proto_name}(process_name={proc}, test_name={test}) @ {path}:".format(
                proto_name=proto_name,
                proc=process_name, test=test_name,
                path=path))
            print(proto)

        once_proto = self.as_proto_once(process_name, test_name)
        if once_proto is not None:
            once_path = get_unit_test_once_path(directory, trace_id, bench_name)
            if self.debug:
                print_debug("IMLUnitTestOnce", once_proto, once_path)
            self._dump_proto(once_path, once_proto)

        multiple_proto = self.as_proto_multiple(process_name, test_name)
        if multiple_proto is not None:
            multiple_path = get_unit_test_multiple_path(directory, trace_id, bench_name)
            if self.debug:
                print_debug("IMLUnitTestMultiple", multiple_proto, multiple_path)
            self._dump_proto(multiple_path, multiple_proto)

    def _dump_proto(self, path, proto):
        with open(path, 'wb') as f:
            f.write(proto.SerializeToString())

    @property
    def _phases_recorded(self):
        phases = sorted(set(self.phase_end.keys()).intersection(self.phase_start.keys()))
        return phases

    def as_proto_once(self, process_name, test_name):
        """
        Create IMLUnitTestOnce instance.

        :return:
        """
        if not self._IMLUnitTestOnce_is_dumpable:
            return None

        proto = IMLUnitTestOnce()
        proto.process_name = process_name
        proto.test_name = test_name

        proto.prof_event.MergeFrom(proto_util.mk_event(
            name="Process start/stop",
            start_us=self.start_t,
            end_us=self.stop_t))

        return proto

    def as_proto_multiple(self, process_name, test_name):
        """
        Create IMLUnitTestMultiple instance.

        :return:
        """
        if not self._IMLUnitTestMultiple_is_dumpable:
            return None

        proto = IMLUnitTestMultiple()
        proto.process_name = process_name
        proto.test_name = test_name

        phases = self._phases_recorded
        for phase in phases:
            for start_us, end_us in zip(self.phase_start[phase], self.phase_end[phase]):
                event = proto_util.mk_event(name=phase, start_us=start_us, end_us=end_us)
                proto.phase_events[phase].events.extend([event])

        return proto

    def _add_time(self, dic, key, time_t):
        if key not in dic:
            dic[key] = []
        dic[key].append(time_t)
