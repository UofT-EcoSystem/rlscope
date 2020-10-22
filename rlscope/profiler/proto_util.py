"""
.. deprecated:: 1.0.0
    Old code for unit-testing / tfprof
"""
from rlscope.protobuf.pyprof_pb2 import CategoryEventsProto, Event

import threading

def mk_event(name, start_us, end_us=None, duration_us=None,
             start_profiling_overhead_us=None,
             duration_profiling_overhead_us=None,
             tid=None, attrs=None):
    if tid is None:
        tid = threading.get_ident()

    if end_us is not None:
        duration_us = end_us - start_us

    event = Event(
        start_time_us=int(start_us),
        duration_us=int(duration_us),
        thread_id=tid,
        name=name,
        # Q: Does this work with attrs=None?
        attrs=attrs)
    if start_profiling_overhead_us is not None:
        event.start_profiling_overhead_us = int(start_profiling_overhead_us)
        event.duration_profiling_overhead_us = int(duration_profiling_overhead_us)

    return event

def event_end_us(event : Event):
    return event.start_time_us + event.duration_us
