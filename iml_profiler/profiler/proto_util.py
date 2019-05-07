from iml_profiler.protobuf.pyprof_pb2 import Pyprof, Event

import threading

def mk_event(name, start_us, end_us=None, duration_us=None, tid=None, attrs=None):
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

    return event

def event_end_us(event : Event):
    return event.start_time_us + event.duration_us
