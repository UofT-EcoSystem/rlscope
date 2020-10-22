"""
Get the current timestamp with microsecond resolution.
"""
import time

from rlscope.parser import constants

# NOTE: This is to avoid a weird bug in tensorflow where a wrapped tensorflow API
# (tensorflow.python.framework.c_api_util.ScopedTFGraph) is called during __del__,
# and for some reason time.time (i.e. the function) is None at that point...
get_time = time.time
assert get_time is not None

def now_us():
    # assert time is not None
    # assert time.time is not None
    # return time.time()*constants.MICROSECONDS_IN_SECOND
    # assert get_time is not None
    if get_time is None:
        print("WARNING: time.time() was none...")
        return None
    return get_time()*constants.MICROSECONDS_IN_SECOND
