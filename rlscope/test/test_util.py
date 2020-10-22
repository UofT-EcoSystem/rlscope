"""
Unit-test helper code for Python event overlap implementation.
"""
from rlscope.parser.common import *

from rlscope.parser.stats import KernelTime

def sec(seconds):
    return seconds*constants.MICROSECONDS_IN_SECOND

def T(start_sec, end_sec, name=None, **kwargs):
    return KernelTime(start_usec=start_sec*constants.MICROSECONDS_IN_SECOND,
                      end_usec=end_sec*constants.MICROSECONDS_IN_SECOND,
                      name=name,
                      **kwargs)

def U(start_usec, end_usec=None, name=None, time_usec=None, **kwargs):
    return KernelTime(start_usec=start_usec,
                      time_usec=time_usec,
                      end_usec=end_usec,
                      name=name,
                      **kwargs)

def flatten_category_times(category_times):
    new_category_times = dict()
    for category in category_times.keys():
        all_times = []
        for times in category_times[category]:
            all_times.extend(times)
        new_category_times[category] = all_times
    return new_category_times
