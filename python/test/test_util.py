from parser.common import *

from parser.stats import KernelTime

def sec(seconds):
    return seconds*MICROSECONDS_IN_SECOND

def T(start_sec, end_sec, name=None):
    return KernelTime(start_usec=start_sec*MICROSECONDS_IN_SECOND,
                      end_usec=end_sec*MICROSECONDS_IN_SECOND,
                      name=name)
