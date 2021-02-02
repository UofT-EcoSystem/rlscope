# NOTE: Avoid importing profiler library stuff to avoid tensorflow import time
import argparse
from rlscope.parser.plot_utils import setup_matplotlib
setup_matplotlib()
import matplotlib.pyplot as plt

# Avoid using None for no bench_name; doesn't play nice with pandas/numpy
# (None == NaN in that context).
NO_BENCH_NAME = "NoBenchName"
NO_PROCESS_NAME = "NoProcessName"
NO_PHASE_NAME = "NoProcessName"
NO_DEVICE_NAME = "NoDeviceName"
NO_IMPL_NAME = "NoImplName"

DPI = plt.figure().get_dpi()
def pixels_as_inches(px):
    px = float(px) / float(DPI)
    return px
