"""
Debugging code; try out different matplotlib plotting options.
"""
import re
import itertools
import sys
import os
import csv
import textwrap
import pprint
from io import StringIO
import json
import codecs
from rlscope.parser.plot_utils import setup_matplotlib
setup_matplotlib()
import matplotlib.pyplot as plt
from os.path import join as _j, abspath as _a, dirname as _d, exists as _e, basename as _b

from rlscope.parser.common import *
from rlscope.parser.plot import TimeBreakdownPlot

def disable_test_grouped_stacked_bar_plot(parser, args):
    # Stacked barplot:
    # https://matplotlib.org/gallery/lines_bars_and_markers/bar_stacked.html
    def stacked_barplot():
        N = 5
        menMeans = (20, 35, 30, 35, 27)
        womenMeans = (25, 32, 34, 20, 25)
        menStd = (2, 3, 4, 1, 2)
        womenStd = (3, 5, 2, 3, 3)
        ind = np.arange(N)    # the x locations for the groups
        width = 0.35       # the width of the bars: can also be len(x) sequence

        p1 = plt.bar(ind, menMeans, width, yerr=menStd)
        p2 = plt.bar(ind, womenMeans, width,
                     bottom=menMeans, yerr=womenStd)

        plt.ylabel('Scores')
        plt.title('Scores by group and gender')
        plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
        plt.yticks(np.arange(0, 81, 10))
        plt.legend((p1[0], p2[0]), ('Men', 'Women'))

        plt.show()

    # Grouped barplot
    # https://python-graph-gallery.com/11-grouped-barplot/
    def grouped_barplot():
        # libraries
        # set width of bar
        bar_width = 0.25

        # set height of bar
        bars1 = [12, 30, 1, 8, 22]
        bars2 = [28, 6, 16, 5, 10]
        bars3 = [29, 3, 24, 25, 17]

        # Set position of bar on X axis
        r1 = np.arange(len(bars1))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]

        # Make the plot
        plt.bar(r1, bars1, color='#7f6d5f', width=bar_width, edgecolor='white', label='var1')
        plt.bar(r2, bars2, color='#557f2d', width=bar_width, edgecolor='white', label='var2')
        plt.bar(r3, bars3, color='#2d7f5e', width=bar_width, edgecolor='white', label='var3')

        # Add xticks on the middle of the group bars
        plt.xlabel('group', fontweight='bold')
        plt.xticks([r + bar_width for r in range(len(bars1))], ['A', 'B', 'C', 'D', 'E'])

        # Create legend & Show graphic
        plt.legend()
        plt.show()

    # How to put patterns (hatches) on bars of a barplot.
    # https://matplotlib.org/gallery/shapes_and_collections/hatch_demo.html#sphx-glr-gallery-shapes-and-collections-hatch-demo-py
    # http://kitchingroup.cheme.cmu.edu/blog/2013/10/26/Hatched-symbols-in-matplotlib/
    # patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.', '/')
    def hatch_barplot():
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse, Polygon

        fig, (ax1, ax2, ax3) = plt.subplots(3)

        ax1.bar(range(1, 5), range(1, 5), color='red', edgecolor='black', hatch="/")
        ax1.bar(range(1, 5), [6] * 4, bottom=range(1, 5),
                color='blue', edgecolor='black', hatch='//')
        ax1.set_xticks([1.5, 2.5, 3.5, 4.5])

        bars = ax2.bar(range(1, 5), range(1, 5), color='yellow', ecolor='black') + \
               ax2.bar(range(1, 5), [6] * 4, bottom=range(1, 5),
                       color='green', ecolor='black')
        ax2.set_xticks([1.5, 2.5, 3.5, 4.5])

        patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
        for bar, pattern in zip(bars, patterns):
            bar.set_hatch(pattern)

        ax3.fill([1, 3, 3, 1], [1, 1, 2, 2], fill=False, hatch='\\')
        ax3.add_patch(Ellipse((4, 1.5), 4, 0.5, fill=False, hatch='*'))
        ax3.add_patch(Polygon([[0, 0], [4, 1.1], [6, 2.5], [2, 1.4]], closed=True,
                              fill=False, hatch='/'))
        ax3.set_xlim((0, 6))
        ax3.set_ylim((0, 2.5))

        plt.show()

    # Multiple legends in a single plot.
    # https://stackoverflow.com/questions/12761806/matplotlib-2-different-legends-on-same-graph
    def multi_legend_plot():
        # I have a plot where different colors are used for different parameters, and
        # where different line styles are used for different algorithms.

        colors = ['b', 'r', 'g', 'c']
        cc = itertools.cycle(colors)
        plot_lines = []
        parameters = np.arange(3)
        n_points = 3

        def algo(algo_num, p):
            slope = p*algo_num
            return np.zeros(n_points) + slope*np.arange(n_points)

        for p in parameters:

            d1 = algo(1, p)
            d2 = algo(2, p)
            d3 = algo(3, p)

            # plt.hold(True)
            c = next(cc)
            # algo1 uses -, algo2 uses --, ...
            l1, = plt.plot(d1, '-', color=c)
            l2, = plt.plot(d2, '--', color=c)
            l3, = plt.plot(d3, '.-', color=c)

            plot_lines.append([l1, l2, l3])

        legend1 = plt.legend(plot_lines[0], ["algo1", "algo2", "algo3"], loc=1)
        plt.legend([l[0] for l in plot_lines], parameters, loc=4)
        plt.gca().add_artist(legend1)

        plt.show()

    def grouped_stacked_barplot():
        png_path = "grouped_stacked_barplot.png"

        def _timings(base_sec, iterations):
            def _t(i, offset):
                return (base_sec + offset) + 0.1*i
            def _ts(offset):
                return [_t(i, offset) for i in range(iterations)]
            return {
                'GPUTimeSec': _ts(offset=0),
                'CppTimeSec': _ts(offset=1),
                'PythonTimeSec': _ts(offset=2),
            }

        def _generate_timings(bench_name_order, base_sec, iterations):
            timings = dict()
            for offset, bench_name in enumerate(bench_name_order):
                timings[bench_name] = _timings(base_sec + offset, iterations)
            return timings

        def _generate_json_datas(time_breakdown):
            iterations = 4

            json_datas = []

            data_quadro_k4000 = {
                'attrs': {
                    'name':'Quadro K4000',
                    'impl_name':'DQN Python',
                },
            }
            data_quadro_k4000.update(_generate_timings(time_breakdown.bench_name_order, 1, iterations))
            json_datas.append(data_quadro_k4000)

            data_quadro_p4000 = {
                'attrs': {
                    'name':'Quadro P4000',
                    'impl_name':'DQN Python',
                },
            }
            data_quadro_p4000.update(_generate_timings(time_breakdown.bench_name_order, 2, iterations))
            json_datas.append(data_quadro_p4000)

            data_gtx_1080 = {
                'attrs': {
                    'name':'GTX 1080',
                    'impl_name':'DQN Python',
                },
            }
            data_gtx_1080.update(_generate_timings(time_breakdown.bench_name_order, 3, iterations))
            json_datas.append(data_gtx_1080)

            return json_datas

        time_breakdown = TimeBreakdownPlot(png_path, show=args.show)

        json_datas = _generate_json_datas(time_breakdown)
        for json_data in json_datas:
            bench_names = [k for k in json_data.keys() if k not in set(['attrs'])]
            for bench_name in bench_names:
                time_breakdown.add_json_data(json_data[bench_name],
                                             bench_name=bench_name,
                                             device=json_data['attrs']['name'],
                                             impl_name=json_data['attrs']['impl_name'])
        time_breakdown.plot()

    # stacked_barplot()
    # grouped_barplot()
    # hatch_barplot()
    # multi_legend_plot()
    grouped_stacked_barplot()


def disable_test_legend_position():
    # https://matplotlib.org/gallery/subplots_axes_and_figures/subplots_adjust.html#sphx-glr-gallery-subplots-axes-and-figures-subplots-adjust-py

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    plt.subplot(211)
    plt.imshow(np.random.random((100, 100)), cmap=plt.cm.BuPu_r)
    plt.subplot(212)
    plt.imshow(np.random.random((100, 100)), cmap=plt.cm.BuPu_r)

    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    # cax = plt.axes([1, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
    plt.show()

def disable_test_plot_position():
    # https://matplotlib.org/gallery/lines_bars_and_markers/scatter_hist.html#sphx-glr-gallery-lines-bars-and-markers-scatter-hist-py

    from matplotlib.ticker import NullFormatter

    # Fixing random state for reproducibility
    np.random.seed(19680801)


    # the random data
    x = np.random.randn(1000)
    y = np.random.randn(1000)

    nullfmt = NullFormatter()         # no labels

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65

    # Q: What's the 0.1 for?
    #   - If we do this, you cannot see the x/y labels; 0.1 leaves space for them!
    #
    # Q: What are the units of the 0.1 (x/y position)?
    #   - A percentage along the axis from the bottom/left
    #   - i.e.
    #     Looks like 0 = bottom/left of plot, 0.5 = center of plot, 1 = top/right of plot
    #
    # Q: If we used percent units, how can you accurately position
    #    two plots relative to each other (e.g on top)...?
    #    - IF we have a way to query the height of the labels box,
    #      then we can set the y-position of the bottom label-box to that.

    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(1, figsize=(8, 8))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    axScatter.scatter(x, y)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    axScatter.set_xlim((-lim, lim))
    axScatter.set_ylim((-lim, lim))

    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(x, bins=bins)
    axHisty.hist(y, bins=bins, orientation='horizontal')

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    plt.show()

def disable_test_subplot():
    # https://matplotlib.org/users/transforms_tutorial.html#axes-coordinates
    #
    # This example code.
    # Shows how to position text inside the plot using the axes coordinate system.

    # https://matplotlib.org/users/transforms_tutorial.html
    #
    # Discussion of the different coordinate systems used in matplotlib
    # TLDR:
    # - axes coordinate system: (0,0) is bottom-left of x-y plot, (1,1) is top-right
    #   You can access the axes of a particular subplot:
    #   subplot_axes = fig.add_subplot(...)
    # - figure coordinate system: (0,0) is bottom-left of entire figure, (1,1) is top-right
    #   I think you access this system via fig.axes

    # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html
    #
    # Legend arguments for positioning the legend.
    # By default, you make a legend via subplot_axes.legend(...).
    # So, the legend uses the axes coordinate system of the subplot.
    # To position the legend in a different coordinate system, such as the whole figure, you do:
    #   fig = plt.figure()
    #   ...
    #   leg = ax.legend(
    #       # Coordinates in the figure coordinate system:
    #       bbox_to_anchor=(0.5, 0.5),
    #       bbox_transform=fig.transFigure)
    #
    # NOTE: I think if you use plt.legend(), it uses the figure coordinate system.

    # https://matplotlib.org/users/artists.html
    #
    # Discussion of "primitives" (e.g. text, lines) and "containers" (e.g. figure, subplot)

    # Options:
    # 1. Make legend and query width/height; NOPE
    # 2. Use preset width/height and add it to bbox_to_anchor arguments for legend.
    # 3. Get the axes of the top-legend, and set the bottom-legend bbox=(x=0, y=-0.04)

    fig = plt.figure()

    for i, label in enumerate(('A', 'B', 'C', 'D')):
        ax = fig.add_subplot(2,2,i+1)
        ax.text(0.05, 0.95, label, transform=ax.transAxes,
          fontsize=16, fontweight='bold', va='top')

        # Default position is relative to the subplot's axes.
        leg = ax.legend(bbox_to_anchor=(0.05, 0.95))

        # This places the legend in the middle of whole figure.
        # leg = ax.legend(
        #     # loc=(0.5, 0.5),
        #     bbox_to_anchor=(0.5, 0.5),
        #     bbox_transform=fig.transFigure)

    plt.show()
