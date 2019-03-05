import argparse
import textwrap
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl

def test_heatmap():
    # https://github.com/mGalarnyk/Python_Tutorials/blob/master/Request/Heat%20Maps%20using%20Matplotlib%20and%20Seaborn.ipynb
    helix = pd.read_csv('helix_parameters.csv')

    couple_columns = helix[['Energy','helix 2 phase', 'helix1 phase']]

    # this is essentially would be taking the average of each unique combination.
    # one important mention is notice how little the data varies from eachother.
    phase_1_2 = couple_columns.groupby(['helix1 phase', 'helix 2 phase']).mean()
    phase_1_2 = phase_1_2.reset_index()

    # seaborn heatmap documentation
    # https://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.heatmap.html

    # cmap choices: http://matplotlib.org/users/colormaps.html
    plt.figure(figsize=(9,9))
    # phase_1_2_keep = phase_1_2
    phase_1_2_keep = phase_1_2[phase_1_2['helix 2 phase'] == 0]
    pivot_table = phase_1_2_keep.pivot('helix1 phase', 'helix 2 phase','Energy')
    plt.xlabel('helix 2 phase', size = 15)
    plt.ylabel('helix1 phase', size = 15)
    plt.title('Energy from Helix Phase Angles', size = 15)
    data = pivot_table
    # data = pivot_table[pivot_table['helix 2 phase'] == 0]
    # sns.heatmap(data, annot=True, fmt=".1f", linewidths=.5, square = True, cmap = 'Blues_r')
    sns.heatmap(data, annot=True, fmt=".1f", linewidths=.5, square = True, cmap = 'Blues_r')
    plt.savefig('heatmap.png')
    # plt.show()

def test_colorbar():
    '''
    https://matplotlib.org/examples/api/colorbar_only.html

    ====================
    Customized colorbars
    ====================

    This example shows how to build colorbars without an attached mappable.
    '''

    # Make a figure and axes with dimensions as desired.
    width_px = 300
    height_px = 100
    # fig = plt.figure(figsize=(8, 3))
    fig = plt.figure(figsize=(width_px*MATPLOTLIB_PIXEL_FACTOR, height_px*MATPLOTLIB_PIXEL_FACTOR))
    # The dimensions [left, bottom, width, height] of the new axes.
    ax1 = fig.add_axes([0.05, 0.65, 0.9, 0.10])
    # ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
    # ax2 = fig.add_axes([0.05, 0.475, 0.9, 0.15])
    # ax3 = fig.add_axes([0.05, 0.15, 0.9, 0.15])

    # Set the colormap and norm to correspond to the data for which
    # the colorbar will be used.
    # cmap = mpl.cm.cool
    cmap = mpl.cm.Blues
    norm = mpl.colors.Normalize(vmin=0, vmax=100)

    # ColorbarBase derives from ScalarMappable and puts a colorbar
    # in a specified axes, so it has everything needed for a
    # standalone colorbar.  There are many more kwargs, but the
    # following gives a basic continuous colorbar with ticks
    # and labels.
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')
    cb1.set_label('Utilization (%)')

    # # The second example illustrates the use of a ListedColormap, a
    # # BoundaryNorm, and extended ends to show the "over" and "under"
    # # value colors.
    # cmap = mpl.colors.ListedColormap(['r', 'g', 'b', 'c'])
    # cmap.set_over('0.25')
    # cmap.set_under('0.75')
    #
    # # If a ListedColormap is used, the length of the bounds array must be
    # # one greater than the length of the color list.  The bounds must be
    # # monotonically increasing.
    # bounds = [1, 2, 4, 7, 8]
    # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    # cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,
    #                                 norm=norm,
    #                                 # to use 'extend', you must
    #                                 # specify two extra boundaries:
    #                                 boundaries=[0] + bounds + [13],
    #                                 extend='both',
    #                                 ticks=bounds,  # optional
    #                                 spacing='proportional',
    #                                 orientation='horizontal')
    # cb2.set_label('Discrete intervals, some other units')
    #
    # # The third example illustrates the use of custom length colorbar
    # # extensions, used on a colorbar with discrete intervals.
    # cmap = mpl.colors.ListedColormap([[0., .4, 1.], [0., .8, 1.],
    #                                   [1., .8, 0.], [1., .4, 0.]])
    # cmap.set_over((1., 0., 0.))
    # cmap.set_under((0., 0., 1.))
    #
    # bounds = [-1., -.5, 0., .5, 1.]
    # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    # cb3 = mpl.colorbar.ColorbarBase(ax3, cmap=cmap,
    #                                 norm=norm,
    #                                 boundaries=[-10] + bounds + [10],
    #                                 extend='both',
    #                                 # Make the length of each extension
    #                                 # the same as the length of the
    #                                 # interior colors:
    #                                 extendfrac='auto',
    #                                 ticks=bounds,
    #                                 spacing='uniform',
    #                                 orientation='horizontal')
    # cb3.set_label('Custom extension lengths, some other units')

    # fig.tight_layout()
    fig.savefig('test_colorbar.png',
                # Remove excess whitespace around the plot.
                bbox_inches='tight')
    # plt.show()

MATPLOTLIB_PIXEL_FACTOR = 1e-2

class HeatScale:
    """
    Make a CPU/GPU utilization color scale.
    """
    def __init__(self, color_value, y_axis, png,
                 width_pixels=480., height_pixels_per_sec=None,
                 cbar_width_px=300, cbar_height_px=100,
                 cmap_name='Blues',
                 debug=False):
        self.color_value = color_value
        self.y_axis = y_axis
        self.png = png
        self.width_pixels = width_pixels
        self.height_pixels_per_sec = height_pixels_per_sec
        self.cbar_width_px = cbar_width_px
        self.cmap_name = cmap_name
        self.cbar_height_px = cbar_height_px
        self.debug = debug

    def add_data(self, dic):
        """
        dic = {
            '<color_value>': [0.1, 0.5, 0.9, 1.0, 0.1, ...],
            '<y_axis>': [0.1, 0.5, 0.9, 1.0, 0.1, ...],
        }

        :param dic:
        :return:
        """
        assert self.color_value in dic
        assert self.y_axis in dic

        data = dict(dic)
        n = len(dic[self.color_value])
        self.df = pd.DataFrame(data)
        # xaxis: We just want a single vertical axis of colored squares, not a matrix heatmap.
        self.df['_heatscale_xaxis'] = 0
        # data['_heatscale_idx'] = np.arange(n)
        self.pivot_table = self.df.pivot(self.y_axis, '_heatscale_xaxis', self.color_value)

    def plot(self):
        self.plot_data()
        self.plot_colorbar()

    def plot_data(self):
        fig = plt.figure()
        if self.height_pixels_per_sec is not None:
            # The last sample.
            max_sec = max(self.df[self.y_axis])
            height = max_sec * self.height_pixels_per_sec
            fig.set_figheight(height*MATPLOTLIB_PIXEL_FACTOR)

        if self.width_pixels is not None:
            # For some reason, matplotlib.Figure gets/set pixels by a factor of 1/100
            # <Figure size 640x500 with 2 Axes>
            # ipdb> fig.get_figwidth()
            # 6.4
            fig.set_figwidth(self.width_pixels*MATPLOTLIB_PIXEL_FACTOR)

        # Fill the entire figure, don't allow any margins.
        left = 0.0
        bottom = 0.0
        width = 1.0
        height = 1.0
        ax = fig.add_axes([left, bottom, width, height])

        # yticklabels = ["{val:.1f}".format(val=val) for val in self.df[self.y_axis]]
        self.plot = sns.heatmap(
            self.pivot_table,
            # Show 1.0 for color label
            fmt=".1f",
            # Don't show _heatscale_xaxis xticks.
            xticklabels=False,
            # Don't show seconds y-labels
            yticklabels=False,
            # Don't show utilization value in cell.
            annot=False,
            # annot=True,
            # yticklabels=yticklabels,

            # If set, use ~ 1 px between color-boxes.
            linewidths=0.5,

            # If set, use NO spacer between color-boxes.
            # NOTE: this does NOT fix the pixel-to-second offset problem.
            # linewidths=0,

            square=True,
            cmap=self.cmap_name,
            # If cbar is False, it won't show the color gradient legend.
            # NOTE: We plot colorbar separately in plot_colorbar()
            cbar=False,
            ax=ax,
        )
        self.plot.set_xlabel(None)
        self.plot.set_ylabel(None)

        if self.debug:
            import ipdb; ipdb.set_trace()
        print("> Save HeatScale @ {path}".format(path=self.png))
        fig.savefig(self.png)

    def plot_colorbar(self):
        fig = plt.figure(figsize=(self.cbar_width_px*MATPLOTLIB_PIXEL_FACTOR, self.cbar_height_px*MATPLOTLIB_PIXEL_FACTOR))
        ax1 = fig.add_axes([0.05, 0.65, 0.9, 0.10])

        # Set the colormap and norm to correspond to the data for which
        # the colorbar will be used.
        cmap = getattr(mpl.cm, self.cmap_name)
        norm = mpl.colors.Normalize(vmin=0, vmax=100)

        # ColorbarBase derives from ScalarMappable and puts a colorbar
        # in a specified axes, so it has everything needed for a
        # standalone colorbar.  There are many more kwargs, but the
        # following gives a basic continuous colorbar with ticks
        # and labels.
        cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                        norm=norm,
                                        orientation='horizontal')
        cb1.set_label('Utilization (%)')
        print("> Save HeatScale cbar @ {path}".format(path=self.cbar_png))
        fig.savefig(self.cbar_png,
                    # Remove excess whitespace around the plot.
                    bbox_inches='tight')

    @property
    def cbar_png(self):
        path = re.sub(r'\.png$', '.cbar.png', self.png)
        assert path != self.png
        return path

def test_heatscale(args):
    # - Sample every 1 second over a period of 10 seconds
    # - A sample at every 1 + 0.1 second (i.e. offset by about 0.1 second)

    # Time between samples in seconds.
    step = 0.1
    # repeats = 1
    repeats = 10
    base_color_values = np.array([
            0.1, 0.5, 0.9, 1.0, 0.8,
            0.1, 0.2, 0.3, 0.4, 0.5,
        ])

    # Useful for testing offset due to color-box spacing.
    repeat_elem_wise = True
    # repeat_elem_wise = False
    if repeat_elem_wise:
        # Repeat element-wise.
        color_value = np.repeat(base_color_values, repeats)
    else:
        # Repeat array.
        color_value = np.concatenate(np.repeat(base_color_values[np.newaxis, ...], repeats, axis=0))

    dic = {
        'color_value': color_value,
        'y_axis': np.arange(start=0, stop=len(base_color_values)*repeats*step, step=step),
    }
    png = 'test_heatscale.png'
    pixels_per_square = 10
    height_pixels_per_sec = pixels_per_square * (1 / step)
    heatscale = HeatScale('color_value', 'y_axis', png,
                          height_pixels_per_sec=height_pixels_per_sec,
                          width_pixels=pixels_per_square,
                          debug=args.debug)
    heatscale.add_data(dic)
    heatscale.plot()

def main():
    parser = argparse.ArgumentParser("Dump protobuf files to txt")
    parser.add_argument("--test-heatmap",
                        action='store_true',
                        help=textwrap.dedent("""
                        Test heat map.
                        """))
    parser.add_argument("--test-colorbar",
                        action='store_true',
                        help=textwrap.dedent("""
                        Test heat map.
                        """))
    parser.add_argument("--test-heatscale",
                        action='store_true',
                        help=textwrap.dedent("""
                        Test heat scale.
                        """))
    parser.add_argument("--debug",
                        action='store_true',
                        help=textwrap.dedent("""
                        Debug
                        """))
    args = parser.parse_args()

    if args.test_colorbar:
        test_colorbar()
        return

    if args.test_heatmap:
        test_heatmap()
        return

    if args.test_heatscale:
        test_heatscale(args)
        return

if __name__ == '__main__':
    main()
