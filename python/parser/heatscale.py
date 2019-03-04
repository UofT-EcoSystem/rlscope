import argparse
import textwrap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


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
    sns.heatmap(data, annot=True, fmt=".1f", linewidths=.5, square = True, cmap = 'Blues_r')
    plt.savefig('heatmap.png')
    # plt.show()

MATPLOTLIB_PIXEL_FACTOR = 1e-2

class HeatScale:
    """
    Make a CPU/GPU utilization color scale.
    """
    def __init__(self, color_value, y_axis, png, width_pixels=480., height_pixels_per_sec=None, debug=False):
        self.color_value = color_value
        self.y_axis = y_axis
        self.png = png
        self.width_pixels = width_pixels
        self.height_pixels_per_sec = height_pixels_per_sec
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
        # xaxis
        self.df['_heatscale_xaxis'] = 0
        # data['_heatscale_idx'] = np.arange(n)
        self.pivot_table = self.df.pivot(self.y_axis, '_heatscale_xaxis', self.color_value)

    def plot(self):
        yticklabels = ["{val:.1f}".format(val=val) for val in self.df[self.y_axis]]
        fig = plt.figure()
        # fig = self.plot.figure

        # Fill the entire figure, don't allow any margins.
        left = 0.0
        bottom = 0.0
        width = 1.0
        height = 1.0
        ax = fig.add_axes([left, bottom, width, height])
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
            linewidths=.5,
            square=True,
            # cmap='Blues_r',
            # xlabel='x!',
            cmap='Blues',
            # If cbar is False, it won't show the color gradient legend.
            cbar=False,
            ax=ax,
        )
        self.plot.set_xlabel(None)
        self.plot.set_ylabel(None)
        # fig.tight_layout()
        # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        # eliminate top/bottom margins?

        # Eliminates the top margin (1 pixel of whitespace, it's probably the box outline.
        # plt.subplots_adjust(top=1.0)

        # plt.subplots_adjust(
        #     # left=1.0, right=1.0,
        #     # This fails an assertion: top >= bottom.
        #     # Q: Why isn't this allowed..?
        #     top=1.0, bottom=1.0,
        # )

        # plt.subplots_adjust(
        #     # left=1.0, right=1.0,
        #     top=1.0
        #     # , bottom=1.0
        # )

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

        if self.debug:
            import ipdb; ipdb.set_trace()
        print("> Save HeatScale @ {path}".format(path=self.png))
        plt.savefig(self.png)

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
    dic = {
        'color_value': np.repeat(base_color_values, repeats),
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

    if args.test_heatmap:
        test_heatmap()
        return

    if args.test_heatscale:
        test_heatscale(args)
        return

if __name__ == '__main__':
    main()
