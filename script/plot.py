"""
Plots the results of the benchmark with all three GPUs

Plots (the figure numbers will be referred in comments)

1. Benchmarks from all three GPUs and grid configurations as a bar chart. y-axis
   in seconds, x-axis no label, the bars are ordered from fastest (left) to
   slowest (right). Grid configurations are not labelled but the GPUs are colour
   coded
2. Bland-Altman difference plot comparing the benchmarks of grid configurations
   with and without padding in shared memory. Each point represents a grid
   configurations and each colour for a different GPU model. Only block
   dimensions with widths 4 and 8 are considered. The x-axis shows the average
   of the two benchmarks and the y-axis the difference. The error bars show the
   combined standard deviation from the 32 repeats. The dashed line is at zero
   and negative values underneath that line show that adding padding improves
   benchmarks
3. Markdown table of benchmarks on the V100, A100 and H100 GPUs. Quoted are the
   mean and standard deviations. Different grid configurations were used, for
   example, 8x4 + 6 means a block dimension of 8x4 with a 6 pixel padding on the
   shared memory
4. Benchmarks for a specific GPU grouped by number of warps per block as a bar
   chart. y-axis in seconds, x-axis number of warps per block. Only the top 5
   grid configurations (with shared memory padding where applicable) are shown
   within each number of warps per block. Within each group, they are ordered
   from fastest to slowest
5. Benchmarks for a specific GPU grouped by block width as a bar chart. y-axis
   in seconds, x-axis block width. All configurations (with shared memory
   padding where applicable) are shown within each block width. Within each
   group, they are ordered from fastest to slowest

How to use:

- pytest tests/benchmark.py > [some_output_file_name]
- Do this for the three different GPUs
- python plot.py [output file from V100] [output file from A100] \
    [output file from H100]
- Charts are displayed with matplotlib
- Tables are printed on screen
"""

import argparse
import re
import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from uncertainties import unumpy

# String format for the labels of each grid configurations IN ORDER
# see tests/benchmark.py for the grid configurations used
GRID_CONFIG = np.asarray(
    [
        "1&times;32",
        "2&times;16",
        "4&times;8",
        "4&times;8 + 2",
        "8&times;4",
        "8&times;4 + 6",
        "16&times;2",
        "32&times;1",
        "1&times;64",
        "2&times;32",
        "4&times;16",
        "4&times;16 + 2",
        "8&times;8",
        "8&times;8 + 6",
        "16&times;4",
        "32&times;2",
        "64&times;1",
        "1&times;128",
        "2&times;64",
        "4&times;32",
        "4&times;32 + 2",
        "8&times;16",
        "8&times;16 + 6",
        "16&times;8",
        "32&times;4",
        "64&times;2",
        "128&times;1",
        "1&times;256",
        "2&times;128",
        "4&times;64",
        "4&times;64 + 2",
        "8&times;32",
        "8&times;32 + 6",
        "16&times;16",
        "32&times;8",
        "64&times;4",
        "128&times;2",
        "256&times;1",
    ]
)

# Indices for GRID_CONFIG
# Each row corresponding to a block dimension
# Column 0 for no padding, columns 1 with padding
DIM_PADDING_INDEX_LIST = [
    [2, 3],
    [4, 5],
    [10, 11],
    [12, 13],
    [19, 20],
    [21, 22],
    [29, 30],
    [31, 32],
]

# which GPU to plot for Figures 4 and 5
PLOT_SINGLE_GPU_INDEX = 1

# for Figure 4
# maximum number of wraps to plot
MAX_N_WRAP = 4
# for each grid config in GRID_CONFIG, the number of wraps
N_WRAP_FOR_EACH_GRID_CONFIG = np.asarray(
    list(
        itertools.chain(
            itertools.repeat(1, 8),
            itertools.repeat(2, 9),
            itertools.repeat(3, 10),
            itertools.repeat(4, 11),
        )
    ),
    dtype=np.int32,
)

# for Figure 5
# maximum block widths to plot
MAX_N_WIDTH = 6
# block widths in GRID_CONFIG up to 32
# this is to include duplicate widths where one hasn't/has padding in shared
# memory
WIDTHS_UP_TO_32 = [1, 2, 4, 4, 8, 8, 16, 32]
# all widths considered in the experiment
WIDTHS = WIDTHS_UP_TO_32 + [64, 128, 256]
# for each grid config in GRID_CONFIG, the block width
WIDTHS_FOR_EACH_GRID_CONFIG = np.asarray(
    WIDTHS_UP_TO_32
    + WIDTHS_UP_TO_32
    + [64]
    + WIDTHS_UP_TO_32
    + [64, 128]
    + WIDTHS_UP_TO_32
    + [64, 128, 256]
)


def get_data(files):
    """Read and process the output files into a list of numpy arrays

    Process the output files, which contains the printed output of
    pytest-benchmark, into a list of numpy arrays containing the grid
    configuration index (pointing to GRID_CONFIG), mean time and standard
    deviation

    Args:
        files (list): list of output files, length three, one for each GPU

    Returns:
        list: length three, one for each GPU. For each element, contains a numpy
            array, each row for a grid configuration, columns: index for
            GRID_CONFIG, mean time, standard deviation time
    """
    data = []
    # a GPU for each file
    for file in files:
        data_i = []
        with open(file) as f:
            for line in f:
                if "dim" in line:
                    # point to the number after the string "dim"
                    dim_index = line.find("dim") + 3
                    # assume the configuration index has two digits
                    dim = line[dim_index : dim_index + 2]
                    # if the configuration index is single digit
                    if "-" in dim:
                        dim = dim[0]
                    # split using whitespace
                    string_splits = re.split("\s+", line)
                    # append the grid config index, mean and standard deviation
                    data_i.append(
                        [
                            int(dim),
                            float(string_splits[5]),
                            float(string_splits[7]),
                        ]
                    )
        data.append(np.asarray(data_i))
    return data


def plot_benchmarks_as_bars(data, gpu_names):
    """Plot all benchmarks as a bar chart

    Plot all benchmarks as a bar chart for all GPUs and grid configurations.
    They are ordered from fastest (left) to slowest (right). Different GPUs
    are colour coded

    The plot is shown with matplotlib

    Args:
        data (list): output from get_data()
        gpu_names (list): list of length 3, string for each GPU
    """
    n_gpu = len(data)
    data_copy = []
    for i, data_i in enumerate(data):
        # replace grid config index with gpu id so that the GPUs can be coloured
        # coded, the grid configs are not labelled here
        data_i_j = data_i.copy()
        data_i_j[:, 0] = i  # replace grid config index with gpu id
        data_copy.append(data_i_j)
    data_copy = np.concatenate(data_copy)

    # sort the data
    # we don't use the standard deviation in column 2, so replace that with
    # the rank, used for plotting benchmarks from fastest to slowest
    sort_index = np.argsort(data_copy[:, 1])
    data_copy[sort_index, 2] = np.arange(len(data_copy))

    plt.figure()
    # for loop so each gpu has a different colour
    for i in range(n_gpu):
        data_i = data_copy[data_copy[:, 0] == i, :]
        plt.bar(data_i[:, 2], data_i[:, 1])
    plt.legend(gpu_names)
    plt.tick_params(
        axis="x",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False,
    )
    plt.ylabel("time (s)")
    plt.show()


def plot_table(data, gpu_names):
    """Plot all benchmarks as a markdown table

    Plot all benchmarks as a markdown table for all GPUs and grid
    configurations. A column for each GPU. Then, the grid configuration, mean
    time and standard deviation time are shown, ordered from fastest (top) to
    slowest (bottom)

    The markdown table is displayed on screen

    Args:
        data (list): output from get_data()
        gpu_names (list): list of length 3, string for each GPU
    """
    table = pd.DataFrame()
    for i, data_i in enumerate(data):
        data_copy = data_i.copy()
        index = np.argsort(data_copy[:, 1])
        data_copy = data_copy[index]

        table[f"{gpu_names[i]}-config"] = GRID_CONFIG[
            data_copy[:, 0].astype(np.int32)
        ]

        # display the mean and standard deviation as one single string
        times = unumpy.uarray(data_copy[:, 1], data_copy[:, 2])
        table[f"{gpu_names[i]}-time"] = [
            str(time).replace("+/-", "&pm;") for time in times
        ]

    print(table.to_markdown())


def plot_padding_bland_altman(data, gpu_names):
    """Bland-Altman plot comparing benchmarks with and without padding

    Bland-Altman difference plot comparing benchmarks with and without padding.
    in shared memory. Each point represents a grid configurations and each
    colour for a different GPU model. Only block dimensions with widths 4 and 8
    are considered. y-axis is the difference and x axis is the average. Error
    bars correspond to the combined standard deviation

    Args:
        data (list): output from get_data()
        gpu_names (list): list of length 3, string for each GPU
    """
    colours = [
        "tab:blue",
        "tab:orange",
        "tab:green",
    ]

    plt.figure()

    # for each gpu
    for i, data_i in enumerate(data):
        data_plot = []
        for dim_index in DIM_PADDING_INDEX_LIST:
            # retrive times for no padding and with padding
            time_0 = data_i[data_i[:, 0] == dim_index[0], 1].item()
            time_1 = data_i[data_i[:, 0] == dim_index[1], 1].item()
            err_0 = data_i[data_i[:, 0] == dim_index[0], 2].item()
            err_1 = data_i[data_i[:, 0] == dim_index[1], 2].item()
            time = (time_0 + time_1) / 2  # mean
            err = math.sqrt(err_0**2 + err_1**2)  # combine std
            data_plot.append([time, time_1 - time_0, err])
        data_plot = np.asarray(data_plot)
        plt.errorbar(
            data_plot[:, 0],
            data_plot[:, 1],
            data_plot[:, 2],
            fmt="none",
            capsize=5,
            ecolor=colours[i],
        )
    ax = plt.gca()
    plt.xlabel("time average (s)")
    plt.ylabel("time difference (s)")
    plt.legend(gpu_names)
    plt.hlines(0, ax.get_xlim()[0], ax.get_xlim()[1], "k", "--")
    plt.show()


def plot_benchmarks_group_by_warps(data):
    """Plot the benchmarks for different number of wraps

    Plot the benchmarks for a specific GPU grouped by number of warps per block
    as a bar chart. y-axis in seconds, x-axis number of warps per block. Only
    the top 5 grid configurations (with shared memory padding where applicable)
    are shown within each number of warps per block. Within each group, they are
    ordered from fastest to slowest

    The plot is shown with matplotlib

    Args:
        data (np.ndarray): an element from the output of get_data()
    """
    data_copy = data.copy()

    # standard deviation not used, replace it with number of wraps
    data_copy[:, 2] = N_WRAP_FOR_EACH_GRID_CONFIG[
        data_copy[:, 0].astype(np.int32)
    ]

    # remove the block configs with no padding
    for i in np.asarray(DIM_PADDING_INDEX_LIST)[:, 0]:
        data_copy = data_copy[data_copy[:, 0] != i, :]

    n_top_selected = 5
    plt.figure()
    # plot different colour for each n_wrap
    for i in range(1, MAX_N_WRAP + 1):
        data_i = data_copy[data_copy[:, 2] == i, :]
        data_i = np.sort(data_i[:, 1])
        # only have top few benchmarks for a n_wrap
        # this is so that we can see trends without poor grid configurations
        # distracting the reader
        data_i = data_i[0:n_top_selected]
        plt.bar(
            np.linspace(i - 0.25, i + 0.25, len(data_i)),
            data_i,
            0.5 / n_top_selected,
        )

    plt.xticks(np.arange(1, MAX_N_WRAP + 1))
    plt.xlabel("warps per block")
    plt.ylabel("time (s)")
    plt.show()


def plot_benchmarks_group_by_widths(data):
    """Plot the benchmarks grouped by block widths

    Plot the benchmarks for a specific GPU grouped by block width as a bar
    chart. y-axis in seconds, x-axis block width. All configurations (with
    shared memory padding where applicable) are shown within each block width.
    Within each group, they are ordered from fastest to slowest

    The plot is shown with matplotlib

    Args:
        data (np.ndarray): an element from the output of get_data()
    """

    data_copy = data.copy()

    # standard deviation not used, replace it with block width
    data_copy[:, 2] = WIDTHS_FOR_EACH_GRID_CONFIG[
        data_copy[:, 0].astype(np.int32)
    ]

    # remove the block configs with no padding
    for i in np.asarray(DIM_PADDING_INDEX_LIST)[:, 0]:
        data_copy = data_copy[data_copy[:, 0] != i, :]

    plt.figure()
    # for each width, use different colour
    for i, width in enumerate(WIDTHS):
        # only plot up to some width
        if i < MAX_N_WIDTH:
            data_i = data_copy[data_copy[:, 2] == width, :]
            data_i = np.sort(data_i[:, 1])
            i += 1  # for plotting starting at one instead of zero
            plt.bar(
                np.linspace(i - 0.25, i + 0.25, len(data_i)),
                data_i,
                0.5 / len(data_i),
            )

    plt.xticks(
        np.arange(1, MAX_N_WIDTH + 1), [str(2**i) for i in range(MAX_N_WIDTH)]
    )
    plt.xlabel("block width")
    plt.ylabel("time (s)")
    plt.show()


def main(files, gpu_names):
    data = get_data(files)
    plot_benchmarks_as_bars(data, gpu_names)
    plot_padding_bland_altman(data, gpu_names)
    plot_table(data, gpu_names)
    plot_benchmarks_group_by_warps(data[PLOT_SINGLE_GPU_INDEX])
    plot_benchmarks_group_by_widths(data[PLOT_SINGLE_GPU_INDEX])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gpu0")
    parser.add_argument("gpu1")
    parser.add_argument("gpu2")
    args = parser.parse_args()
    files = [args.gpu0, args.gpu1, args.gpu2]
    gpu_names = ["V100", "A100", "H100"]
    main(files, gpu_names)
