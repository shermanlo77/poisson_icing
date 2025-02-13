"""GPU implementation of Gibbs sampling on the Poisson-Ising model

GPU implementation of Gibbs sampling on the Poisson-Ising model. This uses the
compiled .ptx file in PTX_FILE_PATH. Usually this is done using nvcc to compile
the source .cu file into a .ptx file. See the README.md for further information.
"""

import ctypes
from importlib.resources import files
import math

import cupy
import numpy as np


PTX_FILE_PATH = str(files("poisson_icing").joinpath("poissonicing.ptx"))


def _set_global_const(module, name, type_func, value):
    """Set global constants for the CUDA module

    Args:
        module: CuPy module
        name (str): name of the constant variable in the CUDA code
        type_func (function): the corresponding ctypes function for the type
        value: value to set the global constant
    """
    device_var = module.get_global(name)
    host_var = type_func(value)
    device_var.copy_from_host(
        ctypes.addressof(host_var), ctypes.sizeof(host_var)
    )


def sample(
    rate_param,
    interaction,
    n_sample,
    n_thin,
    initial_image,
    rng,
    n_thread_per_block,
    shared_mem_buff=0,
):
    """Do Gibbs sampling

    Do Gibbs sampling for a Poisson-Ising model. Each pixel is Poisson
    distributed but with an interaction term with four of its neighbouring
    pixels. Because pixels on black squares on a checkerboard are conditionally
    independent with the pixels on white squares, a sample is taken by sampling
    the black squares given the white squares. For the next sample, vice versa.

    Args:
        rate_param (np.darray): matrix, of size height x width, of non-zero
            poisson rates
        interaction (float): interaction term, real number
        n_sample (int): number of saved samples
        n_thin (int): number of samples (ot steps) between saved samples
        initial_image (np.darray): int32 image of size height x width, initial
            value to sample from
        rng: cupy random number generator
        n_thread_per_block (tuple): size two, number of threads per block
        shared_mem_buff (int): how much to extend the shared memory width by

    Returns:
        np.ndarray: size n_iter x height x width, an image for each sample,
            dtype int32
    """

    with cupy.cuda.Device():
        module = cupy.RawModule(path=PTX_FILE_PATH)

        height = rate_param.shape[0]
        width = rate_param.shape[1]

        max_rate = np.max(rate_param)
        max_value = math.ceil(max_rate + 5 * math.sqrt(max_rate))

        # allocate number of blocks
        # each thread is allocated a pair of pixels
        n_block = (
            math.ceil(width / n_thread_per_block[0] / 2),
            math.ceil(height / n_thread_per_block[1]),
        )

        n_thread_total = (
            n_block[0]
            * n_block[1]
            * n_thread_per_block[0]
            * n_thread_per_block[1]
        )

        # shared memory stores the following:
        #  - Copies of values in image within a block with a one pixel padding
        #  - Probability distribution for each pixel in the block and Poisson
        #    values 0, 1, 2, ... max_value - 1
        #
        # x 2 because each thread is allocated to a pair
        # + 2 for the one pixel padding on left and right
        # shared_mem_buff if requested to avoid bank conflicts
        shared_mem_width = 2 * n_thread_per_block[0] + 2 + shared_mem_buff
        shared_mem_size = shared_mem_width * (
            n_thread_per_block[1] + 2
        ) * ctypes.sizeof(ctypes.c_int32) + max_value * n_thread_per_block[
            0
        ] * n_thread_per_block[1] * ctypes.sizeof(ctypes.c_float)

        # transfer to gpu
        d_image = cupy.asarray(initial_image, cupy.int32)
        d_rate_param = cupy.asarray(rate_param, cupy.float32)
        d_image_array = cupy.zeros((n_sample, height, width), cupy.int32)

        # allocate for random numbers
        d_random_numbers = cupy.empty(n_thread_total, cupy.float32)

        # set gpu constants
        _set_global_const(module, "kHeight", ctypes.c_int32, height)
        _set_global_const(module, "kWidth", ctypes.c_int32, width)
        _set_global_const(module, "kInteraction", ctypes.c_float, interaction)
        _set_global_const(module, "kMaxValue", ctypes.c_int32, max_value)
        _set_global_const(
            module, "kSharedMemoryWidth", ctypes.c_int32, shared_mem_width
        )

        # gibbs sampling
        kernel = module.get_function("PoissonIcing")
        sample_id = 0
        for i_iter in range(n_sample * n_thin):
            # new random numbers for this iteration
            d_random_numbers[:] = rng.uniform(
                0, 1, n_thread_total, dtype=cupy.float32
            )

            kernel_args = (
                d_rate_param,
                cupy.asarray(i_iter, cupy.int32),
                d_random_numbers,
                d_image,
            )
            kernel(
                n_block,
                n_thread_per_block,
                kernel_args,
                shared_mem=shared_mem_size,
            )

            # save sample
            if i_iter % n_thin == n_thin - 1:
                d_image_array[sample_id] = d_image
                sample_id += 1

        return d_image_array.get()
