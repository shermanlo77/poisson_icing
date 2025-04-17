"""CPU implementation of Gibbs sampling on the Poisson-Ising model"""

import itertools
import math

import numpy as np
from scipy import special


def _get_checkerboards(height, width):
    """Get padded boolean matrices representing colours on a checkerboard

    Get padded boolean matrices representing colours on a checkerboard, ie
    True for black squares OR True for white squares. It is padded so that
    a one pixel border of False surrounds the checkerboard

    Args:
        height (int): height of checkerboard_
        width (int): width of checkerboard

    Returns:
        list: list of two boolean matrices, one for black squares and one for
            white_squares
        list: list of two integers, the number of black squares and the number
            of white squares
    """
    x_grid, y_grid = np.meshgrid(range(width), range(height))
    checkerboard = (x_grid + y_grid) % 2
    padded_checkerboards = []
    n_terms = []
    for i in range(2):
        padded_checkerboard_i = np.zeros(
            [height + 2, width + 2], dtype=np.bool_
        )
        padded_checkerboard_i[1:-1, 1:-1] = checkerboard == i
        padded_checkerboards.append(padded_checkerboard_i)
        n_terms.append(np.sum(padded_checkerboard_i))

    return padded_checkerboards, n_terms


def _get_shifted_checkerboards(checkerboards):
    """Get shifted versions of the checkerboards from get_checkerboards()

    For each boolean matrix in checkerboards, return that matrix shifted, one
    for each of the four cardinal directions. They will enter the one pixel
    border introduced in get_checkerboards()

    Args:
        checkerboards (list): checkerboard return value from,
            get_checkerboards(), list of boolean matrices

    Returns:
        list: nested list
            dim 0 for each checkerboard in checkerboards
            dim 1 four boolean matrices, the corresponding checkerboard moved
                one pixel, one for each of the four cardinal directions

    """
    shifted_checkerboards = []

    def shift_index_iter():
        return itertools.chain(
            itertools.product([slice(1, -1)], [slice(0, -2), slice(2, None)]),
            itertools.product([slice(0, -2), slice(2, None)], [slice(1, -1)]),
        )

    for checkerboard_i in checkerboards:
        shifted_checkerboards_i = []
        for slice_i in shift_index_iter():
            checkerboard_slice = np.zeros_like(checkerboard_i)
            checkerboard_slice[slice_i] = checkerboard_i[1:-1, 1:-1]
            shifted_checkerboards_i.append(checkerboard_slice)
        shifted_checkerboards.append(shifted_checkerboards_i)

    return shifted_checkerboards


def _sample(
    rate_param,
    interaction,
    max_value,
    checkerboard_padded,
    shifted_checkerboards,
    n_terms,
    rng,
    image_padded,
    prob_terms,
):
    """Do one iteration of Gibbs sampling

    Args:
        rate_param (np.darray): matrix, of size height x width, of non-zero
            poisson rates
        interaction (float): interaction term, real number
        max_value (int): the highest value + 1 to use
        checkerboard_padded (np.darray): matrix of Booleans of size (height + 2)
            x (width + 2) in a padded checkerboard layout. Can be one of the
            outputs of _get_checkerboards()
        shifted_checkerboards (list):  list of matrices of Booleans of size
            (height + 2) x (width + 2) in a shifted padded checkerboard layout.
            Can be one of the outputs of _get_shifted_checkerboards()
        n_terms (int): number of True values in checkerboard_padded
        rng: random number generator
        image_padded (np.darray): matrix of integers of size (height + 2)
            x (width + 2). The image to be sampled padded
        prob_terms (np.darray): matrix of floats of size n_terms x max_value. A
            temporary matrix to store values of the conditional probability mass
            function
    """
    # get checkerboard without padding
    checkerboard = checkerboard_padded[1:-1, 1:-1]

    # calculate and store the log probability mass, up to a constant, for
    # each poisson value up to max_value
    for poisson_value in range(max_value):
        # poisson pmf
        rate_array = rate_param[checkerboard]
        log_prob = np.float32(
            poisson_value * np.log(rate_array)
            - special.loggamma(poisson_value + 1)
        )

        # interaction terms
        for shifted_checkerboard_i in shifted_checkerboards:
            log_prob += np.float32(
                -interaction
                * np.square(
                    image_padded[shifted_checkerboard_i] - poisson_value
                )
            )

        # store the log probability mass terms
        prob_terms[:, poisson_value] = log_prob

    # normalise terms
    prob_terms = np.exp(prob_terms)
    prob_terms /= np.sum(prob_terms, axis=1, keepdims=True)

    # sample from the normalised distribution
    index = rng.random([n_terms, 1]) < np.cumsum(prob_terms, 1)
    image_padded[checkerboard_padded] = np.argmax(index, 1)


def sample(rate_param, interaction, n_sample, n_thin, initial_image, rng):
    """Do Gibbs sampling

    Do Gibbs sampling for a Poisson Ising model. Each pixel is Poisson
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
        rng: random number generator

    Returns:
        np.ndarray: size n_iter x height x width, an image for each sample,
            dtype int32
    """
    height = rate_param.shape[0]
    width = rate_param.shape[1]

    max_rate = np.max(rate_param)
    max_value = math.ceil(max_rate + 5 * math.sqrt(max_rate))

    # store all samples
    image_array = np.zeros((n_sample, height, width), dtype=np.int32)

    # initial image
    initial_image = np.astype(initial_image, np.int32)
    image_padded = np.pad(initial_image, 1, "symmetric")

    # get checkerboards
    checkerboards, n_terms = _get_checkerboards(height, width)
    shifted_checkerboards = _get_shifted_checkerboards(checkerboards)

    # preallocated temporary variable
    # list of 2 arrays to store unnormalised log terms, one for black squares
    # and another for white squares
    # each array is of dimension of number of white/black squares x max_value
    prob_terms_array = [
        np.zeros((n_terms_i, max_value), dtype=np.float32)
        for n_terms_i in n_terms
    ]

    # gibbs sampling here, switch between white and black after every sample
    sample_id = 0
    for i_iter in range(n_sample * n_thin):
        _sample(
            rate_param,
            interaction,
            max_value,
            checkerboards[i_iter % 2],
            shifted_checkerboards[i_iter % 2],
            n_terms[i_iter % 2],
            rng,
            image_padded,
            prob_terms_array[i_iter % 2],
        )

        image_padded = np.pad(image_padded[1:-1, 1:-1], 1, "symmetric")

        # save the sample
        if (i_iter % n_thin) == (n_thin - 1):
            image_array[sample_id] = image_padded[1:-1, 1:-1]
            sample_id += 1

    return image_array
