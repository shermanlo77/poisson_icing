"""For checking if a grid configuration has a bank conflict

Use check_bank_conflict() to check if a given block dimension and
shared_mem_buff will have a bank conflict. The function main() investigates
various configuration and output ones with no bank conflict

Global variables:

 - BANK_WIDTH the number of banks
 - WARP_SIZE used in main(), ensures the number of threads in a block is a
     multiple of this

"""

import itertools

import numpy as np

import poisson_icing.cpu


BANK_WIDTH = 32
WARP_SIZE = 32


def check_bank_conflict(block_size, shared_mem_buff):
    """Check if a given configuration has a bank conflict

    Check if a given block dimension and shared memory buff has a bank conflict.
    It uses the checkerboard access pattern and check the accessed indices to
    check for bank conflicts.

    Args:
        block_size (list): a list of two integers, the block dimensions in x
            and y
        shared_mem_buff (int): how much to extend the shared memory width by

    Returns:
        boolean: True if no bank conflicts, else False
    """

    # shared memory copies twice the width for checkerboard access
    # one pixel padding for moving the checkerboard up, down, left right
    shared_mem_width = 2 * block_size[0] + 2 + shared_mem_buff
    shared_mem_height = block_size[1] + 2
    shared_mem_size = (shared_mem_height, shared_mem_width)

    # shared_mem contains the bank indices
    shared_mem = np.fromiter(
        itertools.cycle(range(BANK_WIDTH)),
        np.int64,
        shared_mem_size[0] * shared_mem_size[1],
    )
    shared_mem = shared_mem.reshape(shared_mem_size)

    checkerboards = poisson_icing.cpu._get_checkerboards(
        block_size[1], 2 * block_size[0]
    )[0]
    shifted_checkerboards = poisson_icing.cpu._get_shifted_checkerboards(
        checkerboards
    )

    # iterate through black, white, up, down, left right
    for checkerboard in itertools.chain(
        checkerboards, itertools.chain.from_iterable(shifted_checkerboards)
    ):
        # do not access shared memory buffed memory
        if shared_mem_buff > 0:
            column_falses = np.zeros(
                (shared_mem_height, shared_mem_buff), dtype=np.bool_
            )
            checkerboard = np.concatenate([checkerboard, column_falses], axis=1)

        # check if unique banks are accessed
        bank_index = shared_mem[checkerboard]
        n_unique = len(np.unique(bank_index))
        if n_unique != np.sum(checkerboard):
            return False

    return True


def main():
    for y in range(1, WARP_SIZE + 1):
        if WARP_SIZE % y == 0:
            factor = WARP_SIZE // y
        else:
            factor = WARP_SIZE
        n = (WARP_SIZE * WARP_SIZE) // factor
        x_array = factor * np.arange(1, n + 1)

        for x in x_array:
            for offset in range(WARP_SIZE):
                if check_bank_conflict((x, y), offset):
                    print(f"{x}x{y} + {offset}")
                    break


if __name__ == "__main__":
    main()
