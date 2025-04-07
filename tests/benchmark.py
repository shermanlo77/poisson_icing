import numpy as np
import pytest

import poisson_icing.gpu


@pytest.mark.parametrize("width", [2048])
@pytest.mark.parametrize("height", [2048])
@pytest.mark.parametrize("rate", [0.9])
@pytest.mark.parametrize("interaction", [0.8])
@pytest.mark.parametrize("n_sample", [1000])
@pytest.mark.parametrize("n_thin", [500])
@pytest.mark.parametrize("seed", [314655814248387611881343760712275641615])
@pytest.mark.parametrize("gpu_seed", [1843303595])
@pytest.mark.parametrize(
    "block_dim,shared_mem_buff",
    [
        ((1, 32), 0),
        ((2, 16), 0),
        ((4, 8), 0),
        ((4, 8), 2),
        ((8, 4), 0),
        ((8, 4), 6),
        ((16, 2), 0),
        ((32, 1), 0),
        ((1, 64), 0),
        ((2, 32), 0),
        ((4, 16), 0),
        ((4, 16), 2),
        ((8, 8), 0),
        ((8, 8), 6),
        ((16, 4), 0),
        ((32, 2), 0),
        ((64, 1), 0),
        ((1, 128), 0),
        ((2, 64), 0),
        ((4, 32), 0),
        ((4, 32), 2),
        ((8, 16), 0),
        ((8, 16), 6),
        ((16, 8), 0),
        ((32, 4), 0),
        ((64, 2), 0),
        ((128, 1), 0),
        ((1, 256), 0),
        ((2, 128), 0),
        ((4, 64), 0),
        ((4, 64), 2),
        ((8, 32), 0),
        ((8, 32), 6),
        ((16, 16), 0),
        ((32, 8), 0),
        ((64, 4), 0),
        ((128, 2), 0),
        ((256, 2), 0),
    ],
)
@pytest.mark.parametrize("rounds", [32])
def test_gpu_benchmark(
    rate_param,
    initial_image,
    interaction,
    n_sample,
    n_thin,
    rng_gpu,
    block_dim,
    shared_mem_buff,
    benchmark,
    rounds,
):
    """Benchmark for different grid configurations"""
    benchmark.pedantic(
        lambda: poisson_icing.gpu.sample(
            rate_param,
            interaction,
            n_sample,
            n_thin,
            initial_image,
            rng_gpu,
            block_dim,
            shared_mem_buff,
        ),
        rounds=rounds,
        iterations=1,
    )
