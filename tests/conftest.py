import importlib

import numpy as np
import pytest

if importlib.util.find_spec("cupy"):
    cupy = importlib.import_module("cupy")


@pytest.fixture
def rng(seed):
    return np.random.default_rng(seed)


@pytest.fixture
def rng_gpu(gpu_seed):
    return cupy.random.default_rng(gpu_seed)


@pytest.fixture
def rate_param(width, height, rate):
    x_grid, y_grid = np.meshgrid(
        np.linspace(-0.3, 0.3, width), np.linspace(-0.3, 0.3, height)
    )
    return rate * np.exp(-np.square(x_grid)) * np.exp(-np.square(y_grid))


@pytest.fixture
def initial_image(rate_param, rng):
    return rng.poisson(rate_param)
