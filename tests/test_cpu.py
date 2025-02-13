import math

import matplotlib.pyplot as plt
import numpy as np
import pytest

import poisson_icing.cpu


@pytest.mark.parametrize("width", [200])
@pytest.mark.parametrize("height", [200])
@pytest.mark.parametrize("rate", [1.0])
@pytest.mark.parametrize("interaction", [0.8])
@pytest.mark.parametrize("n_sample", [10])
@pytest.mark.parametrize("n_thin", [10])
@pytest.mark.parametrize("seed", [270438581656206183174160986409082595995])
def test_cpu_sample(
    rate,
    rate_param,
    initial_image,
    interaction,
    n_sample,
    n_thin,
    rng,
    tmp_path,
):
    interaction = np.float32(interaction)
    image_array = poisson_icing.cpu.sample(
        rate_param, interaction, n_sample, n_thin, initial_image, rng
    )
    vmax = math.ceil(rate + 1 * math.sqrt(rate))

    plt.figure()
    plt.imshow(initial_image, vmin=0, vmax=vmax)
    plt.colorbar()
    plt.savefig(tmp_path / "cpu-initial.png")
    plt.close()

    for i, image_i in enumerate(image_array):
        plt.figure()
        plt.imshow(np.asarray(image_i), vmin=0, vmax=vmax)
        plt.colorbar()
        plt.savefig(tmp_path / f"cpu-{i}.png")
        plt.close()


@pytest.mark.parametrize("width", [200, 201])
@pytest.mark.parametrize("height", [200])
@pytest.mark.parametrize("rate", [1.0])
@pytest.mark.parametrize("interaction", [0.8])
@pytest.mark.parametrize("n_sample", [10])
@pytest.mark.parametrize("n_thin", [10])
@pytest.mark.parametrize("seed", [142819611142013811249604486498910418401])
@pytest.mark.parametrize("n_repeat", [10])
def test_reproducible(
    rate_param,
    initial_image,
    interaction,
    n_sample,
    n_thin,
    seed,
    n_repeat,
):
    rng = np.random.default_rng(seed)
    image_array_0 = poisson_icing.cpu.sample(
        rate_param,
        interaction,
        n_sample,
        n_thin,
        initial_image,
        rng,
    )

    for _ in range(n_repeat):
        rng = np.random.default_rng(seed)
        image_array_1 = poisson_icing.cpu.sample(
            rate_param,
            interaction,
            n_sample,
            n_thin,
            initial_image,
            rng,
        )

        assert np.array_equal(image_array_0, image_array_1)


@pytest.mark.parametrize("width", [200, 201])
@pytest.mark.parametrize("height", [200])
@pytest.mark.parametrize("rate", [1.0])
@pytest.mark.parametrize("interaction", [0.8])
@pytest.mark.parametrize("n_sample", [10])
@pytest.mark.parametrize("seed", [263746064550849027010058977245320330129])
def test_checkerboard(rate_param, initial_image, interaction, n_sample, rng):
    image_array = poisson_icing.cpu.sample(
        rate_param,
        interaction,
        n_sample,
        1,
        initial_image,
        rng,
    )

    for i in range(1, n_sample):
        x, y = np.where(image_array[i] != image_array[i - 1])
        assert np.unique((x + y) % 2).size == 1
