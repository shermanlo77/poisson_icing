import ctypes
import math

import cupy
import matplotlib.pyplot as plt
import numpy as np
import pytest

import poisson_icing.gpu


@pytest.mark.parametrize("width", [1000, 1001])
@pytest.mark.parametrize("height", [1000])
@pytest.mark.parametrize("rate", [1.0])
@pytest.mark.parametrize("interaction", [0.8])
@pytest.mark.parametrize("n_sample", [10])
@pytest.mark.parametrize("n_thin", [500])
@pytest.mark.parametrize("seed", [314655814248387611881343760712275641615])
@pytest.mark.parametrize(
    "gpu_seed", [184330319631024176335817105491199072267877523595]
)
@pytest.mark.parametrize("block_dim", [(8, 4)])
@pytest.mark.parametrize("shared_mem_buff", [0, 6])
def test_gpu_sample(
    rate,
    rate_param,
    initial_image,
    interaction,
    n_sample,
    n_thin,
    rng_gpu,
    block_dim,
    shared_mem_buff,
    tmp_path,
):
    interaction = np.float32(interaction)
    image_array = poisson_icing.gpu.sample(
        rate_param,
        interaction,
        n_sample,
        n_thin,
        initial_image,
        rng_gpu,
        block_dim,
        shared_mem_buff,
    )

    vmax = math.ceil(rate + 1 * math.sqrt(rate))

    plt.figure()
    plt.imshow(initial_image, vmin=0, vmax=vmax)
    plt.colorbar()
    plt.savefig(tmp_path / "cuda-initial.png")
    plt.close()

    for i, image_i in enumerate(image_array):
        plt.figure()
        plt.imshow(np.asarray(image_i), vmin=0, vmax=vmax)
        plt.colorbar()
        plt.savefig(tmp_path / f"cuda-{i}.png")
        plt.close()


@pytest.mark.parametrize("width", [1000, 1001])
@pytest.mark.parametrize("height", [1000])
@pytest.mark.parametrize("rate", [1.0])
@pytest.mark.parametrize("interaction", [0.8])
@pytest.mark.parametrize("n_sample", [10])
@pytest.mark.parametrize("n_thin", [500])
@pytest.mark.parametrize("seed", [237334714993000014785369363746842320564])
@pytest.mark.parametrize("gpu_seed", [13507076133892765674107682974097366481])
@pytest.mark.parametrize("block_dim", [(8, 4)])
@pytest.mark.parametrize("shared_mem_buff", [0, 6])
@pytest.mark.parametrize("n_repeat", [10])
def test_reproducible(
    rate_param,
    initial_image,
    interaction,
    n_sample,
    n_thin,
    gpu_seed,
    block_dim,
    shared_mem_buff,
    n_repeat,
):
    rng = cupy.random.default_rng(gpu_seed)
    image_array_0 = poisson_icing.gpu.sample(
        rate_param,
        interaction,
        n_sample,
        n_thin,
        initial_image,
        rng,
        block_dim,
        shared_mem_buff,
    )

    for _ in range(n_repeat):
        rng = cupy.random.default_rng(gpu_seed)
        image_array_1 = poisson_icing.gpu.sample(
            rate_param,
            interaction,
            n_sample,
            n_thin,
            initial_image,
            rng,
            block_dim,
            shared_mem_buff,
        )
        assert np.array_equal(image_array_0, image_array_1)


@pytest.mark.parametrize("width", [1000, 1001])
@pytest.mark.parametrize("height", [1000])
@pytest.mark.parametrize("rate", [1.0])
@pytest.mark.parametrize("interaction", [0.8])
@pytest.mark.parametrize("n_sample", [10])
@pytest.mark.parametrize("seed", [65761717142118063498015790500159065881])
@pytest.mark.parametrize("gpu_seed", [319436614167081617057973460223105734853])
@pytest.mark.parametrize("block_dim", [(8, 4)])
@pytest.mark.parametrize("shared_mem_buff", [0, 6])
def test_checkerboard(
    rate_param,
    initial_image,
    interaction,
    n_sample,
    rng_gpu,
    block_dim,
    shared_mem_buff,
):
    image_array = poisson_icing.gpu.sample(
        rate_param,
        interaction,
        n_sample,
        1,
        initial_image,
        rng_gpu,
        block_dim,
        shared_mem_buff,
    )

    for i in range(1, n_sample):
        x, y = np.where(image_array[i] != image_array[i - 1])
        assert np.unique((x + y) % 2).size == 1


@pytest.mark.parametrize("width", [24, 25])
@pytest.mark.parametrize("height", [12])
@pytest.mark.parametrize("rate", [1.0])
@pytest.mark.parametrize("seed", [319177124816097325326554460856810870872])
@pytest.mark.parametrize("block_dim", [(4, 4)])
@pytest.mark.parametrize("shared_mem_buff", [0, 2])
def test_shared_mem(width, height, initial_image, block_dim, shared_mem_buff):
    n_block = (
        math.ceil(width / block_dim[0] / 2),
        math.ceil(height / block_dim[1]),
    )

    for block_id_y in range(n_block[1]):
        for block_id_x in range(n_block[0]):
            with cupy.cuda.Device():
                module = cupy.RawModule(path=poisson_icing.gpu.PTX_FILE_PATH)
                shared_mem_width = 2 * block_dim[0] + 2 + shared_mem_buff
                shared_mem_size = (
                    shared_mem_width
                    * (block_dim[1] + 2)
                    * ctypes.sizeof(ctypes.c_int32)
                )

                # transfer to gpu
                d_image = cupy.asarray(initial_image, cupy.int32)

                d_shared_mem = cupy.zeros(
                    (block_dim[1] + 2, shared_mem_width), cupy.int32
                )

                kernel = module.get_function("TestSharedMem")

                poisson_icing.gpu._set_global_const(
                    module, "kHeight", ctypes.c_int32, height
                )
                poisson_icing.gpu._set_global_const(
                    module, "kWidth", ctypes.c_int32, width
                )
                poisson_icing.gpu._set_global_const(
                    module,
                    "kSharedMemoryWidth",
                    ctypes.c_int32,
                    shared_mem_width,
                )

                kernel_args = (
                    d_image,
                    cupy.asarray(block_id_x, cupy.int32),
                    cupy.asarray(block_id_y, cupy.int32),
                    d_shared_mem,
                )

                kernel(
                    n_block, block_dim, kernel_args, shared_mem=shared_mem_size
                )

                shared_mem = d_shared_mem.get()

            if shared_mem_buff > 0:
                shared_mem = shared_mem[:, 0:-shared_mem_buff]

            image = np.pad(initial_image, 1, "symmetric")

            image_id_y_0 = block_id_y * block_dim[1]
            image_id_y_1 = (block_id_y + 1) * block_dim[1] + 2
            if image_id_y_1 > height + 2:
                image_id_y_1 = height + 2
            image_id_x_0 = block_id_x * 2 * block_dim[0]
            image_id_x_1 = (block_id_x + 1) * 2 * block_dim[0] + 2
            if image_id_x_1 > width + 2:
                image_id_x_1 = width + 2

            image = image[image_id_y_0:image_id_y_1, image_id_x_0:image_id_x_1]

            shared_mem = shared_mem[0 : image.shape[0], 0 : image.shape[1]]

            image[0, 0] = 0
            image[-1, 0] = 0

            if shared_mem.shape[1] > 3:
                image[0, -1] = 0
                image[-1, -1] = 0
            else:
                image[0, -1] = shared_mem[0, -1]
                image[-1, -1] = shared_mem[-1, -1]

            assert np.array_equal(shared_mem, image)
