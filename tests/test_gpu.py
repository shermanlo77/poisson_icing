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


@pytest.mark.parametrize("width", [24, 25, 26, 27, 28, 29, 30])
@pytest.mark.parametrize("height", [12, 13])
@pytest.mark.parametrize("rate", [1.0])
@pytest.mark.parametrize(
    "seed",
    [
        319177124816097325326554460856810870872,
        18780873665392869462041247026533354832,
        18879499111372849442521197495859824697,
    ],
)
@pytest.mark.parametrize("block_dim", [(4, 4)])
@pytest.mark.parametrize("shared_mem_buff", [0, 2, 6])
def test_shared_mem(width, height, initial_image, block_dim, shared_mem_buff):
    n_block = (
        math.ceil(width / block_dim[0] / 2),
        math.ceil(height / block_dim[1]),
    )

    for block_id_y in range(n_block[1]):
        for block_id_x in range(n_block[0]):
            with cupy.cuda.Device():
                module = cupy.RawModule(path=poisson_icing.gpu.PTX_FILE_PATH)

                shared_mem_width, shared_mem_size = (
                    poisson_icing.gpu._get_shared_mem_size(
                        0, block_dim, shared_mem_buff
                    )
                )

                # transfer to gpu
                d_image = cupy.asarray(initial_image, cupy.int32)

                # for copying content of shared memory (to global memory)
                d_shared_mem = cupy.zeros(
                    (block_dim[1] + 2, shared_mem_width), cupy.int32
                )

                kernel = module.get_function("TestSharedMem")

                # set other parameters to zero as they are unused
                poisson_icing.gpu._set_all_global_const(
                    module, height, width, 0.0, 0, shared_mem_width
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

            # ignore buffed shared memory content, they are not used in
            # calculations
            if shared_mem_buff > 0:
                shared_mem = shared_mem[:, 0:-shared_mem_buff]

            image = np.pad(initial_image, 1, "symmetric")

            # calculate the indices to get the within-block image from the image
            # for image_id_x_1 and image_id_y_1, we add 2 for the one pixel
            # padding
            image_id_y_0 = block_id_y * block_dim[1]
            image_id_y_1 = (block_id_y + 1) * block_dim[1] + 2
            image_id_x_0 = block_id_x * 2 * block_dim[0]
            image_id_x_1 = (block_id_x + 1) * 2 * block_dim[0] + 2

            # case if the block goes over the image boundary
            if image_id_y_1 > height + 2:
                image_id_y_1 = height + 2
            if image_id_x_1 > width + 2:
                image_id_x_1 = width + 2

            # extract the corresponding within-block image
            image = image[image_id_y_0:image_id_y_1, image_id_x_0:image_id_x_1]
            # crop the image in shared memory, this removes the buffed shared
            # memory content and unused pixels if they fall outside the image
            # boundary
            shared_mem = shared_mem[0 : image.shape[0], 0 : image.shape[1]]

            # the four corners do not need to be tested as they are unused and
            # not assigned a value in GPU code
            image[0, 0] = shared_mem[0, 0]
            image[-1, 0] = shared_mem[-1, 0]
            image[0, -1] = shared_mem[0, -1]
            image[-1, -1] = shared_mem[-1, -1]

            assert np.array_equal(shared_mem, image)
