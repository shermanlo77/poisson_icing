/**
 * Gibbs sampling for a Poisson Ising model
 *
 * See PoissonIcing for the main GPU kernel
 *
 * To use, set the global constants before using, set the size of shared
 * memory accordingly and provide sufficient number of blocks
 *
 * A thread is allocated to pairs of pixels and the number of blocks should
 * cover this. That is:
 *   blockDim.x * gridDim.x > kWidth / 2
 *   blockDim.y * gridDim.y >= kHeight
 *
 * This kernel uses shared memory for the following:
 *  - Copies of values in image within a block with a one pixel padding
 *  - Probability distribution for each pixel in the block and Poisson values
 *    0, 1, 2, ... kMaxValue - 1
 *
 * Set shared memory to be of size:
 * (kSharedMemoryWidth * (blockDim.x + 2)) * sizeof(int)
 * + kMaxValue * blockDim.x * blockDim.y * sizeof(float)
 *
 * kSharedMemoryWidth must be at least 2 * blockDim.x  + 2 and can be larger
 * to avoid bank conflicts
 */

#include <cuda.h>

/** Height of the image */
__constant__ int kHeight;
/** Width of the image */
__constant__ int kWidth;
/** Constant interaction term */
__constant__ float kInteraction;
/** Truncate the Poission distribution at kMaxValue - 1 */
__constant__ int kMaxValue;
/** Width of shared memory, must be at least 2 * blockDim.x  + 2 */
__constant__ int kSharedMemoryWidth;

/**
 * Get index of shared memory for this thread
 *
 * @return index of shared memory for this thread
 */
__device__ int SharedIndex() {
  return (threadIdx.y + 1) * kSharedMemoryWidth + 2 * threadIdx.x + 1;
}

/**
 * Copy the image to shared memory
 *
 * Copy this thread's pixel to shared memory, captured by the block. Also copy
 * one pixel outside the block to shared memory. This is required for the
 * checkerboard pattern
 *
 * @param image location of this thread's pixel to copy from
 * @param image_idy row index of this thread
 * @param image_idx column index of this thread
 * @param shared_mem location of this thread's shared memory to copy into
 */
__device__ void CopyToSharedMem(int* image, int image_idy, int image_idx,
                                int* shared_mem) {
  // pairs of pixels are allocated to a thread
  // behaviour is different depending if kWidth is odd or even
  //
  // diagram of how threads are allocated in a row of the image
  // O = thread position in image
  // X = thread position if shifted by checkerboard_shift (+1)
  // | O X O X |   example of even kWidth
  // | O X O |     example of odd kWidth
  //
  // use symmetric padding
  //
  // the 4 neighbours (up, left, right, down) are required for the interation
  // terms
  //
  // for even kWidth, the most right thread will need to do a copy right AND a
  // copy right right
  //
  // for odd kWidth, the most right thread will need to do a copy right but a
  // right right is not needed

  int offset;  // offset of image to copy

  // copy this pixel
  *shared_mem = *image;

  if (image_idx < kWidth - 1) {
    // copy pixel to right
    offset = 1;
  } else {
    // right not available, do symmetric padding
    // this only happens for odd kWidth
    offset = 0;
  }
  *(shared_mem + 1) = *(image + offset);

  // copy values outside the block

  // top and left edge of image will line up with top and left edge of a block

  // top row in block
  if (threadIdx.y == 0) {
    if (image_idy > 0) {
      // copy up
      offset = -kWidth;
    } else {
      // top row in image does symmetric padding
      offset = 0;
    }
    *(shared_mem - kSharedMemoryWidth) = *(image + offset);

    // copy up + right if available
    if ((image_idy > 0) && (image_idx < kWidth - 1)) {
      offset = -kWidth + 1;
    } else if ((image_idy == 0) && (image_idx < kWidth - 1)) {
      // top row and the right is available, do symmetric padding
      offset = 1;
    } else {
      // corresponds to top right beyond boundary
      // top row in block correspond to top row in image and
      // image_idx + 1 >= kWidth
      // => image_idx == kWidth - 1 (this happens for odd kWidth)
      offset = 0;  // not strictly required as unused
    }
    *(shared_mem - kSharedMemoryWidth + 1) = *(image + offset);
  }

  // left column in block
  if (threadIdx.x == 0) {
    // copy left
    if (image_idx > 0) {
      offset = -1;
    } else {
      // symmetric padding for left column
      offset = 0;
    }
    *(shared_mem - 1) = *(image + offset);
  }

  // for right column and bottom row, blocks will overlap the image

  // right column of block or on right column of image
  if ((threadIdx.x == blockDim.x - 1) || (image_idx >= kWidth - 2)) {
    // copy right right
    if (image_idx < kWidth - 2) {
      offset = 2;
    } else if (image_idx == kWidth - 2) {
      // even kWidth and right right is one pixel outside image
      // so do symmetric padding
      offset = 1;
    } else {
      // odd kWidth, right right copy not needed
      offset = 0;  // strictly not needed
    }
    *(shared_mem + 2) = *(image + offset);
  }

  // bottom row in block or image
  if ((threadIdx.y == blockDim.y - 1) || (image_idy == kHeight - 1)) {
    // copy down
    if (image_idy < kHeight - 1) {
      offset = kWidth;
    } else {
      // bottom row of image, do symmetric padding
      offset = 0;
    }
    *(shared_mem + kSharedMemoryWidth) = *(image + offset);

    // copy down right
    if ((image_idy < kHeight - 1) && (image_idx < kWidth - 1)) {
      offset = kWidth + 1;
    } else if ((image_idy == kHeight - 1) && (image_idx < kWidth - 1)) {
      // bottom row of image, do symmetric padding
      offset = 1;
    } else {
      // corresponds to bottom right beyond boundary
      offset = 0;  // strictly not needed
    }
    *(shared_mem + kSharedMemoryWidth + 1) = *(image + offset);
  }
}

/**
 * Interaction term in the Poission distribution
 *
 * Take the squared difference of poisson_value and this thread's neighbour
 *
 * @param shared_mem this thread's shared memory
 * @param neighbour_offset offset of shared_mem to one of its neighbours
 * @param poisson_value the Poisson value to calculate the interaction term for
 */
__device__ float InteractionTerm(int* shared_mem, int neighbour_offset,
                                 int poisson_value) {
  float diff = __int2float_rn(*(shared_mem + neighbour_offset) - poisson_value);
  return -kInteraction * diff * diff;
}

/**
 * Calculate the the probability distribution up to a constant
 *
 * For this pixel (or thread), calculate the unnormalised probability
 * distribution, for all Poission values up to kMaxValue. They are stored in
 * the parameter prob_array
 *
 * @param shared_mem this thread's shared memory
 * @param log_rate_param_i log of the rate parameter for this thread
 * @param prob_array modified, to store the resulting unnormalised probability
 * distribution for this thread, the address is shifted by prob_array_shift
 * for every poisson value
 * @param prob_array_shift the amount to shift the address of prob_array for
 * every poisson value
 */
__device__ void CalcUnnormalisedProb(int* shared_mem, float log_rate_param_i,
                                     float* prob_array, int prob_array_shift) {
  for (int poisson_value = 0; poisson_value < kMaxValue; ++poisson_value) {
    float f_poisson_value = __int2float_rn(poisson_value);
    float log_prob =
        f_poisson_value * log_rate_param_i - lgammaf(f_poisson_value + 1);

    // interaction terms

    // up
    log_prob += InteractionTerm(shared_mem, -kSharedMemoryWidth, poisson_value);

    // left
    log_prob += InteractionTerm(shared_mem, -1, poisson_value);

    // right
    log_prob += InteractionTerm(shared_mem, 1, poisson_value);

    // down
    log_prob += InteractionTerm(shared_mem, kSharedMemoryWidth, poisson_value);

    *prob_array = expf(log_prob);
    prob_array += prob_array_shift;
  }
}

/**
 * Calculate the normalisation constant
 *
 * @param prob_array unnormalised probability distribution for this thread,
 * the address is shifted by prob_array_shift for every poisson value
 * @param prob_array_shift the amount to shift the address of prob_array for
 * every poisson value
 */
__device__ float GetNormalisationConstant(float* prob_array,
                                          int prob_array_shift) {
  float normalisation_constant = 0;
  for (int poisson_value = 0; poisson_value < kMaxValue; ++poisson_value) {
    normalisation_constant += *prob_array;
    prob_array += prob_array_shift;
  }
  return normalisation_constant;
}

/**
 * Do Gibbs sampling for this thread or pixel
 *
 * @param prob_array unnormalised probability distribution for this thread,
 * the address is shifted by prob_array_shift for every poisson value
 * @param prob_array_shift the amount to shift the address of prob_array for
 * every poisson value
 * @param random_value a random uniform (0, 1) value
 *
 * @return the sampled Poisson value
 */
__device__ int Sample(float* prob_array, int prob_array_shift,
                      float random_value) {
  float normalisation_constant =
      GetNormalisationConstant(prob_array, prob_array_shift);
  float sum = 0.0f;
  for (int poisson_value = 0; poisson_value < kMaxValue; ++poisson_value) {
    sum += *prob_array / normalisation_constant;
    prob_array += prob_array_shift;
    if (sum > random_value) {
      return poisson_value;
    }
  }
  return kMaxValue - 1;
}

/**
 * One iteration of Gibbs sampling for the Poisson image
 *
 * Does one iteration of Gibbs sampling for the Poisson image, sampling pixels
 * on a checkerboard. The colour of the checkerboard changes for every
 * iteration
 *
 * The image and rate_param shall have height kHeight and width kWidth,
 * assuming row major
 *
 * @param rate_param image of Poisson rate parameters
 * @param i_iter the iteration number
 * @param random_numbers array of random numbers, available for each thread,
 * size blockDim.x * gridDim.x * blockDim.y * gridDim.y
 * @param image modified, image of Poisson values, pixels on the checkerboard
 * for this iteration shall be sampled and replaced
 * @param prob_array array to store the unnormalised probability distribution
 * for each Poisson value and thread, size blockDim.x * gridDim.x * blockDim.y
 * * gridDim.y * kMaxValue
 */
extern "C" __global__ void PoissonIcing(float* rate_param, int* i_iter,
                                        float* random_numbers, int* image) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;  // row index
  int idy = threadIdx.y + blockIdx.y * blockDim.y;  // columns index
  int id = idx + idy * blockDim.x * gridDim.x;      // thread index

  // checkerboard access done by either shifting the thread to the pixel to
  // the right or not hence threads are distributed over pairs of pixels
  int image_idx = 2 * idx;
  int image_id = idy * kWidth + image_idx;

  if ((idy >= kHeight) || (image_idx >= kWidth)) {
    return;
  }

  // shift address to be relative to this thread or pixel
  image += image_id;
  rate_param += image_id;
  random_numbers += id;

  // shift shared memory address to be relative to this thread or pixel
  extern __shared__ int shared_mem_array[];
  int* shared_mem = shared_mem_array;

  float* prob_array = reinterpret_cast<float*>(
      shared_mem + kSharedMemoryWidth * (blockDim.y + 2));

  shared_mem += SharedIndex();
  prob_array += threadIdx.x + threadIdx.y * blockDim.x;

  CopyToSharedMem(image, idy, image_idx, shared_mem);
  __syncthreads();

  // shift addresses according to the checkerboard pattern
  // the "colour" of the checkerboard changes for each iteration, ie sample
  // black squares, then white squares, black squares, ...etc
  bool checkboard_shift = (*i_iter % 2) == (idy % 2);
  image_idx += checkboard_shift;
  if (image_idx >= kWidth) {
    return;
  }
  shared_mem += checkboard_shift;
  image += checkboard_shift;
  rate_param += checkboard_shift;

  CalcUnnormalisedProb(shared_mem, logf(*rate_param), prob_array,
                       blockDim.x * blockDim.y);

  *image = Sample(prob_array, blockDim.x * blockDim.y, *random_numbers);
}

/**
 * Test Shared Memory
 *
 * Only for testing. Copies the content of shared memory which stores copies of
 * image within a block with a one pixel padding
 *
 * @param image image of Poisson values
 * @param block_id_x which block to copy values from
 * @param block_id_y which block to copy values from
 * @param shared_mem_device modified, where to copy the content of shared memory
 * to
 */
extern "C" __global__ void TestSharedMem(int* image, int* block_id_x,
                                         int* block_id_y,
                                         int* shared_mem_device) {
  // only copy for a block
  // this is fine for testing purposes
  if (blockIdx.x != *block_id_x || blockIdx.y != *block_id_y) {
    return;
  }

  int idx = threadIdx.x + blockIdx.x * blockDim.x;  // row index
  int idy = threadIdx.y + blockIdx.y * blockDim.y;  // columns index

  // checkerboard access done by either shifting the thread to the pixel to
  // the right or not hence threads are distributed over pairs of pixels
  int image_idx = 2 * idx;
  int image_id = idy * kWidth + image_idx;

  if ((idy >= kHeight) || (image_idx >= kWidth)) {
    return;
  }

  // shift address to be relative to this thread or pixel
  image += image_id;

  // shift shared memory address to be relative to this thread or pixel
  extern __shared__ int shared_mem_array[];
  int* shared_mem = shared_mem_array;
  shared_mem += SharedIndex();

  CopyToSharedMem(image, idy, image_idx, shared_mem);
  __syncthreads();

  // single thread copies
  // fine for testing purposes
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    shared_mem = shared_mem_array;
    for (int y = 0; y < blockDim.y + 2; ++y) {
      for (int x = 0; x < kSharedMemoryWidth; ++x) {
        *shared_mem_device++ = *shared_mem++;
      }
    }
  }
}
