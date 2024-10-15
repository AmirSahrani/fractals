#include <cuComplex.h>
#include <mandelbrot.c>
#include <vector_types.h>

__device__ cuDoubleComplex cpow(cuDoubleComplex z, int power) {
  cuDoubleComplex result = make_cuDoubleComplex(1.0, 0.0);
  for (int i = 0; i < power; i++) {
    result = cuCmul(result, z);
  }
  return result;
}

__device__ int iterate_mandelbrotGPU(cuDoubleComplex start_c) {
  float bound = 2;
  cuDoubleComplex z = start_c;
  int iter;

  for (iter = 0; (iter < MAX_ITERATIONS); iter++) {
    z = cuCadd(cpow((cuDoubleComplex)z, 5.0), start_c);
    if (cuCabs(z) > 2.0) {
      return iter;
    }
  };
  return MAX_ITERATIONS;
}

__global__ void GPUIterations(double complex_plane[SCREENWIDTH][2],
                              int *results, int width, int height,
                              int max_iterations) {
  int index = blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < width * height; i += stride) {
    if (i < SCREENWIDTH * 2) {
      cuDoubleComplex c =
          make_cuDoubleComplex(complex_plane[i][0], complex_plane[i][1]);
      results[i] = iterate_mandelbrotGPU(c);
    }
  }
}
