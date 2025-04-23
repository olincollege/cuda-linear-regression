#include <cuda_runtime.h>
#include <stdio.h>

#include "gpu_matrix.h"
#include "matrix.h"

/**
 * CUDA kernel for matrix multiplication.
 */
__global__ void matrix_multiply_kernel(const Matrix* left_mat,
                                       const Matrix* right_mat,
                                       Matrix* result) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < result->rows && col < result->cols) {
    float sum = 0.0f;
    for (int i = 0; i < left_mat->cols; ++i) {
      float a = left_mat->elements[row * left_mat->cols + i];
      float b = right_mat->elements[i * right_mat->cols + col];
      sum += a * b;
    }
    result->elements[row * result->cols + col] = sum;
  }
}

/**
 * Multiply two matrices using the GPU.
 */
Matrix* gpu_matrix_multiply(const Matrix* left_mat, const Matrix* right_mat) {
  if (left_mat->cols != right_mat->rows) {
    return NULL;
  }

  // Create device copies
  Matrix* d_left = copy_matrix_host_to_device(left_mat);
  Matrix* d_right = copy_matrix_host_to_device(right_mat);
  Matrix* d_result = create_matrix_device(left_mat->rows, right_mat->cols);

  // Set up kernel launch dimensions
  dim3 blockDim(16, 16);
  dim3 gridDim((right_mat->cols + 15) / 16, (left_mat->rows + 15) / 16);

  // Launch kernel
  matrix_multiply_kernel<<<gridDim, blockDim>>>(d_left, d_right, d_result);
  check_device_error("Kernel launch failed", cudaGetLastError());
  cudaDeviceSynchronize();

  // Copy result back to host
  Matrix* h_result = copy_matrix_device_to_host(d_result);

  // Cleanup
  free_matrix_device(d_left);
  free_matrix_device(d_right);
  free_matrix_device(d_result);

  return h_result;
}

__global__ void matrix_transpose_kernel(const Matrix* input, Matrix* output) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < input->rows && col < input->cols) {
    output->elements[col * output->cols + row] =
        input->elements[row * input->cols + col];
  }
}

Matrix* gpu_matrix_transpose(const Matrix* mat) {
  if (!mat || !mat->elements) return nullptr;

  // Allocate device memory for input matrix
  Matrix* d_input = copy_matrix_host_to_device(mat);
  if (!d_input) return nullptr;

  // Allocate device memory for output (transposed) matrix
  Matrix* d_output = create_matrix_device(mat->cols, mat->rows);
  if (!d_output) return nullptr;

  // Launch kernel
  dim3 block_size(16, 16);
  dim3 grid_size((mat->cols + block_size.x - 1) / block_size.x,
                 (mat->rows + block_size.y - 1) / block_size.y);

  matrix_transpose_kernel<<<grid_size, block_size>>>(d_input, d_output);
  check_device_error("Transpose Kernel Launch Failed", cudaGetLastError());

  // Copy result back to host
  Matrix* h_result = copy_matrix_device_to_host(d_output);

  // Free device memory
  free_matrix_device(d_input);
  free_matrix_device(d_output);

  return h_result;
}
