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
void gpu_matrix_multiply(const Matrix* d_left, const Matrix* d_right, Matrix* d_result) {
  // Set up kernel launch dimensions
  dim3 blockDim(16, 16);
  dim3 gridDim((d_result->cols + 15) / 16, (d_result->rows + 15) / 16);

  // Launch kernel
  matrix_multiply_kernel<<<gridDim, blockDim>>>(d_left, d_right, d_result);
  check_device_error("Kernel launch failed", cudaGetLastError());
}

__global__ void matrix_transpose_kernel(const Matrix* input, Matrix* output) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < input->rows && col < input->cols) {
    output->elements[col * output->cols + row] =
        input->elements[row * input->cols + col];
  }
}

void gpu_matrix_transpose(const Matrix* d_input, Matrix* d_output) {
  // Launch kernel
  dim3 block_size(16, 16);
  dim3 grid_size((d_input->cols + 15) / 16, (d_input->rows + 15) / 16);
  matrix_transpose_kernel<<<grid_size, block_size>>>(d_input, d_output);
  check_device_error("matrix_transpose_kernel", cudaGetLastError());
}

__global__ void scalar_multiply_kernel(const Matrix* input, float scalar,
                                       Matrix* output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = input->rows * input->cols;

  if (idx < total_elements) {
    output->elements[idx] = input->elements[idx] * scalar;
  }
}

void gpu_scalar_multiply(const Matrix* d_input, float scalar, Matrix* d_output) {
  int total_elements = d_input->rows * d_input->cols;
  int threads_per_block = 256;
  int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
  scalar_multiply_kernel<<<blocks, threads_per_block>>>(d_input, scalar, d_output);
  check_device_error("scalar_multiply_kernel", cudaGetLastError());

}

__global__ void matrix_add_kernel(const Matrix* mat_a, const Matrix* mat_b,
                                  Matrix* result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = mat_a->rows * mat_a->cols;

  if (idx < total_elements) {
    result->elements[idx] = mat_a->elements[idx] + mat_b->elements[idx];
  }
}


void gpu_matrix_add(const Matrix* d_a, const Matrix* d_b, Matrix* d_result) {
  int total_elements = d_a->rows * d_a->cols;
  int threads_per_block = 256;
  int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
  matrix_add_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_result);
  check_device_error("matrix_add_kernel", cudaGetLastError());
}


__global__ void matrix_subtract_kernel(const Matrix* mat_a, const Matrix* mat_b,
                                       Matrix* result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = mat_a->rows * mat_a->cols;

  if (idx < total_elements) {
    result->elements[idx] = mat_a->elements[idx] - mat_b->elements[idx];
  }
}

void gpu_matrix_subtract(const Matrix* d_a, const Matrix* d_b, Matrix* d_result) {
  int total_elements = d_a->rows * d_a->cols;
  int threads_per_block = 256;
  int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
  matrix_subtract_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_result);
  check_device_error("matrix_subtract_kernel", cudaGetLastError());
}
