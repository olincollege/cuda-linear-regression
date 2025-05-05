#include <cuda_runtime.h>
#include <stdio.h>

#include "gpu_matrix.h"
#include "matrix.h"

/**
 * CUDA kernel for matrix multiplication.
 */
__global__ void matrix_multiply_kernel(const Matrix* left_mat,
                                       const Matrix* right_mat,
                                       Matrix* result, bool* error_flag) {
  // Check for incompatible matrix size                                      
  if (left_mat-> cols != right_mat-> rows){
    *error_flag = true;
    return;
  }

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
Matrix* gpu_matrix_multiply(const Matrix* d_left, const Matrix* d_right,
                            size_t out_rows, size_t out_cols) {
  // Set up kernel launch dimensions
  dim3 blockDim(16, 16);
  dim3 gridDim((out_cols + 15) / 16, (out_rows + 15) / 16);
  Matrix* d_result = create_matrix_device(out_rows, out_cols);

  // Set up error flag
  bool* d_error_flag;
  cudaMalloc(&d_error_flag, sizeof(bool));
  cudaMemset(d_error_flag, 0, sizeof(bool));

  // Launch kernel
  matrix_multiply_kernel<<<gridDim, blockDim>>>(d_left, d_right, d_result, d_error_flag);
  check_device_error("Kernel launch failed", cudaGetLastError());
  cudaDeviceSynchronize();
  
  // Check for error 
  bool error_flag = false;
  cudaMemcpy(&error_flag, d_error_flag, sizeof(bool), cudaMemcpyDeviceToHost);
  cudaFree(d_error_flag);

  if (error_flag){
    fprintf(stderr, "Matrix dimension mismatch\n");
    return nullptr;
  }
  return d_result;
}

__global__ void matrix_transpose_kernel(const Matrix* input, Matrix* output) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < input->rows && col < input->cols) {
    output->elements[col * output->cols + row] =
        input->elements[row * input->cols + col];
  }
}

Matrix* gpu_matrix_transpose(const Matrix* d_input, size_t out_rows,
                             size_t out_cols) {
  // Launch kernel
  dim3 block_size(16, 16);
  dim3 grid_size((out_rows + 15) / 16, (out_cols + 15) / 16);

  Matrix* d_output = create_matrix_device(out_rows, out_cols);

  matrix_transpose_kernel<<<grid_size, block_size>>>(d_input, d_output);
  check_device_error("matrix_transpose_kernel", cudaGetLastError());
  cudaDeviceSynchronize();

  return d_output;
}

__global__ void scalar_multiply_kernel(const Matrix* input, float scalar,
                                       Matrix* output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = input->rows * input->cols;

  if (idx < total_elements) {
    output->elements[idx] = input->elements[idx] * scalar;
  }
}

Matrix* gpu_scalar_multiply(const Matrix* d_input, float scalar,
                            size_t out_rows, size_t out_cols) {
  int total_elements = out_rows * out_cols;
  int threads_per_block = 256;
  int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

  Matrix* d_output = create_matrix_device(out_rows, out_cols);

  scalar_multiply_kernel<<<blocks, threads_per_block>>>(d_input, scalar,
                                                        d_output);
  check_device_error("scalar_multiply_kernel", cudaGetLastError());
  cudaDeviceSynchronize();

  return d_output;
}

__global__ void matrix_add_kernel(const Matrix* mat_a, const Matrix* mat_b,
                                  Matrix* result, bool* error_flag) {
  // Check for incompatible matrix size                                      
  if (mat_a-> cols != mat_b-> cols || mat_a-> rows != mat_b-> rows){
    *error_flag = true;
    return;
  }                            
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = mat_a->rows * mat_a->cols;

  if (idx < total_elements) {
    result->elements[idx] = mat_a->elements[idx] + mat_b->elements[idx];
  }
}

Matrix* gpu_matrix_add(const Matrix* d_a, const Matrix* d_b, size_t out_rows,
                       size_t out_cols) {
  int total_elements = out_rows * out_cols;
  int threads_per_block = 256;
  int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

  Matrix* d_result = create_matrix_device(out_rows, out_cols);
  
  // Set up error flag
  bool* d_error_flag;
  cudaMalloc(&d_error_flag, sizeof(bool));
  cudaMemset(d_error_flag, 0, sizeof(bool));

  // Launch kernel
  matrix_add_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_result, d_error_flag);
  check_device_error("matrix_add_kernel", cudaGetLastError());
  cudaDeviceSynchronize();

  // Check for error 
  bool error_flag = false;
  cudaMemcpy(&error_flag, d_error_flag, sizeof(bool), cudaMemcpyDeviceToHost);
  cudaFree(d_error_flag);

  if (error_flag){
    fprintf(stderr, "Matrix dimension mismatch\n");
    return nullptr;
  }

  return d_result;
}

__global__ void matrix_subtract_kernel(const Matrix* mat_a, const Matrix* mat_b,
                                       Matrix* result, bool* error_flag) {
  // Check for incompatible matrix size    
  if (mat_a-> cols != mat_b-> cols || mat_a-> rows != mat_b-> rows){
    *error_flag = true;
    return;
  }  
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = mat_a->rows * mat_a->cols;

  if (idx < total_elements) {
    result->elements[idx] = mat_a->elements[idx] - mat_b->elements[idx];
  }
}

Matrix* gpu_matrix_subtract(const Matrix* d_a, const Matrix* d_b,
                            size_t out_rows, size_t out_cols) {
  int total_elements = out_rows * out_cols;
  int threads_per_block = 256;
  int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
  Matrix* d_result = create_matrix_device(out_rows, out_cols);
  
  // Set up error flag
  bool* d_error_flag;
  cudaMalloc(&d_error_flag, sizeof(bool));
  cudaMemset(d_error_flag, 0, sizeof(bool));

  // Launch kernel
  matrix_subtract_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_result, d_error_flag);
  check_device_error("matrix_subtract_kernel", cudaGetLastError());
  cudaDeviceSynchronize();
  
  // Check for error
  bool error_flag = false;
  cudaMemcpy(&error_flag, d_error_flag, sizeof(bool), cudaMemcpyDeviceToHost);
  cudaFree(d_error_flag);

  if (error_flag){
    fprintf(stderr, "Matrix dimension mismatch\n");
    return nullptr;
  }

  return d_result;
}
