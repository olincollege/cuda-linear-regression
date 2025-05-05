#include <stdbool.h>

#pragma once

#include <cuda_runtime.h>

#include "matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Multiply two matrices using the GPU.
 *
 * @param d_left Pointer to device memory for left matrix.
 * @param d_right Pointer to device memory for right matrix.
 * @param out_rows The number of rows of the output matrix.
 * @param out_cols The number of cols of the output matrix.
 *
 * @return Pointer to the product of the two matrices or NULL if operation not
 * feasible
 */
Matrix* gpu_matrix_multiply(const Matrix* d_left, const Matrix* d_right,
                            size_t out_rows, size_t out_cols);

/**
 * Kernel to run part of matrix multiplication.
 *
 * @param left_mat Pointer to left matrix in multiplication.
 * @param right_mat Pointer to right matrix in multiplication.
 * @param result Pointer to result matrix in multiplication.
 * @param error_flag Pointer to a boolean value indicating whether an error
 * occurred in the kernel
 */
__global__ void matrix_multiply_kernel(const Matrix* left_mat,
                                       const Matrix* right_mat, Matrix* result,
                                       bool* error_flag);

/**
 * Transpose a matrix using the GPU.
 *
 * @param d_input Pointer to device memory for input matrix.
 * @param out_rows The number of rows of the output matrix.
 * @param out_cols The number of cols of the output matrix.
 *
 * @return Pointer to the transposed matrix.
 */
Matrix* gpu_matrix_transpose(const Matrix* d_input, size_t out_rows,
                             size_t out_cols);

/**
 * Kernel to perform matrix transposition.
 *
 * @param input Pointer to the input matrix.
 * @param output Pointer to the transposed matrix.
 */
__global__ void matrix_transpose_kernel(const Matrix* input, Matrix* output);

/**
 * Multiply every element of a matrix by a scalar value on the GPU.
 *
 * @param d_input Pointer to device memory for input matrix.
 * @param scalar Scalar multiplier.
 * @param out_rows The number of rows of the output matrix.
 * @param out_cols The number of cols of the output matrix.
 *
 * @return Pointer to the product of the matrix and scalar value.
 */
Matrix* gpu_scalar_multiply(const Matrix* d_input, float scalar,
                            size_t out_rows, size_t out_cols);

/**
 * Kernel to perform element-wise scalar multiplication.
 *
 * @param input Pointer to the input matrix.
 * @param scalar The scalar multiplier.
 * @param output Pointer to the result matrix.
 */
__global__ void scalar_multiply_kernel(const Matrix* input, const float scalar,
                                       Matrix* output);

/**
 * Add two matrices element-wise using the GPU.
 *
 * @param d_a Pointer to device memory for first matrix.
 * @param d_b Pointer to device memory for second matrix.
 * @param out_rows The number of rows of the output matrix.
 * @param out_cols The number of cols of the output matrix.
 *
 * @return Pointer to the sum of the two matrices.
 */
Matrix* gpu_matrix_add(const Matrix* d_a, const Matrix* d_b, size_t out_rows,
                       size_t out_cols);

/**
 * Kernel to perform element-wise matrix addition.
 *
 * @param mat_a First matrix.
 * @param mat_b Second matrix.
 * @param result Output matrix containing the sum.
 */
__global__ void matrix_add_kernel(const Matrix* mat_a, const Matrix* mat_b,
                                  Matrix* result);

/**
 * Subtract one matrix from another element-wise using the GPU.
 * @param d_a Pointer to device memory for first matrix.
 * @param d_b Pointer to device memory for second matrix.
 * @param out_rows The number of rows of the output matrix.
 * @param out_cols The number of cols of the output matrix.
 *
 * @return Pointer to the transposed matrix.
 */
Matrix* gpu_matrix_subtract(const Matrix* d_a, const Matrix* d_b,
                            size_t out_rows, size_t out_cols);

/**
 * Kernel to perform element-wise matrix subtraction.
 *
 * @param mat_a Matrix to subtract from.
 * @param mat_b Matrix to subtract.
 * @param result Output matrix containing the difference.
 */
__global__ void matrix_subtract_kernel(const Matrix* mat_a, const Matrix* mat_b,
                                       Matrix* result);

#ifdef __cplusplus
}
#endif
