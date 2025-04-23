#pragma once

#include <cuda_runtime.h>

#include "matrix.h"

/**
 * Multiply two matrices using the GPU.
 *
 * Dynamically allocates a third matrix for the result. The caller is
 * responsible for cleaning up this memory. If the matrices are incompatible for
 * multiplication, a null pointer is returned.
 *
 * @param left_mat The left matrix in the multiplication.
 * @param right_mat The right matrix in multiplication.
 * @return pointer to result matrix or NULL if incompatible matrices
 */
Matrix* gpu_matrix_multiply(const Matrix* left_mat, const Matrix* right_mat);

/**
 * Kernel to run part of matrix multiplication.
 *
 * @param left_mat Pointer to left matrix in multiplication.
 * @param right_mat Pointer to right matrix in multiplication.
 * @param result Pointer to result matrix in multiplication.
 */
__global__ void matrix_multiply_kernel(const Matrix* left_mat, const Matrix* right_mat,
                                       Matrix* result);

/**
 * Transpose a matrix using the GPU.
 *
 * Dynamically allocates a new matrix for the result. The caller is responsible
 * for freeing this memory.
 *
 * @param mat Pointer to the input matrix.
 * @return Pointer to the transposed matrix.
 */
Matrix* gpu_matrix_transpose(const Matrix* mat);

/**
 * Kernel to perform matrix transposition.
 *
 * @param input Pointer to the input matrix.
 * @param output Pointer to the transposed matrix.
 */
__global__ void matrix_transpose_kernel(const Matrix* input, Matrix* output);

/**
 * Compute the inverse of a matrix using the GPU.
 *
 * Allocates and returns a new matrix representing the inverse. If the matrix is
 * non-invertible or not square, a null pointer is returned.
 *
 * @param mat Pointer to the matrix to invert.
 * @return Pointer to the inverted matrix or NULL if not invertible.
 */
Matrix* gpu_matrix_inverse(const Matrix* mat);

/**
 * Kernel to assist in computing matrix inverse.
 *
 * @param input Pointer to the input matrix.
 * @param output Pointer to the inverted matrix.
 */
__global__ void matrix_inverse_kernel(const Matrix* input, Matrix* output);

/**
 * Multiply every element of a matrix by a scalar value on the GPU.
 *
 * Allocates and returns a new matrix with scaled values.
 *
 * @param mat Pointer to the matrix to scale.
 * @param scalar The scalar value to multiply each element by.
 * @return Pointer to the scaled matrix.
 */
Matrix* gpu_scalar_multiply(const Matrix* mat, const float scalar);

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
 * Allocates and returns a new matrix as the result. If matrix sizes are
 * incompatible, a null pointer is returned.
 *
 * @param mat_a First matrix.
 * @param mat_b Second matrix.
 * @return Pointer to the result matrix or NULL if dimensions mismatch.
 */
Matrix* gpu_matrix_add(const Matrix* mat_a, const Matrix* mat_b);

/**
 * Kernel to perform element-wise matrix addition.
 *
 * @param mat_a First matrix.
 * @param mat_b Second matrix.
 * @param result Output matrix containing the sum.
 */
__global__ void matrix_add_kernel(const Matrix* mat_a, const Matrix* mat_b, Matrix* result);

/**
 * Subtract one matrix from another element-wise using the GPU.
 *
 * Allocates and returns a new matrix as the result. If matrix sizes are
 * incompatible, a null pointer is returned.
 *
 * @param mat_a Matrix to subtract from.
 * @param mat_b Matrix to subtract.
 * @return Pointer to the result matrix or NULL if dimensions mismatch.
 */
Matrix* gpu_matrix_subtract(const Matrix* mat_a, const Matrix* mat_b);

/**
 * Kernel to perform element-wise matrix subtraction.
 *
 * @param mat_a Matrix to subtract from.
 * @param mat_b Matrix to subtract.
 * @param result Output matrix containing the difference.
 */
__global__ void matrix_subtract_kernel(const Matrix* mat_a, const Matrix* mat_b,
                                       Matrix* result);
