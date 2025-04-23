#pragma once

#include "matrix.h"

/**
 * Multiply matrix a by matrix b on the CPU.
 *
 * Dynamically allocates a third matrix for the result. The caller is
 * responsible for cleaning up this memory. If the matrices are incompatible for
 * multiplication, a null pointer is returned.
 *
 * @param left_mat The left matrix in the multiplication.
 * @param right_mat The right matrix in multiplication.
 * @return Pointer to result matrix or NULL if incompatible matrices.
 */
Matrix* cpu_matrix_multiply(Matrix* left_mat, Matrix* right_mat);

/**
 * Transpose a matrix using the CPU.
 *
 * Dynamically allocates a new matrix for the result. The caller is responsible
 * for freeing this memory.
 *
 * @param mat Pointer to the input matrix.
 * @return Pointer to the transposed matrix.
 */
Matrix* cpu_matrix_transpose(Matrix* mat);

/**
 * Compute the inverse of a matrix using the CPU.
 *
 * Allocates and returns a new matrix representing the inverse. If the matrix is
 * non-invertible or not square, a null pointer is returned.
 *
 * @param mat Pointer to the matrix to invert.
 * @return Pointer to the inverted matrix or NULL if not invertible.
 */
Matrix* cpu_matrix_inverse(Matrix* mat);

/**
 * Multiply every element of a matrix by a scalar value on the CPU.
 *
 * Allocates and returns a new matrix with scaled values.
 *
 * @param mat Pointer to the matrix to scale.
 * @param scalar The scalar value to multiply each element by.
 * @return Pointer to the scaled matrix.
 */
Matrix* cpu_scalar_multiply(Matrix* mat, float scalar);

/**
 * Add two matrices element-wise using the CPU.
 *
 * Allocates and returns a new matrix as the result. If matrix sizes are
 * incompatible, a null pointer is returned.
 *
 * @param mat_a First matrix.
 * @param mat_b Second matrix.
 * @return Pointer to the result matrix or NULL if dimensions mismatch.
 */
Matrix* cpu_matrix_add(Matrix* mat_a, Matrix* mat_b);

/**
 * Subtract one matrix from another element-wise using the CPU.
 *
 * Allocates and returns a new matrix as the result. If matrix sizes are
 * incompatible, a null pointer is returned.
 *
 * @param mat_a Matrix to subtract from.
 * @param mat_b Matrix to subtract.
 * @return Pointer to the result matrix or NULL if dimensions mismatch.
 */
Matrix* cpu_matrix_subtract(Matrix* mat_a, Matrix* mat_b);
