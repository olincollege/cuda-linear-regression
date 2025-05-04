#pragma once

#include "matrix.h"

extern const float ZERO_THRESH;

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
Matrix* cpu_matrix_multiply(const Matrix* left_mat, const Matrix* right_mat);

/**
 * Transpose a matrix using the CPU.
 *
 * Dynamically allocates a new matrix for the result. The caller is responsible
 * for freeing this memory.
 *
 * @param mat Pointer to the input matrix.
 * @return Pointer to the transposed matrix.
 */
Matrix* cpu_matrix_transpose(const Matrix* mat);

/**
 * Ensure a non-zero pivot element by row swapping if needed.
 *
 * Checks if the pivot element is near zero. If so, searches a lower row with
 * a sufficiently large value in the pivot column and swaps rows in both the
 * input and inverse matrices. If no such row is found, the matrix is not
 * invertible.
 *
 * @param input_mat Pointer to the matrix being transformed.
 * @param inverse_mat Pointer to the matrix tracking the inverse (initially
 * identity).
 * @param pivot_row Index of the current pivot row.
 * @return 1 if a non-zero pivot was ensured; 0 if the matrix is not invertible.
 */
int unzero_pivot(Matrix* input_mat, Matrix* inverse_mat, size_t pivot_row);

/**
 * Perform row elimination for Gauss-Jordan elimination.
 *
 * Eliminates all rows except the pivot row by subtracting a multiple of the
 * pivot row from each target row. This operation is applied to both the input
 * matrix and the identity matrix being transformed into the inverse.
 *
 * @param input_mat Pointer to the matrix being transformed.
 * @param inverse_mat Pointer to the matrix tracking the inverse (initially
 * identity).
 * @param pivot_row Index of the current pivot row.
 */
void eliminate_rows(Matrix* input_mat, Matrix* inverse_mat, size_t pivot_row);

/**
 * Compute the inverse of a matrix using the CPU.
 *
 * Allocates and returns a new matrix representing the inverse. If the matrix is
 * non-invertible or not square, a null pointer is returned.
 *
 * @param mat Pointer to the matrix to invert.
 * @return Pointer to the inverted matrix or NULL if not invertible.
 */
Matrix* cpu_matrix_inverse(const Matrix* mat);

/**
 * Multiply every element of a matrix by a scalar value on the CPU.
 *
 * Allocates and returns a new matrix with scaled values.
 *
 * @param mat Pointer to the matrix to scale.
 * @param scalar The scalar value to multiply each element by.
 * @return Pointer to the scaled matrix.
 */
Matrix* cpu_scalar_multiply(const Matrix* mat, float scalar);

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
Matrix* cpu_matrix_add(const Matrix* mat_a, const Matrix* mat_b);

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
Matrix* cpu_matrix_subtract(const Matrix* mat_a, const Matrix* mat_b);
