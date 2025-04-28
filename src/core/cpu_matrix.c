#include "cpu_matrix.h"

#include <math.h>

#include "matrix.h"

const float ZERO_THRESH = (float)1e-7;

Matrix* cpu_matrix_multiply(const Matrix* left_mat, const Matrix* right_mat) {
  if (!left_mat || !right_mat || !left_mat->elements || !right_mat->elements)
    return NULL;

  if (left_mat->cols != right_mat->rows) return NULL;

  Matrix* result = create_matrix_host(left_mat->rows, right_mat->cols);
  if (!result) return NULL;

  for (int i = 0; i < left_mat->rows; ++i) {
    for (int j = 0; j < right_mat->cols; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < left_mat->cols; ++k) {
        sum += left_mat->elements[i * left_mat->cols + k] *
               right_mat->elements[k * right_mat->cols + j];
      }
      result->elements[i * result->cols + j] = sum;
    }
  }

  return result;
}

Matrix* cpu_matrix_transpose(const Matrix* mat) {
  if (!mat || !mat->elements) return NULL;

  Matrix* transposed = create_matrix_host(mat->cols, mat->rows);
  if (!transposed) return NULL;

  for (int i = 0; i < mat->rows; ++i) {
    for (int j = 0; j < mat->cols; ++j) {
      transposed->elements[j * transposed->cols + i] =
          mat->elements[i * mat->cols + j];
    }
  }

  return transposed;
}

Matrix* cpu_matrix_inverse(const Matrix* mat) {
  if (mat->cols != mat->rows) {
    return NULL;
  }
  size_t dim = mat->rows;
  size_t num_elements = dim * dim;

  // Create identity matrix
  Matrix* i_mat = create_matrix_host(dim, dim);

  for (size_t row = 0; row < dim; row++) {
    for (size_t col = 0; col < dim; col++) {
      if (row == col) {
        i_mat->elements[row * dim + col] = (float)1;
      } else {
        i_mat->elements[row * dim + col] = (float)0;
      }
    }
  }

  // Create a copy of the original matrix
  Matrix* mat_c = create_matrix_host(dim, dim);
  for (size_t i = 0; i < num_elements; i++) {
    mat_c->elements[i] = mat->elements[i];
  }

  // Do gauss jordan elimination
  for (size_t pivot_row = 0; pivot_row < dim; pivot_row++) {
    // If pivot is zero, swap with lower row
    if (fabsf(mat_c->elements[pivot_row * dim + pivot_row]) < ZERO_THRESH) {
      for (size_t swap_row = pivot_row + 1; swap_row < dim; swap_row++) {
        if (fabsf(mat_c->elements[swap_row * dim + pivot_row]) > ZERO_THRESH) {
          // Swap mat_c pivot row and swap row for mat_c and identity
          for (size_t col = 0; col < dim; col++) {
            float temp = mat_c->elements[pivot_row * dim + col];
            mat_c->elements[pivot_row * dim + col] =
                mat_c->elements[swap_row * dim + col];
            mat_c->elements[swap_row * dim + col] = temp;

            temp = i_mat->elements[pivot_row * dim + col];
            i_mat->elements[pivot_row * dim + col] =
                i_mat->elements[swap_row * dim + col];
            i_mat->elements[swap_row * dim + col] = temp;
          }
          break;
        }
      }
      // If still zero after trying to swap, it isn't invertible
      if (fabsf(mat_c->elements[pivot_row * dim + pivot_row]) < ZERO_THRESH) {
        free_matrix_host(i_mat);
        free_matrix_host(mat_c);
        return NULL;
      }
    }
    // Normalize pivot row
    float pivot_val = mat_c->elements[pivot_row * dim + pivot_row];
    for (size_t col = 0; col < dim; col++) {
      mat_c->elements[pivot_row * dim + col] /= pivot_val;
      i_mat->elements[pivot_row * dim + col] /= pivot_val;
    }
    // Eliminate the other rows
    for (size_t target_row = 0; target_row < dim; target_row++) {
      if (target_row != pivot_row) {
        float factor = mat_c->elements[target_row * dim + pivot_row];
        for (size_t col = 0; col < dim; col++) {
          mat_c->elements[target_row * dim + col] =
              mat_c->elements[target_row * dim + col] -
              factor * mat_c->elements[pivot_row * dim + col];
          i_mat->elements[target_row * dim + col] =
              i_mat->elements[target_row * dim + col] -
              factor * i_mat->elements[pivot_row * dim + col];
        }
      }
    }
  }
  free_matrix_host(mat_c);
  return i_mat;
}

Matrix* cpu_scalar_multiply(const Matrix* mat, const float scalar) {
  if (!mat || !mat->elements) return NULL;

  Matrix* result = create_matrix_host(mat->rows, mat->cols);
  if (!result) return NULL;

  for (int i = 0; i < mat->rows * mat->cols; ++i) {
    result->elements[i] = mat->elements[i] * scalar;
  }

  return result;
}

Matrix* cpu_matrix_add(const Matrix* mat_a, const Matrix* mat_b) {
  if (!mat_a || !mat_b || !mat_a->elements || !mat_b->elements) return NULL;

  if (mat_a->rows != mat_b->rows || mat_a->cols != mat_b->cols) return NULL;

  Matrix* result = create_matrix_host(mat_a->rows, mat_a->cols);
  if (!result) return NULL;

  for (int i = 0; i < mat_a->rows * mat_a->cols; ++i) {
    result->elements[i] = mat_a->elements[i] + mat_b->elements[i];
  }

  return result;
}

Matrix* cpu_matrix_subtract(const Matrix* mat_a, const Matrix* mat_b) {
  if (!mat_a || !mat_b || !mat_a->elements || !mat_b->elements) return NULL;

  if (mat_a->rows != mat_b->rows || mat_a->cols != mat_b->cols) return NULL;

  Matrix* result = create_matrix_host(mat_a->rows, mat_a->cols);
  if (!result) return NULL;

  for (int i = 0; i < mat_a->rows * mat_a->cols; ++i) {
    result->elements[i] = mat_a->elements[i] - mat_b->elements[i];
  }

  return result;
}
