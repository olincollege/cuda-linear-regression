#include "cpu_matrix.h"

#include <math.h>

#include "matrix.h"

const float ZERO_THRESH = (float)1e-7;

Matrix* cpu_matrix_multiply(const Matrix* left_mat, const Matrix* right_mat) {
  if (!left_mat || !right_mat || !left_mat->elements || !right_mat->elements) {
    return NULL;
  }

  if (left_mat->cols != right_mat->rows) {
    return NULL;
  }

  Matrix* result = create_matrix_host(left_mat->rows, right_mat->cols);
  if (!result) {
    return NULL;
  }

  for (size_t i = 0; i < left_mat->rows; ++i) {
    for (size_t j = 0; j < right_mat->cols; ++j) {
      float sum = 0.0F;
      for (size_t k = 0; k < left_mat->cols; ++k) {
        sum += left_mat->elements[i * left_mat->cols + k] *
               right_mat->elements[k * right_mat->cols + j];
      }
      result->elements[i * result->cols + j] = sum;
    }
  }

  return result;
}

Matrix* cpu_matrix_transpose(const Matrix* mat) {
  if (!mat || !mat->elements) {
    return NULL;
  }

  Matrix* transposed = create_matrix_host(mat->cols, mat->rows);
  if (!transposed) {
    return NULL;
  }

  for (size_t i = 0; i < mat->rows; ++i) {
    for (size_t j = 0; j < mat->cols; ++j) {
      transposed->elements[j * transposed->cols + i] =
          mat->elements[i * mat->cols + j];
    }
  }

  return transposed;
}

int unzero_pivot(Matrix* input_mat, Matrix* inverse_mat,
                 const size_t pivot_row) {
  size_t dim = input_mat->rows;
  if (fabsf(input_mat->elements[pivot_row * dim + pivot_row]) < ZERO_THRESH) {
    for (size_t swap_row = pivot_row + 1; swap_row < dim; swap_row++) {
      if (fabsf(input_mat->elements[swap_row * dim + pivot_row]) >
          ZERO_THRESH) {
        // Swap mat pivot row and swap row for mat and identity
        for (size_t col = 0; col < dim; col++) {
          float temp = input_mat->elements[pivot_row * dim + col];
          input_mat->elements[pivot_row * dim + col] =
              input_mat->elements[swap_row * dim + col];
          input_mat->elements[swap_row * dim + col] = temp;

          temp = inverse_mat->elements[pivot_row * dim + col];
          inverse_mat->elements[pivot_row * dim + col] =
              inverse_mat->elements[swap_row * dim + col];
          inverse_mat->elements[swap_row * dim + col] = temp;
        }
        break;
      }
    }
    // If still zero after trying to swap, it isn't invertible
    if (fabsf(input_mat->elements[pivot_row * dim + pivot_row]) < ZERO_THRESH) {
      return 0;
    }
  }
  return 1;
}

void eliminate_rows(Matrix* input_mat, Matrix* inverse_mat,
                    const size_t pivot_row) {
  size_t dim = input_mat->rows;
  // Eliminate the other rows
  for (size_t target_row = 0; target_row < dim; target_row++) {
    if (target_row != pivot_row) {
      float factor = input_mat->elements[target_row * dim + pivot_row];
      for (size_t col = 0; col < dim; col++) {
        input_mat->elements[target_row * dim + col] =
            input_mat->elements[target_row * dim + col] -
            factor * input_mat->elements[pivot_row * dim + col];
        inverse_mat->elements[target_row * dim + col] =
            inverse_mat->elements[target_row * dim + col] -
            factor * inverse_mat->elements[pivot_row * dim + col];
      }
    }
  }
}

Matrix* cpu_matrix_inverse(const Matrix* mat) {
  if (mat->cols != mat->rows) {
    return NULL;
  }
  size_t dim = mat->rows;
  size_t num_elements = dim * dim;

  // Create identity matrix
  Matrix* inv_mat = create_matrix_host(dim, dim);

  for (size_t row = 0; row < dim; row++) {
    for (size_t col = 0; col < dim; col++) {
      if (row == col) {
        inv_mat->elements[row * dim + col] = (float)1;
      } else {
        inv_mat->elements[row * dim + col] = (float)0;
      }
    }
  }

  // Create a copy of the original matrix
  Matrix* input_mat = create_matrix_host(dim, dim);
  for (size_t i = 0; i < num_elements; i++) {
    input_mat->elements[i] = mat->elements[i];
  }

  // Do gauss jordan elimination
  for (size_t pivot_row = 0; pivot_row < dim; pivot_row++) {
    // If pivot is zero, swap with lower row
    if (unzero_pivot(input_mat, inv_mat, pivot_row) == 0) {
      free_matrix_host(inv_mat);
      free_matrix_host(input_mat);
      return NULL;
    }

    // Normalize pivot row
    float pivot_val = input_mat->elements[pivot_row * dim + pivot_row];
    for (size_t col = 0; col < dim; col++) {
      input_mat->elements[pivot_row * dim + col] /= pivot_val;
      inv_mat->elements[pivot_row * dim + col] /= pivot_val;
    }
    eliminate_rows(input_mat, inv_mat, pivot_row);
  }
  free_matrix_host(input_mat);
  return inv_mat;
}

Matrix* cpu_scalar_multiply(const Matrix* mat, const float scalar) {
  if (!mat || !mat->elements) {
    return NULL;
  }

  Matrix* result = create_matrix_host(mat->rows, mat->cols);
  if (!result) {
    return NULL;
  }

  for (size_t i = 0; i < mat->rows * mat->cols; ++i) {
    result->elements[i] = mat->elements[i] * scalar;
  }

  return result;
}

Matrix* cpu_matrix_add(const Matrix* mat_a, const Matrix* mat_b) {
  if (!mat_a || !mat_b || !mat_a->elements || !mat_b->elements) {
    return NULL;
  }

  if (mat_a->rows != mat_b->rows || mat_a->cols != mat_b->cols) {
    return NULL;
  }

  Matrix* result = create_matrix_host(mat_a->rows, mat_a->cols);
  if (!result) {
    return NULL;
  }

  for (size_t i = 0; i < mat_a->rows * mat_a->cols; ++i) {
    result->elements[i] = mat_a->elements[i] + mat_b->elements[i];
  }

  return result;
}

Matrix* cpu_matrix_subtract(const Matrix* mat_a, const Matrix* mat_b) {
  if (!mat_a || !mat_b || !mat_a->elements || !mat_b->elements) {
    return NULL;
  }

  if (mat_a->rows != mat_b->rows || mat_a->cols != mat_b->cols) {
    return NULL;
  }

  Matrix* result = create_matrix_host(mat_a->rows, mat_a->cols);
  if (!result) {
    return NULL;
  }

  for (size_t i = 0; i < mat_a->rows * mat_a->cols; ++i) {
    result->elements[i] = mat_a->elements[i] - mat_b->elements[i];
  }

  return result;
}
