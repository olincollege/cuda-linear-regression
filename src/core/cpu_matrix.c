#include "cpu_matrix.h"

#include <math.h>

#include "matrix.h"

const float ZERO_THRESH = (float)1e-7;

Matrix* cpu_matrix_inverse(Matrix* mat) {
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
