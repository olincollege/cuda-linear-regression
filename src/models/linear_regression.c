#include "linear_regression.h"

#include <math.h>

#include "cpu_matrix.h"
#include "gpu_matrix.h"
#include "matrix.h"

Matrix* cpu_regression(const Matrix* X_mat, const Matrix* y_mat) {
  if (X_mat->rows != y_mat->rows) {
    return NULL;
  }
  Matrix* X_transpose = cpu_matrix_transpose(X_mat);
  Matrix* X_t_X = cpu_matrix_multiply(X_transpose, X_mat);
  Matrix* X_t_X_inv = cpu_matrix_inverse(X_t_X);
  if (X_t_X_inv == NULL) {
    free_matrix_host(X_transpose);
    free_matrix_host(X_t_X);
    return NULL;
  }
  Matrix* X_t_y = cpu_matrix_multiply(X_transpose, y_mat);
  Matrix* weights = cpu_matrix_multiply(X_t_X_inv, X_t_y);

  free_matrix_host(X_transpose);
  free_matrix_host(X_t_X);
  free_matrix_host(X_t_X_inv);
  free_matrix_host(X_t_y);

  return weights;
}

Matrix* gpu_regression(const Matrix* X_mat, const Matrix* y_mat) {
  if (X_mat->rows != y_mat->rows) {
    return NULL;
  }
  Matrix* X_transpose = gpu_matrix_transpose(X_mat);
  Matrix* X_t_X = gpu_matrix_multiply(X_transpose, X_mat);
  Matrix* X_t_X_inv = cpu_matrix_inverse(X_t_X);
  if (X_t_X_inv == NULL) {
    free_matrix_host(X_transpose);
    free_matrix_host(X_t_X);
    return NULL;
  }
  Matrix* X_t_y = gpu_matrix_multiply(X_transpose, y_mat);
  Matrix* weights = gpu_matrix_multiply(X_t_X_inv, X_t_y);

  free_matrix_host(X_transpose);
  free_matrix_host(X_t_X);
  free_matrix_host(X_t_X_inv);
  free_matrix_host(X_t_y);

  return weights;
}

float mae_loss(const Matrix* targets, const Matrix* truths) {
  if (targets->rows != truths->rows || targets->cols != 1 ||
      truths->cols != 1) {
    return -1.0F;
  }
  Matrix* diff = cpu_matrix_subtract(targets, truths);
  float total_diff = 0;
  for (size_t i = 0; i < diff->rows; i++) {
    total_diff += fabsf(diff->elements[i]);
  }
  free_matrix_host(diff);
  return total_diff / (float)targets->rows;
}
