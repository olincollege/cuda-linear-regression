#include "linear_regression.h"

#include "cpu_matrix.h"
#include "gpu_matrix.h"
#include "matrix.h"

Matrix* cpu_regression(Matrix* X_mat, Matrix* y_mat) {
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

Matrix* gpu_regression(Matrix* X_mat, Matrix* y_mat) {
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
