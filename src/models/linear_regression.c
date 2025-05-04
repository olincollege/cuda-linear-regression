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

Matrix* gpu_regression(const Matrix* d_X_mat, const Matrix* d_y_mat,
                       size_t X_rows, size_t X_cols) {
  // Intentional swap of parameters because transposing
  // NOLINTNEXTLINE(readability-suspicious-call-argument)
  Matrix* d_X_transpose = gpu_matrix_transpose(d_X_mat, X_cols, X_rows);

  Matrix* d_X_t_X = gpu_matrix_multiply(d_X_transpose, d_X_mat, X_cols, X_cols);

  Matrix* h_X_t_X = copy_matrix_device_to_host(d_X_t_X);
  Matrix* h_X_t_X_inv = cpu_matrix_inverse(h_X_t_X);
  if (h_X_t_X_inv == NULL) {
    free_matrix_device(d_X_transpose);
    free_matrix_device(d_X_t_X);
    free_matrix_host(h_X_t_X);
    return NULL;
  }
  Matrix* d_X_t_X_inv = copy_matrix_host_to_device(h_X_t_X_inv);

  Matrix* d_X_t_y = gpu_matrix_multiply(d_X_transpose, d_y_mat, X_cols, 1);

  Matrix* d_weights = gpu_matrix_multiply(d_X_t_X_inv, d_X_t_y, X_cols, 1);

  free_matrix_device(d_X_transpose);
  free_matrix_device(d_X_t_X);
  free_matrix_host(h_X_t_X);
  free_matrix_host(h_X_t_X_inv);
  free_matrix_device(d_X_t_X_inv);
  free_matrix_device(d_X_t_y);

  return d_weights;
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
