#include <criterion/criterion.h>
#include <criterion/new/assert.h>
#include <stdlib.h>

#include "linear_regression.h"
#include "matrix.h"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers,cppcoreguidelines-pro-bounds-pointer-arithmetic)
// Since we're testing, magic numbers are needed to configure specific cases
// Also ignore pointer arithmetic warning, since in all cases it is in a loop
// which prevents out of bounds

// Test small-sized matrix multiplication on CPU
Test(CPU_regression, correct_weight_bias_exact) {
  Matrix* X_mat = create_matrix_host(100, 2);
  Matrix* y_mat = create_matrix_host(100, 1);

  float weight = 5.0F;
  float bias = 32.0F;

  for (size_t i = 0; i < (X_mat->rows * X_mat->cols); i++) {
    if (i % 2 == 0) {
      X_mat->elements[i] = (float)i / (2.0F);
    } else {
      X_mat->elements[i] = 1;
    }
    if (i < y_mat->rows) {
      y_mat->elements[i] = ((float)i * weight) + bias;
    }
  }

  Matrix* weights = cpu_regression(X_mat, y_mat);

  cr_assert_not_null(weights);
  cr_assert_float_eq(weights->elements[0], weight, 1e-3,
                     "%f weight expected but got %f", (double)weight,
                     (double)weights->elements[0]);
  cr_assert_float_eq(weights->elements[1], bias, 1e-3,
                     "%f bias expected but got %f", (double)bias,
                     (double)weights->elements[1]);

  free_matrix_host(X_mat);
  free_matrix_host(y_mat);
  free_matrix_host(weights);
}

// Test small-sized matrix multiplication on CPU
Test(CPU_regression, correct_weight_bias_noise) {
  Matrix* X_mat = create_matrix_host(100, 2);
  Matrix* y_mat = create_matrix_host(100, 1);

  float weight = 5.0F;
  float bias = 32.0F;

  for (size_t i = 0; i < (X_mat->rows * X_mat->cols); i++) {
    if (i % 2 == 0) {
      X_mat->elements[i] = (float)i / (2.0F);
    } else {
      X_mat->elements[i] = 1;
    }
    if (i < y_mat->rows) {
      // NOLINTNEXTLINE(cert-msc30-c,cert-msc50-cpp,concurrency-mt-unsafe)
      float random_number = (float)rand() / (float)(RAND_MAX / 2) - 1;
      y_mat->elements[i] = ((float)i * weight) + bias + random_number;
    }
  }

  Matrix* weights = cpu_regression(X_mat, y_mat);

  cr_assert_not_null(weights);
  cr_assert_float_eq(weights->elements[0], weight, .1,
                     "%f weight expected but got %f", (double)weight,
                     (double)weights->elements[0]);
  cr_assert_float_eq(weights->elements[1], bias, .1,
                     "%f bias expected but got %f", (double)bias,
                     (double)weights->elements[1]);

  free_matrix_host(X_mat);
  free_matrix_host(y_mat);
  free_matrix_host(weights);
}

// Test incompatible dimensions is NULL
Test(CPU_regression, incompatible_dim) {
  Matrix* X_mat = create_matrix_host(3, 2);
  Matrix* y_mat = create_matrix_host(2, 1);

  for (size_t i = 0; i < (X_mat->rows * X_mat->cols); i++) {
    X_mat->elements[i] = 1.0F;
  }
  for (size_t i = 0; i < (y_mat->rows * y_mat->cols); i++) {
    y_mat->elements[i] = 1.0F;
  }

  Matrix* weights = cpu_regression(X_mat, y_mat);

  cr_assert(eq(ptr, weights, NULL), "NULL expected but got %p", weights);

  free_matrix_host(X_mat);
  free_matrix_host(y_mat);
}

// Test non-invertible is NULL
Test(CPU_regression, non_invertible) {
  Matrix* X_mat = create_matrix_host(3, 2);
  Matrix* y_mat = create_matrix_host(3, 1);

  // Cols are linearly dependant
  float X_vals[] = {1, 2, 2, 4, 3, 6};

  for (size_t i = 0; i < (X_mat->rows * X_mat->cols); i++) {
    X_mat->elements[i] = X_vals[i];
  }
  for (size_t i = 0; i < (y_mat->rows * y_mat->cols); i++) {
    y_mat->elements[i] = 1.0F;
  }

  Matrix* weights = cpu_regression(X_mat, y_mat);

  cr_assert(eq(ptr, weights, NULL), "NULL expected but got %p", weights);

  free_matrix_host(X_mat);
  free_matrix_host(y_mat);
}

// Test small-sized matrix multiplication on GPU
Test(GPU_regression, correct_weight_bias_exact) {
  Matrix* X_mat = create_matrix_host(100, 2);
  Matrix* y_mat = create_matrix_host(100, 1);

  float weight = 5.0F;
  float bias = 32.0F;

  for (size_t i = 0; i < (X_mat->rows * X_mat->cols); i++) {
    if (i % 2 == 0) {
      X_mat->elements[i] = (float)i / (2.0F);
    } else {
      X_mat->elements[i] = 1;
    }
    if (i < y_mat->rows) {
      y_mat->elements[i] = ((float)i * weight) + bias;
    }
  }

  Matrix* d_X_mat = copy_matrix_host_to_device(X_mat);
  Matrix* d_y_mat = copy_matrix_host_to_device(y_mat);

  Matrix* d_weights =
      gpu_regression(d_X_mat, d_y_mat, X_mat->rows, X_mat->cols);
  Matrix* weights = copy_matrix_device_to_host(d_weights);

  cr_assert_not_null(weights);
  cr_assert_float_eq(weights->elements[0], weight, 1e-3,
                     "%f weight expected but got %f", (double)weight,
                     (double)weights->elements[0]);
  cr_assert_float_eq(weights->elements[1], bias, 1e-3,
                     "%f bias expected but got %f", (double)bias,
                     (double)weights->elements[1]);

  free_matrix_host(X_mat);
  free_matrix_host(y_mat);
  free_matrix_host(weights);

  free_matrix_device(d_X_mat);
  free_matrix_device(d_y_mat);
  free_matrix_device(d_weights);
}

// Test small-sized matrix multiplication on GPU
Test(GPU_regression, correct_weight_bias_noise) {
  Matrix* X_mat = create_matrix_host(100, 2);
  Matrix* y_mat = create_matrix_host(100, 1);

  float weight = 5.0F;
  float bias = 32.0F;

  for (size_t i = 0; i < (X_mat->rows * X_mat->cols); i++) {
    if (i % 2 == 0) {
      X_mat->elements[i] = (float)i / (2.0F);
    } else {
      X_mat->elements[i] = 1;
    }
    if (i < y_mat->rows) {
      // NOLINTNEXTLINE(cert-msc30-c,cert-msc50-cpp,concurrency-mt-unsafe)
      float random_number = (float)rand() / (float)(RAND_MAX / 2) - 1;
      y_mat->elements[i] = ((float)i * weight) + bias + random_number;
    }
  }

  Matrix* d_X_mat = copy_matrix_host_to_device(X_mat);
  Matrix* d_y_mat = copy_matrix_host_to_device(y_mat);

  Matrix* d_weights =
      gpu_regression(d_X_mat, d_y_mat, X_mat->rows, X_mat->cols);
  Matrix* weights = copy_matrix_device_to_host(d_weights);

  cr_assert_not_null(weights);
  cr_assert_float_eq(weights->elements[0], weight, .1,
                     "%f weight expected but got %f", (double)weight,
                     (double)weights->elements[0]);
  cr_assert_float_eq(weights->elements[1], bias, .1,
                     "%f bias expected but got %f", (double)bias,
                     (double)weights->elements[1]);

  free_matrix_host(X_mat);
  free_matrix_host(y_mat);
  free_matrix_host(weights);

  free_matrix_device(d_X_mat);
  free_matrix_device(d_y_mat);
  free_matrix_device(d_weights);
}

// Test incompatible dimensions is NULL
Test(GPU_regression, incompatible_dim) {
  Matrix* X_mat = create_matrix_host(3, 2);
  Matrix* y_mat = create_matrix_host(2, 1);

  for (size_t i = 0; i < (X_mat->rows * X_mat->cols); i++) {
    X_mat->elements[i] = 1.0F;
  }
  for (size_t i = 0; i < (y_mat->rows * y_mat->cols); i++) {
    y_mat->elements[i] = 1.0F;
  }

  Matrix* d_X_mat = copy_matrix_host_to_device(X_mat);
  Matrix* d_y_mat = copy_matrix_host_to_device(y_mat);

  Matrix* d_weights =
      gpu_regression(d_X_mat, d_y_mat, X_mat->rows, X_mat->cols);

  cr_assert(eq(ptr, d_weights, NULL), "NULL expected but got %p", d_weights);

  free_matrix_host(X_mat);
  free_matrix_host(y_mat);

  free_matrix_device(d_X_mat);
  free_matrix_device(d_y_mat);
}

// Test non-invertible is NULL
Test(GPU_regression, non_invertible) {
  Matrix* X_mat = create_matrix_host(3, 2);
  Matrix* y_mat = create_matrix_host(3, 1);

  // Cols are linearly dependant
  float X_vals[] = {1, 2, 2, 4, 3, 6};

  for (size_t i = 0; i < (X_mat->rows * X_mat->cols); i++) {
    X_mat->elements[i] = X_vals[i];
  }
  for (size_t i = 0; i < (y_mat->rows * y_mat->cols); i++) {
    y_mat->elements[i] = 1.0F;
  }

  Matrix* d_X_mat = copy_matrix_host_to_device(X_mat);
  Matrix* d_y_mat = copy_matrix_host_to_device(y_mat);

  Matrix* d_weights =
      gpu_regression(d_X_mat, d_y_mat, X_mat->rows, X_mat->cols);

  cr_assert(eq(ptr, d_weights, NULL), "NULL expected but got %p", d_weights);

  free_matrix_host(X_mat);
  free_matrix_host(y_mat);

  free_matrix_device(d_X_mat);
  free_matrix_device(d_y_mat);
}

// Test mean absolute error with variety of differences
Test(mean_absolute_error, mean_error_value) {
  Matrix* arr_1 = create_matrix_host(4, 1);
  Matrix* arr_2 = create_matrix_host(4, 1);

  float arr_1_val[] = {4, 2, -1, 3};
  float arr_2_val[] = {-1, 2, 3, 2};

  for (size_t i = 0; i < arr_1->rows; i++) {
    arr_1->elements[i] = arr_1_val[i];
    arr_2->elements[i] = arr_2_val[i];
  }

  float loss = mae_loss(arr_1, arr_2);

  cr_assert_float_eq(loss, 2.5, 1e-3, "2.5 loss expected but got %f",
                     (double)loss);

  free_matrix_host(arr_1);
  free_matrix_host(arr_2);
}

// Test error when unequal number of elements
Test(mean_absolute_error, unequal_error) {
  Matrix* arr_1 = create_matrix_host(4, 1);
  Matrix* arr_2 = create_matrix_host(5, 1);

  float loss = mae_loss(arr_1, arr_2);

  cr_assert(lt(dbl, loss, 0), "Error should be negative, got %f", (double)loss);

  free_matrix_host(arr_1);
  free_matrix_host(arr_2);
}

// Test error when non column vector
Test(mean_absolute_error, non_col_vector_error) {
  Matrix* arr_1 = create_matrix_host(4, 2);
  Matrix* arr_2 = create_matrix_host(4, 1);

  float loss = mae_loss(arr_1, arr_2);

  cr_assert(lt(dbl, loss, 0), "Error should be negative, got %f", (double)loss);

  free_matrix_host(arr_1);
  free_matrix_host(arr_2);
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers,cppcoreguidelines-pro-bounds-pointer-arithmetic)
