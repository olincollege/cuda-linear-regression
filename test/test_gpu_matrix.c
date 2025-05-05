#include <criterion/criterion.h>

#include "gpu_matrix.h"
#include "matrix.h"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers,cppcoreguidelines-pro-bounds-pointer-arithmetic)
// Since we're testing, magic numbers are needed to configure specific cases
// Also ignore pointer arithmetic warning, since in all cases it is in a loop
// which prevents out of bounds

// Test small-sized matrix multiplication on GPU
Test(GPU_Matrix, Multiply_SmallMatrix) {
  Matrix* A = create_matrix_host(2, 3);
  Matrix* B = create_matrix_host(3, 2);

  float a_vals[] = {1, 2, 3, 4, 5, 6};     // 2x3
  float b_vals[] = {7, 8, 9, 10, 11, 12};  // 3x2

  for (int i = 0; i < 6; ++i) A->elements[i] = a_vals[i];
  for (int i = 0; i < 6; ++i) B->elements[i] = b_vals[i];

  // Copy A and B to device
  Matrix* d_A = copy_matrix_host_to_device(A);
  Matrix* d_B = copy_matrix_host_to_device(B);

  // Run GPU multiplication
  Matrix* d_C = gpu_matrix_multiply(d_A, d_B, 2, 2);
  Matrix* C = copy_matrix_device_to_host(d_C);

  // Expected result
  float expected[] = {58, 64, 139, 154};
  for (int i = 0; i < 4; ++i) {
    cr_assert_float_eq(C->elements[i], expected[i], 1e-5);
  }

  // Free all matrices
  free_matrix_host(A);
  free_matrix_host(B);
  free_matrix_host(C);
  free_matrix_device(d_A);
  free_matrix_device(d_B);
  free_matrix_device(d_C);
}

// Test identity matrix multiplication on GPU
Test(GPU_Matrix, Multiply_IdentityMatrix) {
  Matrix* mat = create_matrix_host(2, 2);
  float values[] = {1, 2, 3, 4};
  for (int i = 0; i < 4; ++i) mat->elements[i] = values[i];

  Matrix* identity = create_matrix_host(2, 2);
  float identity_values[] = {1, 0, 0, 1};
  for (int i = 0; i < 4; ++i) identity->elements[i] = identity_values[i];

  Matrix* d_mat = copy_matrix_host_to_device(mat);
  Matrix* d_identity = copy_matrix_host_to_device(identity);

  Matrix* d_result = gpu_matrix_multiply(d_mat, d_identity, 2, 2);
  Matrix* result = copy_matrix_device_to_host(d_result);

  cr_assert_not_null(result);
  for (int i = 0; i < 4; ++i) {
    cr_assert_float_eq(result->elements[i], values[i], 1e-5);
  }

  free_matrix_host(mat);
  free_matrix_host(identity);
  free_matrix_host(result);
  free_matrix_device(d_mat);
  free_matrix_device(d_identity);
  free_matrix_device(d_result);
}

// Test zero matrix multiplication on GPU
Test(GPU_Matrix, Multiply_ZeroMatrix) {
  Matrix* mat = create_matrix_host(2, 2);
  float values[] = {5, 6, 7, 8};
  for (int i = 0; i < 4; ++i) mat->elements[i] = values[i];

  Matrix* zero = create_matrix_host(2, 2);
  for (int i = 0; i < 4; ++i) zero->elements[i] = 0.0f;
  Matrix* d_mat = copy_matrix_host_to_device(mat);
  Matrix* d_zero = copy_matrix_host_to_device(zero);

  Matrix* d_result = gpu_matrix_multiply(d_mat, d_zero, 2, 2);
  Matrix* result = copy_matrix_device_to_host(d_result);

  cr_assert_not_null(result);
  for (int i = 0; i < 4; ++i) {
    cr_assert_float_eq(result->elements[i], 0.0f, 1e-5);
  }

  free_matrix_host(mat);
  free_matrix_host(zero);
  free_matrix_host(result);
  free_matrix_device(d_mat);
  free_matrix_device(d_zero);
  free_matrix_device(d_result);
}

// Test invalid matrix multiplication on GPU
Test(GPU_Matrix, Multiply_MismatchedDimensions) {
  Matrix* A = create_matrix_host(2, 3);
  Matrix* B = create_matrix_host(4, 2);  // incompatible

  Matrix* d_A = copy_matrix_host_to_device(A);
  Matrix* d_B = copy_matrix_host_to_device(B);

  Matrix* d_result = gpu_matrix_multiply(d_A, d_B, 2, 2);

  cr_assert_null(d_result);

  free_matrix_host(A);
  free_matrix_host(B);
  free_matrix_device(d_A);
  free_matrix_device(d_B);
}

// Test large matrix multiplication on GPU
Test(GPU_Matrix, Multiply_LargeMatrix) {
  Matrix* A = create_matrix_host(100, 200);
  Matrix* B = create_matrix_host(200, 50);

  for (int i = 0; i < 100 * 200; ++i) A->elements[i] = 1.0f;
  for (int i = 0; i < 200 * 50; ++i) B->elements[i] = 1.0f;
  Matrix* d_A = copy_matrix_host_to_device(A);
  Matrix* d_B = copy_matrix_host_to_device(B);
  Matrix* d_result = gpu_matrix_multiply(d_A, d_B, 100, 50);
  Matrix* result = copy_matrix_device_to_host(d_result);

  cr_assert_not_null(result);
  cr_assert_eq(result->rows, 100);
  cr_assert_eq(result->cols, 50);

  for (int i = 0; i < result->rows * result->cols; ++i) {
    cr_assert_float_eq(result->elements[i], 200.0f, 1e-3);
  }

  free_matrix_host(A);
  free_matrix_host(B);
  free_matrix_host(result);
  free_matrix_device(d_A);
  free_matrix_device(d_B);
  free_matrix_device(d_result);
}

// Test non-square matrix multiplication on GPU
Test(GPU_Matrix, Multiply_NonSquareMatrix) {
  Matrix* A = create_matrix_host(2, 4);
  Matrix* B = create_matrix_host(4, 3);

  for (int i = 0; i < 8; ++i) A->elements[i] = i + 1;
  for (int i = 0; i < 12; ++i) B->elements[i] = i + 1;

  Matrix* d_A = copy_matrix_host_to_device(A);
  Matrix* d_B = copy_matrix_host_to_device(B);

  Matrix* d_result = gpu_matrix_multiply(d_A, d_B, 2, 3);
  Matrix* result = copy_matrix_device_to_host(d_result);

  cr_assert_not_null(result);
  cr_assert_eq(result->rows, 2);
  cr_assert_eq(result->cols, 3);

  free_matrix_host(A);
  free_matrix_host(B);
  free_matrix_host(result);
  free_matrix_device(d_A);
  free_matrix_device(d_B);
  free_matrix_device(d_result);
}

// Test small-sized matrix transpose on GPU
Test(GPU_Matrix, Transpose_SmallMatrix) {
  Matrix* mat = create_matrix_host(2, 3);
  float values[] = {1, 2, 3, 4, 5, 6};  // 2x3
  for (int i = 0; i < 6; ++i) mat->elements[i] = values[i];

  Matrix* d_mat = copy_matrix_host_to_device(mat);

  Matrix* d_transposed = gpu_matrix_transpose(d_mat, 3, 2);
  Matrix* transposed = copy_matrix_device_to_host(d_transposed);

  cr_assert_not_null(transposed);
  cr_assert_eq(transposed->rows, 3);
  cr_assert_eq(transposed->cols, 2);

  float expected[] = {1, 4, 2, 5, 3, 6};  // Transposed 3x2
  for (int i = 0; i < 6; ++i) {
    cr_assert_float_eq(transposed->elements[i], expected[i], 1e-5);
  }

  free_matrix_host(mat);
  free_matrix_host(transposed);
  free_matrix_device(d_mat);
  free_matrix_device(d_transposed);
}

// Test identity matrix transpose on GPU
Test(GPU_Matrix, Transpose_IdentityMatrix) {
  Matrix* mat = create_matrix_host(3, 3);
  float values[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};  // 3x3 Identity
  for (int i = 0; i < 9; ++i) mat->elements[i] = values[i];
  Matrix* d_mat = copy_matrix_host_to_device(mat);

  Matrix* d_result = gpu_matrix_transpose(d_mat, 3, 3);

  Matrix* result = copy_matrix_device_to_host(d_result);

  cr_assert_not_null(result);
  cr_assert_eq(result->rows, 3);
  cr_assert_eq(result->cols, 3);
  for (int i = 0; i < 9; ++i) {
    cr_assert_float_eq(result->elements[i], values[i], 1e-5);
  }

  free_matrix_host(mat);
  free_matrix_host(result);
  free_matrix_device(d_mat);
  free_matrix_device(d_result);
}

// Test single value matrix transpose on GPU
Test(GPU_Matrix, Transpose_OneByOneMatrix) {
  Matrix* mat = create_matrix_host(1, 1);
  mat->elements[0] = 42.0f;

  Matrix* d_mat = copy_matrix_host_to_device(mat);

  Matrix* d_result = gpu_matrix_transpose(d_mat, 1, 1);

  Matrix* result = copy_matrix_device_to_host(d_result);

  cr_assert_not_null(result);
  cr_assert_eq(result->rows, 1);
  cr_assert_eq(result->cols, 1);
  cr_assert_float_eq(result->elements[0], 42.0f, 1e-5);

  free_matrix_host(mat);
  free_matrix_host(result);
  free_matrix_device(d_mat);
  free_matrix_device(d_result);
}

// Test large matrix transpose on GPU
Test(GPU_Matrix, Transpose_LargeMatrix) {
  int rows = 100;
  int cols = 200;
  Matrix* mat = create_matrix_host(rows, cols);

  for (int i = 0; i < rows * cols; ++i) {
    mat->elements[i] = (float)i;
  }
  Matrix* d_mat = copy_matrix_host_to_device(mat);
  Matrix* d_result = gpu_matrix_transpose(d_mat, cols, rows);
  Matrix* result = copy_matrix_device_to_host(d_result);

  cr_assert_not_null(result);
  cr_assert_eq(result->rows, cols);
  cr_assert_eq(result->cols, rows);

  cr_assert_float_eq(result->elements[0], mat->elements[0], 1e-5);
  cr_assert_float_eq(result->elements[1], mat->elements[cols], 1e-5);

  free_matrix_host(mat);
  free_matrix_host(result);
  free_matrix_device(d_mat);
  free_matrix_device(d_result);
}

// Test zero matrix transpose on GPU
Test(GPU_Matrix, Transpose_ZeroMatrix) {
  Matrix* mat = create_matrix_host(3, 4);
  for (int i = 0; i < 12; ++i) mat->elements[i] = 0.0f;

  Matrix* d_mat = copy_matrix_host_to_device(mat);

  Matrix* d_result = gpu_matrix_transpose(d_mat, 4, 3);

  Matrix* result = copy_matrix_device_to_host(d_result);

  cr_assert_not_null(result);
  cr_assert_eq(result->rows, 4);
  cr_assert_eq(result->cols, 3);

  for (int i = 0; i < 12; ++i) {
    cr_assert_float_eq(result->elements[i], 0.0f, 1e-5);
  }

  free_matrix_host(mat);
  free_matrix_host(result);
  free_matrix_device(d_mat);
  free_matrix_device(d_result);
}

// Test small-sized matrix scalar multiplication on GPU
Test(GPU_Matrix, ScalarMultiply_SmallMatrix) {
  Matrix* mat = create_matrix_host(2, 2);

  float values[] = {1, 2, 3, 4};  // 2x2
  for (int i = 0; i < 4; ++i) {
    mat->elements[i] = values[i];
  }
  float scalar = 2.0f;

  Matrix* d_mat = copy_matrix_host_to_device(mat);

  Matrix* d_result = gpu_scalar_multiply(d_mat, scalar, 2, 2);
  Matrix* result = copy_matrix_device_to_host(d_result);
  float expected[] = {2, 4, 6, 8};  // 2x2

  cr_assert_not_null(result);
  cr_assert_eq(result->rows, 2);
  cr_assert_eq(result->cols, 2);

  for (int i = 0; i < 4; ++i) {
    cr_assert_float_eq(result->elements[i], expected[i], 1e-5);
  }

  free_matrix_host(mat);
  free_matrix_host(result);
  free_matrix_device(d_mat);
  free_matrix_device(d_result);
}

// Test scalar multiply zero to matrix on GPU
Test(GPU_Matrix, ScalarMultiply_Zero) {
  Matrix* mat = create_matrix_host(1, 3);
  mat->elements[0] = 5;
  mat->elements[1] = -2;
  mat->elements[2] = 9;

  Matrix* d_mat = copy_matrix_host_to_device(mat);

  Matrix* d_result = gpu_scalar_multiply(d_mat, 0.0f, 1, 3);

  Matrix* result = copy_matrix_device_to_host(d_result);

  cr_assert_not_null(result);
  for (int i = 0; i < 3; ++i) {
    cr_assert_float_eq(result->elements[i], 0.0f, 1e-5);
  }

  free_matrix_host(mat);
  free_matrix_host(result);
  free_matrix_device(d_mat);
  free_matrix_device(d_result);
}

// Test scalar multiply negative number on GPU
Test(GPU_Matrix, ScalarMultiply_Negative) {
  Matrix* mat = create_matrix_host(2, 1);
  mat->elements[0] = 4;
  mat->elements[1] = -3;

  Matrix* d_mat = copy_matrix_host_to_device(mat);
  Matrix* d_result = gpu_scalar_multiply(d_mat, -2.0f, 2, 1);
  Matrix* result = copy_matrix_device_to_host(d_result);

  cr_assert_not_null(result);
  cr_assert_float_eq(result->elements[0], -8.0f, 1e-5);
  cr_assert_float_eq(result->elements[1], 6.0f, 1e-5);

  free_matrix_host(mat);
  free_matrix_host(result);
  free_matrix_device(d_mat);
  free_matrix_device(d_result);
}

// Test large matrix scalar multiplication on GPU
Test(GPU_Matrix, ScalarMultiply_LargeMatrix) {
  int rows = 1000, cols = 1000;
  Matrix* mat = create_matrix_host(rows, cols);
  for (int i = 0; i < rows * cols; ++i) {
    mat->elements[i] = 1.0f;
  }
  Matrix* d_mat = copy_matrix_host_to_device(mat);
  Matrix* d_result = gpu_scalar_multiply(d_mat, 3.0f, rows, cols);
  Matrix* result = copy_matrix_device_to_host(d_result);

  cr_assert_not_null(result);
  for (int i = 0; i < rows * cols; ++i) {
    cr_assert_float_eq(result->elements[i], 3.0f, 1e-5);
  }

  free_matrix_host(mat);
  free_matrix_host(result);
  free_matrix_device(d_mat);
  free_matrix_device(d_result);
}

// Test small-sized matrix addition on GPU
Test(GPU_Matrix, Add_SmallMatrix) {
  Matrix* A = create_matrix_host(2, 2);
  Matrix* B = create_matrix_host(2, 2);

  float valuesA[] = {1, 2, 3, 4};  // 2x2
  float valuesB[] = {5, 6, 7, 8};  // 2x2
  for (int i = 0; i < 4; ++i) {
    A->elements[i] = valuesA[i];
    B->elements[i] = valuesB[i];
  }
  Matrix* d_A = copy_matrix_host_to_device(A);
  Matrix* d_B = copy_matrix_host_to_device(B);
  Matrix* d_result = gpu_matrix_add(d_A, d_B, 2, 2);
  Matrix* result = copy_matrix_device_to_host(d_result);

  cr_assert_not_null(result);
  cr_assert_eq(result->rows, 2);
  cr_assert_eq(result->cols, 2);

  float expected[] = {6, 8, 10, 12};  // 2x2
  for (int i = 0; i < 4; ++i) {
    cr_assert_float_eq(result->elements[i], expected[i], 1e-5);
  }

  free_matrix_host(A);
  free_matrix_host(B);
  free_matrix_host(result);
  free_matrix_device(d_A);
  free_matrix_device(d_B);
  free_matrix_device(d_result);
}

// Test adding negative elements on GPU
Test(GPU_Matrix, MatrixAdd_NegativeElements) {
  Matrix* A = create_matrix_host(1, 3);
  Matrix* B = create_matrix_host(1, 3);
  float vals_a[] = {-1, 2, -3};
  float vals_b[] = {4, -5, 6};
  for (int i = 0; i < 3; ++i) {
    A->elements[i] = vals_a[i];
    B->elements[i] = vals_b[i];
  }
  Matrix* d_A = copy_matrix_host_to_device(A);
  Matrix* d_B = copy_matrix_host_to_device(B);
  Matrix* d_result = gpu_matrix_add(d_A, d_B, 1, 3);
  Matrix* result = copy_matrix_device_to_host(d_result);

  cr_assert_not_null(result);
  float expected[] = {3, -3, 3};
  for (int i = 0; i < 3; ++i) {
    cr_assert_float_eq(result->elements[i], expected[i], 1e-5);
  }

  free_matrix_host(A);
  free_matrix_host(B);
  free_matrix_host(result);
  free_matrix_device(d_A);
  free_matrix_device(d_B);
  free_matrix_device(d_result);
}

// Test adding matrices with different dimension on GPU
Test(GPU_Matrix, MatrixAdd_DimensionMismatch) {
  Matrix* A = create_matrix_host(2, 2);
  Matrix* B = create_matrix_host(3, 2);  // Different dimensions

  Matrix* result = gpu_matrix_add(A, B, 2, 2);

  cr_assert_null(result);

  free_matrix_host(A);
  free_matrix_host(B);
}

// Test adding large matrix on GPU
Test(GPU_Matrix, MatrixAdd_LargeMatrix) {
  int rows = 1000, cols = 1000;
  Matrix* A = create_matrix_host(rows, cols);
  Matrix* B = create_matrix_host(rows, cols);

  for (int i = 0; i < rows * cols; ++i) {
    A->elements[i] = 2.0f;
    B->elements[i] = 5.0f;
  }
  Matrix* d_A = copy_matrix_host_to_device(A);
  Matrix* d_B = copy_matrix_host_to_device(B);
  Matrix* d_result = gpu_matrix_add(d_A, d_B, rows, cols);
  Matrix* result = copy_matrix_device_to_host(d_result);

  cr_assert_not_null(result);
  for (int i = 0; i < rows * cols; ++i) {
    cr_assert_float_eq(result->elements[i], 7.0f, 1e-5);
  }

  free_matrix_host(A);
  free_matrix_host(B);
  free_matrix_host(result);
  free_matrix_device(d_A);
  free_matrix_device(d_B);
  free_matrix_device(d_result);
}

// Test small-sized matrix subtraction on GPU
Test(GPU_Matrix, Subtract_SmallMatrix) {
  Matrix* A = create_matrix_host(2, 2);
  Matrix* B = create_matrix_host(2, 2);

  float valuesA[] = {5, 6, 7, 8};  // 2x2
  float valuesB[] = {1, 2, 3, 4};  // 2x2
  for (int i = 0; i < 4; ++i) {
    A->elements[i] = valuesA[i];
    B->elements[i] = valuesB[i];
  }
  Matrix* d_A = copy_matrix_host_to_device(A);
  Matrix* d_B = copy_matrix_host_to_device(B);
  Matrix* d_result = gpu_matrix_subtract(d_A, d_B, 2, 2);
  Matrix* result = copy_matrix_device_to_host(d_result);

  cr_assert_not_null(result);
  cr_assert_eq(result->rows, 2);
  cr_assert_eq(result->cols, 2);

  float expected[] = {4, 4, 4, 4};  // 2x2
  for (int i = 0; i < 4; ++i) {
    cr_assert_float_eq(result->elements[i], expected[i], 1e-5);
  }

  free_matrix_host(A);
  free_matrix_host(B);
  free_matrix_host(result);
  free_matrix_device(d_A);
  free_matrix_device(d_B);
  free_matrix_device(d_result);
}

// Test subtracting negative elements on GPU
Test(GPU_Matrix, MatrixSubtract_NegativeResults) {
  Matrix* A = create_matrix_host(1, 2);
  Matrix* B = create_matrix_host(1, 2);
  A->elements[0] = 2;
  A->elements[1] = 3;
  B->elements[0] = 5;
  B->elements[1] = 1;
  Matrix* d_A = copy_matrix_host_to_device(A);
  Matrix* d_B = copy_matrix_host_to_device(B);
  Matrix* d_result = gpu_matrix_subtract(d_A, d_B, 1, 2);
  Matrix* result = copy_matrix_device_to_host(d_result);

  cr_assert_not_null(result);
  cr_assert_float_eq(result->elements[0], -3.0f, 1e-5);
  cr_assert_float_eq(result->elements[1], 2.0f, 1e-5);

  free_matrix_host(A);
  free_matrix_host(B);
  free_matrix_host(result);
  free_matrix_device(d_A);
  free_matrix_device(d_B);
  free_matrix_device(d_result);
}

// Test subtracting matrices with different dimensions on GPU
Test(GPU_Matrix, MatrixSubtract_DimensionMismatch) {
  Matrix* a = create_matrix_host(2, 2);
  Matrix* b = create_matrix_host(2, 3);  // Incompatible dimensions

  Matrix* result = gpu_matrix_subtract(a, b, 2, 3);

  cr_assert_null(result);

  free_matrix_host(a);
  free_matrix_host(b);
}

// Test subtracting large matrix on GPU
Test(GPU_Matrix, MatrixSubtract_LargeMatrix) {
  int rows = 1000, cols = 1000;
  Matrix* A = create_matrix_host(rows, cols);
  Matrix* B = create_matrix_host(rows, cols);

  for (int i = 0; i < rows * cols; ++i) {
    A->elements[i] = 10.0f;
    B->elements[i] = 1.0f;
  }
  Matrix* d_A = copy_matrix_host_to_device(A);
  Matrix* d_B = copy_matrix_host_to_device(B);
  Matrix* d_result = gpu_matrix_subtract(d_A, d_B, rows, cols);
  Matrix* result = copy_matrix_device_to_host(d_result);

  cr_assert_not_null(result);
  for (int i = 0; i < rows * cols; ++i) {
    cr_assert_float_eq(result->elements[i], 9.0f, 1e-5);
  }

  free_matrix_host(A);
  free_matrix_host(B);
  free_matrix_host(result);
  free_matrix_device(d_A);
  free_matrix_device(d_B);
  free_matrix_device(d_result);
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers,cppcoreguidelines-pro-bounds-pointer-arithmetic)
