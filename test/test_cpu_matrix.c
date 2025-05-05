#include <criterion/criterion.h>

#include "cpu_matrix.h"
#include "matrix.h"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers,cppcoreguidelines-pro-bounds-pointer-arithmetic)
// Since we're testing, magic numbers are needed to configure specific cases
// Also ignore pointer arithmetic warning, since in all cases it is in a loop
// which prevents out of bounds

// Test small-sized matrix multiplication on CPU
Test(CPU_Matrix, Multiply_SmallMatrix) {
  Matrix* h_A = create_matrix_host(2, 3);
  Matrix* h_B = create_matrix_host(3, 2);

  float a_vals[] = {1, 2, 3, 4, 5, 6};     // 2x3
  float b_vals[] = {7, 8, 9, 10, 11, 12};  // 3x2

  for (int i = 0; i < 6; ++i) {
    h_A->elements[i] = a_vals[i];
  }
  for (int i = 0; i < 6; ++i) {
    h_B->elements[i] = b_vals[i];
  }

  Matrix* h_c = cpu_matrix_multiply(h_A, h_B);

  cr_assert_not_null(h_c);
  cr_assert_eq(h_c->rows, 2);
  cr_assert_eq(h_c->cols, 2);

  float expected[] = {58, 64, 139, 154};
  for (int i = 0; i < 4; ++i) {
    cr_assert_float_eq(h_c->elements[i], expected[i], 1e-5);
  }
}

// Test identity matrix multiplication on CPU
Test(CPU_Matrix, Multiply_IdentityMatrix) {
  Matrix* mat = create_matrix_host(2, 2);
  float values[] = {1, 2, 3, 4};
  for (int i = 0; i < 4; ++i) {
    mat->elements[i] = values[i];
  }

  Matrix* identity = create_matrix_host(2, 2);
  float identity_values[] = {1, 0, 0, 1};
  for (int i = 0; i < 4; ++i) {
    identity->elements[i] = identity_values[i];
  }

  Matrix* result = cpu_matrix_multiply(mat, identity);

  cr_assert_not_null(result);
  for (int i = 0; i < 4; ++i) {
    cr_assert_float_eq(result->elements[i], values[i], 1e-5);
  }
}

// Test zero matrix multiplication on CPU
Test(CPU_Matrix, Multiply_ZeroMatrix) {
  Matrix* mat = create_matrix_host(2, 2);
  float values[] = {5, 6, 7, 8};
  for (int i = 0; i < 4; ++i) {
    mat->elements[i] = values[i];
  }

  Matrix* zero = create_matrix_host(2, 2);
  for (int i = 0; i < 4; ++i) {
    zero->elements[i] = 0.0F;
  }

  Matrix* result = cpu_matrix_multiply(mat, zero);

  cr_assert_not_null(result);
  for (int i = 0; i < 4; ++i) {
    cr_assert_float_eq(result->elements[i], 0.0F, 1e-5);
  }
}

// Test invalid matrix multiplication on CPU
Test(CPU_Matrix, Multiply_MismatchedDimensions) {
  Matrix* mat_a = create_matrix_host(2, 3);
  Matrix* mat_b = create_matrix_host(4, 2);  // incompatible

  Matrix* result = cpu_matrix_multiply(mat_a, mat_b);

  cr_assert_null(result);
}

// Test large matrix multiplication on CPU
Test(CPU_Matrix, Multiply_LargeMatrix) {
  Matrix* mat_a = create_matrix_host(100, 200);
  Matrix* mat_b = create_matrix_host(200, 50);

  for (int i = 0; i < 100 * 200; ++i) {
    mat_a->elements[i] = 1.0F;
  }
  for (int i = 0; i < 200 * 50; ++i) {
    mat_b->elements[i] = 1.0F;
  }

  Matrix* result = cpu_matrix_multiply(mat_a, mat_b);

  cr_assert_not_null(result);
  cr_assert_eq(result->rows, 100);
  cr_assert_eq(result->cols, 50);

  for (size_t i = 0; i < result->rows * result->cols; ++i) {
    cr_assert_float_eq(result->elements[i], 200.0F, 1e-3);
  }
}

// Test non-square matrix multiplication on CPU
Test(CPU_Matrix, Multiply_NonSquareMatrix) {
  Matrix* mat_a = create_matrix_host(2, 4);
  Matrix* mat_b = create_matrix_host(4, 3);

  for (int i = 0; i < 8; ++i) {
    mat_a->elements[i] = (float)(i + 1);
  }
  for (int i = 0; i < 12; ++i) {
    mat_b->elements[i] = (float)(i + 1);
  }

  Matrix* result = cpu_matrix_multiply(mat_a, mat_b);

  cr_assert_not_null(result);
  cr_assert_eq(result->rows, 2);
  cr_assert_eq(result->cols, 3);
}

// Test small-sized matrix transpose on CPU
Test(CPU_Matrix, Transpose_SmallMatrix) {
  Matrix* mat = create_matrix_host(2, 3);
  float values[] = {1, 2, 3, 4, 5, 6};  // 2x3
  for (int i = 0; i < 6; ++i) {
    mat->elements[i] = values[i];
  }

  Matrix* result = cpu_matrix_transpose(mat);

  cr_assert_not_null(result);
  cr_assert_eq(result->rows, 3);
  cr_assert_eq(result->cols, 2);

  float expected[] = {1, 4, 2, 5, 3, 6};  // Transposed 3x2
  for (int i = 0; i < 6; ++i) {
    cr_assert_float_eq(result->elements[i], expected[i], 1e-5);
  }
}

// Test identity matrix transpose on CPU
Test(CPU_Matrix, Transpose_IdentityMatrix) {
  Matrix* mat = create_matrix_host(3, 3);
  float values[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};  // 3x3 Identity
  for (int i = 0; i < 9; ++i) {
    mat->elements[i] = values[i];
  }

  Matrix* result = cpu_matrix_transpose(mat);

  cr_assert_not_null(result);
  cr_assert_eq(result->rows, 3);
  cr_assert_eq(result->cols, 3);
  for (int i = 0; i < 9; ++i) {
    cr_assert_float_eq(result->elements[i], values[i], 1e-5);
  }
}

// Test single value matrix transpose on CPU
Test(CPU_Matrix, Transpose_OneByOneMatrix) {
  Matrix* mat = create_matrix_host(1, 1);
  mat->elements[0] = 42.0F;

  Matrix* result = cpu_matrix_transpose(mat);

  cr_assert_not_null(result);
  cr_assert_eq(result->rows, 1);
  cr_assert_eq(result->cols, 1);
  cr_assert_float_eq(result->elements[0], 42.0F, 1e-5);
}

// Test large matrix transpose on CPU
Test(CPU_Matrix, Transpose_LargeMatrix) {
  size_t rows = 100;
  size_t cols = 200;
  Matrix* mat = create_matrix_host(rows, cols);

  for (size_t i = 0; i < rows * cols; ++i) {
    mat->elements[i] = (float)i;
  }

  Matrix* result = cpu_matrix_transpose(mat);

  cr_assert_not_null(result);
  cr_assert_eq(result->rows, cols);
  cr_assert_eq(result->cols, rows);

  cr_assert_float_eq(result->elements[0], mat->elements[0], 1e-5);
  cr_assert_float_eq(result->elements[1], mat->elements[cols], 1e-5);
}

// Test zero matrix transpose on CPU
Test(CPU_Matrix, Transpose_ZeroMatrix) {
  Matrix* mat = create_matrix_host(3, 4);
  for (int i = 0; i < 12; ++i) {
    mat->elements[i] = 0.0F;
  }

  Matrix* result = cpu_matrix_transpose(mat);

  cr_assert_not_null(result);
  cr_assert_eq(result->rows, 4);
  cr_assert_eq(result->cols, 3);

  for (int i = 0; i < 12; ++i) {
    cr_assert_float_eq(result->elements[i], 0.0F, 1e-5);
  }
}

// Test 2x2 matrix inversion on CPU
Test(CPU_Matrix, Inverse_2x2Matrix) {
  Matrix* mat = create_matrix_host(2, 2);
  float values[] = {4, 7, 2, 6};
  for (int i = 0; i < 4; ++i) {
    mat->elements[i] = values[i];
  }

  Matrix* result = cpu_matrix_inverse(mat);

  cr_assert_not_null(result);
  cr_assert_eq(result->rows, 2);
  cr_assert_eq(result->cols, 2);

  float expected[] = {0.6F, -0.7F, -0.2F, 0.4F};
  for (int i = 0; i < 4; ++i) {
    cr_assert_float_eq(result->elements[i], expected[i], 1e-5);
  }
}

// Test non-invertible matrix on CPU
Test(CPU_Matrix, Inverse_SingularMatrix) {
  Matrix* mat = create_matrix_host(2, 2);
  float values[] = {1, 2, 2, 4};  // determinant = 0 -> singular
  for (int i = 0; i < 4; ++i) {
    mat->elements[i] = values[i];
  }

  Matrix* result = cpu_matrix_inverse(mat);

  cr_assert_null(result);  // Should fail to invert
}

// Test identity matrix inversion on CPU
Test(CPU_Matrix, Inverse_IdentityMatrix) {
  Matrix* mat = create_matrix_host(3, 3);
  for (int i = 0; i < 9; ++i) {
    mat->elements[i] = (i % 4 == 0) ? 1.0F : 0.0F;
  }

  Matrix* result = cpu_matrix_inverse(mat);

  cr_assert_not_null(result);
  cr_assert_eq(result->rows, 3);
  cr_assert_eq(result->cols, 3);

  for (int i = 0; i < 9; ++i) {
    cr_assert_float_eq(result->elements[i], mat->elements[i], 1e-5);
  }
}

// Test inversion on matrix with negative elements on CPU
Test(CPU_Matrix, Inverse_NegativeElements) {
  Matrix* mat = create_matrix_host(2, 2);
  float values[] = {-2, -3, -1, -4};
  for (int i = 0; i < 4; ++i) {
    mat->elements[i] = values[i];
  }

  Matrix* result = cpu_matrix_inverse(mat);

  cr_assert_not_null(result);
  cr_assert_eq(result->rows, 2);
  cr_assert_eq(result->cols, 2);
  float expected[] = {-0.8F, 0.6F, 0.2F, -0.4F};

  for (int i = 0; i < 4; ++i) {
    cr_assert_float_eq(result->elements[i], expected[i], 1e-5);
  }
}

// Test small-sized matrix scalar multiplication on CPU
Test(CPU_Matrix, ScalarMultiply_SmallMatrix) {
  Matrix* mat = create_matrix_host(2, 2);

  float values[] = {1, 2, 3, 4};  // 2x2
  for (int i = 0; i < 4; ++i) {
    mat->elements[i] = values[i];
  }

  float scalar = 2.0F;
  Matrix* result = cpu_scalar_multiply(mat, scalar);

  cr_assert_not_null(result);
  cr_assert_eq(result->rows, 2);
  cr_assert_eq(result->cols, 2);

  float expected[] = {2, 4, 6, 8};  // 2x2
  for (int i = 0; i < 4; ++i) {
    cr_assert_float_eq(result->elements[i], expected[i], 1e-5);
  }
}

// Test scalar multiply zero to matrix on CPU
Test(CPU_Matrix, ScalarMultiply_Zero) {
  Matrix* mat = create_matrix_host(1, 3);
  mat->elements[0] = 5;
  mat->elements[1] = -2;
  mat->elements[2] = 9;

  Matrix* result = cpu_scalar_multiply(mat, 0.0F);

  cr_assert_not_null(result);
  for (int i = 0; i < 3; ++i) {
    cr_assert_float_eq(result->elements[i], 0.0F, 1e-5);
  }
}

// Test scalar multiply negative number on CPU
Test(CPU_Matrix, ScalarMultiply_Negative) {
  Matrix* mat = create_matrix_host(2, 1);
  mat->elements[0] = 4;
  mat->elements[1] = -3;

  Matrix* result = cpu_scalar_multiply(mat, -2.0F);

  cr_assert_not_null(result);
  cr_assert_float_eq(result->elements[0], -8.0F, 1e-5);
  cr_assert_float_eq(result->elements[1], 6.0F, 1e-5);
}

// Test scalar multiply when input is null on CPU
Test(CPU_Matrix, ScalarMultiply_NullInput) {
  Matrix* result = cpu_scalar_multiply(NULL, 2.0F);

  cr_assert_null(result, "input matrix is NULL");
}

// Test large matrix scalar multiplication on PGU
Test(CPU_Matrix, ScalarMultiply_LargeMatrix) {
  size_t rows = 1000;
  size_t cols = 1000;
  Matrix* mat = create_matrix_host(rows, cols);
  for (size_t i = 0; i < rows * cols; ++i) {
    mat->elements[i] = 1.0F;
  }

  Matrix* result = cpu_scalar_multiply(mat, 3.0F);

  cr_assert_not_null(result);
  for (size_t i = 0; i < rows * cols; ++i) {
    cr_assert_float_eq(result->elements[i], 3.0F, 1e-5);
  }
}

// Test small-sized matrix addition on CPU
Test(CPU_Matrix, Add_SmallMatrix) {
  Matrix* matA = create_matrix_host(2, 2);
  Matrix* matB = create_matrix_host(2, 2);

  float valuesA[] = {1, 2, 3, 4};  // 2x2
  float valuesB[] = {5, 6, 7, 8};  // 2x2
  for (int i = 0; i < 4; ++i) {
    matA->elements[i] = valuesA[i];
    matB->elements[i] = valuesB[i];
  }

  Matrix* result = cpu_matrix_add(matA, matB);

  cr_assert_not_null(result);
  cr_assert_eq(result->rows, 2);
  cr_assert_eq(result->cols, 2);

  float expected[] = {6, 8, 10, 12};  // 2x2
  for (int i = 0; i < 4; ++i) {
    cr_assert_float_eq(result->elements[i], expected[i], 1e-5);
  }
}

// Test adding negative elements on CPU
Test(CPU_Matrix, MatrixAdd_NegativeElements) {
  Matrix* mat_a = create_matrix_host(1, 3);
  Matrix* mat_b = create_matrix_host(1, 3);
  float vals_a[] = {-1, 2, -3};
  float vals_b[] = {4, -5, 6};
  for (int i = 0; i < 3; ++i) {
    mat_a->elements[i] = vals_a[i];
    mat_b->elements[i] = vals_b[i];
  }

  Matrix* result = cpu_matrix_add(mat_a, mat_b);

  cr_assert_not_null(result);
  float expected[] = {3, -3, 3};
  for (int i = 0; i < 3; ++i) {
    cr_assert_float_eq(result->elements[i], expected[i], 1e-5);
  }
}

// Test adding matrices with different dimension on CPU
Test(CPU_Matrix, MatrixAdd_DimensionMismatch) {
  Matrix* mat_a = create_matrix_host(2, 2);
  Matrix* mat_b = create_matrix_host(3, 2);  // Different dimensions

  Matrix* result = cpu_matrix_add(mat_a, mat_b);

  cr_assert_null(result);
}

// Test adding large matrix on CPU
Test(CPU_Matrix, MatrixAdd_LargeMatrix) {
  size_t rows = 1000;
  size_t cols = 1000;
  Matrix* mat_a = create_matrix_host(rows, cols);
  Matrix* mat_b = create_matrix_host(rows, cols);

  for (size_t i = 0; i < rows * cols; ++i) {
    mat_a->elements[i] = 2.0F;
    mat_b->elements[i] = 5.0F;
  }

  Matrix* result = cpu_matrix_add(mat_a, mat_b);

  cr_assert_not_null(result);
  for (size_t i = 0; i < rows * cols; ++i) {
    cr_assert_float_eq(result->elements[i], 7.0F, 1e-5);
  }
}
// Test small-sized matrix subtraction on CPU
Test(CPU_Matrix, Subtract_SmallMatrix) {
  Matrix* matA = create_matrix_host(2, 2);
  Matrix* matB = create_matrix_host(2, 2);

  float valuesA[] = {5, 6, 7, 8};  // 2x2
  float valuesB[] = {1, 2, 3, 4};  // 2x2
  for (int i = 0; i < 4; ++i) {
    matA->elements[i] = valuesA[i];
    matB->elements[i] = valuesB[i];
  }

  Matrix* difference = cpu_matrix_subtract(matA, matB);

  cr_assert_not_null(difference);
  cr_assert_eq(difference->rows, 2);
  cr_assert_eq(difference->cols, 2);

  float expected[] = {4, 4, 4, 4};  // 2x2
  for (int i = 0; i < 4; ++i) {
    cr_assert_float_eq(difference->elements[i], expected[i], 1e-5);
  }
}

// Test subtracting negative elements on CPU
Test(CPU_Matrix, MatrixSubtract_NegativeResults) {
  Matrix* mat_a = create_matrix_host(1, 2);
  Matrix* mat_b = create_matrix_host(1, 2);
  mat_a->elements[0] = 2;
  mat_a->elements[1] = 3;
  mat_b->elements[0] = 5;
  mat_b->elements[1] = 1;

  Matrix* result = cpu_matrix_subtract(mat_a, mat_b);

  cr_assert_not_null(result);
  cr_assert_float_eq(result->elements[0], -3.0F, 1e-5);
  cr_assert_float_eq(result->elements[1], 2.0F, 1e-5);
}

// Test subtracting matrices with different dimensions on CPU
Test(CPU_Matrix, MatrixSubtract_DimensionMismatch) {
  Matrix* mat_a = create_matrix_host(2, 2);
  Matrix* mat_b = create_matrix_host(2, 3);  // Incompatible dimensions

  Matrix* result = cpu_matrix_subtract(mat_a, mat_b);

  cr_assert_null(result);
}

// Test subtracting large matrix on CPU
Test(CPU_Matrix, MatrixSubtract_LargeMatrix) {
  size_t rows = 1000;
  size_t cols = 1000;
  Matrix* mat_a = create_matrix_host(rows, cols);
  Matrix* mat_b = create_matrix_host(rows, cols);

  for (size_t i = 0; i < rows * cols; ++i) {
    mat_a->elements[i] = 10.0F;
    mat_b->elements[i] = 1.0F;
  }

  Matrix* result = cpu_matrix_subtract(mat_a, mat_b);

  cr_assert_not_null(result);
  for (size_t i = 0; i < rows * cols; ++i) {
    cr_assert_float_eq(result->elements[i], 9.0F, 1e-5);
  }
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers,cppcoreguidelines-pro-bounds-pointer-arithmetic)
