#include <criterion/criterion.h>

#include "cpu_matrix.h"
#include "matrix.h"

// Test small-sized matrix multiplication on CPU
Test(CPU_Matrix, Multiply_SmallMatrix) {
  Matrix* A = create_matrix_host(2, 3);
  Matrix* B = create_matrix_host(3, 2);

  float a_vals[] = {1, 2, 3, 4, 5, 6};     // 2x3
  float b_vals[] = {7, 8, 9, 10, 11, 12};  // 3x2

  for (int i = 0; i < 6; ++i) A->elements[i] = a_vals[i];
  for (int i = 0; i < 6; ++i) B->elements[i] = b_vals[i];

  Matrix* C = cpu_matrix_multiply(A, B);

  cr_assert_not_null(C);
  cr_assert_eq(C->rows, 2);
  cr_assert_eq(C->cols, 2);

  float expected[] = {58, 64, 139, 154};
  for (int i = 0; i < 4; ++i) {
    cr_assert_float_eq(C->elements[i], expected[i], 1e-5);
  }
}

// Test identity matrix multiplication on CPU
Test(CPU_Matrix, Multiply_IdentityMatrix) {
  Matrix* mat = create_matrix_host(2, 2);
  float values[] = {1, 2, 3, 4};
  for (int i = 0; i < 4; ++i) mat->elements[i] = values[i];

  Matrix* identity = create_matrix_host(2, 2);
  float identity_values[] = {1, 0, 0, 1};
  for (int i = 0; i < 4; ++i) identity->elements[i] = identity_values[i];

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
  for (int i = 0; i < 4; ++i) mat->elements[i] = values[i];

  Matrix* zero = create_matrix_host(2, 2);
  for (int i = 0; i < 4; ++i) zero->elements[i] = 0.0f;

  Matrix* result = cpu_matrix_multiply(mat, zero);

  cr_assert_not_null(result);
  for (int i = 0; i < 4; ++i) {
    cr_assert_float_eq(result->elements[i], 0.0f, 1e-5);
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

  for (int i = 0; i < 100 * 200; ++i) mat_a->elements[i] = 1.0f;
  for (int i = 0; i < 200 * 50; ++i) mat_b->elements[i] = 1.0f;

  Matrix* result = cpu_matrix_multiply(mat_a, mat_b);

  cr_assert_not_null(result);
  cr_assert_eq(result->rows, 100);
  cr_assert_eq(result->cols, 50);

  for (int i = 0; i < result->rows * result->cols; ++i) {
    cr_assert_float_eq(result->elements[i], 200.0f, 1e-3);
  }
}

// Test non-square matrix multiplication on CPU
Test(CPU_Matrix, Multiply_NonSquareMatrix) {
  Matrix* mat_a = create_matrix_host(2, 4);
  Matrix* mat_b = create_matrix_host(4, 3);

  for (int i = 0; i < 8; ++i) mat_a->elements[i] = i + 1;
  for (int i = 0; i < 12; ++i) mat_b->elements[i] = i + 1;

  Matrix* result = cpu_matrix_multiply(mat_a, mat_b);

  cr_assert_not_null(result);
  cr_assert_eq(result->rows, 2);
  cr_assert_eq(result->cols, 3);
}

// Test small-sized matrix transpose on CPU
Test(CPU_Matrix, Transpose_SmallMatrix) {
  Matrix* mat = create_matrix_host(2, 3);
  float values[] = {1, 2, 3, 4, 5, 6};  // 2x3
  for (int i = 0; i < 6; ++i) mat->elements[i] = values[i];

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
  for (int i = 0; i < 9; ++i) mat->elements[i] = values[i];

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
  mat->elements[0] = 42.0f;

  Matrix* result = cpu_matrix_transpose(mat);

  cr_assert_not_null(result);
  cr_assert_eq(result->rows, 1);
  cr_assert_eq(result->cols, 1);
  cr_assert_float_eq(result->elements[0], 42.0f, 1e-5);
}

// Test large matrix transpose on CPU
Test(CPU_Matrix, Transpose_LargeMatrix) {
  int rows = 100;
  int cols = 200;
  Matrix* mat = create_matrix_host(rows, cols);

  for (int i = 0; i < rows * cols; ++i) {
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
  for (int i = 0; i < 12; ++i) mat->elements[i] = 0.0f;

  Matrix* result = cpu_matrix_transpose(mat);

  cr_assert_not_null(result);
  cr_assert_eq(result->rows, 4);
  cr_assert_eq(result->cols, 3);

  for (int i = 0; i < 12; ++i) {
    cr_assert_float_eq(result->elements[i], 0.0f, 1e-5);
  }
}

// Test 2x2 matrix inversion on CPU
Test(CPU_Matrix, Inverse_2x2Matrix) {
  Matrix* mat = create_matrix_host(2, 2);
  float values[] = {4, 7, 2, 6};
  for (int i = 0; i < 4; ++i) mat->elements[i] = values[i];

  Matrix* result = cpu_matrix_inverse(mat);

  cr_assert_not_null(result);
  cr_assert_eq(result->rows, 2);
  cr_assert_eq(result->cols, 2);

  float expected[] = {0.6, -0.7, -0.2, 0.4};
  for (int i = 0; i < 4; ++i) {
    cr_assert_float_eq(result->elements[i], expected[i], 1e-5);
  }
}

// Test non-invertible matrix on CPU
Test(CPU_Matrix, Inverse_SingularMatrix) {
  Matrix* mat = create_matrix_host(2, 2);
  float values[] = {1, 2, 2, 4};  // determinant = 0 -> singular
  for (int i = 0; i < 4; ++i) mat->elements[i] = values[i];

  Matrix* result = cpu_matrix_inverse(mat);

  cr_assert_null(result);  // Should fail to invert
}

// Test identity matrix inversion on CPU
Test(CPU_Matrix, Inverse_IdentityMatrix) {
  Matrix* mat = create_matrix_host(3, 3);
  for (int i = 0; i < 9; ++i) mat->elements[i] = (i % 4 == 0) ? 1.0f : 0.0f;

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
  for (int i = 0; i < 4; ++i) mat->elements[i] = values[i];

  Matrix* result = cpu_matrix_inverse(mat);

  cr_assert_not_null(result);
  cr_assert_eq(result->rows, 2);
  cr_assert_eq(result->cols, 2);
  float expected[] = {-0.8, 0.6, 0.2, -0.4};

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

  float scalar = 2.0f;
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

  Matrix* result = cpu_scalar_multiply(mat, 0.0f);

  cr_assert_not_null(result);
  for (int i = 0; i < 3; ++i) {
    cr_assert_float_eq(result->elements[i], 0.0f, 1e-5);
  }
}

// Test scalar multiply negative number on CPU
Test(CPU_Matrix, ScalarMultiply_Negative) {
  Matrix* mat = create_matrix_host(2, 1);
  mat->elements[0] = 4;
  mat->elements[1] = -3;

  Matrix* result = cpu_scalar_multiply(mat, -2.0f);

  cr_assert_not_null(result);
  cr_assert_float_eq(result->elements[0], -8.0f, 1e-5);
  cr_assert_float_eq(result->elements[1], 6.0f, 1e-5);
}

// Test scalar multiply when input is null on CPU
Test(CPU_Matrix, ScalarMultiply_NullInput) {
  Matrix* result = cpu_scalar_multiply(NULL, 2.0f);

  cr_assert_null(result, "input matrix is NULL");
}

// Test large matrix scalar multiplication on PGU
Test(CPU_Matrix, ScalarMultiply_LargeMatrix) {
  int rows = 1000, cols = 1000;
  Matrix* mat = create_matrix_host(rows, cols);
  for (int i = 0; i < rows * cols; ++i) {
    mat->elements[i] = 1.0f;
  }

  Matrix* result = cpu_scalar_multiply(mat, 3.0f);

  cr_assert_not_null(result);
  for (int i = 0; i < rows * cols; ++i) {
    cr_assert_float_eq(result->elements[i], 3.0f, 1e-5);
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
  Matrix* a = create_matrix_host(1, 3);
  Matrix* b = create_matrix_host(1, 3);
  float vals_a[] = {-1, 2, -3};
  float vals_b[] = {4, -5, 6};
  for (int i = 0; i < 3; ++i) {
    a->elements[i] = vals_a[i];
    b->elements[i] = vals_b[i];
  }

  Matrix* result = cpu_matrix_add(a, b);

  cr_assert_not_null(result);
  float expected[] = {3, -3, 3};
  for (int i = 0; i < 3; ++i) {
    cr_assert_float_eq(result->elements[i], expected[i], 1e-5);
  }
}

// Test adding matrices with different dimension on CPU
Test(CPU_Matrix, MatrixAdd_DimensionMismatch) {
  Matrix* a = create_matrix_host(2, 2);
  Matrix* b = create_matrix_host(3, 2);  // Different dimensions

  Matrix* result = cpu_matrix_add(a, b);

  cr_assert_null(result);
}

// Test adding large matrix on CPU
Test(CPU_Matrix, MatrixAdd_LargeMatrix) {
  int rows = 1000, cols = 1000;
  Matrix* a = create_matrix_host(rows, cols);
  Matrix* b = create_matrix_host(rows, cols);

  for (int i = 0; i < rows * cols; ++i) {
    a->elements[i] = 2.0f;
    b->elements[i] = 5.0f;
  }

  Matrix* result = cpu_matrix_add(a, b);

  cr_assert_not_null(result);
  for (int i = 0; i < rows * cols; ++i) {
    cr_assert_float_eq(result->elements[i], 7.0f, 1e-5);
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
  Matrix* a = create_matrix_host(1, 2);
  Matrix* b = create_matrix_host(1, 2);
  a->elements[0] = 2;
  a->elements[1] = 3;
  b->elements[0] = 5;
  b->elements[1] = 1;

  Matrix* result = cpu_matrix_subtract(a, b);

  cr_assert_not_null(result);
  cr_assert_float_eq(result->elements[0], -3.0f, 1e-5);
  cr_assert_float_eq(result->elements[1], 2.0f, 1e-5);
}

// Test subtracting matrices with different dimensions on CPU
Test(CPU_Matrix, MatrixSubtract_DimensionMismatch) {
  Matrix* a = create_matrix_host(2, 2);
  Matrix* b = create_matrix_host(2, 3);  // Incompatible dimensions

  Matrix* result = cpu_matrix_subtract(a, b);

  cr_assert_null(result);
}

// Test subtracting large matrix on CPU
Test(CPU_Matrix, MatrixSubtract_LargeMatrix) {
  int rows = 1000, cols = 1000;
  Matrix* a = create_matrix_host(rows, cols);
  Matrix* b = create_matrix_host(rows, cols);

  for (int i = 0; i < rows * cols; ++i) {
    a->elements[i] = 10.0f;
    b->elements[i] = 1.0f;
  }

  Matrix* result = cpu_matrix_subtract(a, b);

  cr_assert_not_null(result);
  for (int i = 0; i < rows * cols; ++i) {
    cr_assert_float_eq(result->elements[i], 9.0f, 1e-5);
  }
}
