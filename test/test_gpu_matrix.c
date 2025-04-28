#include <criterion/criterion.h>

#include "gpu_matrix.h"
#include "matrix.h"

// Test small-sized matrix multiplication on GPU
Test(GPU_Matrix, Multiply_SmallMatrix) {
  Matrix* A = create_matrix_host(2, 3);
  Matrix* B = create_matrix_host(3, 2);

  float a_vals[] = {1, 2, 3, 4, 5, 6};     // 2x3
  float b_vals[] = {7, 8, 9, 10, 11, 12};  // 3x2

  for (int i = 0; i < 6; ++i) A->elements[i] = a_vals[i];
  for (int i = 0; i < 6; ++i) B->elements[i] = b_vals[i];

  Matrix* C = gpu_matrix_multiply(A, B);

  cr_assert_not_null(C);
  cr_assert_eq(C->rows, 2);
  cr_assert_eq(C->cols, 2);

  float expected[] = {58, 64, 139, 154};
  for (int i = 0; i < 4; ++i) {
    cr_assert_float_eq(C->elements[i], expected[i], 1e-5);
  }

  free_matrix_host(A);
  free_matrix_host(B);
  free_matrix_host(C);
}

// Test identity matrix multiplication on GPU
Test(GPU_Matrix, Multiply_IdentityMatrix) {
  Matrix* mat = create_matrix_host(2, 2);
  float values[] = {1, 2, 3, 4};
  for (int i = 0; i < 4; ++i) mat->elements[i] = values[i];

  Matrix* identity = create_matrix_host(2, 2);
  float identity_values[] = {1, 0, 0, 1};
  for (int i = 0; i < 4; ++i) identity->elements[i] = identity_values[i];

  Matrix* result = gpu_matrix_multiply(mat, identity);

  cr_assert_not_null(result);
  for (int i = 0; i < 4; ++i) {
    cr_assert_float_eq(result->elements[i], values[i], 1e-5);
  }

  free_matrix_host(mat);
  free_matrix_host(identity);
  free_matrix_host(result);
}

// Test zero matrix multiplication on GPU
Test(GPU_Matrix, Multiply_ZeroMatrix) {
  Matrix* mat = create_matrix_host(2, 2);
  float values[] = {5, 6, 7, 8};
  for (int i = 0; i < 4; ++i) mat->elements[i] = values[i];

  Matrix* zero = create_matrix_host(2, 2);
  for (int i = 0; i < 4; ++i) zero->elements[i] = 0.0f;

  Matrix* result = gpu_matrix_multiply(mat, zero);

  cr_assert_not_null(result);
  for (int i = 0; i < 4; ++i) {
    cr_assert_float_eq(result->elements[i], 0.0f, 1e-5);
  }

  free_matrix_host(mat);
  free_matrix_host(zero);
  free_matrix_host(result);
}

// Test invalid matrix multiplication on GPU
Test(GPU_Matrix, Multiply_MismatchedDimensions) {
  Matrix* mat_a = create_matrix_host(2, 3);
  Matrix* mat_b = create_matrix_host(4, 2);  // incompatible

  Matrix* result = gpu_matrix_multiply(mat_a, mat_b);

  cr_assert_null(result);

  free_matrix_host(mat_a);
  free_matrix_host(mat_b);
}

// Test large matrix multiplication on GPU
Test(GPU_Matrix, Multiply_LargeMatrix) {
  Matrix* mat_a = create_matrix_host(100, 200);
  Matrix* mat_b = create_matrix_host(200, 50);

  for (int i = 0; i < 100 * 200; ++i) mat_a->elements[i] = 1.0f;
  for (int i = 0; i < 200 * 50; ++i) mat_b->elements[i] = 1.0f;

  Matrix* result = gpu_matrix_multiply(mat_a, mat_b);

  cr_assert_not_null(result);
  cr_assert_eq(result->rows, 100);
  cr_assert_eq(result->cols, 50);

  for (int i = 0; i < result->rows * result->cols; ++i) {
    cr_assert_float_eq(result->elements[i], 200.0f, 1e-3);
  }

  free_matrix_host(mat_a);
  free_matrix_host(mat_b);
  free_matrix_host(result);
}

// Test non-square matrix multiplication on GPU
Test(GPU_Matrix, Multiply_NonSquareMatrix) {
  Matrix* mat_a = create_matrix_host(2, 4);
  Matrix* mat_b = create_matrix_host(4, 3);

  for (int i = 0; i < 8; ++i) mat_a->elements[i] = i + 1;
  for (int i = 0; i < 12; ++i) mat_b->elements[i] = i + 1;

  Matrix* result = gpu_matrix_multiply(mat_a, mat_b);

  cr_assert_not_null(result);
  cr_assert_eq(result->rows, 2);
  cr_assert_eq(result->cols, 3);

  free_matrix_host(mat_a);
  free_matrix_host(mat_b);
  free_matrix_host(result);
}

// Test small-sized matrix transpose on GPU
Test(GPU_Matrix, Transpose_SmallMatrix) {
  Matrix* mat = create_matrix_host(2, 3);
  float values[] = {1, 2, 3, 4, 5, 6};  // 2x3
  for (int i = 0; i < 6; ++i) mat->elements[i] = values[i];

  Matrix* transposed = gpu_matrix_transpose(mat);

  cr_assert_not_null(transposed);
  cr_assert_eq(transposed->rows, 3);
  cr_assert_eq(transposed->cols, 2);

  float expected[] = {1, 4, 2, 5, 3, 6};  // Transposed 3x2
  for (int i = 0; i < 6; ++i) {
    cr_assert_float_eq(transposed->elements[i], expected[i], 1e-5);
  }

  free_matrix_host(mat);
  free_matrix_host(transposed);
}
// Test 2x2 matrix inversion on GPU
// Test(GPU_Matrix, Inverse_2x2Matrix) {
//   Matrix* mat = create_matrix_host(2, 2);
//   float values[] = {4, 7, 2, 6};
//   for (int i = 0; i < 4; ++i) mat->elements[i] = values[i];

//   Matrix* inverse = gpu_matrix_inverse(mat);

//   cr_assert_not_null(inverse);
//   cr_assert_eq(inverse->rows, 2);
//   cr_assert_eq(inverse->cols, 2);

//   float expected[] = {0.6, -0.7, -0.2, 0.4};
//   for (int i = 0; i < 4; ++i) {
//     cr_assert_float_eq(inverse->elements[i], expected[i], 1e-5);
//   }

//   free_matrix_host(mat);
//   free_matrix_host(inverse);
// }

// Test small-sized matrix scalar multiplication on GPU
Test(GPU_Matrix, ScalarMultiply_SmallMatrix) {
  Matrix* mat = create_matrix_host(2, 2);

  float values[] = {1, 2, 3, 4};  // 2x2
  for (int i = 0; i < 4; ++i) {
    mat->elements[i] = values[i];
  }

  float scalar = 2.0f;
  Matrix* result = gpu_scalar_multiply(mat, scalar);

  cr_assert_not_null(result);
  cr_assert_eq(result->rows, 2);
  cr_assert_eq(result->cols, 2);

  float expected[] = {2, 4, 6, 8};  // 2x2
  for (int i = 0; i < 4; ++i) {
    cr_assert_float_eq(result->elements[i], expected[i], 1e-5);
  }

  free_matrix_host(mat);
  free_matrix_host(result);
}

// Test small-sized matrix addition on GPU
Test(GPU_Matrix, Add_SmallMatrix) {
  Matrix* matA = create_matrix_host(2, 2);
  Matrix* matB = create_matrix_host(2, 2);

  float valuesA[] = {1, 2, 3, 4};  // 2x2
  float valuesB[] = {5, 6, 7, 8};  // 2x2
  for (int i = 0; i < 4; ++i) {
    matA->elements[i] = valuesA[i];
    matB->elements[i] = valuesB[i];
  }

  Matrix* sum = gpu_matrix_add(matA, matB);

  cr_assert_not_null(sum);
  cr_assert_eq(sum->rows, 2);
  cr_assert_eq(sum->cols, 2);

  float expected[] = {6, 8, 10, 12};  // 2x2
  for (int i = 0; i < 4; ++i) {
    cr_assert_float_eq(sum->elements[i], expected[i], 1e-5);
  }

  free_matrix_host(matA);
  free_matrix_host(matB);
  free_matrix_host(sum);
}

// Test small-sized matrix subtraction on GPU
Test(GPU_Matrix, Subtract_SmallMatrix) {
  Matrix* matA = create_matrix_host(2, 2);
  Matrix* matB = create_matrix_host(2, 2);

  float valuesA[] = {5, 6, 7, 8};  // 2x2
  float valuesB[] = {1, 2, 3, 4};  // 2x2
  for (int i = 0; i < 4; ++i) {
    matA->elements[i] = valuesA[i];
    matB->elements[i] = valuesB[i];
  }

  Matrix* difference = gpu_matrix_subtract(matA, matB);

  cr_assert_not_null(difference);
  cr_assert_eq(difference->rows, 2);
  cr_assert_eq(difference->cols, 2);

  float expected[] = {4, 4, 4, 4};  // 2x2
  for (int i = 0; i < 4; ++i) {
    cr_assert_float_eq(difference->elements[i], expected[i], 1e-5);
  }

  free_matrix_host(matA);
  free_matrix_host(matB);
  free_matrix_host(difference);
}
