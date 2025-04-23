#include <criterion/criterion.h>
#include "matrix.h"
#include "gpu_matrix.h" 

// Test small-sized matrix multiplication on GPU
Test(GPU_Matrix, Multiply_SmallMatrix) {
    Matrix* A = create_matrix_host(2, 3);
    Matrix* B = create_matrix_host(3, 2);

    float a_vals[] = {1, 2, 3, 4, 5, 6};  // 2x3
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
