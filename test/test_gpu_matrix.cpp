#include <cassert>
#include <iostream>
#include "../utils/matrix.hpp"

#include <criterion/criterion.h>
#include "matrix.hpp"
#include "gpu_matrix.hpp" 

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
