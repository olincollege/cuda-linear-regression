#pragma once
#include "matrix.h"
/**
 * Perform linear regression on CPU using the Normal Equation.
 *
 * This function computes the optimal weight matrix `W` for the linear
 * regression model: W = (XᵀX)⁻¹Xᵀy where:
 *  - X is the input feature matrix of shape (n_samples x n_features)
 *  - y is the target output matrix of shape (n_samples x 1)
 *  - W is the resulting weight matrix of shape (n_features x 1)
 *
 * If the operation can't be mathmatically performed, returns NULL. If
 * possible, it dynamically allocates the weight matrix. The caller is
 * responsible for freeing.
 *
 * @param X_mat Pointer to the input matrix (n_samples x n_features)
 * @param y_mat Pointer to the target matrix (n_samples x 1)
 * @return Pointer to the resulting weight matrix on host (n_features x 1)
 * or NULL
 */
Matrix* cpu_regression(Matrix* X_mat, Matrix* y_mat);

/**
 * Perform linear regression on GPU using the Normal Equation.
 *
 * This function computes the optimal weight matrix `W` for the linear
 * regression model: W = (XᵀX)⁻¹Xᵀy where:
 *  - X is the input feature matrix of shape (n_samples x n_features)
 *  - y is the target output matrix of shape (n_samples x 1)
 *  - W is the resulting weight matrix of shape (n_features x 1)
 *
 * If the operation can't be mathmatically performed, returns NULL. If
 * possible, it dynamically allocates the weight matrix. The caller is
 * responsible for freeing.
 *
 * @param X_mat Pointer to the input matrix (n_samples x n_features)
 * @param y_mat Pointer to the target matrix (n_samples x 1)
 * @return Pointer to the resulting weight matrix on host (n_features x 1)
 * or NULL
 */
Matrix* gpu_regression(Matrix* X_mat, Matrix* y_mat);
