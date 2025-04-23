#include "../utils/matrix.h"
/** 
 * Perform linear regression on CPU using the Normal Equation.
 * This function computes the optimal weight matrix `W` for the linear regression model:
 *      W = (XᵀX)⁻¹Xᵀy
 * where:
 *  - X is the input feature matrix of shape (n_samples x n_features)
 *  - y is the target output matrix of shape (n_samples x 1)
 *  - W is the resulting weight matrix of shape (n_features x 1)
 *
 * Note: This version uses only CPU-based matrix operations.
 *
 * @param X Pointer to the input matrix (n_samples x n_features)
 * @param y Pointer to the target matrix (n_samples x 1)
 * @return Pointer to the resulting weight matrix (n_features x 1)
 */
Matrix* cpu_regression(Matrix* X, Matrix* y);
/** 
 * Perform linear regression on GPU using the Normal Equation.
 * This function computes the optimal weight matrix `W` for the linear regression model:
 *      W = (XᵀX)⁻¹Xᵀy
 * where:
 *  - X is the input feature matrix of shape (n_samples x n_features)
 *  - y is the target output matrix of shape (n_samples x 1)
 *  - W is the resulting weight matrix of shape (n_features x 1)
 *
 * Note: This version uses only GPU-based matrix operations.
 *
 * @param X Pointer to the input matrix (n_samples x n_features)
 * @param y Pointer to the target matrix (n_samples x 1)
 * @return Pointer to the resulting weight matrix (n_features x 1)
 */
Matrix* gpu_regression(Matrix* X, Matrix* y);
