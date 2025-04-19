#pragma once

#include <cuda_runtime.h>

typedef struct {
  int rows;
  int cols;
  float* elements;  // height x width

} Matrix;

/**
 * Print an error message and exit with a failure status code.
 *
 * Upon an error, print an error message with a desired prefix. The prefix
 * error_msg should describe the context in which the error occurred, followed
 * by a more specific message corresponding to errno set by whatever function or
 * system call that encountered the error. This function exits the program and
 * thus does not return.
 *
 * @param error_msg The error message to print.
 */
void host_error_and_exit(const char* error_msg);

/**
 * Check if cuda errored and print message if so
 *
 * If err isn't cudaSuccess, print error message and exit.
 * The function won't return if this is the case.
 *
 * @param error_msg The error message to print if errored
 * @param err Return from cuda function
 */
void check_device_error(const char* error_msg, cudaError_t err);

/**
 * Create new matrix with uninitialized values on host.
 *
 * Dynamically allocates the matrix on the host heap. Caller is responsible for
 * freeing.
 *
 * @param num_rows Number of rows in the matrix
 * @param num_cols Number of columbs in the matrix
 * @return Pointer to the dynamically allocated matrix.
 */
Matrix* create_matrix_host(int num_rows, int num_cols);

/**
 * Create new matrix with uninitialized values on device.
 *
 * Dynamically allocates the matrix on the device heap. Caller is responsible
 * for freeing.
 *
 * @param num_rows Number of rows in the matrix
 * @param num_cols Number of columbs in the matrix
 * @return Pointer to the dynamically allocated matrix.
 */
Matrix* create_matrix_device(int num_rows, int num_cols);

/**
 * Frees the memory allocated for a matrix on host.
 *
 * Frees both the struct and elements array.
 *
 * @param h_mat Pointer to the Matrix whose memory should be released.
 */
void free_matrix_host(Matrix* h_mat);

/**
 * Frees the memory allocated for a matrix on device.
 *
 * Frees both the struct and elements array.
 *
 * @param d_mat Pointer to the Matrix whose memory should be released.
 */
void free_matrix_device(Matrix* d_mat);

/**
 * Copy a matrix from the host memory to device memory.
 *
 * Dynamically allocates memory on device. Caller is responsible for freeing
 * matrix on GPU.
 *
 * @param h_mat Pointer to matrix on host
 * @return Pointer to matrix location on device.
 */
Matrix* copy_matrix_host_to_device(const Matrix* h_mat);

/**
 * Copy a matrix from the gpu memory to host memory.
 *
 * Dynamically allocates memory on host. Caller is responsible for freeing
 * matrix on GPU.
 *
 * @param d_mat Pointer to matrix on device
 * @return Pointer to matrix location on host.
 */
Matrix* copy_matrix_device_to_host(const Matrix* d_mat);

/**
 * Loads a matrix from a CSV file.
 *
 * The CSV file should contain numeric values separated by commas, with one row
 * per line. Creates the matrix on host and dynamically allocates memory. Caller
 * is responsible for freeing.
 *
 * @param filename Path to the CSV file.
 * @return Pointer to a newly allocated Matrix containing the parsed data.
 *         The caller is responsible for freeing this matrix using
 * `free_matrix`.
 */
Matrix* create_matrix_from_csv(const char* filename);
