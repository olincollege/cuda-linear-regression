#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "matrix.hpp"

void host_error_and_exit(const char* error_msg) {
  perror(error_msg);
  // NOLINTNEXTLINE(concurrency-mt-unsafe)s
  exit(EXIT_FAILURE);
}

void check_device_error(const char* error_msg, cudaError_t err) {
  if (err != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", error_msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

Matrix* create_matrix_host(int num_rows, int num_cols) {
  Matrix* mat = (Matrix*)malloc(sizeof(Matrix));

  if (mat == NULL) {
    host_error_and_exit("Couldn't allocate host matrix struct space");
  }

  mat->rows = num_rows;
  mat->cols = num_cols;

  size_t num_elements = (size_t)(num_rows * num_cols);

  mat->elements = (float*)malloc(sizeof(float) * (num_rows * num_cols));

  if (mat->elements == NULL) {
    host_error_and_exit("Couldn't allocate matrix array space");
  }
  return mat;
}

Matrix* create_matrix_device(int num_rows, int num_cols) {
  // First we create the memory for matrix struct on the device
  // with a pointer to it on host
  Matrix* d_mat;
  check_device_error("Allocating Matrix struct",
                     cudaMalloc(&d_mat, sizeof(Matrix)));

  // Then create the memory for the elements on the device with
  // a pointer on host
  float* d_elems;
  check_device_error("alloc elements",
                     cudaMalloc(&d_elems, sizeof(float) * num_rows * num_cols));

  // Then create the whole struct on host
  Matrix h_mat;
  h_mat.rows = num_rows;
  h_mat.cols = num_cols;
  h_mat.elements = d_elems;

  // Finally move the whole struct to device
  check_device_error(
      "copy struct to device",
      cudaMemcpy(d_mat, &h_mat, sizeof(Matrix), cudaMemcpyHostToDevice));

  return d_mat;
}

void free_matrix_host(Matrix* mat) {}

void free_matrix_device(Matrix* mat) {}

Matrix* copy_matrix_host_to_device(Matrix* host_matrix) {}

Matrix* copy_matrix_device_to_host(Matrix* device_matrix) {}

Matrix* create_matrix_from_csv(const char* filename) {}
