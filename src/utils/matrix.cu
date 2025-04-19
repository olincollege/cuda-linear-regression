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
    host_error_and_exit("Allocating host matrix struct");
  }

  mat->rows = num_rows;
  mat->cols = num_cols;

  mat->elements = (float*)malloc(sizeof(float) * (num_rows * num_cols));

  if (mat->elements == NULL) {
    host_error_and_exit("Allocating elements array");
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
  check_device_error("Allocating elements array",
                     cudaMalloc(&d_elems, sizeof(float) * num_rows * num_cols));

  // Then create the whole struct on host
  Matrix h_mat;
  h_mat.rows = num_rows;
  h_mat.cols = num_cols;
  h_mat.elements = d_elems;

  // Finally move the whole struct to device
  check_device_error(
      "Copying struct to device",
      cudaMemcpy(d_mat, &h_mat, sizeof(Matrix), cudaMemcpyHostToDevice));

  return d_mat;
}

void free_matrix_host(Matrix* h_mat) {
  free(h_mat->elements);
  free(h_mat);
}

void free_matrix_device(Matrix* d_mat) {
  // First, copy the struct back to host so we can read the pointers
  Matrix h_mat;
  check_device_error(
      "Copy struct to host",
      cudaMemcpy(&h_mat, d_mat, sizeof(Matrix), cudaMemcpyDeviceToHost));

  check_device_error("Free elements", cudaFree(h_mat.elements));
  check_device_error("Free struct", cudaFree(d_mat));
}

Matrix* copy_matrix_host_to_device(const Matrix* h_mat) {
  // First we create the memory for matrix struct on the device
  // with a pointer to it on host
  Matrix* d_mat;
  check_device_error("Allocating Matrix struct",
                     cudaMalloc(&d_mat, sizeof(Matrix)));

  // Then create the memory for the elements on the device with
  // a pointer on host
  float* d_elems;
  check_device_error(
      "Allocating elements array",
      cudaMalloc(&d_elems, sizeof(float) * h_mat->rows * h_mat->cols));

  // Then create the whole struct on host
  Matrix temp_h_mat;
  temp_h_mat.rows = h_mat->rows;
  temp_h_mat.cols = h_mat->cols;
  temp_h_mat.elements = d_elems;

  // Copy the whole struct to device
  check_device_error(
      "Copying struct to device",
      cudaMemcpy(d_mat, &temp_h_mat, sizeof(Matrix), cudaMemcpyHostToDevice));

  // Copy the elements to the elements
  check_device_error("Copy elements to device",
                     cudaMemcpy(d_elems, h_mat->elements,
                                sizeof(float) * (h_mat->rows * h_mat->cols),
                                cudaMemcpyHostToDevice));

  return d_mat;
}

Matrix* copy_matrix_device_to_host(const Matrix* d_mat) {
  Matrix* h_mat = (Matrix*)malloc(sizeof(Matrix));
  if (h_mat == NULL) {
    host_error_and_exit("Allocate memory for matrix struct on host");
  }
  check_device_error(
      "Copy struct to host",
      cudaMemcpy(h_mat, d_mat, sizeof(Matrix), cudaMemcpyDeviceToHost));

  float* elements = (float*)malloc(sizeof(float) * h_mat->rows * h_mat->cols);
  if (elements == NULL) {
    host_error_and_exit("Allocate memory for elements on host");
  }

  check_device_error("Copy elements to host",
                     cudaMemcpy(elements, h_mat->elements,
                                sizeof(float) * (h_mat->rows * h_mat->cols),
                                cudaMemcpyDeviceToHost));
  h_mat->elements = elements;
  return h_mat;
}

Matrix* create_matrix_from_csv(const char* filename) {}
