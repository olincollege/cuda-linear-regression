#include <stdio.h>

#include "cpu_matrix.h"
#include "gpu_matrix.h"
#include "linear_regression.h"
#include "matrix.h"
#include "timing.h"

const size_t num_runs = 20;

int main(void) {
  // Read benchmarking data matrices
  puts("Reading data matrices...");
  Matrix* square_mat = create_matrix_from_csv("../../data/random_300x300.csv");
  Matrix* d_square_mat = copy_matrix_host_to_device(square_mat);

  Matrix* col_mat = create_matrix_from_csv("../../data/random_300x1.csv");
  Matrix* d_col_mat = copy_matrix_host_to_device(col_mat);

  puts("Warming up GPU...");
  // Warm up GPU
  for (size_t i = 0; i < num_runs; i++) {
    Matrix* d_weights = gpu_regression(d_square_mat, d_col_mat,
                                       square_mat->rows, square_mat->cols);
    free_matrix_device(d_weights);
  }

  // Benchmarking
  puts("Starting benchmarking...\n");

  // Matrix Multiplication
  puts("Matrix Multiplication:");
  double cpu_total_time = 0;
  for (size_t i = 0; i < num_runs; i++) {
    double start_time = get_current_time();
    Matrix* result = cpu_matrix_multiply(square_mat, square_mat);
    double end_time = get_current_time();
    cpu_total_time += end_time - start_time;
    free_matrix_host(result);
  }
  double cpu_avg_time = cpu_total_time / (double)num_runs;
  printf("  Average CPU time: %f sec\n", cpu_avg_time);

  double gpu_total_time = 0;
  for (size_t i = 0; i < num_runs; i++) {
    double start_time = get_current_time();
    Matrix* d_result = gpu_matrix_multiply(d_square_mat, d_square_mat,
                                           square_mat->rows, square_mat->cols);
    double end_time = get_current_time();
    gpu_total_time += end_time - start_time;
    free_matrix_device(d_result);
  }
  double gpu_avg_time = gpu_total_time / (double)num_runs;
  printf("  Average GPU time: %f sec\n", gpu_avg_time);
  printf("  Speedup: %fx\n\n", cpu_avg_time / gpu_avg_time);

  // Matrix Transpose
  puts("Matrix Transpose:");
  cpu_total_time = 0;
  for (size_t i = 0; i < num_runs; i++) {
    double start_time = get_current_time();
    Matrix* result = cpu_matrix_transpose(square_mat);
    double end_time = get_current_time();
    cpu_total_time += end_time - start_time;
    free_matrix_host(result);
  }
  cpu_avg_time = cpu_total_time / (double)num_runs;
  printf("  Average CPU time: %f sec\n", cpu_avg_time);

  gpu_total_time = 0;
  for (size_t i = 0; i < num_runs; i++) {
    double start_time = get_current_time();
    Matrix* d_result =
        gpu_matrix_transpose(d_square_mat, square_mat->cols, square_mat->rows);
    double end_time = get_current_time();
    gpu_total_time += end_time - start_time;
    free_matrix_device(d_result);
  }
  gpu_avg_time = gpu_total_time / (double)num_runs;
  printf("  Average GPU time: %f sec\n", gpu_avg_time);
  printf("  Speedup: %fx\n\n", cpu_avg_time / gpu_avg_time);

  // Element-wise Operation
  puts("Elementwise Operation (adding):");
  cpu_total_time = 0;
  for (size_t i = 0; i < num_runs; i++) {
    double start_time = get_current_time();
    Matrix* result = cpu_matrix_add(square_mat, square_mat);
    double end_time = get_current_time();
    cpu_total_time += end_time - start_time;
    free_matrix_host(result);
  }
  cpu_avg_time = cpu_total_time / (double)num_runs;
  printf("  Average CPU time: %f sec\n", cpu_avg_time);

  gpu_total_time = 0;
  for (size_t i = 0; i < num_runs; i++) {
    double start_time = get_current_time();
    Matrix* d_result = gpu_matrix_add(d_square_mat, d_square_mat,
                                      square_mat->rows, square_mat->cols);
    double end_time = get_current_time();
    gpu_total_time += end_time - start_time;
    free_matrix_device(d_result);
  }
  gpu_avg_time = gpu_total_time / (double)num_runs;
  printf("  Average GPU time: %f sec\n", gpu_avg_time);
  printf("  Speedup: %fx\n\n", cpu_avg_time / gpu_avg_time);

  // Inverse on CPU
  puts("CPU inverse time:");
  cpu_total_time = 0;
  for (size_t i = 0; i < num_runs; i++) {
    double start_time = get_current_time();
    Matrix* result = cpu_matrix_inverse(square_mat);
    double end_time = get_current_time();
    cpu_total_time += end_time - start_time;
    free_matrix_host(result);
  }
  cpu_avg_time = cpu_total_time / (double)num_runs;
  printf("  Average CPU inverse time: %f sec\n", cpu_avg_time);
  puts("  Not implemented on GPU\n");

  // Linear Regression
  puts("Linear Regression:");
  cpu_total_time = 0;
  for (size_t i = 0; i < num_runs; i++) {
    double start_time = get_current_time();
    Matrix* result = cpu_regression(square_mat, col_mat);
    double end_time = get_current_time();
    cpu_total_time += end_time - start_time;
    free_matrix_host(result);
  }
  cpu_avg_time = cpu_total_time / (double)num_runs;
  printf("  Average CPU time: %f sec\n", cpu_avg_time);

  gpu_total_time = 0;
  for (size_t i = 0; i < num_runs; i++) {
    double start_time = get_current_time();
    Matrix* d_result = gpu_regression(d_square_mat, d_col_mat, square_mat->rows,
                                      square_mat->cols);
    double end_time = get_current_time();
    gpu_total_time += end_time - start_time;
    free_matrix_device(d_result);
  }
  gpu_avg_time = gpu_total_time / (double)num_runs;
  printf("  Average GPU time: %f sec\n", gpu_avg_time);
  printf("  Speedup: %fx\n\n", cpu_avg_time / gpu_avg_time);

  // Matrix Transfer Time
  puts("Matrix transfer time:");
  cpu_total_time = 0;
  for (size_t i = 0; i < num_runs; i++) {
    double start_time = get_current_time();
    Matrix* d_result = copy_matrix_host_to_device(square_mat);
    free_matrix_device(d_result);
    double end_time = get_current_time();
    cpu_total_time += end_time - start_time;
  }
  cpu_avg_time = cpu_total_time / (double)num_runs;
  printf("  Average CPU to GPU time: %f sec\n", cpu_avg_time);

  gpu_total_time = 0;
  for (size_t i = 0; i < num_runs; i++) {
    double start_time = get_current_time();
    Matrix* result = copy_matrix_device_to_host(d_square_mat);
    free_matrix_host(result);
    double end_time = get_current_time();
    gpu_total_time += end_time - start_time;
  }
  gpu_avg_time = gpu_total_time / (double)num_runs;
  printf("  Average GPU to CPU time: %f sec\n", gpu_avg_time);

  free_matrix_host(square_mat);
  free_matrix_host(col_mat);
  free_matrix_device(d_square_mat);
  free_matrix_device(d_col_mat);
  return 0;
}
