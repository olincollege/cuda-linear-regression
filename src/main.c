#include <stdio.h>

#include "cpu_matrix.h"
#include "linear_regression.h"
#include "matrix.h"
#include "timing.h"

const size_t num_runs = 100;
char* const weight_meaning[] = {"Gestation length (days)",
                                "If first pregnancy",
                                "Mother's age (yrs)",
                                "Mother's height (in)",
                                "Mother's weight (lbs)",
                                "If mother smokes",
                                "Bias (oz)"};

int main(void) {
  // Read data matrices
  Matrix* X_train = create_matrix_from_csv("../../data/X_train.csv");
  Matrix* y_train = create_matrix_from_csv("../../data/y_train.csv");
  Matrix* X_test = create_matrix_from_csv("../../data/X_test.csv");
  Matrix* y_test = create_matrix_from_csv("../../data/y_test.csv");

  // Benchmark CPU
  double cpu_total_time = 0;
  for (size_t i = 0; i < num_runs; i++) {
    double start_time = get_current_time();
    Matrix* weights = cpu_regression(X_train, y_train);
    double end_time = get_current_time();
    cpu_total_time += end_time - start_time;
    free_matrix_host(weights);
  }
  double cpu_avg_time = cpu_total_time / (double)num_runs;
  printf("Average CPU time: %f\n", cpu_avg_time);

  // Benchmark GPU
  // Warm up
  for (size_t i = 0; i < num_runs; i++) {
    Matrix* weights = gpu_regression(X_train, y_train);
    free_matrix_host(weights);
  }

  // Real GPU trial
  double gpu_total_time = 0;
  for (size_t i = 0; i < num_runs; i++) {
    double start_time = get_current_time();
    Matrix* weights = gpu_regression(X_train, y_train);
    double end_time = get_current_time();
    gpu_total_time += end_time - start_time;
    free_matrix_host(weights);
  }
  double gpu_avg_time = gpu_total_time / (double)num_runs;
  printf("Average GPU time: %f\n", gpu_avg_time);

  // Calculate weights to use
  Matrix* weights = cpu_regression(X_train, y_train);

  // Print weights gestation,parity,age,height,weight,smoke
  puts("Calculated Weights: ");
  for (size_t i = 0; i < weights->rows; i++) {
    printf("  %s: %f\n", weight_meaning[i], (double)weights->elements[i]);
  }

  // Calculate train loss
  Matrix* y_train_pred = cpu_matrix_multiply(X_train, weights);
  float train_loss = mae_loss(y_train, y_train_pred);
  free_matrix_host(y_train_pred);
  printf("Train mean absolute difference: %f\n", (double)train_loss);

  // Calculate test loss
  Matrix* y_test_pred = cpu_matrix_multiply(X_test, weights);
  float test_loss = mae_loss(y_test, y_test_pred);
  free_matrix_host(y_test_pred);
  printf("Test mean absolute difference: %f\n", (double)test_loss);

  free_matrix_host(X_train);
  free_matrix_host(y_train);
  free_matrix_host(X_test);
  free_matrix_host(y_test);
  return 0;
}
