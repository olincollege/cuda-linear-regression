#include <stdio.h>

#include "cpu_matrix.h"
#include "linear_regression.h"
#include "matrix.h"

char* const weight_meaning[] = {"Gestation length (days)",
                                "If not first pregnancy",
                                "Mother's age (yrs)",
                                "Mother's height (in)",
                                "Mother's weight (lbs)",
                                "If mother smokes",
                                "Bias (oz)"};

int main(void) {
  // Read data matrices
  puts("Reading data matrices...");
  Matrix* X_train = create_matrix_from_csv("../../data/X_train.csv");
  Matrix* y_train = create_matrix_from_csv("../../data/y_train.csv");
  Matrix* X_test = create_matrix_from_csv("../../data/X_test.csv");
  Matrix* y_test = create_matrix_from_csv("../../data/y_test.csv");

  // Calculate weights to use
  puts("Running linear regression...");
  Matrix* weights = cpu_regression(X_train, y_train);

  // Print weights
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

  free_matrix_host(weights);
  free_matrix_host(X_train);
  free_matrix_host(y_train);
  free_matrix_host(X_test);
  free_matrix_host(y_test);
  return 0;
}
