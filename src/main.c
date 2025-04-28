#include "cpu_matrix.h"
#include "linear_regression.h"
#include "matrix.h"
#include "stdio.h"

int main(int argc, char** argv) {
  Matrix* X_train = create_matrix_from_csv(
      "/home/akurtz/cuda-neural-network/data/X_train.csv");
  Matrix* y_train = create_matrix_from_csv(
      "/home/akurtz/cuda-neural-network/data/y_train.csv");
  Matrix* weights = gpu_regression(X_train, y_train);

  printf("(%zu, %zu)\n", weights->rows, weights->cols);
  for (size_t i = 0; i < weights->rows * weights->cols; i++) {
    printf("%f, ", weights->elements[i]);
    if (((i + 1) % weights->rows) == 0) {
      puts("");
    }
  }

  // printf("%p\n", mat);
  // for (size_t i = 0; i < mat->rows * mat->cols; i++) {
  //   printf("%f, ", mat->elements[i]);
  //   if (((i + 1) % mat->rows) == 0) {
  //     puts("");
  //   }
  // }
  // puts("");

  // printf("(%zu, %zu)\n", mat->rows, mat->cols);

  // free_matrix_host(mat);
  return 0;
}
