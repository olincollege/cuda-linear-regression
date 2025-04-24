#include "cpu_matrix.h"
#include "linear_regression.h"
#include "matrix.h"
#include "stdio.h"

int main(int argc, char** argv) {
  printf("Cuda‑Linear‑Regression demo\n");
  Matrix* mat = create_matrix_from_csv(
      "/home/akurtz/cuda-neural-network/test/data/3x3_matrix_2.csv");

  Matrix* inv = cpu_matrix_inverse(mat);
  for (size_t i = 0; i < mat->rows * mat->cols; i++) {
    printf("%f, ", mat->elements[i]);
    if (((i + 1) % mat->rows) == 0) {
      puts("");
    }
  }
  puts("");

  printf("%p\n", inv);
  for (size_t i = 0; i < mat->rows * mat->cols; i++) {
    printf("%f, ", inv->elements[i]);
    if (((i + 1) % mat->rows) == 0) {
      puts("");
    }
  }
  puts("");

  printf("(%zu, %zu)\n", mat->rows, mat->cols);

  free_matrix_host(mat);
  free_matrix_host(inv);
  return 0;
}
