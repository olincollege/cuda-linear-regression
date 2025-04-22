#include <iostream>

#include "linear_regression.hpp"
#include "matrix.hpp"

int main(int argc, char** argv) {
  std::cout << "Cuda‑Linear‑Regression demo\n";
  Matrix* mat = create_matrix_from_csv(
      "/home/akurtz/cuda-neural-network/test/data/1000x1000.csv");

  // for (size_t i = 0; i < mat->rows * mat->cols; i++) {
  //   printf("%f, ", mat->elements[i]);
  //   if (((i + 1) % 101) == 0) {
  //     puts("");
  //   }
  // }
  // puts("");
  printf("(%zu, %zu)\n", mat->rows, mat->cols);
  printf("%p\n", mat);
  free_matrix_host(mat);
  return 0;
}
