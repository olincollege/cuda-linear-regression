#include <iostream>

#include "linear_regression.hpp"
#include "matrix.hpp"

int main(int argc, char** argv) {
  std::cout << "Cuda‑Linear‑Regression demo\n";
  // TODO: load data, build Matrix objects, call cpu_regression / gpu_regression
  // …
  Matrix* h_mat = create_matrix_host(3, 4);
  for (int i = 0; i < 12; i++) {
    h_mat->elements[i] = (float)(i + 1);
  }
  Matrix* d_mat = copy_matrix_host_to_device(h_mat);
  Matrix* new_h_mat = copy_matrix_device_to_host(d_mat);
  free_matrix_host(h_mat);
  free_matrix_host(new_h_mat);
  free_matrix_device(d_mat);
  return 0;
}
