#include <criterion/criterion.h>
#include <criterion/new/assert.h>

#include "matrix.h"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers,cppcoreguidelines-pro-bounds-pointer-arithmetic)
// Since we're testing, magic numbers are needed to configure specific cases
// Also ignore pointer arithmetic warning, since in all cases it is in a loop
// which prevents out of bounds

// Test rows and cols initialized correctly on host
Test(create_matrix_host, rows_and_cols_set_correctly) {
  Matrix* h_mat = create_matrix_host(3, 4);
  cr_assert(eq(int, h_mat->rows, 3), "3 expected but got %d rows", h_mat->rows);
  cr_assert(eq(int, h_mat->cols, 4), "4 expected but got %d cols", h_mat->cols);
  free_matrix_host(h_mat);
}

// Test elements pointer is not NULL on host
Test(create_matrix_host, elements_not_null) {
  Matrix* h_mat = create_matrix_host(3, 4);
  cr_assert(ne(ptr, h_mat->elements, NULL), "Element pointer is NULL");
  free_matrix_host(h_mat);
}

// Test rows and cols initialized correctly on device
Test(create_matrix_device, rows_and_cols_set_correctly) {
  Matrix* d_mat = create_matrix_device(3, 4);
  Matrix* h_mat = copy_matrix_device_to_host(d_mat);
  cr_assert(eq(int, h_mat->rows, 3), "3 expected but got %d rows", h_mat->rows);
  cr_assert(eq(int, h_mat->cols, 4), "4 expected but got %d cols", h_mat->cols);
  free_matrix_host(h_mat);
  free_matrix_device(d_mat);
}

// Test elements pointer is not NULL on device
Test(create_matrix_device, elements_not_null) {
  Matrix* d_mat = create_matrix_device(3, 4);
  Matrix* h_mat = copy_matrix_device_to_host(d_mat);
  cr_assert(ne(ptr, h_mat->elements, NULL), "Element pointer is NULL");
  free_matrix_host(h_mat);
  free_matrix_device(d_mat);
}

// Test struct values are the same when copied in loop
Test(copy_matrix, copies_struct_correctly) {
  Matrix* h_mat = create_matrix_host(3, 4);
  Matrix* d_mat = copy_matrix_host_to_device(h_mat);
  Matrix* new_h_mat = copy_matrix_device_to_host(d_mat);
  cr_assert(eq(int, h_mat->rows, new_h_mat->rows),
            "%d expected but got %d rows", h_mat->rows, new_h_mat->rows);
  cr_assert(eq(int, h_mat->cols, new_h_mat->cols),
            "%d expected but got %d cols", h_mat->cols, new_h_mat->cols);
  free_matrix_host(h_mat);
  free_matrix_host(new_h_mat);
  free_matrix_device(d_mat);
}

// Test element values are the same when copied in loop
Test(copy_matrix, copies_elements_correctly) {
  Matrix* h_mat = create_matrix_host(3, 4);
  for (int i = 0; i < 12; i++) {
    h_mat->elements[i] = (float)(i + 1);
  }
  Matrix* d_mat = copy_matrix_host_to_device(h_mat);
  Matrix* new_h_mat = copy_matrix_device_to_host(d_mat);
  for (int i = 0; i < 12; i++) {
    cr_assert(ieee_ulp_eq(flt, h_mat->elements[i], new_h_mat->elements[i], 3),
              "%f expected but got %f", (double)h_mat->elements[i],
              (double)new_h_mat->elements[i]);
  }
  free_matrix_host(h_mat);
  free_matrix_host(new_h_mat);
  free_matrix_device(d_mat);
}

// Test elements of load a regular small CSV
Test(load_matrix, small_CSV_elements) {
  Matrix* loaded = create_matrix_from_csv("../../data/3x3_matrix.csv");
  for (int i = 0; i < 9; i++) {
    cr_assert(ieee_ulp_eq(flt, loaded->elements[i], (float)(i + 1), 3),
              "%f expected but got %f", (double)(i + 1),
              (double)loaded->elements[i]);
  }
  free_matrix_host(loaded);
}

// Test correct rows and columns of small matrix
Test(load_matrix, small_CSV_size) {
  Matrix* loaded = create_matrix_from_csv("../../data/3x3_matrix.csv");
  cr_assert(eq(int, loaded->rows, 3), "3 expected but got %d rows",
            loaded->rows);
  cr_assert(eq(int, loaded->cols, 3), "3 expected but got %d cols",
            loaded->cols);
  free_matrix_host(loaded);
}

// Test elements of huge matrix
Test(load_matrix, huge_csv_elements) {
  Matrix* loaded = create_matrix_from_csv("../../data/1000x1000.csv");
  for (int i = 0; i < (1000000); i++) {
    cr_assert(ieee_ulp_eq(flt, loaded->elements[i], (float)(i), 3),
              "%f expected but got %f", (double)i, (double)loaded->elements[i]);
  }
  free_matrix_host(loaded);
}

// Test correct rows and columns of huge matrix
Test(load_matrix, huge_csv_size) {
  Matrix* loaded = create_matrix_from_csv("../../data/1000x1000.csv");
  cr_assert(eq(int, loaded->rows, 1000), "1000 expected but got %d rows",
            loaded->rows);
  cr_assert(eq(int, loaded->cols, 1000), "1000 expected but got %d cols",
            loaded->cols);
  free_matrix_host(loaded);
}

// Test unevenly sized Matrix is NULL
Test(load_matrix, uneven_size) {
  Matrix* loaded = create_matrix_from_csv("../../data/uneven_matrix.csv");
  cr_assert(eq(ptr, loaded, NULL), "NULL expected but got %p", loaded);
}

// Test trailing commas read correclty
Test(load_matrix, trailing_comma_elements) {
  Matrix* loaded =
      create_matrix_from_csv("../../data/3x3_matrix_ending_commas.csv");
  for (int i = 0; i < 9; i++) {
    cr_assert(ieee_ulp_eq(flt, loaded->elements[i], (float)(i + 1), 3),
              "%f expected but got %f", (double)(i + 1),
              (double)loaded->elements[i]);
  }
  free_matrix_host(loaded);
}

// Test correct rows and columns of small matrix
Test(load_matrix, trailing_comma_size) {
  Matrix* loaded =
      create_matrix_from_csv("../../data/3x3_matrix_ending_commas.csv");
  cr_assert(eq(int, loaded->rows, 3), "3 expected but got %d rows",
            loaded->rows);
  cr_assert(eq(int, loaded->cols, 3), "3 expected but got %d cols",
            loaded->cols);
  free_matrix_host(loaded);
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers,cppcoreguidelines-pro-bounds-pointer-arithmetic)
