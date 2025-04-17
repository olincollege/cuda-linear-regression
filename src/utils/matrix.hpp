#pragma once

typedef struct {
    int height;
    int width;
    float *elements; // height x width

} Matrix;

/**
 * Frees the memory allocated for a matrix.
 *
 * @param mat Pointer to the Matrix whose memory should be released.
 */
void free_matrix(Matrix* mat);

/**
 * Initializes an existing matrix with a specific value.
 *
 * @param mat Pointer to the Matrix to initialize.
 * @param value Value to assign to every element in the matrix.
 */
void init_matrix(Matrix* mat, float value);

/**
 * Loads a matrix from a CSV file.
 *
 * The CSV file should contain numeric values separated by commas, with one row per line.
 *
 * @param filename Path to the CSV file.
 * @return Pointer to a newly allocated Matrix containing the parsed data.
 *         The caller is responsible for freeing this matrix using `free_matrix`.
 */
Matrix* create_matrix_from_csv(const char* filename);
