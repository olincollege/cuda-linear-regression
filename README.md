# CUDA Linear Regression

Linear Regression implemented for both CPU and GPU (with CUDA) using the least squares solution, built with matrix operations libraries fors each platform. Both libraries are benchmarked to compare speed. To test the implementation, a model is trained to predict child birth weight from gestation details using this [Kaggle dataset](https://www.kaggle.com/datasets/jacopoferretti/child-weight-at-birth-and-gestation-details).

### Features

- CPU and GPU Support: Implements core matrix operations on both CPU (C) and GPU (CUDA) backends.

- Exact Least Squares Regression: Solves w = (XᵀX)⁻¹ Xᵀy using matrix operations for accurate model weights.

- Benchmarking Included: Measures and compares average runtime across 100 trials on CPU and GPU.

- Practical Example: Trains on real-world birth weight data and evaluates with mean absolute error.

## Structure

- `src`: Implementation code
    - `src/main.c`: Runs linear regression and benchmarking
    - `core`: Matrix operations libraries
        - `cpu_matrix.c/.h`: CPU matrix operations library
        - `gpu_matrix.cu/.h`: GPU matrix operations written in CUDA
    - `models`: Linear regression library
        - `linear_regression.c/.h`: Library to perform linear regression on CPU and GPU and calculate loss
    - `utils`: Helper functions used throughout
        - `matrix.cu/.h`: Define Matrix struct, load Matrix from CSV, and manage memory
        - `timing.c/.h`: Get the current time for benchmarking
- `test`: Unit tests to ensure correct implementations of all five `cpu_matrix`, `gpu_matrix`, `linear_regression`, `matrix`, and `timing`
- `data`: Data for model training and testing
    - `X_train/X_test.csv`: Training and testing gestation data with last cols 1 for bias term
    - `y_train/y_test.csv`: Training and testing target data
    - The remainder of the files are to test reading CSV to matrix

## Data

As discussed, this [Kaggle gestation dataset](https://www.kaggle.com/datasets/jacopoferretti/child-weight-at-birth-and-gestation-details) was used to predict child birth weight. In the `data` folder the `X_test` and `X_train` are the feature datapoints and the `y_test`, and `y_train` contains the corresponding birthweights. They are split into a train test split to validate model generalization.

The data columns of the feature data (X files) are as follows:
1. Gestation length (days)
2. If first pregnancy
3. Mother's age (yrs)
4. Mother's height (in)
5. Mother's weight (lbs)
6. If mother smokes
7. 1s to be bias (y-intercept) term

The target data (y files) contain the child birthweight in ounces.

## Running Instructions

This project will only run on a system with an NVIDIA GPU due to the need for CUDA.

### Dependencies
- `CUDA Toolkit`: Used to compile CUDA to run on GPU. Install with:
```bash 
$ sudo apt install cuda
```
Note 1: You may need to ensure this is compatible with your nvidia GPU drivers. You can find more details on the official [Nvidia documentation](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html).

Note 2: For your computer to find NVCC, you may need to add it to your PATH using an environment variable temporarily or ~/.bashrc or ~/.zshrc for persistance.

### Build and Run

Clone the repository:
```bash
$ git clone https://github.com/olincollege/cuda-neural-network.git
```
Build it with:
```bash
$ mkdir build && cd build
$ cmake ..
$ make
```

If you run into issues building, you may have a different GPU architecture and need to update the top `CMakeLists.txt` `set(CMAKE_CUDA_ARCHITECTURES 75)` line. Here is a source with an explanation and list of values depending on GPU.

Run main with:
```bash
./src/main
```

