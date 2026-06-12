#include "bidiagonalization_cuda.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " : " << cudaGetErrorString(err__) << std::endl;       \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t stat__ = (call);                                         \
        if (stat__ != CUBLAS_STATUS_SUCCESS) {                                  \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__     \
                      << " : status = " << static_cast<int>(stat__)             \
                      << std::endl;                                             \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

static double vector_norm_host(const std::vector<double> &v) {
    double sum = 0.0;
    for (double x : v) {
        sum += x * x;
    }
    return std::sqrt(sum);
}

static void matrix_to_col_major(const Matrix &M, std::vector<double> &out) {
    const int rows = M.rows();
    const int cols = M.cols();
    out.assign(static_cast<size_t>(rows) * cols, 0.0);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            out[static_cast<size_t>(i) + static_cast<size_t>(j) * rows] = M.at(i, j);
        }
    }
}

static Matrix col_major_to_matrix(const std::vector<double> &in, int rows, int cols) {
    Matrix M(rows, cols, 0.0);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            M.at(i, j) = in[static_cast<size_t>(i) + static_cast<size_t>(j) * rows];
        }
    }

    return M;
}

static void make_identity_col_major(std::vector<double> &out, int n) {
    out.assign(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) {
        out[static_cast<size_t>(i) + static_cast<size_t>(i) * n] = 1.0;
    }
}

__global__ void zero_below_col_kernel(double *B, int m, int n, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = k + 1 + idx;

    if (i < m) {
        B[i + k * m] = 0.0;
    }
}

__global__ void zero_right_row_kernel(double *B, int m, int n, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int j = k + 2 + idx;

    if (j < n) {
        B[k + j * m] = 0.0;
    }
}

__global__ void ger_update_kernel(double *A,
                                  int lda,
                                  const double *x,
                                  const double *y,
                                  double alpha,
                                  int rows,
                                  int cols) {
    // Column-major layout:
    // A(row, col) = A[row + col * lda]
    //
    // blockDim.x is assigned to consecutive rows, so neighboring lanes
    // mainly access contiguous memory locations for the same column.
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        A[row + col * lda] += alpha * x[row] * y[col];
    }
}

static void ger_update(cublasHandle_t handle,
                       bool use_custom_ger,
                       int rows,
                       int cols,
                       const double *alpha,
                       const double *x,
                       int incx,
                       const double *y,
                       int incy,
                       double *A,
                       int lda) {
    // The current bidiagonalization code always calls GER with incx=incy=1.
    // If a non-unit stride appears in the future, fall back to cublasDger.
    if (use_custom_ger && incx == 1 && incy == 1) {
        dim3 block(32, 8);
        dim3 grid((rows + block.x - 1) / block.x,
                  (cols + block.y - 1) / block.y);

        ger_update_kernel<<<grid, block>>>(A, lda, x, y, *alpha, rows, cols);
        CUDA_CHECK(cudaGetLastError());
    } else {
        CUBLAS_CHECK(cublasDger(handle,
                                rows,
                                cols,
                                alpha,
                                x,
                                incx,
                                y,
                                incy,
                                A,
                                lda));
    }
}

Matrix to_bidiagonal_cuda(const Matrix &A, Matrix &U, Matrix &V) {
    if (A.rows() < A.cols()) {
        throw std::invalid_argument("to_bidiagonal_cuda: requires m >= n");
    }

    const int m = A.rows();
    const int n = A.cols();

    std::vector<double> hB;
    std::vector<double> hU;
    std::vector<double> hV;

    matrix_to_col_major(A, hB);
    make_identity_col_major(hU, m);
    make_identity_col_major(hV, n);

    double *dB = nullptr;
    double *dU = nullptr;
    double *dV = nullptr;
    double *dv = nullptr;
    double *dw = nullptr;

    const size_t sizeB = static_cast<size_t>(m) * n * sizeof(double);
    const size_t sizeU = static_cast<size_t>(m) * m * sizeof(double);
    const size_t sizeV = static_cast<size_t>(n) * n * sizeof(double);
    const int max_len = std::max(m, n);

    CUDA_CHECK(cudaMalloc(&dB, sizeB));
    CUDA_CHECK(cudaMalloc(&dU, sizeU));
    CUDA_CHECK(cudaMalloc(&dV, sizeV));
    CUDA_CHECK(cudaMalloc(&dv, static_cast<size_t>(max_len) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dw, static_cast<size_t>(max_len) * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(dB, hB.data(), sizeB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dU, hU.data(), sizeU, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV.data(), sizeV, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const char *ger_env = std::getenv("SVD_GER");
    const bool use_custom_ger = ger_env && std::string(ger_env) == "custom";

    static bool printed_ger_backend = false;
    if (!printed_ger_backend) {
        std::cout << "GER backend: "
                  << (use_custom_ger ? "custom CUDA kernel" : "cuBLAS cublasDger")
                  << std::endl;
        printed_ger_backend = true;
    }

    const double one = 1.0;
    const double zero = 0.0;

    std::vector<double> hx(max_len);
    std::vector<double> hv(max_len);

    for (int k = 0; k < n; ++k) {
        // ================================================================
        // Step 1: left Householder
        // B[k:m, k:n] = B[k:m, k:n] - beta * v * (v^T * B[k:m, k:n])
        // U[:, k:m]   = U[:, k:m]   - beta * (U[:, k:m] * v) * v^T
        // ================================================================
        const int rows_left = m - k;
        const int cols_left = n - k;

        CUDA_CHECK(cudaMemcpy(hx.data(),
                              dB + k + k * m,
                              static_cast<size_t>(rows_left) * sizeof(double),
                              cudaMemcpyDeviceToHost));

        double norm_x = 0.0;
        for (int i = 0; i < rows_left; ++i) {
            norm_x += hx[i] * hx[i];
        }
        norm_x = std::sqrt(norm_x);

        if (norm_x > 1e-14 && k < m - 1) {
            double sigma = (hx[0] >= 0.0 ? 1.0 : -1.0) * norm_x;

            for (int i = 0; i < rows_left; ++i) {
                hv[i] = hx[i];
            }
            hv[0] += sigma;

            double vTv = 0.0;
            for (int i = 0; i < rows_left; ++i) {
                vTv += hv[i] * hv[i];
            }

            if (vTv > 1e-28) {
                const double beta = 2.0 / vTv;
                const double alpha = -beta;

                CUDA_CHECK(cudaMemcpy(dv,
                                      hv.data(),
                                      static_cast<size_t>(rows_left) * sizeof(double),
                                      cudaMemcpyHostToDevice));

                // w = Bsub^T * v
                CUBLAS_CHECK(cublasDgemv(handle,
                                         CUBLAS_OP_T,
                                         rows_left,
                                         cols_left,
                                         &one,
                                         dB + k + k * m,
                                         m,
                                         dv,
                                         1,
                                         &zero,
                                         dw,
                                         1));

                // Bsub = Bsub - beta * v * w^T
                ger_update(handle,
                           use_custom_ger,
                           rows_left,
                           cols_left,
                           &alpha,
                           dv,
                           1,
                           dw,
                           1,
                           dB + k + k * m,
                           m);

                // wU = Usub * v, Usub = U[:, k:m]
                const int ucols = m - k;
                CUBLAS_CHECK(cublasDgemv(handle,
                                         CUBLAS_OP_N,
                                         m,
                                         ucols,
                                         &one,
                                         dU + k * m,
                                         m,
                                         dv,
                                         1,
                                         &zero,
                                         dw,
                                         1));

                // Usub = Usub - beta * wU * v^T
                ger_update(handle,
                           use_custom_ger,
                           m,
                           ucols,
                           &alpha,
                           dw,
                           1,
                           dv,
                           1,
                           dU + k * m,
                           m);
            }
        }

        // force B[k+1:m, k] = 0
        if (m - k - 1 > 0) {
            int threads = 256;
            int blocks = (m - k - 1 + threads - 1) / threads;
            zero_below_col_kernel<<<blocks, threads>>>(dB, m, n, k);
            CUDA_CHECK(cudaGetLastError());
        }

        // ================================================================
        // Step 2: right Householder
        // B[k:m, k+1:n] = B[k:m, k+1:n] - beta * (B[k:m, k+1:n] * v) * v^T
        // V[:, k+1:n]   = V[:, k+1:n]   - beta * (V[:, k+1:n] * v) * v^T
        // ================================================================
        if (k < n - 2) {
            const int cols_right = n - k - 1;
            const int rows_right = m - k;

            // Copy row B[k, k+1:n] with stride m into contiguous dv.
            CUBLAS_CHECK(cublasDcopy(handle,
                                     cols_right,
                                     dB + k + (k + 1) * m,
                                     m,
                                     dv,
                                     1));

            CUDA_CHECK(cudaMemcpy(hx.data(),
                                  dv,
                                  static_cast<size_t>(cols_right) * sizeof(double),
                                  cudaMemcpyDeviceToHost));

            double norm_y = 0.0;
            for (int j = 0; j < cols_right; ++j) {
                norm_y += hx[j] * hx[j];
            }
            norm_y = std::sqrt(norm_y);

            if (norm_y > 1e-14) {
                double sigma = (hx[0] >= 0.0 ? 1.0 : -1.0) * norm_y;

                for (int j = 0; j < cols_right; ++j) {
                    hv[j] = hx[j];
                }
                hv[0] += sigma;

                double vTv = 0.0;
                for (int j = 0; j < cols_right; ++j) {
                    vTv += hv[j] * hv[j];
                }

                if (vTv > 1e-28) {
                    const double beta = 2.0 / vTv;
                    const double alpha = -beta;

                    CUDA_CHECK(cudaMemcpy(dv,
                                          hv.data(),
                                          static_cast<size_t>(cols_right) * sizeof(double),
                                          cudaMemcpyHostToDevice));

                    // w = Bsub * v
                    CUBLAS_CHECK(cublasDgemv(handle,
                                             CUBLAS_OP_N,
                                             rows_right,
                                             cols_right,
                                             &one,
                                             dB + k + (k + 1) * m,
                                             m,
                                             dv,
                                             1,
                                             &zero,
                                             dw,
                                             1));

                    // Bsub = Bsub - beta * w * v^T
                    ger_update(handle,
                               use_custom_ger,
                               rows_right,
                               cols_right,
                               &alpha,
                               dw,
                               1,
                               dv,
                               1,
                               dB + k + (k + 1) * m,
                               m);

                    // wV = Vsub * v, Vsub = V[:, k+1:n]
                    const int vcols = n - k - 1;
                    CUBLAS_CHECK(cublasDgemv(handle,
                                             CUBLAS_OP_N,
                                             n,
                                             vcols,
                                             &one,
                                             dV + (k + 1) * n,
                                             n,
                                             dv,
                                             1,
                                             &zero,
                                             dw,
                                             1));

                    // Vsub = Vsub - beta * wV * v^T
                    ger_update(handle,
                               use_custom_ger,
                               n,
                               vcols,
                               &alpha,
                               dw,
                               1,
                               dv,
                               1,
                               dV + (k + 1) * n,
                               n);
                }
            }

            // force B[k, k+2:n] = 0
            if (n - k - 2 > 0) {
                int threads = 256;
                int blocks = (n - k - 2 + threads - 1) / threads;
                zero_right_row_kernel<<<blocks, threads>>>(dB, m, n, k);
                CUDA_CHECK(cudaGetLastError());
            }
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hB.data(), dB, sizeB, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hU.data(), dU, sizeU, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hV.data(), dV, sizeV, cudaMemcpyDeviceToHost));

    Matrix B = col_major_to_matrix(hB, m, n);
    U = col_major_to_matrix(hU, m, m);
    V = col_major_to_matrix(hV, n, n);

    CUBLAS_CHECK(cublasDestroy(handle));

    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dU));
    CUDA_CHECK(cudaFree(dV));
    CUDA_CHECK(cudaFree(dv));
    CUDA_CHECK(cudaFree(dw));

    return B;
}
