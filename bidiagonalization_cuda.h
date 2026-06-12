#pragma once

#include "matrix.h"

// CUDA/cuBLAS 版本的上二对角化。
// 输出满足 A = U * B * V^T，其中 B 为上二对角矩阵。
Matrix to_bidiagonal_cuda(const Matrix &A, Matrix &U, Matrix &V);
