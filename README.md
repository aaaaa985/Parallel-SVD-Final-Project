# CUDA-SVD-Bidiagonalization

基于 CUDA/cuBLAS 的矩阵 SVD 上二对角化阶段 GPU 加速实现。

本项目是**南开大学计算机学院《并行程序设计》课程大作业之一**。项目围绕矩阵奇异值分解（Singular Value Decomposition, SVD）中的 Householder 上二对角化阶段进行 GPU 编程优化，并对 CPU baseline、CUDA/cuBLAS 后端和手写 CUDA kernel 后端进行正确性验证与性能对比。

## 项目简介

矩阵 SVD 分解可表示为：

```text
A = U * Sigma * V^T
```

本项目原始 SVD 实现采用两阶段流程：

1. 使用 Householder 变换将输入矩阵化为上二对角矩阵；
2. 在上二对角矩阵上执行 Golub-Kahan SVD 迭代，得到奇异值与正交矩阵。

本项目主要优化第一阶段，即 **Householder 上二对角化阶段**。第二阶段 Golub-Kahan 迭代仍保留在 CPU 上执行。

Householder 上二对角化过程中包含大量规则的矩阵-向量乘法和秩一更新操作，可抽象为：

* GEMV：矩阵-向量乘法；
* GER：秩一矩阵更新。

因此，本项目使用 CUDA/cuBLAS 将这些核心 BLAS2 操作迁移到 GPU，并进一步实现了手写 CUDA GER kernel，与 `cublasDger` 进行性能对比。

## 功能特点

* 保留 CPU baseline，用于正确性与性能对比；
* 使用 CUDA/cuBLAS 加速 Householder 上二对角化阶段；
* 使用 `cublasDgemv` 完成 GEMV 操作；
* 使用 `cublasDger` 或手写 CUDA kernel 完成 GER 操作；
* 支持通过环境变量切换 CPU / CUDA 后端；
* 支持通过环境变量切换 cuBLAS GER / custom GER；
* 对重构误差、相对重构误差、正交性误差、对角结构误差、奇异值排序与非负性进行检查；
* 输出上二对角化阶段耗时、GKH 阶段耗时和端到端运行时间。

## 文件结构

项目主要文件结构如下：

```text
.
├── main.cpp                    # 测试入口、正确性验证和计时统计
├── matrix.h                    # 矩阵类与基本矩阵运算
├── bidiagonalization.cpp       # CPU 上二对角化实现
├── bidiagonalization.h         # CPU 上二对角化接口
├── bidiagonalization_cuda.cu   # CUDA/cuBLAS 上二对角化实现
├── bidiagonalization_cuda.h    # CUDA 上二对角化接口
├── gkh.cpp                     # Golub-Kahan SVD 迭代实现
├── gkh.h                       # GKH 迭代接口
├── givens.h                    # Givens 旋转相关函数
├── Makefile                    # 编译规则
└── README.md                   # 项目说明文档
```

## 实验环境

GPU 版本在如下环境中测试通过：

```text
GPU: NVIDIA GeForce RTX 3090
显存: 24GB
CUDA Toolkit: 11.8
GPU 后端库: cuBLAS
编译器: g++ / nvcc
```

原 CPU SIMD 代码基于 ARM NEON intrinsic。由于实验 GPU 服务器为 x86_64 架构，因此使用 SIMDe 对 NEON intrinsic 进行兼容，使原 CPU baseline 能够在 x86_64 平台上编译运行。

在 Ubuntu 中可通过如下命令安装 SIMDe：

```bash
sudo apt update
sudo apt install -y libsimde-dev
```

## 编译方法

编译 CPU 和 GPU 两个版本：

```bash
make clean
make all
```

只编译 CPU 版本：

```bash
make cpu
```

只编译 GPU 版本：

```bash
make gpu
```

清理编译产物：

```bash
make clean
```

## 运行方法

运行 CPU baseline：

```bash
SVD_SEED=20260410 ./svd_cpu
```

运行 CUDA/cuBLAS 后端：

```bash
SVD_SEED=20260410 SVD_BACKEND=cuda ./svd_gpu
```

运行 CUDA 后端，并使用 cuBLAS GER：

```bash
SVD_SEED=20260410 SVD_BACKEND=cuda SVD_GER=cublas ./svd_gpu
```

运行 CUDA 后端，并使用手写 CUDA GER kernel：

```bash
SVD_SEED=20260410 SVD_BACKEND=cuda SVD_GER=custom ./svd_gpu
```

## 环境变量说明

| 环境变量          | 可选值      | 含义                         |
| ------------- | -------- | -------------------------- |
| `SVD_SEED`    | 整数       | 设置测试矩阵生成所使用的随机种子           |
| `SVD_BACKEND` | `cuda`   | 使用 CUDA/cuBLAS 上二对角化后端     |
| `SVD_GER`     | `cublas` | 使用 `cublasDger` 完成 GER 操作  |
| `SVD_GER`     | `custom` | 使用手写 CUDA kernel 完成 GER 操作 |

如果不设置 `SVD_BACKEND=cuda`，程序默认使用 CPU 上二对角化后端。

## 正确性验证

每个测试用例会检查如下指标：

* GKH 迭代是否收敛；
* 重构误差 `||A - U*S*V^T||_F`；
* 相对重构误差；
* `U` 的正交性误差；
* `V` 的正交性误差；
* 对角结构误差；
* 奇异值是否按降序排列；
* 奇异值是否非负。

只有全部指标满足要求时，该测试用例才记为 `PASS`。

测试用例包括：

```text
1. 固定值 5x5 矩阵
2. 随机 8x8 矩阵
3. 近秩亏损 10x8 矩阵
4. 随机 10x8 矩阵
5. 随机 1000x1000 矩阵
```

## 实现细节

### CPU Baseline

CPU baseline 保留原始两阶段 SVD 流程：

```text
输入矩阵 A
    -> Householder 上二对角化
    -> 上二对角矩阵 B
    -> Golub-Kahan SVD 迭代
    -> 奇异值和正交矩阵
```

原始 SIMD 代码使用 ARM NEON intrinsic。为了在 x86_64 GPU 服务器上编译运行，本项目使用 SIMDe 对 NEON intrinsic 进行兼容。

### CUDA/cuBLAS 上二对角化

项目中的 `Matrix` 类采用 row-major 存储，而 cuBLAS 默认采用 column-major 存储。因此 CUDA 版本在进入 GPU 计算前，会将矩阵从 row-major 转换为 column-major；计算完成后，再将 GPU 结果转换回原项目使用的 row-major `Matrix` 格式。

在每一轮 Householder 变换中，核心计算可写为：

```text
w = v^T * Bsub
Bsub = Bsub - beta * v * w^T
```

以及：

```text
w = Bsub * v
Bsub = Bsub - beta * w * v^T
```

对应到 cuBLAS 中，主要使用：

```text
cublasDgemv
cublasDger
```

### 手写 CUDA GER Kernel

GER 操作的数学形式为：

```text
A = A + alpha * x * y^T
```

其中每个矩阵元素的更新互不依赖：

```text
A(i, j) = A(i, j) + alpha * x(i) * y(j)
```

因此可以让每个 CUDA 线程负责一个矩阵元素。手写 kernel 如下：

```cpp
__global__ void ger_update_kernel(double *A, int lda,
                                  const double *x,
                                  const double *y,
                                  double alpha,
                                  int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        A[row + col * lda] += alpha * x[row] * y[col];
    }
}
```

由于 GPU 端矩阵采用 column-major 存储，因此元素访问方式为：

```text
A(row, col) = A[row + col * lda]
```

手写 GER kernel 主要用于与 cuBLAS 提供的 `cublasDger` 进行对比，观察简单自定义 CUDA kernel 在固定调用模式下是否能够接近库函数性能。

## 实验结果

固定随机种子为 `20260410` 时，CPU 版本和 CUDA/cuBLAS 版本均通过全部 5 组测试。

单次正式测试结果如下：

| 后端          |    上二对角化耗时 |   GKH 迭代耗时 | 通过情况 |   端到端时间 |
| ----------- | ---------: | ---------: | ---: | ------: |
| CPU/SIMDe   | 3756.79 ms | 13570.9 ms |  5/5 | 18.88 s |
| CUDA/cuBLAS | 512.364 ms | 13302.5 ms |  5/5 | 15.50 s |

上二对角化阶段加速比为：

```text
3756.79 / 512.364 ≈ 7.33x
```

端到端加速比为：

```text
18.88 / 15.50 ≈ 1.22x
```

端到端加速比较低的主要原因是 Golub-Kahan 迭代阶段仍在 CPU 上执行，成为完整 SVD 流程中的主要剩余瓶颈。

重复实验结果如下：

| 后端          |    Trial 1 |    Trial 2 |    Trial 3 |       平均耗时 |
| ----------- | ---------: | ---------: | ---------: | ---------: |
| CPU/SIMDe   | 3445.21 ms | 3258.59 ms | 2745.83 ms | 3149.88 ms |
| CUDA/cuBLAS | 555.558 ms | 412.867 ms | 418.139 ms |  462.19 ms |

重复实验中的平均加速比为：

```text
3149.88 / 462.19 ≈ 6.82x
```

GER 后端对比结果如下：

| GER 后端              |    Trial 1 |    Trial 2 |    Trial 3 |       平均耗时 | 正确性 |
| ------------------- | ---------: | ---------: | ---------: | ---------: | --: |
| cuBLAS `cublasDger` | 499.915 ms | 412.337 ms | 414.092 ms | 442.115 ms | 5/5 |
| 手写 CUDA kernel      | 505.088 ms | 413.194 ms | 404.778 ms | 441.020 ms | 5/5 |

可以看到，手写 CUDA GER kernel 与 cuBLAS `cublasDger` 的性能基本持平。在本项目固定的调用模式下，简单的一线程一元素 GER kernel 能够接近 cuBLAS 的表现。

## 性能分析

CUDA/cuBLAS 后端能够显著加速 Householder 上二对角化阶段，但完整程序的端到端加速比低于局部阶段加速比。主要原因包括：

* Householder 外层循环存在串行依赖，第 `k+1` 轮必须依赖第 `k` 轮更新后的矩阵；
* GEMV 和 GER 属于 BLAS2 操作，算术强度低于 GEMM，更容易受到访存带宽限制；
* 当前实现中 Householder 向量仍在 CPU 端构造，需要小向量在 CPU 与 GPU 之间往返传输；
* 小规模矩阵测试中，GPU 初始化、cuBLAS handle 创建和 kernel launch 开销相对计算量过大；
* Golub-Kahan SVD 迭代阶段仍在 CPU 上执行，限制了端到端加速效果。

## 后续优化方向

后续可考虑的优化方向包括：

* 将 Householder 向量构造也迁移到 GPU；
* 减少每轮迭代中的 CPU-GPU 同步和小向量传输；
* 尝试融合部分 GEMV/GER 相关操作，降低 kernel launch 次数；
* 研究 Golub-Kahan SVD 迭代阶段的 GPU 并行化；
* 针对大矩阵进一步分析访存模式和算术强度。

## 课程背景

本项目是**南开大学计算机学院《并行程序设计》课程大作业之一**。

项目展示了如何将 GPU 编程技术应用到数值线性代数问题中，并通过实验分析局部阶段加速与完整算法端到端加速之间的差异。

## License

This project is released under the MIT License.
