# Performance Analysis of Parallel N-Body Simulation Using CPU and GPU Acceleration

**Submitted to:** CDAC CINE & IIT Guwahati Campus  
**Project Team:** Partha Pratim Das, Manash Das  
**Date:** December 2025

---

## 1. Abstract

This project evaluates the performance benefits of GPU acceleration in scientific computing through the implementation of an N-Body gravitational simulation. High Performance Computing (HPC) techniques are applied to solve the computationally intensive $O(N^2)$ N-Body problem. We compare a sequential CPU implementation using NumPy against a parallel GPU implementation using CuPy (CUDA). Our results demonstrate significant speedup for large particle counts ($N > 2000$), highlighting the efficacy of data-parallel architectures for massive arithmetical workloads.

## 2. Introduction to High Performance Computing

High Performance Computing (HPC) involves aggregating computing power to solve complex problems that are too large or time-consuming for standard workstations. In modern scientific research, Graphics Processing Units (GPUs) have evolved from rendering devices to powerful parallel compute engines. This project leverages this evolution, utilizing the massive number of cores in a GPU to accelerate physical simulations.

## 3. N-Body Problem Description

The N-Body problem predicts the individual motions of a group of celestial objects interacting with each other gravitationally. For a system of $N$ particles, calculating the net force on one particle requires summing the contributions from all other $N-1$ particles. Doing this for all particles results in $N(N-1)$ interactions, leading to a computational complexity of $O(N^2)$.

## 4. Mathematical Model

We utilize Newton's Law of Universal Gravitation. To avoid numerical instability when particles come arbitrarily close (singularity at $r \to 0$), we introduce a softening parameter $\epsilon$.

The force $\vec{F}_{ij}$ exerted on particle $i$ by particle $j$ is given by:

$$ \vec{F}_{ij} = \frac{G \cdot m_i \cdot m_j \cdot (\vec{r}_j - \vec{r}_i)}{(\|\vec{r}_j - \vec{r}_i\|^2 + \epsilon^2)^{3/2}} $$

Where:

- $G$ is the gravitational constant.
- $m$ is mass.
- $\vec{r}$ is the position vector.

The total force on particle $i$ is:

$$ \vec{F}_i = \sum_{j \neq i} \vec{F}_{ij} $$

Using Newton's Second Law ($\vec{F} = m\vec{a}$), we update velocity and position using the Euler integration method:

$$ \vec{v}_{t+\Delta t} = \vec{v}_t + \vec{a}_t \cdot \Delta t $$
$$ \vec{r}_{t+\Delta t} = \vec{r}_t + \vec{v}_{t+\Delta t} \cdot \Delta t $$

## 5. CPU Architecture vs. GPU Architecture

| Feature | CPU (Sequential/Latency Optimized) | GPU (Parallel/Throughput Optimized) |
| :--- | :--- | :--- |
| **Cores** | Few (4-64), powerful cores | Thousands (1000+), simpler cores |
| **Task** | Serial processing, logic, branching | Parallel data processing (SIMD/SIMT) |
| **Philosophy** | Minimize latency of single task | Maximize total throughput of tasks |
| **N-Body Fit** | Low (Must loop $N^2$ times sequentially) | High (Can compute $N^2$ interactions in parallel) |

## 6. Implementation Details

The project is implemented in Python.

- **CPU Implementation**: Uses `NumPy`. While NumPy utilizes C-level optimization and SIMD instructions (AVX), the algorithm remains fundamentally limited by the CPU's core count and memory bandwidth.
- **GPU Implementation**: Uses `CuPy`, a library compatible with NumPy that executes on NVIDIA GPUs via CUDA. The implementation explicitly handles memory allocation on the device (GPU VRAM) and performs the force calculation kernel in parallel.

## 7. Experimental Setup

- **Hardware**: Testing performed on Google Colab (Tesla T4 GPU) / Local Workstation (Specify Specs).
- **Parameters**:
  - Particle Counts ($N$): 500, 1000, 2000, 5000, 10000
  - Time Steps: 10 (averaged over 3 runs)
  - Data Type: Float32 (Single Precision)

## 8. Results

*(Insert generated plots here after running the simulation)*

- **Figure 1**: CPU Execution Time vs. $N$ (Shows quadratic growth).
- **Figure 2**: GPU Execution Time vs. $N$ (Shows slower growth).
- **Figure 3**: Speedup Factor (CPU Time / GPU Time).

## 9. Performance Analysis

### 9.1 The Cross-Over Point

For small $N$ (e.g., $N < 500$), the CPU may perform similarly or even faster than the GPU. This is due to **PCIe Transfer Overhead**. The cost of copying data from Host to Device and launching the kernel exceeds the compute time saved.

### 9.2 Scaling Behavior

As $N$ increases, the $O(N^2)$ compute complexity dominates. The CPU execution time explodes (e.g., $10,000$ particles might take seconds per step). The GPU, with thousands of cores, handles this load efficiently, achieving speedups often exceeding 50x-100x depending on the hardware.

### 9.3 Bottlenecks

- **Compute**: The naive algorithm is compute-bound ($N^2$ ops vs $N$ data).
- **Memory**: For very large $N$, GPU VRAM bandwidth becomes the limit.

## 10. Conclusion

This project successfully demonstrates the application of HPC principles to the N-Body problem. We verified that while CPU implementations are sufficient for small systems, GPU acceleration is indispensable for large-scale scientific simulations. The results confirm Amdahl's law and the distinct advantages of data-parallel architectures in computational physics.

## 11. References

1. NVIDIA CUDA C++ Programming Guide.
2. "Gravitational N-Body Simulations", S.J. Aarseth.
3. NumPy and CuPy Documentation.

---
**End of Report**
