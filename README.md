# Performance Analysis of Parallel N-Body Simulation Using CPU and GPU Acceleration

**Submitted under:** CDAC CINE, IIT Guwahati Campus  
**Contributors:** Partha Pratim Das, Manash Das

---

## Project Overview

This project implements a classic N-Body gravitational simulation to demonstrate the power of High Performance Computing (HPC) using GPU acceleration. We compare a baseline sequential CPU implementation (NumPy) against a parallel GPU implementation (CuPy) to measure speedup and scalability.

This work addresses the $O(N^2)$ computational complexity of the N-Body problem and answers the question: **When does GPU acceleration become beneficial?**

## Key Features

- **Scientific Correctness:** Uses Newton's Law of Gravitation with softening.
- **Dual Implementation:**
  - CPU: Sequential, vectorized implementation using NumPy.
  - GPU: Massive parallel implementation using CuPy (CUDA).
- **Benchmarking:** Rigorous performance analysis across varying particle counts ($N$).
- **Visualization:** Speedup curves, scalability plots, and particle animations.
- **Portability:** Designed to run seamlessly on Google Colab and local implementations with NVIDIA GPUs.

## HPC Concepts Demonstrated

- **Data Parallelism:** Distributing particle force calculations across thousands of GPU cores.
- **Memory Bandwidth vs. Compute Bound:** Analyzing the transfer overhead vs. computational gain.
- **Amdahl's Law:** Understanding the limits of parallel speedup.
- **Vectorization:** Efficient array operations in Python.

## Folder Structure

```
HPC-NBody-Simulation/
├── notebooks/          # Main logical notebook for benchmarking & visualization
├── src/                # Python source modules for N-Body logic
│   ├── cpu_nbody.py    # NumPy implementation
│   └── gpu_nbody.py    # CuPy implementation
├── data/               # Directory for initial conditions (auto-generated)
├── results/            # Generated plots and animations
├── report/             # Project report and documentation
└── requirements.txt    # Python dependencies
```

## How to Run

### Option 1: Google Colab (Recommended)

1. Upload the entire `HPC-NBody-Simulation` folder to your Google Drive.
2. Open `notebooks/nbody_cpu_vs_gpu.ipynb` in Google Colab.
3. Mount Google Drive.
4. Change the runtime type to **GPU**.
5. Run all cells to execute benchmarks and generate results.

### Option 2: Local Execution

**Prerequisites:** NVIDIA GPU, CUDA Toolkit installed.

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd HPC-NBody-Simulation
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   *Note: You may need to install the specific CuPy version for your CUDA toolkit (e.g., `pip install cupy-cuda12x`)*.
4. detailed steps are in the notebook.

## Expected Outputs

- Console output showing execution times for different $N$.
- Plots in `results/`:
  - `cpu_time_vs_particles.png`
  - `gpu_time_vs_particles.png`
  - `speedup_curve.png`
- `nbody_simulation.gif`: Animation of the planetary motion.

---
*Developed for academic evaluation.*
