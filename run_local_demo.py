import sys
import os
import time

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from cpu_nbody import NBodyCPU
    print("✅ Successfully imported NBodyCPU")
except ImportError as e:
    print(f"❌ Failed to import NBodyCPU: {e}")
    sys.exit(1)

try:
    from gpu_nbody import NBodyGPU, HAS_GPU
    print(f"✅ Successfully imported NBodyGPU (HAS_GPU={HAS_GPU})")
except ImportError as e:
    print(f"⚠️ Failed to import NBodyGPU: {e}")
    HAS_GPU = False

def run_demo():
    print("\n--- Starting Local Demo Run ---")
    
    # 1. CPU Run
    n_particles = 500
    steps = 10
    dt = 0.01
    
    print(f"\nrunning CPU Simulation (N={n_particles}, Steps={steps})...")
    sim_cpu = NBodyCPU(n_particles)
    start = time.perf_counter()
    sim_cpu.run_simulation(steps, dt)
    end = time.perf_counter()
    print(f"CPU Verification Complete! Time: {end-start:.4f}s")

    # 2. GPU Run
    if HAS_GPU:
        print(f"\nrunning GPU Simulation (N={n_particles}, Steps={steps})...")
        try:
            sim_gpu = NBodyGPU(n_particles)
            start = time.perf_counter()
            sim_gpu.run_simulation(steps, dt)
            end = time.perf_counter()
            print(f"GPU Verification Complete! Time: {end-start:.4f}s")
        except Exception as e:
            print(f"GPU Runtime Error: {e}")
            print("Note: This is expected if you don't have a CUDA-capable GPU or correct CuPy version installed.")
    else:
        print("\nSkipping GPU run (CuPy not installed or no GPU detected).")
        print("To run the GPU version, please use Google Colab or install 'cupy-cudaXX'.")

if __name__ == "__main__":
    run_demo()
