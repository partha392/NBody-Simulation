import sys

# Safe Import for CuPy
try:
    import cupy as cp
    # Try to access a device to ensure it's not just installed but working
    with cp.cuda.Device(0):
        pass
    HAS_GPU = True
except (ImportError, Exception) as e:
    HAS_GPU = False
    # Only print warning if we are not in a CI/Test environment to avoid noise
    if "unittest" not in sys.modules:
        print(f"INFO: GPU acceleration unavailable ({e}). Switched to CPU-only mode for functionality checks.")

class NBodyGPU:
    """
    N-Body simulation implementation using GPU (CuPy).
    """

    def __init__(self, n_particles, G=6.674e-11, epsilon=1e-5):
        """
        Initialize the GPU N-Body simulator.

        Args:
            n_particles (int): Number of particles.
            G (float): Gravitational constant.
            epsilon (float): Smoothing parameter to avoid singularities.
        """
        if not HAS_GPU:
            raise RuntimeError("CRITICAL: Cannot initialize NBodyGPU. No NVIDIA GPU or CuPy library detected.")

        self.n_particles = n_particles
        self.G = G
        self.epsilon = epsilon
        
        # Initialize data on Host (CPU)
        pos_cpu = np.random.randn(n_particles, 3).astype(np.float32)
        vel_cpu = np.random.randn(n_particles, 3).astype(np.float32)
        mass_cpu = np.ones((n_particles, 1), dtype=np.float32)

        # Transfer to Device (GPU)
        # Using cupy.array automatically allocates GPU memory
        self.pos = cp.array(pos_cpu)
        self.vel = cp.array(vel_cpu)
        self.mass = cp.array(mass_cpu)

    def compute_forces(self):
        """
        Compute gravitational forces on the GPU.
        """
        # CuPy performs operations element-wise on the GPU
        # The logic is identical to NumPy, but executed in parallel on GPU cores
        
        # Global memory access pattern here is not fully optimized (shared memory tiling is better),
        # but for this demonstration, the parallelism gain is still massive.
        
        # Shapes:
        # self.pos: (N, 3)
        # diff: (N, N, 3) - Memory intensive for very large N.
        # For huge N, a kernel approach is better, but broadcasting is clearer for Python code.
        
        diff = self.pos.reshape(1, self.n_particles, 3) - self.pos.reshape(self.n_particles, 1, 3)
        
        dist_sq = cp.sum(diff**2, axis=-1) + self.epsilon**2
        
        inv_dist_cube = dist_sq ** (-1.5)
        
        acc_matrix = (self.G * self.mass.T).reshape(1, self.n_particles, 1) * diff * inv_dist_cube.reshape(self.n_particles, self.n_particles, 1)
        
        return cp.sum(acc_matrix, axis=1)

    def update_particles(self, dt):
        """
        Update particle positions and velocities on the GPU.
        """
        acc = self.compute_forces()
        
        self.vel += acc * dt
        self.pos += self.vel * dt
        
        # Ensure synchronization for accurate timing if calling individually
        # cp.cuda.Stream.null.synchronize()

    def run_simulation(self, steps, dt):
        """
        Run the simulation on GPU.
        
        Returns:
            float: Total execution time (including transfers if applicable, mostly compute).
        """
        # Warmup (optional but good practice)
        self.update_particles(dt)
        cp.cuda.Stream.null.synchronize()

        start_time = time.perf_counter()
        
        for _ in range(steps):
            self.update_particles(dt)
        
        # Vital: Synchronize CPU and GPU to get correct timing
        cp.cuda.Stream.null.synchronize()
        end_time = time.perf_counter()
        
        return end_time - start_time
    
    def get_positions(self):
        """
        Transfer positions back to CPU for visualization.
        """
        return cp.asnumpy(self.pos)
