import numpy as np
import time

class NBodyCPU:
    """
    N-Body simulation implementation using CPU (NumPy).
    """

    def __init__(self, n_particles, G=6.674e-11, epsilon=1e-5):
        """
        Initialize the CPU N-Body simulator.

        Args:
            n_particles (int): Number of particles.
            G (float): Gravitational constant.
            epsilon (float): Smoothing parameter to avoid singularities.
        """
        self.n_particles = n_particles
        self.G = G
        self.epsilon = epsilon
        
        # Initialize positions and velocities with random data
        # Positions: (x, y, z)
        # Velocities: (vx, vy, vz)
        # masses: (m)
        self.pos = np.random.randn(n_particles, 3).astype(np.float32)
        self.vel = np.random.randn(n_particles, 3).astype(np.float32)
        self.mass = np.ones((n_particles, 1), dtype=np.float32)

    def compute_forces(self):
        """
        Compute gravitational forces between all pairs of particles.
        
        Returns:
            np.ndarray: Acceleration vectors for each particle.
        """
        # Broadcasting to create matrices of position differences
        # shape: (N, 1, 3) - (1, N, 3) -> (N, N, 3)
        # This requires O(N^2) memory which dictates the limit for this simple implementation
        
        # To avoid O(N^2) memory for large N, we can iterate, but for 
        # reasonable benchmarking N (<10000), broadcasting is faster in Python.
        
        # diff[i, j] = pos[j] - pos[i]
        diff = self.pos.reshape(1, self.n_particles, 3) - self.pos.reshape(self.n_particles, 1, 3)
        
        # Distance squared + softening
        dist_sq = np.sum(diff**2, axis=-1) + self.epsilon**2
        
        # Inverse distance cubed: 1 / (r^2 + eps^2)^(3/2)
        inv_dist_cube = dist_sq ** (-1.5)
        
        # F_ij = G * m_i * m_j * r_ij / |r_ij|^3 (Scalar part handles magnitude)
        # a_i = sum(F_ij) / m_i = G * sum(m_j * r_ij / |r_ij|^3)
        
        # Acceleration contribution from j on i: G * m_j * diff_ij * inv_dist_cube_ij
        # shape: (N, N, 1) * (N, N, 3) -> (N, N, 3)
        acc_matrix = (self.G * self.mass.T).reshape(1, self.n_particles, 1) * diff * inv_dist_cube.reshape(self.n_particles, self.n_particles, 1)
        
        return np.sum(acc_matrix, axis=1)

    def update_particles(self, dt):
        """
        Update particle positions and velocities using Euler integration.

        Args:
            dt (float): Time step.
        """
        acc = self.compute_forces()
        
        self.vel += acc * dt
        self.pos += self.vel * dt
        
    def run_simulation(self, steps, dt):
        """
        Run the simulation for a number of steps.
        
        Args:
            steps (int): Number of steps.
            dt (float): Time step size.
        
        Returns:
            float: Total execution time.
        """
        start_time = time.perf_counter()
        
        for _ in range(steps):
            self.update_particles(dt)
            
        end_time = time.perf_counter()
        return end_time - start_time
