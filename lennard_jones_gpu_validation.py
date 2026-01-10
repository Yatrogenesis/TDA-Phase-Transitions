#!/usr/bin/env python3
"""
Lennard-Jones 2D Phase Transition - GPU ACCELERATED VERSION
============================================================

Uses CuPy + Numba CUDA for ~10-50x speedup on RTX 3050.
Runs N=900 and N=1600 validations.

Author: Francisco Molina Burgos
Date: 2026-01-10
"""

import numpy as np
import cupy as cp
from numba import cuda, float64
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ripser import ripser
import time
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Verify GPU
print(f"GPU: {cp.cuda.runtime.getDeviceCount()} device(s)")
print(f"Device 0: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")

# === CUDA KERNELS ===

@cuda.jit
def pbc_distance_matrix_kernel(pos, dm, box, N):
    """CUDA kernel for PBC distance matrix calculation."""
    i, j = cuda.grid(2)
    if i < N and j < N and i < j:
        dx = pos[i, 0] - pos[j, 0]
        dy = pos[i, 1] - pos[j, 1]
        # PBC
        dx = dx - box * round(dx / box)
        dy = dy - box * round(dy / box)
        d = math.sqrt(dx * dx + dy * dy)
        dm[i, j] = d
        dm[j, i] = d


@cuda.jit
def compute_forces_kernel(pos, forces, box, N):
    """CUDA kernel for LJ force calculation."""
    i = cuda.grid(1)
    if i >= N:
        return

    fx, fy = 0.0, 0.0
    r_cut2 = 6.25  # 2.5^2
    r_min2 = 0.49  # 0.7^2

    for j in range(N):
        if i == j:
            continue
        dx = pos[i, 0] - pos[j, 0]
        dy = pos[i, 1] - pos[j, 1]
        dx = dx - box * round(dx / box)
        dy = dy - box * round(dy / box)
        r2 = dx * dx + dy * dy

        if r2 < r_cut2:
            if r2 < r_min2:
                r2 = r_min2
            r2_inv = 1.0 / r2
            r6_inv = r2_inv * r2_inv * r2_inv
            f_mag = 48.0 * r2_inv * r6_inv * (r6_inv - 0.5)
            if f_mag > 50.0:
                f_mag = 50.0
            elif f_mag < -50.0:
                f_mag = -50.0
            fx += f_mag * dx
            fy += f_mag * dy

    forces[i, 0] = fx
    forces[i, 1] = fy


@cuda.jit
def hexatic_kernel(pos, dm, psi6_real, psi6_imag, box, N):
    """CUDA kernel for hexatic order calculation."""
    i = cuda.grid(1)
    if i >= N:
        return

    cutoff = 1.8
    min_dist = 0.3
    psi_r, psi_i = 0.0, 0.0
    n_neigh = 0

    for j in range(N):
        if i == j:
            continue
        d = dm[i, j]
        if d > min_dist and d < cutoff:
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dx = dx - box * round(dx / box)
            dy = dy - box * round(dy / box)
            theta = math.atan2(dy, dx)
            psi_r += math.cos(6.0 * theta)
            psi_i += math.sin(6.0 * theta)
            n_neigh += 1

    if n_neigh >= 3:
        psi6_real[i] = psi_r / n_neigh
        psi6_imag[i] = psi_i / n_neigh
    else:
        psi6_real[i] = 0.0
        psi6_imag[i] = 0.0


class GPUSimulator:
    """GPU-accelerated Lennard-Jones simulator."""

    def __init__(self, n_particles, density=0.7):
        self.N = n_particles
        self.density = density
        self.box = np.sqrt(n_particles / density)
        self.dt = 0.002

        # CUDA grid configuration
        self.threads_1d = 256
        self.blocks_1d = (self.N + self.threads_1d - 1) // self.threads_1d
        self.threads_2d = (16, 16)
        self.blocks_2d = ((self.N + 15) // 16, (self.N + 15) // 16)

        # GPU arrays
        self.d_pos = None
        self.d_vel = None
        self.d_forces = None
        self.d_dm = None

    def init_random(self, seed):
        """Initialize random configuration."""
        np.random.seed(seed)
        pos = np.random.rand(self.N, 2).astype(np.float64) * self.box
        vel = np.random.randn(self.N, 2).astype(np.float64) * np.sqrt(2.0)
        vel -= np.mean(vel, axis=0)

        self.d_pos = cuda.to_device(pos)
        self.d_vel = cuda.to_device(vel)
        self.d_forces = cuda.device_array((self.N, 2), dtype=np.float64)
        self.d_dm = cuda.device_array((self.N, self.N), dtype=np.float64)

        # Initial forces
        compute_forces_kernel[self.blocks_1d, self.threads_1d](
            self.d_pos, self.d_forces, self.box, self.N
        )
        cuda.synchronize()

    def step(self, target_T, thermostat_tau=50, step_num=0):
        """Single velocity Verlet step with thermostat."""
        pos = self.d_pos.copy_to_host()
        vel = self.d_vel.copy_to_host()
        forces = self.d_forces.copy_to_host()

        # Velocity Verlet
        vel_half = vel + 0.5 * self.dt * forces
        pos = (pos + self.dt * vel_half) % self.box

        self.d_pos = cuda.to_device(pos)
        compute_forces_kernel[self.blocks_1d, self.threads_1d](
            self.d_pos, self.d_forces, self.box, self.N
        )
        cuda.synchronize()

        forces = self.d_forces.copy_to_host()
        vel = vel_half + 0.5 * self.dt * forces

        # Thermostat
        if step_num % thermostat_tau == 0:
            ke = 0.5 * np.sum(vel**2)
            current_T = ke / self.N
            if current_T > 1e-6:
                scale = np.sqrt(target_T / current_T)
                scale = np.clip(scale, 0.9, 1.1)
                vel *= scale

        self.d_vel = cuda.to_device(vel)

    def compute_distance_matrix(self):
        """Compute PBC distance matrix on GPU."""
        # Reset matrix
        self.d_dm = cuda.device_array((self.N, self.N), dtype=np.float64)
        pbc_distance_matrix_kernel[self.blocks_2d, self.threads_2d](
            self.d_pos, self.d_dm, self.box, self.N
        )
        cuda.synchronize()
        return self.d_dm.copy_to_host()

    def compute_hexatic_order(self):
        """Compute |ψ6| on GPU."""
        # Ensure distance matrix is computed
        dm = self.compute_distance_matrix()

        d_psi_r = cuda.device_array(self.N, dtype=np.float64)
        d_psi_i = cuda.device_array(self.N, dtype=np.float64)

        hexatic_kernel[self.blocks_1d, self.threads_1d](
            self.d_pos, self.d_dm, d_psi_r, d_psi_i, self.box, self.N
        )
        cuda.synchronize()

        psi_r = d_psi_r.copy_to_host()
        psi_i = d_psi_i.copy_to_host()

        psi6_mag = np.sqrt(psi_r**2 + psi_i**2)
        valid = psi6_mag > 0
        return np.mean(psi6_mag[valid]) if np.any(valid) else 0.0

    def compute_persistence_entropy(self):
        """Compute H1 persistence entropy (ripser on CPU, matrix from GPU)."""
        dm = self.compute_distance_matrix()
        dgms = ripser(dm, distance_matrix=True, maxdim=1)['dgms']

        if len(dgms) < 2 or len(dgms[1]) == 0:
            return 0.0

        lifetimes = dgms[1][:, 1] - dgms[1][:, 0]
        valid = (lifetimes > 1e-5) & np.isfinite(lifetimes)
        lifetimes = lifetimes[valid]

        if len(lifetimes) == 0:
            return 0.0

        L_sum = np.sum(lifetimes)
        if L_sum == 0:
            return 0.0

        p = lifetimes / L_sum
        return -np.sum(p * np.log(p))


def detect_crystal_persistent(psi6_series, times, threshold=0.5, persistence=4):
    above = np.array(psi6_series) > threshold
    for i in range(len(above) - persistence):
        if all(above[i:i+persistence]):
            return int(times[i])
    return None


def detect_topo_cusum(s_h1_series, times, baseline_fraction=0.3):
    baseline_end = int(len(s_h1_series) * baseline_fraction)
    if baseline_end < 5:
        return None

    baseline = np.array(s_h1_series[:baseline_end])
    mu = np.mean(baseline)
    sigma = np.std(baseline)

    if sigma < 1e-6:
        return None

    threshold = 3.0 * sigma
    cusum = 0.0

    for i in range(baseline_end, len(s_h1_series)):
        cusum = max(0, cusum + (mu - s_h1_series[i]) - 0.5 * sigma)
        if cusum > threshold:
            return int(times[i])

    return None


def run_gpu_trial(n_particles, seed, steps_equil=2000, steps_prod=5000, sample_interval=25):
    """Run single GPU-accelerated trial."""
    sim = GPUSimulator(n_particles)
    sim.init_random(seed)

    T_HIGH, T_LOW = 2.0, 0.1

    # Equilibration
    for step in range(steps_equil):
        sim.step(T_HIGH, thermostat_tau=50, step_num=step)

    # Production
    times, psi6_series, s_h1_series = [], [], []

    for step in range(steps_prod):
        progress = step / steps_prod
        target_T = T_HIGH + (T_LOW - T_HIGH) * progress
        sim.step(target_T, thermostat_tau=50, step_num=step)

        if step % sample_interval == 0:
            psi6 = sim.compute_hexatic_order()
            s_h1 = sim.compute_persistence_entropy()
            times.append(step)
            psi6_series.append(psi6)
            s_h1_series.append(s_h1)

    # Detection
    times = np.array(times)
    t_phys = detect_crystal_persistent(psi6_series, times)
    t_topo = detect_topo_cusum(s_h1_series, times)

    gap = None
    if t_phys is not None and t_topo is not None:
        gap = t_phys - t_topo

    psi6_final = np.mean(psi6_series[-5:])
    phase = "CRYSTAL" if psi6_final > 0.5 else "LIQUID/GLASS"

    return {
        'seed': seed,
        'phase': phase,
        'psi6_final': float(psi6_final),
        't_phys': t_phys,
        't_topo': t_topo,
        'gap': gap,
        'psi6_series': psi6_series,
        's_h1_series': s_h1_series
    }


def run_validation(n_particles, n_trials, output_dir):
    """Run full validation for given N."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"GPU VALIDATION: N={n_particles}, {n_trials} trials")
    print(f"{'='*70}")

    results = []
    total_start = time.time()

    for i in range(n_trials):
        seed = n_particles * 10 + i
        t0 = time.time()
        result = run_gpu_trial(n_particles, seed)
        elapsed = time.time() - t0
        results.append(result)
        print(f"  Trial {i+1}/{n_trials}: {result['phase']}, |ψ6|={result['psi6_final']:.3f}, gap={result['gap']} ({elapsed:.1f}s)")

    total_time = time.time() - total_start

    # Analysis
    crystal = [r for r in results if r['phase'] == "CRYSTAL"]
    n_crystal = len(crystal)
    gaps = [r['gap'] for r in crystal if r['gap'] is not None]
    n_detected = len(gaps)
    n_precursor = sum(1 for g in gaps if g > 0)
    precursor_rate = n_precursor / n_detected if n_detected > 0 else 0
    mean_gap = np.mean(gaps) if gaps else 0
    std_gap = np.std(gaps) if gaps else 0

    print(f"\n--- RESULTS N={n_particles} ---")
    print(f"Crystallization: {n_crystal}/{n_trials}")
    print(f"Precursor rate: {n_precursor}/{n_detected} ({100*precursor_rate:.1f}%)")
    print(f"Mean gap: {mean_gap:.1f} ± {std_gap:.1f}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Time/trial: {total_time/n_trials:.1f}s")

    # Save
    summary = {
        'N_PARTICLES': n_particles,
        'N_TRIALS': n_trials,
        'total_time_sec': total_time,
        'time_per_trial_sec': total_time / n_trials,
        'n_crystal': n_crystal,
        'n_detected': n_detected,
        'n_precursor': n_precursor,
        'precursor_rate': precursor_rate,
        'mean_gap': mean_gap,
        'std_gap': std_gap,
        'gaps': gaps,
        'trials': [{k: v for k, v in r.items() if k not in ['psi6_series', 's_h1_series']} for r in results]
    }

    with open(f'{output_dir}/validation_N{n_particles}_gpu.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Figure
    if crystal:
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))
        for r in crystal[:5]:
            axs[0].plot(r['psi6_series'], alpha=0.5)
            axs[1].plot(r['s_h1_series'], alpha=0.5)
        axs[0].axhline(0.5, ls='--', color='red')
        axs[0].set_ylabel('|ψ₆|')
        axs[0].set_title(f'N={n_particles} GPU Validation: {100*precursor_rate:.1f}% precursor')
        axs[1].set_ylabel('S_H1')
        axs[1].set_xlabel('Sample')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/N{n_particles}_gpu.png', dpi=150)
        plt.close()

    return summary


if __name__ == "__main__":
    print("="*70)
    print("TDA-CUSUM GPU VALIDATION (RTX 3050)")
    print("="*70)

    # Run N=900 (5 trials)
    res_900 = run_validation(900, 5, 'results_N900')

    # Run N=1600 (3 trials)
    res_1600 = run_validation(1600, 3, 'results_N1600')

    # Summary
    print("\n" + "="*70)
    print("GPU VALIDATION COMPLETE")
    print("="*70)
    print(f"N=900:  {res_900['precursor_rate']*100:.1f}% precursor, {res_900['mean_gap']:.1f} gap, {res_900['time_per_trial_sec']:.1f}s/trial")
    print(f"N=1600: {res_1600['precursor_rate']*100:.1f}% precursor, {res_1600['mean_gap']:.1f} gap, {res_1600['time_per_trial_sec']:.1f}s/trial")
    print("="*70)
