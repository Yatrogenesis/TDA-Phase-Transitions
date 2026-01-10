#!/usr/bin/env python3
"""
Lennard-Jones 2D Phase Transition - CuPy GPU VERSION
=====================================================

Pure CuPy implementation (no Numba CUDA required).
Uses GPU for distance matrix and vectorized operations.

Author: Francisco Molina Burgos
Date: 2026-01-10
"""

import numpy as np
import cupy as cp
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
print(f"CuPy GPU: {cp.cuda.runtime.getDeviceCount()} device(s)")
dev = cp.cuda.Device(0)
props = cp.cuda.runtime.getDeviceProperties(0)
print(f"Device: {props['name'].decode() if isinstance(props['name'], bytes) else props['name']}")
print(f"Memory: {dev.mem_info[1] / 1e9:.1f} GB total")

# === PARAMETERS ===
DT = 0.002
T_HIGH = 2.0
T_LOW = 0.1
STEPS_EQUIL = 2000
STEPS_PROD = 5000
SAMPLE_INTERVAL = 25
THERMOSTAT_TAU = 50
DENSITY = 0.7


def pbc_distance_matrix_gpu(pos_gpu, box):
    """Compute PBC distance matrix on GPU using CuPy vectorization."""
    N = pos_gpu.shape[0]

    # Expand for broadcasting: (N, 1, 2) - (1, N, 2) -> (N, N, 2)
    delta = pos_gpu[:, None, :] - pos_gpu[None, :, :]

    # Apply PBC
    delta = delta - box * cp.round(delta / box)

    # Compute distances
    dm = cp.sqrt(cp.sum(delta ** 2, axis=2))

    return dm


def compute_forces_gpu(pos_gpu, box):
    """Compute LJ forces on GPU."""
    N = pos_gpu.shape[0]
    r_cut = 2.5
    r_min = 0.7

    # Pairwise differences with PBC
    delta = pos_gpu[:, None, :] - pos_gpu[None, :, :]
    delta = delta - box * cp.round(delta / box)

    # Distances squared
    r2 = cp.sum(delta ** 2, axis=2)

    # Avoid self-interaction and apply cutoff
    mask = (r2 > 0) & (r2 < r_cut**2)

    # Clamp minimum distance
    r2_safe = cp.maximum(r2, r_min**2)

    # LJ force magnitude: 48 * r^-2 * (r^-12 - 0.5 * r^-6)
    r2_inv = 1.0 / r2_safe
    r6_inv = r2_inv ** 3
    f_mag = 48.0 * r2_inv * r6_inv * (r6_inv - 0.5)
    f_mag = cp.clip(f_mag, -50.0, 50.0)
    f_mag = cp.where(mask, f_mag, 0.0)

    # Force vectors
    forces = cp.sum(f_mag[:, :, None] * delta, axis=1)

    return forces


def hexatic_order_gpu(pos_gpu, dm_gpu, box):
    """Compute |psi_6| on GPU."""
    N = pos_gpu.shape[0]
    cutoff = 1.8
    min_dist = 0.3

    # Neighbor mask
    mask = (dm_gpu > min_dist) & (dm_gpu < cutoff)

    # Pairwise angles with PBC
    delta = pos_gpu[None, :, :] - pos_gpu[:, None, :]
    delta = delta - box * cp.round(delta / box)
    theta = cp.arctan2(delta[:, :, 1], delta[:, :, 0])

    # psi_6 components
    psi6_real = cp.exp(6j * theta)

    # Sum over neighbors
    psi6_sum = cp.sum(cp.where(mask[:, :, None], psi6_real[:, :, None], 0), axis=1)
    n_neigh = cp.sum(mask, axis=1)

    # Average magnitude
    valid = n_neigh >= 3
    psi6_mag = cp.abs(psi6_sum[:, 0]) / cp.maximum(n_neigh, 1)
    psi6_mag = cp.where(valid, psi6_mag, 0)

    return float(cp.mean(psi6_mag[valid]).get()) if cp.any(valid) else 0.0


def persistence_entropy_h1(dm_cpu):
    """Compute H1 persistence entropy (ripser on CPU)."""
    dgms = ripser(dm_cpu, distance_matrix=True, maxdim=1)['dgms']

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


def detect_crystal(psi6_series, times, threshold=0.5, persistence=4):
    above = np.array(psi6_series) > threshold
    for i in range(len(above) - persistence):
        if all(above[i:i+persistence]):
            return int(times[i])
    return None


def detect_cusum(s_h1_series, times, baseline_fraction=0.3):
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


def run_trial_cupy(n_particles, seed):
    """Run single trial with CuPy GPU acceleration."""
    box = np.sqrt(n_particles / DENSITY)

    np.random.seed(seed)

    # Initialize on GPU
    pos = cp.array(np.random.rand(n_particles, 2) * box, dtype=cp.float64)
    vel = cp.array(np.random.randn(n_particles, 2) * np.sqrt(T_HIGH), dtype=cp.float64)
    vel -= cp.mean(vel, axis=0)

    forces = compute_forces_gpu(pos, box)

    # Equilibration
    for step in range(STEPS_EQUIL):
        # Velocity Verlet
        vel += 0.5 * DT * forces
        pos = (pos + DT * vel) % box
        forces = compute_forces_gpu(pos, box)
        vel += 0.5 * DT * forces

        if step % THERMOSTAT_TAU == 0:
            ke = 0.5 * float(cp.sum(vel**2).get())
            current_T = ke / n_particles
            if current_T > 1e-6:
                scale = min(max(np.sqrt(T_HIGH / current_T), 0.9), 1.1)
                vel *= scale

    # Production
    times, psi6_series, s_h1_series = [], [], []

    for step in range(STEPS_PROD):
        progress = step / STEPS_PROD
        target_T = T_HIGH + (T_LOW - T_HIGH) * progress

        # Velocity Verlet
        vel += 0.5 * DT * forces
        pos = (pos + DT * vel) % box
        forces = compute_forces_gpu(pos, box)
        vel += 0.5 * DT * forces

        if step % THERMOSTAT_TAU == 0:
            ke = 0.5 * float(cp.sum(vel**2).get())
            current_T = ke / n_particles
            if current_T > 1e-6:
                scale = min(max(np.sqrt(target_T / current_T), 0.9), 1.1)
                vel *= scale

        if step % SAMPLE_INTERVAL == 0:
            dm_gpu = pbc_distance_matrix_gpu(pos, box)
            psi6 = hexatic_order_gpu(pos, dm_gpu, box)

            # Transfer to CPU for ripser
            dm_cpu = cp.asnumpy(dm_gpu)
            s_h1 = persistence_entropy_h1(dm_cpu)

            times.append(step)
            psi6_series.append(psi6)
            s_h1_series.append(s_h1)

    # Detection
    times = np.array(times)
    t_phys = detect_crystal(psi6_series, times)
    t_topo = detect_cusum(s_h1_series, times)

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
        'gap': gap
    }


def run_validation(n_particles, n_trials, output_dir):
    """Run full validation."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"CuPy GPU VALIDATION: N={n_particles}, {n_trials} trials")
    print(f"{'='*70}")

    results = []
    total_start = time.time()

    for i in range(n_trials):
        seed = n_particles * 10 + i
        t0 = time.time()
        result = run_trial_cupy(n_particles, seed)
        elapsed = time.time() - t0
        results.append(result)
        print(f"  Trial {i+1}/{n_trials}: {result['phase']}, |psi6|={result['psi6_final']:.3f}, gap={result['gap']} ({elapsed:.1f}s)")

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
    print(f"Mean gap: {mean_gap:.1f} +/- {std_gap:.1f}")
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
        'trials': results
    }

    with open(f'{output_dir}/validation_N{n_particles}_cupy.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    print("="*70)
    print("TDA-CUSUM CuPy GPU VALIDATION (RTX 3050)")
    print("="*70)

    # Run N=900
    res_900 = run_validation(900, 5, 'results_N900')

    # Run N=1600
    res_1600 = run_validation(1600, 3, 'results_N1600')

    # Summary
    print("\n" + "="*70)
    print("CuPy GPU VALIDATION COMPLETE")
    print("="*70)
    print(f"N=900:  {res_900['precursor_rate']*100:.1f}% precursor, {res_900['mean_gap']:.1f} gap, {res_900['time_per_trial_sec']:.1f}s/trial")
    print(f"N=1600: {res_1600['precursor_rate']*100:.1f}% precursor, {res_1600['mean_gap']:.1f} gap, {res_1600['time_per_trial_sec']:.1f}s/trial")
    print("="*70)
