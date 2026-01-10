#!/usr/bin/env python3
"""
Lennard-Jones 2D Phase Transition - N=400 VALIDATION
=====================================================

Finite-size validation with N=400 particles (vs N=144 in main paper).
Expected: Higher precursor rate as fluctuations decrease.

Author: Francisco Molina Burgos
Date: 2026-01-10
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ripser import ripser
from scipy.signal import savgol_filter
import time
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# --- PARAMETERS (N=400 validation) ---
N_PARTICLES = 400         # Validation: 400 vs 144
DENSITY = 0.7
BOX_SIZE = np.sqrt(N_PARTICLES / DENSITY)
DT = 0.002
T_HIGH = 2.0
T_LOW = 0.1
STEPS_EQUIL = 2000
STEPS_PROD = 5000
SAMPLE_INTERVAL = 25
N_TRIALS = 10             # Reduced for speed (N=400 is ~7x slower)
THERMOSTAT_TAU = 50

# Detection parameters
CRYSTAL_PERSISTENCE = 4
SAVGOL_WINDOW = 11
SAVGOL_ORDER = 2

OUTPUT_DIR = '/Users/yatrogenesis/Desktop/PAPER_FINAL/N400_validation'


@dataclass
class TrialResultN400:
    seed: int
    final_phase: str
    psi6_final: float
    t_phys: Optional[int]
    t_topo_cusum: Optional[int]
    gap_cusum: Optional[int]
    t_series: List[int]
    psi6_series: List[float]
    s_h1_series: List[float]


def get_pbc_distance_matrix(pos, box):
    """Compute distance matrix with periodic boundary conditions."""
    N = len(pos)
    dm = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            delta = pos[i] - pos[j]
            delta = delta - box * np.round(delta / box)
            d = np.sqrt(np.dot(delta, delta))
            dm[i, j] = d
            dm[j, i] = d
    return dm


def compute_forces_lj(pos, box):
    """Compute LJ forces and potential energy."""
    N = len(pos)
    forces = np.zeros((N, 2))
    pe = 0.0
    r_cut = 2.5
    r_min = 0.7

    for i in range(N):
        for j in range(i+1, N):
            delta = pos[i] - pos[j]
            delta = delta - box * np.round(delta / box)
            r2 = np.dot(delta, delta)

            if r2 < r_cut**2:
                if r2 < r_min**2:
                    r2 = r_min**2

                r2_inv = 1.0 / r2
                r6_inv = r2_inv ** 3
                f_mag = 48.0 * r2_inv * r6_inv * (r6_inv - 0.5)
                f_mag = np.clip(f_mag, -50.0, 50.0)

                f_vec = f_mag * delta
                forces[i] += f_vec
                forces[j] -= f_vec
                pe += 4.0 * r6_inv * (r6_inv - 1.0)

    return forces, pe


def velocity_verlet_step(pos, vel, forces, dt, box):
    vel_half = vel + 0.5 * dt * forces
    pos_new = pos + dt * vel_half
    pos_new = pos_new % box
    forces_new, pe = compute_forces_lj(pos_new, box)
    vel_new = vel_half + 0.5 * dt * forces_new
    return pos_new, vel_new, forces_new, pe


def apply_thermostat(vel, target_T):
    ke = 0.5 * np.sum(vel**2)
    current_T = ke / len(vel)
    if current_T > 1e-6:
        scale = np.sqrt(target_T / current_T)
        scale = np.clip(scale, 0.9, 1.1)
        vel *= scale
    return vel


def hexatic_order_pbc(pos, box):
    """Compute hexatic order parameter |ψ₆|."""
    N = len(pos)
    psi6_mags = []
    dm = get_pbc_distance_matrix(pos, box)
    cutoff = 1.8

    for i in range(N):
        neigh = np.where((dm[i] > 0.3) & (dm[i] < cutoff))[0]
        if len(neigh) < 3:
            continue

        psi_local = 0j
        for j in neigh:
            delta = pos[j] - pos[i]
            delta = delta - box * np.round(delta / box)
            theta = np.arctan2(delta[1], delta[0])
            psi_local += np.exp(6j * theta)

        psi6_mags.append(np.abs(psi_local / len(neigh)))

    return np.mean(psi6_mags) if psi6_mags else 0.0


def persistent_entropy_h1(pos, box):
    """Compute H1 persistence entropy."""
    dm = get_pbc_distance_matrix(pos, box)
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
    entropy = -np.sum(p * np.log(p))
    return entropy


def detect_crystal_persistent(psi6_series, times, threshold=0.5, persistence=CRYSTAL_PERSISTENCE):
    """Detect crystallization with persistence criterion."""
    above = np.array(psi6_series) > threshold
    for i in range(len(above) - persistence):
        if all(above[i:i+persistence]):
            return int(times[i])
    return None


def detect_topo_cusum(s_h1_series, times, baseline_fraction=0.3):
    """CUSUM change point detection."""
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


def run_trial_n400(seed):
    """Run single N=400 simulation."""
    np.random.seed(seed)

    # Random initialization
    pos = np.random.rand(N_PARTICLES, 2) * BOX_SIZE
    vel = np.random.randn(N_PARTICLES, 2) * np.sqrt(T_HIGH)
    vel -= np.mean(vel, axis=0)
    forces, _ = compute_forces_lj(pos, BOX_SIZE)

    # Equilibration
    for step in range(STEPS_EQUIL):
        pos, vel, forces, _ = velocity_verlet_step(pos, vel, forces, DT, BOX_SIZE)
        if step % THERMOSTAT_TAU == 0:
            vel = apply_thermostat(vel, T_HIGH)

    # Production
    times = []
    psi6_series = []
    s_h1_series = []

    for step in range(STEPS_PROD):
        progress = step / STEPS_PROD
        target_T = T_HIGH + (T_LOW - T_HIGH) * progress

        pos, vel, forces, _ = velocity_verlet_step(pos, vel, forces, DT, BOX_SIZE)

        if step % THERMOSTAT_TAU == 0:
            vel = apply_thermostat(vel, target_T)

        if step % SAMPLE_INTERVAL == 0:
            psi6 = hexatic_order_pbc(pos, BOX_SIZE)
            s_h1 = persistent_entropy_h1(pos, BOX_SIZE)

            times.append(step)
            psi6_series.append(psi6)
            s_h1_series.append(s_h1)

    # Event detection
    times = np.array(times)
    t_phys = detect_crystal_persistent(psi6_series, times)
    t_topo_cusum = detect_topo_cusum(s_h1_series, times)

    gap_cusum = None
    if t_phys is not None and t_topo_cusum is not None:
        gap_cusum = t_phys - t_topo_cusum

    psi6_final = np.mean(psi6_series[-5:])
    phase = "CRYSTAL" if psi6_final > 0.5 else "LIQUID/GLASS"

    return TrialResultN400(
        seed=seed,
        final_phase=phase,
        psi6_final=psi6_final,
        t_phys=t_phys,
        t_topo_cusum=t_topo_cusum,
        gap_cusum=gap_cusum,
        t_series=list(times),
        psi6_series=list(psi6_series),
        s_h1_series=list(s_h1_series)
    )


if __name__ == "__main__":
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("LENNARD-JONES 2D - N=400 VALIDATION")
    print("=" * 70)
    print(f"N={N_PARTICLES}, ρ={DENSITY}, Box={BOX_SIZE:.2f}")
    print(f"T: {T_HIGH} -> {T_LOW}")
    print(f"Trials: {N_TRIALS}")
    print("=" * 70)

    # Run trials
    print(f"\n1. Running {N_TRIALS} trials (N=400 is ~7x slower than N=144)...")
    results = []
    start_time = time.time()

    for i in range(N_TRIALS):
        seed = 4000 + i
        t0 = time.time()
        result = run_trial_n400(seed)
        results.append(result)
        elapsed_trial = time.time() - t0
        print(f"   Trial {i+1}/{N_TRIALS}: {result.final_phase}, |ψ6|={result.psi6_final:.3f} ({elapsed_trial:.1f}s)")

    elapsed = time.time() - start_time
    print(f"   Total: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Analysis
    print("\n" + "=" * 70)
    print("N=400 VALIDATION RESULTS")
    print("=" * 70)

    crystal_trials = [r for r in results if r.final_phase == "CRYSTAL"]
    n_crystal = len(crystal_trials)

    print(f"\nCrystallization: {n_crystal}/{N_TRIALS} ({100*n_crystal/N_TRIALS:.1f}%)")

    if n_crystal > 0:
        gaps_cusum = [r.gap_cusum for r in crystal_trials if r.gap_cusum is not None]
        n_detected = len(gaps_cusum)
        n_precursor = sum(1 for g in gaps_cusum if g > 0)

        print(f"\n--- CUSUM METHOD ---")
        print(f"Events detected: {n_detected}/{n_crystal}")
        if n_detected > 0:
            precursor_rate = n_precursor / n_detected
            print(f"Precursor (gap > 0): {n_precursor}/{n_detected} ({100*precursor_rate:.1f}%)")
            if gaps_cusum:
                print(f"Mean gap: {np.mean(gaps_cusum):.1f} ± {np.std(gaps_cusum):.1f} steps")

            # Comparison with N=144
            print(f"\n--- COMPARISON WITH N=144 ---")
            print(f"N=144: 73.3% precursor rate, 750.8 mean gap")
            print(f"N=400: {100*precursor_rate:.1f}% precursor rate, {np.mean(gaps_cusum):.1f} mean gap")

            if precursor_rate > 0.733:
                print("✓ IMPROVED: Larger system shows higher precursor rate")
            elif precursor_rate >= 0.60:
                print("≈ CONSISTENT: Results within expected variance")
            else:
                print("? UNEXPECTED: Lower rate may indicate need for more trials")

    # Figure
    if crystal_trials:
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        ref_times = np.array(crystal_trials[0].t_series)
        psi6_ens = np.mean([r.psi6_series for r in crystal_trials], axis=0)
        s_h1_ens = np.mean([r.s_h1_series for r in crystal_trials], axis=0)
        psi6_std = np.std([r.psi6_series for r in crystal_trials], axis=0)
        s_h1_std = np.std([r.s_h1_series for r in crystal_trials], axis=0)

        axs[0].plot(ref_times, psi6_ens, 'm-', lw=2, label='|ψ₆|')
        axs[0].fill_between(ref_times, psi6_ens - psi6_std, psi6_ens + psi6_std, alpha=0.3, color='m')
        axs[0].axhline(0.5, ls='--', color='gray')
        axs[0].set_ylabel('|ψ₆|')
        axs[0].set_title(f'N=400 Validation ({n_crystal} crystal trials)', fontweight='bold')
        axs[0].legend()

        axs[1].plot(ref_times, s_h1_ens, 'b-', lw=2, label='S_H1')
        axs[1].fill_between(ref_times, s_h1_ens - s_h1_std, s_h1_ens + s_h1_std, alpha=0.3, color='b')
        axs[1].set_ylabel('S_H1')
        axs[1].set_xlabel('Step')
        axs[1].legend()

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/N400_ensemble.png', dpi=150)
        print(f"\nSaved: {OUTPUT_DIR}/N400_ensemble.png")
        plt.close()

    # Save summary
    summary = {
        'parameters': {
            'N_PARTICLES': N_PARTICLES,
            'DENSITY': DENSITY,
            'N_TRIALS': N_TRIALS
        },
        'results': {
            'n_crystal': n_crystal,
            'cusum_method': {
                'n_detected': n_detected if n_crystal > 0 else 0,
                'n_precursor': n_precursor if n_crystal > 0 else 0,
                'precursor_rate': float(precursor_rate) if n_crystal > 0 and n_detected > 0 else None,
                'mean_gap': float(np.mean(gaps_cusum)) if gaps_cusum else None,
                'std_gap': float(np.std(gaps_cusum)) if gaps_cusum else None,
                'gaps': gaps_cusum if gaps_cusum else []
            }
        },
        'comparison_n144': {
            'n144_precursor_rate': 0.733,
            'n144_mean_gap': 750.8
        }
    }

    with open(f'{OUTPUT_DIR}/N400_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {OUTPUT_DIR}/N400_results.json")

    print("\n" + "=" * 70)
    print("N=400 VALIDATION COMPLETE")
    print("=" * 70)
