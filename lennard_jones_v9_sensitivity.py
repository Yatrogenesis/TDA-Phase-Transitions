#!/usr/bin/env python3
"""
Lennard-Jones 2D Phase Transition - V9.0 SENSITIVITY ANALYSIS
==============================================================

V8 Issue: 53% precursor rate is noise due to fragile argmax detection.

V9 Improvements:
1. Robust crystallization detection with PERSISTENCE criterion
2. Change Point Detection for topological event (not argmax)
3. Derivative-based (Savitzky-Golay) and CUSUM methods
4. Larger system (N=144) to reduce thermal fluctuations

The goal: Distinguish whether topology STARTS to change before matter,
not whether it has a random peak before.

Author: Francisco Molina Burgos
Date: 2026-01-05
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ripser import ripser
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d
import time
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# --- PARAMETERS (Larger system) ---
N_PARTICLES = 144         # Larger to reduce fluctuations
DENSITY = 0.7
BOX_SIZE = np.sqrt(N_PARTICLES / DENSITY)
DT = 0.002
T_HIGH = 2.0
T_LOW = 0.1
STEPS_EQUIL = 2000
STEPS_PROD = 5000
SAMPLE_INTERVAL = 25      # Finer sampling for derivative analysis
N_TRIALS = 30
THERMOSTAT_TAU = 50

# Detection parameters
CRYSTAL_PERSISTENCE = 4   # Must stay > 0.5 for this many samples (~100 steps)
SAVGOL_WINDOW = 11        # Savitzky-Golay window (must be odd)
SAVGOL_ORDER = 2          # Polynomial order

OUTPUT_DIR = '/Users/yatrogenesis/Desktop/CODIGO_6_V9_SENSITIVITY'


@dataclass
class TrialResultV9:
    seed: int
    final_phase: str
    psi6_final: float

    # New robust event times
    t_phys: Optional[int]           # Crystal onset (persistent)
    t_topo_derivative: Optional[int] # Max negative derivative of S_H1
    t_topo_cusum: Optional[int]      # CUSUM change point

    # Gaps (positive = precursor)
    gap_derivative: Optional[int]
    gap_cusum: Optional[int]

    # Raw series for ensemble analysis
    t_series: List[int]
    psi6_series: List[float]
    s_h1_series: List[float]
    s_h1_derivative: List[float]


# --- PHYSICS FUNCTIONS (Same as V8) ---

def get_pbc_distance_matrix(pos, box):
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


# --- NEW: ROBUST EVENT DETECTION ---

def detect_crystal_persistent(psi6_series: np.ndarray, times: np.ndarray,
                               threshold: float = 0.5,
                               persistence: int = CRYSTAL_PERSISTENCE) -> Optional[int]:
    """
    Detect crystallization with PERSISTENCE criterion.
    Returns the time when |ψ₆| first crosses threshold AND stays above for 'persistence' samples.
    """
    above = psi6_series > threshold

    for i in range(len(above) - persistence):
        if all(above[i:i+persistence]):
            return int(times[i])

    return None


def detect_topo_derivative(s_h1_series: np.ndarray, times: np.ndarray,
                           window: int = SAVGOL_WINDOW,
                           order: int = SAVGOL_ORDER) -> Tuple[Optional[int], np.ndarray]:
    """
    Detect topological event as point of maximum NEGATIVE derivative.
    This is when S_H1 starts falling fastest (collapse of topological degrees of freedom).

    Returns: (event_time, derivative_series)
    """
    if len(s_h1_series) < window:
        return None, np.zeros_like(s_h1_series)

    # Smooth derivative using Savitzky-Golay
    derivative = savgol_filter(s_h1_series, window, order, deriv=1)

    # Skip first 20% (transient) and last 10% (boundary)
    start_idx = len(derivative) // 5
    end_idx = int(len(derivative) * 0.9)

    if start_idx >= end_idx:
        return None, derivative

    analysis_region = derivative[start_idx:end_idx]
    times_region = times[start_idx:end_idx]

    # Find minimum (most negative) derivative
    min_idx = np.argmin(analysis_region)

    # Only count if derivative is actually negative (S_H1 falling)
    if analysis_region[min_idx] < 0:
        return int(times_region[min_idx]), derivative

    return None, derivative


def detect_topo_cusum(s_h1_series: np.ndarray, times: np.ndarray,
                      baseline_fraction: float = 0.3) -> Optional[int]:
    """
    CUSUM (Cumulative Sum) change point detection.
    Detects when S_H1 deviates significantly from its baseline mean.

    Returns: time of first significant deviation
    """
    # Establish baseline from first portion (liquid phase)
    baseline_end = int(len(s_h1_series) * baseline_fraction)
    if baseline_end < 5:
        return None

    baseline = s_h1_series[:baseline_end]
    mu = np.mean(baseline)
    sigma = np.std(baseline)

    if sigma < 1e-6:
        return None

    # CUSUM for negative shift (S_H1 decreasing)
    threshold = 3.0 * sigma  # 3-sigma threshold
    cusum = 0.0

    for i in range(baseline_end, len(s_h1_series)):
        # Accumulate negative deviations
        cusum = max(0, cusum + (mu - s_h1_series[i]) - 0.5 * sigma)

        if cusum > threshold:
            return int(times[i])

    return None


# --- SIMULATION ---

def run_trial_v9(seed: int) -> TrialResultV9:
    """Run simulation and apply robust event detection."""
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

    # Convert to arrays
    times = np.array(times)
    psi6_arr = np.array(psi6_series)
    s_h1_arr = np.array(s_h1_series)

    # --- ROBUST EVENT DETECTION ---

    # 1. Physical event (persistent crystallization)
    t_phys = detect_crystal_persistent(psi6_arr, times)

    # 2. Topological event (derivative method)
    t_topo_deriv, s_h1_derivative = detect_topo_derivative(s_h1_arr, times)

    # 3. Topological event (CUSUM method)
    t_topo_cusum = detect_topo_cusum(s_h1_arr, times)

    # --- COMPUTE GAPS ---
    gap_derivative = None
    gap_cusum = None

    if t_phys is not None:
        if t_topo_deriv is not None:
            gap_derivative = t_phys - t_topo_deriv  # Positive = precursor
        if t_topo_cusum is not None:
            gap_cusum = t_phys - t_topo_cusum  # Positive = precursor

    # Phase diagnosis
    psi6_final = np.mean(psi6_arr[-5:])
    if psi6_final > 0.5:
        phase = "CRYSTAL"
    else:
        phase = "LIQUID/GLASS"

    return TrialResultV9(
        seed=seed,
        final_phase=phase,
        psi6_final=psi6_final,
        t_phys=t_phys,
        t_topo_derivative=t_topo_deriv,
        t_topo_cusum=t_topo_cusum,
        gap_derivative=gap_derivative,
        gap_cusum=gap_cusum,
        t_series=list(times),
        psi6_series=list(psi6_arr),
        s_h1_series=list(s_h1_arr),
        s_h1_derivative=list(s_h1_derivative)
    )


# --- MAIN ---

if __name__ == "__main__":
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("LENNARD-JONES 2D - V9.0 SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"N={N_PARTICLES} (larger), ρ={DENSITY}, Box={BOX_SIZE:.2f}")
    print(f"T: {T_HIGH} -> {T_LOW}")
    print(f"Trials: {N_TRIALS}")
    print(f"Detection: Derivative (Savitzky-Golay) + CUSUM")
    print(f"Crystal persistence: {CRYSTAL_PERSISTENCE} samples")
    print("=" * 70)

    # Run trials
    print(f"\n1. Running {N_TRIALS} trials with robust detection...")
    results = []
    start_time = time.time()

    for i in range(N_TRIALS):
        seed = 3000 + i
        result = run_trial_v9(seed)
        results.append(result)
        if (i+1) % 5 == 0:
            print(f"   Trial {i+1}/{N_TRIALS}: {result.final_phase}, |ψ6|={result.psi6_final:.2f}")

    elapsed = time.time() - start_time
    print(f"   Completed in {elapsed:.1f}s")

    # --- ANALYSIS ---
    print("\n" + "=" * 70)
    print("ROBUST EVENT DETECTION ANALYSIS")
    print("=" * 70)

    # Filter crystallized trials
    crystal_trials = [r for r in results if r.final_phase == "CRYSTAL"]
    n_crystal = len(crystal_trials)

    print(f"\nCrystallization: {n_crystal}/{N_TRIALS} ({100*n_crystal/N_TRIALS:.1f}%)")

    if n_crystal == 0:
        print("\n⚠ No crystallization - cannot analyze precursors")
    else:
        # Derivative method analysis
        gaps_deriv = [r.gap_derivative for r in crystal_trials if r.gap_derivative is not None]
        n_detected_deriv = len(gaps_deriv)
        n_precursor_deriv = sum(1 for g in gaps_deriv if g > 0)

        print(f"\n--- DERIVATIVE METHOD (Savitzky-Golay) ---")
        print(f"Events detected: {n_detected_deriv}/{n_crystal}")
        if n_detected_deriv > 0:
            precursor_rate_deriv = n_precursor_deriv / n_detected_deriv
            print(f"Precursor (gap > 0): {n_precursor_deriv}/{n_detected_deriv} ({100*precursor_rate_deriv:.1f}%)")
            if gaps_deriv:
                print(f"Mean gap: {np.mean(gaps_deriv):.1f} ± {np.std(gaps_deriv):.1f} steps")
                print(f"Median gap: {np.median(gaps_deriv):.1f} steps")

        # CUSUM method analysis
        gaps_cusum = [r.gap_cusum for r in crystal_trials if r.gap_cusum is not None]
        n_detected_cusum = len(gaps_cusum)
        n_precursor_cusum = sum(1 for g in gaps_cusum if g > 0)

        print(f"\n--- CUSUM METHOD ---")
        print(f"Events detected: {n_detected_cusum}/{n_crystal}")
        if n_detected_cusum > 0:
            precursor_rate_cusum = n_precursor_cusum / n_detected_cusum
            print(f"Precursor (gap > 0): {n_precursor_cusum}/{n_detected_cusum} ({100*precursor_rate_cusum:.1f}%)")
            if gaps_cusum:
                print(f"Mean gap: {np.mean(gaps_cusum):.1f} ± {np.std(gaps_cusum):.1f} steps")
                print(f"Median gap: {np.median(gaps_cusum):.1f} steps")

        # Determine outcome
        print("\n" + "=" * 70)
        print("HYPOTHESIS EVALUATION")
        print("=" * 70)

        best_rate = max(
            precursor_rate_deriv if n_detected_deriv > 0 else 0,
            precursor_rate_cusum if n_detected_cusum > 0 else 0
        )

        if best_rate >= 0.8:
            conclusion = "SCENARIO A: STRONG SUPPORT"
            interpretation = """
The topological instability (S_H1 derivative minimum) consistently
precedes crystallization. This supports the refined hypothesis:

  "The transition is triggered by topological instability -
   the collapse of informational degrees of freedom (S_H1 ↓)
   ENABLES the metric to order."
"""
        elif best_rate >= 0.6:
            conclusion = "SCENARIO A: MODERATE SUPPORT"
            interpretation = """
The topological change precedes crystallization in most trials.
The signal is real but not deterministic. Refinement needed.
"""
        elif best_rate <= 0.55 and best_rate >= 0.45:
            conclusion = "SCENARIO B: CO-EMERGENCE (SYNCHRONY)"
            interpretation = """
The 50/50 result persists even with robust detection.
This suggests topology and thermodynamics are CONJUGATE VARIABLES:

  "There is no temporal hierarchy. Topology and matter
   co-emerge during the transition. The 'chaos' regime is
   where these descriptions decouple; 'order' is their
   resynchronization. (Structural autopoiesis)"
"""
        else:
            conclusion = "INCONCLUSIVE"
            interpretation = "Results do not clearly support either scenario."

        print(f"\nBest precursor rate: {100*best_rate:.1f}%")
        print(f"\n>>> {conclusion} <<<")
        print(interpretation)

    # --- FIGURES ---
    print("\n2. Generating figures...")

    # Figure 1: Ensemble with derivative
    if crystal_trials:
        fig, axs = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

        ref_times = np.array(crystal_trials[0].t_series)

        # Ensemble averages
        psi6_ens = np.mean([r.psi6_series for r in crystal_trials], axis=0)
        s_h1_ens = np.mean([r.s_h1_series for r in crystal_trials], axis=0)
        deriv_ens = np.mean([r.s_h1_derivative for r in crystal_trials], axis=0)

        psi6_std = np.std([r.psi6_series for r in crystal_trials], axis=0)
        s_h1_std = np.std([r.s_h1_series for r in crystal_trials], axis=0)
        deriv_std = np.std([r.s_h1_derivative for r in crystal_trials], axis=0)

        # Temperature
        T_sched = [T_HIGH + (T_LOW - T_HIGH) * (t / STEPS_PROD) for t in ref_times]
        axs[0].plot(ref_times, T_sched, 'r-', lw=2)
        axs[0].set_ylabel('Temperature')
        axs[0].set_title(f'V9 Robust Detection (N={N_PARTICLES}, {n_crystal} crystal trials)',
                         fontweight='bold', fontsize=12)

        # Hexatic order
        axs[1].plot(ref_times, psi6_ens, 'm-', lw=2)
        axs[1].fill_between(ref_times, psi6_ens - psi6_std, psi6_ens + psi6_std, alpha=0.3, color='m')
        axs[1].axhline(0.5, ls='--', color='gray', label='Crystal threshold')
        axs[1].set_ylabel('|ψ₆|')
        axs[1].legend(loc='upper left')

        # S_H1
        axs[2].plot(ref_times, s_h1_ens, 'b-', lw=2)
        axs[2].fill_between(ref_times, s_h1_ens - s_h1_std, s_h1_ens + s_h1_std, alpha=0.3, color='b')
        axs[2].set_ylabel('S_H1')

        # Derivative
        axs[3].plot(ref_times, deriv_ens, 'g-', lw=2)
        axs[3].fill_between(ref_times, deriv_ens - deriv_std, deriv_ens + deriv_std, alpha=0.3, color='g')
        axs[3].axhline(0, ls='--', color='gray')
        axs[3].set_ylabel('dS_H1/dt')
        axs[3].set_xlabel('Step')

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/FIG1_ensemble_with_derivative.png', dpi=150)
        print(f"   Saved: FIG1_ensemble_with_derivative.png")
        plt.close()

    # Figure 2: Gap distributions comparison
    if gaps_deriv or gaps_cusum:
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        if gaps_deriv:
            axs[0].hist(gaps_deriv, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
            axs[0].axvline(0, color='red', ls='--', lw=2, label='Synchrony')
            axs[0].axvline(np.mean(gaps_deriv), color='orange', ls='-', lw=2,
                          label=f'Mean={np.mean(gaps_deriv):.0f}')
            axs[0].set_xlabel('Gap (t_phys - t_topo)')
            axs[0].set_ylabel('Count')
            axs[0].set_title(f'Derivative Method\n{n_precursor_deriv}/{n_detected_deriv} precursors',
                            fontweight='bold')
            axs[0].legend()

        if gaps_cusum:
            axs[1].hist(gaps_cusum, bins=15, edgecolor='black', alpha=0.7, color='coral')
            axs[1].axvline(0, color='red', ls='--', lw=2, label='Synchrony')
            axs[1].axvline(np.mean(gaps_cusum), color='orange', ls='-', lw=2,
                          label=f'Mean={np.mean(gaps_cusum):.0f}')
            axs[1].set_xlabel('Gap (t_phys - t_topo)')
            axs[1].set_ylabel('Count')
            axs[1].set_title(f'CUSUM Method\n{n_precursor_cusum}/{n_detected_cusum} precursors',
                            fontweight='bold')
            axs[1].legend()

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/FIG2_gap_comparison.png', dpi=150)
        print(f"   Saved: FIG2_gap_comparison.png")
        plt.close()

    # Figure 3: Individual trial examples
    if len(crystal_trials) >= 3:
        fig, axs = plt.subplots(3, 3, figsize=(15, 12))

        for idx, trial in enumerate(crystal_trials[:3]):
            times = np.array(trial.t_series)

            # psi6
            axs[idx, 0].plot(times, trial.psi6_series, 'm-', lw=1.5)
            axs[idx, 0].axhline(0.5, ls='--', color='gray')
            if trial.t_phys:
                axs[idx, 0].axvline(trial.t_phys, color='red', ls='-', lw=2, label=f't_phys={trial.t_phys}')
            axs[idx, 0].set_ylabel('|ψ₆|')
            axs[idx, 0].legend(fontsize=8)
            if idx == 0:
                axs[idx, 0].set_title('Hexatic Order', fontweight='bold')

            # S_H1
            axs[idx, 1].plot(times, trial.s_h1_series, 'b-', lw=1.5)
            if trial.t_topo_derivative:
                axs[idx, 1].axvline(trial.t_topo_derivative, color='green', ls='-', lw=2,
                                   label=f't_deriv={trial.t_topo_derivative}')
            if trial.t_topo_cusum:
                axs[idx, 1].axvline(trial.t_topo_cusum, color='orange', ls='--', lw=2,
                                   label=f't_cusum={trial.t_topo_cusum}')
            axs[idx, 1].set_ylabel('S_H1')
            axs[idx, 1].legend(fontsize=8)
            if idx == 0:
                axs[idx, 1].set_title('Topological Entropy', fontweight='bold')

            # Derivative
            axs[idx, 2].plot(times, trial.s_h1_derivative, 'g-', lw=1.5)
            axs[idx, 2].axhline(0, ls='--', color='gray')
            if trial.t_topo_derivative:
                axs[idx, 2].axvline(trial.t_topo_derivative, color='green', ls='-', lw=2)
            axs[idx, 2].set_ylabel('dS_H1/dt')
            if idx == 0:
                axs[idx, 2].set_title('Derivative', fontweight='bold')

            axs[idx, 0].set_title(f'Trial {trial.seed}', loc='left', fontsize=10)

        for ax in axs[-1, :]:
            ax.set_xlabel('Step')

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/FIG3_individual_trials.png', dpi=150)
        print(f"   Saved: FIG3_individual_trials.png")
        plt.close()

    # --- SAVE RESULTS ---
    summary = {
        'parameters': {
            'N_PARTICLES': N_PARTICLES,
            'DENSITY': DENSITY,
            'T_HIGH': T_HIGH,
            'T_LOW': T_LOW,
            'N_TRIALS': N_TRIALS,
            'CRYSTAL_PERSISTENCE': CRYSTAL_PERSISTENCE,
            'SAVGOL_WINDOW': SAVGOL_WINDOW
        },
        'results': {
            'n_crystal': n_crystal,
            'derivative_method': {
                'n_detected': n_detected_deriv if n_crystal > 0 else 0,
                'n_precursor': n_precursor_deriv if n_crystal > 0 else 0,
                'precursor_rate': float(precursor_rate_deriv) if n_crystal > 0 and n_detected_deriv > 0 else None,
                'mean_gap': float(np.mean(gaps_deriv)) if gaps_deriv else None,
                'std_gap': float(np.std(gaps_deriv)) if gaps_deriv else None
            },
            'cusum_method': {
                'n_detected': n_detected_cusum if n_crystal > 0 else 0,
                'n_precursor': n_precursor_cusum if n_crystal > 0 else 0,
                'precursor_rate': float(precursor_rate_cusum) if n_crystal > 0 and n_detected_cusum > 0 else None,
                'mean_gap': float(np.mean(gaps_cusum)) if gaps_cusum else None,
                'std_gap': float(np.std(gaps_cusum)) if gaps_cusum else None
            }
        },
        'conclusion': conclusion if n_crystal > 0 else 'NO_CRYSTALLIZATION'
    }

    with open(f'{OUTPUT_DIR}/results_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   Saved: results_summary.json")

    # --- FINAL REPORT ---
    print("\n" + "=" * 70)
    print("FINAL REPORT - V9.0")
    print("=" * 70)
    print(f"""
PARAMETERS:
  N = {N_PARTICLES} (increased from V8's 100)
  ρ = {DENSITY}
  T: {T_HIGH} → {T_LOW}
  Crystal persistence: {CRYSTAL_PERSISTENCE} samples

DETECTION RESULTS:
  Crystallization: {n_crystal}/{N_TRIALS}
""")

    if n_crystal > 0:
        print(f"""  Derivative method:
    - Events detected: {n_detected_deriv}/{n_crystal}
    - Precursor rate: {100*precursor_rate_deriv:.1f}%
    - Mean gap: {np.mean(gaps_deriv):.1f if gaps_deriv else 'N/A'} steps

  CUSUM method:
    - Events detected: {n_detected_cusum}/{n_crystal}
    - Precursor rate: {100*precursor_rate_cusum:.1f}%
    - Mean gap: {np.mean(gaps_cusum):.1f if gaps_cusum else 'N/A'} steps

CONCLUSION: {conclusion}
""")

    print(f"Output: {OUTPUT_DIR}")
