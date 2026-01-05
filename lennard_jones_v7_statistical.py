#!/usr/bin/env python3
"""
Lennard-Jones 2D Phase Transition - V7.0 STATISTICAL VALIDATION
================================================================

This version implements the full Capa A/B/C protocol:
- Capa A: Multiple runs (N_TRIALS) with different seeds
- Capa B: Null controls (constant T, randomized positions)
- Capa C: Formal precursor criterion with statistical reporting

Key improvements:
- Stable velocity Verlet + velocity rescaling (no Langevin instability)
- Parameters tuned for actual crystallization
- PBC distance matrix for TDA
- Comprehensive statistical analysis

Author: Francisco Molina Burgos
Date: 2026-01-05
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ripser import ripser
import time
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# --- A. SIMULATION PARAMETERS ---
N_PARTICLES = 64          # Smaller for faster multiple runs
DENSITY = 0.9             # High density to favor crystallization
BOX_SIZE = np.sqrt(N_PARTICLES / DENSITY)
DT = 0.002                # Small timestep for stability
T_HIGH = 1.5              # Start temperature
T_LOW = 0.01              # End temperature (very cold)
STEPS_EQUIL = 1000        # Equilibration
STEPS_PROD = 3000         # Production (slower cooling)
SAMPLE_INTERVAL = 30      # Sample every N steps
N_TRIALS = 30             # Number of independent runs
THERMOSTAT_TAU = 50       # Rescale every N steps

OUTPUT_DIR = '/Users/yatrogenesis/Desktop/CODIGO_4_V7_STATISTICAL'

# --- B. DATA STRUCTURES ---
@dataclass
class TrialResult:
    seed: int
    final_phase: str
    psi6_final: float
    msd_final: float
    s_h1_max_step: int
    s_h1_max_value: float
    crystal_step: Optional[int]  # Step where |psi6| > 0.5
    precursor_gap: Optional[int]  # s_h1_max_step - crystal_step (if both exist)
    t_series: List[float]
    psi6_series: List[float]
    s_h1_series: List[float]
    msd_series: List[float]


# --- C. PHYSICS FUNCTIONS (STABLE) ---

def get_pbc_distance_matrix(pos, box):
    """Periodic boundary condition distance matrix."""
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
    """Lennard-Jones forces with PBC and numerical safeguards."""
    N = len(pos)
    forces = np.zeros((N, 2))
    pe = 0.0
    r_cut = 2.5
    r_min = 0.7  # Prevent singularity

    for i in range(N):
        for j in range(i+1, N):
            delta = pos[i] - pos[j]
            delta = delta - box * np.round(delta / box)
            r2 = np.dot(delta, delta)

            if r2 < r_cut**2:
                if r2 < r_min**2:
                    r2 = r_min**2  # Soft cap

                r2_inv = 1.0 / r2
                r6_inv = r2_inv ** 3
                f_mag = 48.0 * r2_inv * r6_inv * (r6_inv - 0.5)

                # Cap force
                f_mag = np.clip(f_mag, -50.0, 50.0)

                f_vec = f_mag * delta
                forces[i] += f_vec
                forces[j] -= f_vec
                pe += 4.0 * r6_inv * (r6_inv - 1.0)

    return forces, pe


def velocity_verlet_step(pos, vel, forces, dt, box):
    """Velocity Verlet integration."""
    # Half-step velocity
    vel_half = vel + 0.5 * dt * forces

    # Full position step
    pos_new = pos + dt * vel_half
    pos_new = pos_new % box

    # New forces
    forces_new, pe = compute_forces_lj(pos_new, box)

    # Complete velocity step
    vel_new = vel_half + 0.5 * dt * forces_new

    return pos_new, vel_new, forces_new, pe


def apply_thermostat(vel, target_T):
    """Simple velocity rescaling thermostat."""
    ke = 0.5 * np.sum(vel**2)
    current_T = ke / len(vel)
    if current_T > 1e-6:
        scale = np.sqrt(target_T / current_T)
        scale = np.clip(scale, 0.9, 1.1)  # Gentle rescaling
        vel *= scale
    return vel


def hexatic_order_pbc(pos, box):
    """Hexatic order parameter with PBC."""
    N = len(pos)
    psi6_mags = []
    dm = get_pbc_distance_matrix(pos, box)
    cutoff = 1.5

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
    """H1 persistent entropy with PBC."""
    dm = get_pbc_distance_matrix(pos, box)
    dgms = ripser(dm, distance_matrix=True, maxdim=1)['dgms']

    if len(dgms) < 2 or len(dgms[1]) == 0:
        return 0.0, 0

    lifetimes = dgms[1][:, 1] - dgms[1][:, 0]
    valid = (lifetimes > 1e-5) & np.isfinite(lifetimes)
    lifetimes = lifetimes[valid]

    if len(lifetimes) == 0:
        return 0.0, 0

    L_sum = np.sum(lifetimes)
    if L_sum == 0:
        return 0.0, len(lifetimes)

    p = lifetimes / L_sum
    entropy = -np.sum(p * np.log(p))
    return entropy, len(lifetimes)


# --- D. SINGLE TRIAL SIMULATION ---

def run_trial(seed: int, constant_T: bool = False) -> TrialResult:
    """Run a single trial with given seed."""
    np.random.seed(seed)

    # Random initialization
    pos = np.random.rand(N_PARTICLES, 2) * BOX_SIZE
    vel = np.random.randn(N_PARTICLES, 2) * np.sqrt(T_HIGH)
    vel -= np.mean(vel, axis=0)  # Remove COM velocity
    forces, _ = compute_forces_lj(pos, BOX_SIZE)

    # Equilibration at T_HIGH
    for step in range(STEPS_EQUIL):
        pos, vel, forces, _ = velocity_verlet_step(pos, vel, forces, DT, BOX_SIZE)
        if step % THERMOSTAT_TAU == 0:
            vel = apply_thermostat(vel, T_HIGH)

    # Production run
    pos_0 = pos.copy()
    pos_unwrapped = pos.copy()
    prev_pos = pos.copy()

    times = []
    psi6_series = []
    s_h1_series = []
    msd_series = []

    for step in range(STEPS_PROD):
        # Temperature schedule
        if constant_T:
            target_T = T_HIGH  # Null control
        else:
            progress = step / STEPS_PROD
            target_T = T_HIGH + (T_LOW - T_HIGH) * progress

        # Integration
        pos, vel, forces, pe = velocity_verlet_step(pos, vel, forces, DT, BOX_SIZE)

        # Thermostat
        if step % THERMOSTAT_TAU == 0:
            vel = apply_thermostat(vel, target_T)

        # MSD (unwrapped)
        delta = pos - prev_pos
        delta = delta - BOX_SIZE * np.round(delta / BOX_SIZE)
        pos_unwrapped += delta
        prev_pos = pos.copy()

        # Sample
        if step % SAMPLE_INTERVAL == 0:
            msd = np.mean(np.sum((pos_unwrapped - pos_0)**2, axis=1))
            psi6 = hexatic_order_pbc(pos, BOX_SIZE)
            s_h1, _ = persistent_entropy_h1(pos, BOX_SIZE)

            times.append(step)
            psi6_series.append(psi6)
            s_h1_series.append(s_h1)
            msd_series.append(msd)

    # Analysis
    psi6_arr = np.array(psi6_series)
    s_h1_arr = np.array(s_h1_series)

    # Find crystal transition (|psi6| > 0.5)
    crystal_indices = np.where(psi6_arr > 0.5)[0]
    crystal_step = times[crystal_indices[0]] if len(crystal_indices) > 0 else None

    # Find S_H1 maximum (after first 10% to skip transient)
    start_idx = len(s_h1_arr) // 10
    if start_idx < len(s_h1_arr):
        s_h1_analysis = s_h1_arr[start_idx:]
        times_analysis = np.array(times)[start_idx:]
        max_idx = np.argmax(s_h1_analysis)
        s_h1_max_step = times_analysis[max_idx]
        s_h1_max_value = s_h1_analysis[max_idx]
    else:
        s_h1_max_step = 0
        s_h1_max_value = 0

    # Precursor gap
    if crystal_step is not None and s_h1_max_step < crystal_step:
        precursor_gap = crystal_step - s_h1_max_step
    else:
        precursor_gap = None

    # Phase diagnosis
    psi6_final = np.mean(psi6_arr[-5:])
    msd_final = msd_series[-1]
    msd_growth = msd_series[-1] - msd_series[len(msd_series)//2] if len(msd_series) > 1 else 0

    if psi6_final > 0.5:
        phase = "CRYSTAL"
    elif msd_growth < 1.0:
        phase = "GLASS"
    else:
        phase = "LIQUID"

    return TrialResult(
        seed=seed,
        final_phase=phase,
        psi6_final=psi6_final,
        msd_final=msd_final,
        s_h1_max_step=s_h1_max_step,
        s_h1_max_value=s_h1_max_value,
        crystal_step=crystal_step,
        precursor_gap=precursor_gap,
        t_series=times,
        psi6_series=psi6_series,
        s_h1_series=s_h1_series,
        msd_series=msd_series
    )


# --- E. MAIN EXECUTION ---

if __name__ == "__main__":
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("LENNARD-JONES 2D - V7.0 STATISTICAL VALIDATION")
    print("=" * 70)
    print(f"N={N_PARTICLES}, ρ={DENSITY}, Box={BOX_SIZE:.2f}")
    print(f"T: {T_HIGH} -> {T_LOW}")
    print(f"Trials: {N_TRIALS} + null controls")
    print("=" * 70)

    # --- Run main trials ---
    print(f"\n1. Running {N_TRIALS} quench trials...")
    results_quench = []
    start_time = time.time()

    for i in range(N_TRIALS):
        seed = 1000 + i
        result = run_trial(seed, constant_T=False)
        results_quench.append(result)
        if (i+1) % 5 == 0:
            print(f"   Trial {i+1}/{N_TRIALS}: {result.final_phase}, |ψ6|={result.psi6_final:.2f}")

    quench_time = time.time() - start_time
    print(f"   Completed in {quench_time:.1f}s")

    # --- Run null controls (constant T) ---
    print(f"\n2. Running {N_TRIALS//3} null controls (constant T)...")
    results_null = []
    n_null = N_TRIALS // 3

    for i in range(n_null):
        seed = 2000 + i
        result = run_trial(seed, constant_T=True)
        results_null.append(result)

    print(f"   Completed {n_null} null controls")

    # --- Statistical Analysis ---
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    # Phase distribution
    phases_quench = [r.final_phase for r in results_quench]
    n_crystal = sum(1 for p in phases_quench if p == "CRYSTAL")
    n_glass = sum(1 for p in phases_quench if p == "GLASS")
    n_liquid = sum(1 for p in phases_quench if p == "LIQUID")

    print(f"\nPhase Distribution (Quench Trials):")
    print(f"  CRYSTAL: {n_crystal}/{N_TRIALS} ({100*n_crystal/N_TRIALS:.1f}%)")
    print(f"  GLASS:   {n_glass}/{N_TRIALS} ({100*n_glass/N_TRIALS:.1f}%)")
    print(f"  LIQUID:  {n_liquid}/{N_TRIALS} ({100*n_liquid/N_TRIALS:.1f}%)")

    # Precursor analysis (only for crystallized trials)
    crystal_trials = [r for r in results_quench if r.final_phase == "CRYSTAL"]

    if len(crystal_trials) > 0:
        gaps = [r.precursor_gap for r in crystal_trials if r.precursor_gap is not None]
        if len(gaps) > 0:
            mean_gap = np.mean(gaps)
            std_gap = np.std(gaps)
            n_precursor = len(gaps)
            print(f"\nPrecursor Gap Analysis (Crystal Trials):")
            print(f"  Trials with S_H1 peak BEFORE crystal: {n_precursor}/{len(crystal_trials)}")
            print(f"  Mean gap: {mean_gap:.1f} ± {std_gap:.1f} steps")

            # Statistical significance
            if n_precursor >= len(crystal_trials) * 0.7:
                print(f"  ** SIGNIFICANT: {100*n_precursor/len(crystal_trials):.0f}% show precursor **")
            else:
                print(f"  Not significant: only {100*n_precursor/len(crystal_trials):.0f}% show precursor")
        else:
            print("\nNo trials showed S_H1 peak before crystallization")
    else:
        print("\nNo crystallization observed - cannot test precursor hypothesis")

    # Null control comparison
    phases_null = [r.final_phase for r in results_null]
    n_crystal_null = sum(1 for p in phases_null if p == "CRYSTAL")
    print(f"\nNull Controls (Constant T):")
    print(f"  Crystallization: {n_crystal_null}/{n_null} (should be ~0)")

    # --- Generate Figures ---
    print("\n3. Generating figures...")

    # Figure 1: Ensemble averages
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Compute ensemble average for crystal trials (if any)
    if len(crystal_trials) > 0:
        ref_times = crystal_trials[0].t_series
        psi6_ensemble = np.mean([r.psi6_series for r in crystal_trials], axis=0)
        s_h1_ensemble = np.mean([r.s_h1_series for r in crystal_trials], axis=0)
        msd_ensemble = np.mean([r.msd_series for r in crystal_trials], axis=0)
        psi6_std = np.std([r.psi6_series for r in crystal_trials], axis=0)
        s_h1_std = np.std([r.s_h1_series for r in crystal_trials], axis=0)
    else:
        # Use all trials if no crystals
        ref_times = results_quench[0].t_series
        psi6_ensemble = np.mean([r.psi6_series for r in results_quench], axis=0)
        s_h1_ensemble = np.mean([r.s_h1_series for r in results_quench], axis=0)
        msd_ensemble = np.mean([r.msd_series for r in results_quench], axis=0)
        psi6_std = np.std([r.psi6_series for r in results_quench], axis=0)
        s_h1_std = np.std([r.s_h1_series for r in results_quench], axis=0)

    # Panel 1: Temperature schedule
    T_schedule = [T_HIGH + (T_LOW - T_HIGH) * (t / STEPS_PROD) for t in ref_times]
    axs[0].plot(ref_times, T_schedule, 'r-', linewidth=2)
    axs[0].set_ylabel('Target Temperature')
    axs[0].set_title(f'Quench Protocol (N={N_TRIALS} trials)', fontweight='bold')

    # Panel 2: Order parameter
    axs[1].plot(ref_times, psi6_ensemble, 'm-', linewidth=2, label='|ψ₆| mean')
    axs[1].fill_between(ref_times, psi6_ensemble - psi6_std, psi6_ensemble + psi6_std,
                        alpha=0.3, color='m')
    axs[1].axhline(0.5, ls='--', color='gray', label='Crystal threshold')
    axs[1].set_ylabel('Hexatic Order |ψ₆|')
    axs[1].legend()
    axs[1].set_title('Structural Order (ensemble average ± std)', fontweight='bold')

    # Panel 3: Topological entropy
    axs[2].plot(ref_times, s_h1_ensemble, 'b-', linewidth=2, label='S_H1 mean')
    axs[2].fill_between(ref_times, s_h1_ensemble - s_h1_std, s_h1_ensemble + s_h1_std,
                        alpha=0.3, color='b')
    axs[2].set_ylabel('Persistence Entropy S_H1')
    axs[2].set_xlabel('Production Step')
    axs[2].legend()
    axs[2].set_title('Topological Information (ensemble average ± std)', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/FIG1_ensemble_average.png', dpi=150)
    print(f"   Saved: FIG1_ensemble_average.png")
    plt.close()

    # Figure 2: Precursor gap distribution
    if len(crystal_trials) > 0 and len(gaps) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(gaps, bins=15, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(gaps), color='r', linestyle='--', label=f'Mean = {np.mean(gaps):.1f}')
        ax.set_xlabel('Precursor Gap (S_H1 max step - Crystal step)')
        ax.set_ylabel('Count')
        ax.set_title(f'Precursor Gap Distribution (n={len(gaps)} crystal trials)', fontweight='bold')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/FIG2_precursor_gap_distribution.png', dpi=150)
        print(f"   Saved: FIG2_precursor_gap_distribution.png")
        plt.close()

    # Figure 3: Phase diagram
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'CRYSTAL': 'blue', 'GLASS': 'orange', 'LIQUID': 'red'}
    for r in results_quench:
        ax.scatter(r.psi6_final, r.msd_final, c=colors[r.final_phase],
                   alpha=0.7, s=50, edgecolors='black')
    # Add legend
    for phase, color in colors.items():
        ax.scatter([], [], c=color, label=phase, s=50, edgecolors='black')
    ax.axvline(0.5, ls='--', color='gray', alpha=0.5)
    ax.set_xlabel('Final |ψ₆|')
    ax.set_ylabel('Final MSD')
    ax.set_title('Phase Diagram (all trials)', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/FIG3_phase_diagram.png', dpi=150)
    print(f"   Saved: FIG3_phase_diagram.png")
    plt.close()

    # --- Save results to JSON ---
    summary = {
        'parameters': {
            'N_PARTICLES': N_PARTICLES,
            'DENSITY': DENSITY,
            'T_HIGH': T_HIGH,
            'T_LOW': T_LOW,
            'N_TRIALS': N_TRIALS
        },
        'results': {
            'n_crystal': n_crystal,
            'n_glass': n_glass,
            'n_liquid': n_liquid,
            'precursor_gaps': gaps if len(crystal_trials) > 0 else [],
            'mean_gap': float(np.mean(gaps)) if len(gaps) > 0 else None,
            'std_gap': float(np.std(gaps)) if len(gaps) > 0 else None
        }
    }

    with open(f'{OUTPUT_DIR}/results_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   Saved: results_summary.json")

    # --- Final Summary ---
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"""
Parameters:
  N = {N_PARTICLES}, ρ = {DENSITY}
  T: {T_HIGH} → {T_LOW}
  Trials: {N_TRIALS}

Phase Results:
  Crystal: {n_crystal}/{N_TRIALS} ({100*n_crystal/N_TRIALS:.1f}%)
  Glass:   {n_glass}/{N_TRIALS} ({100*n_glass/N_TRIALS:.1f}%)
  Liquid:  {n_liquid}/{N_TRIALS} ({100*n_liquid/N_TRIALS:.1f}%)
""")

    if len(crystal_trials) > 0 and len(gaps) > 0:
        significance = "SIGNIFICANT" if len(gaps) >= len(crystal_trials) * 0.7 else "NOT SIGNIFICANT"
        print(f"""
Precursor Analysis:
  Trials with precursor: {len(gaps)}/{len(crystal_trials)}
  Mean gap: {np.mean(gaps):.1f} ± {np.std(gaps):.1f} steps
  Statistical significance: {significance}
""")
    else:
        print("""
Precursor Analysis:
  Cannot evaluate - insufficient crystallization
""")

    print(f"Results saved to: {OUTPUT_DIR}")
