#!/usr/bin/env python3
"""
Lennard-Jones 2D Phase Transition - V8.0 CORRECTED METHODOLOGY
================================================================

V7 Issue: Density too high (0.9) caused crystallization even at high T,
invalidating null controls.

V8 Corrections:
- Lower density (0.7) where system is liquid at high T
- Proper phase boundary crossing during quench
- Extended equilibration

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
from dataclasses import dataclass
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

# --- CORRECTED PARAMETERS ---
N_PARTICLES = 100         # Larger system for better statistics
DENSITY = 0.7             # Lower density - liquid at high T
BOX_SIZE = np.sqrt(N_PARTICLES / DENSITY)
DT = 0.002
T_HIGH = 2.0              # Higher start temp (clearly liquid)
T_LOW = 0.1               # End temp (should crystallize)
STEPS_EQUIL = 2000        # Longer equilibration
STEPS_PROD = 5000         # Longer production for slower cooling
SAMPLE_INTERVAL = 50
N_TRIALS = 30
THERMOSTAT_TAU = 50

OUTPUT_DIR = '/Users/yatrogenesis/Desktop/CODIGO_5_V8_CORRECTED'


@dataclass
class TrialResult:
    seed: int
    final_phase: str
    psi6_final: float
    msd_final: float
    s_h1_max_step: int
    s_h1_max_value: float
    crystal_step: Optional[int]
    precursor_gap: Optional[int]
    t_series: List[float]
    psi6_series: List[float]
    s_h1_series: List[float]
    msd_series: List[float]
    temp_series: List[float]


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
    """Velocity Verlet integration."""
    vel_half = vel + 0.5 * dt * forces
    pos_new = pos + dt * vel_half
    pos_new = pos_new % box
    forces_new, pe = compute_forces_lj(pos_new, box)
    vel_new = vel_half + 0.5 * dt * forces_new
    return pos_new, vel_new, forces_new, pe


def apply_thermostat(vel, target_T):
    """Simple velocity rescaling thermostat."""
    ke = 0.5 * np.sum(vel**2)
    current_T = ke / len(vel)
    if current_T > 1e-6:
        scale = np.sqrt(target_T / current_T)
        scale = np.clip(scale, 0.9, 1.1)
        vel *= scale
    return vel


def hexatic_order_pbc(pos, box):
    """Hexatic order parameter with PBC."""
    N = len(pos)
    psi6_mags = []
    dm = get_pbc_distance_matrix(pos, box)
    cutoff = 1.8  # Slightly larger cutoff for lower density

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


def measure_temperature(vel):
    """Instantaneous kinetic temperature."""
    ke = 0.5 * np.sum(vel**2)
    return ke / len(vel)


def run_trial(seed: int, constant_T: bool = False) -> TrialResult:
    """Run a single trial with given seed."""
    np.random.seed(seed)

    # Random initialization
    pos = np.random.rand(N_PARTICLES, 2) * BOX_SIZE
    vel = np.random.randn(N_PARTICLES, 2) * np.sqrt(T_HIGH)
    vel -= np.mean(vel, axis=0)
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
    temp_series = []

    for step in range(STEPS_PROD):
        # Temperature schedule
        if constant_T:
            target_T = T_HIGH
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
            T_inst = measure_temperature(vel)

            times.append(step)
            psi6_series.append(psi6)
            s_h1_series.append(s_h1)
            msd_series.append(msd)
            temp_series.append(T_inst)

    # Analysis
    psi6_arr = np.array(psi6_series)
    s_h1_arr = np.array(s_h1_series)

    # Find crystal transition (|psi6| > 0.5)
    crystal_indices = np.where(psi6_arr > 0.5)[0]
    crystal_step = times[crystal_indices[0]] if len(crystal_indices) > 0 else None

    # Find S_H1 maximum (after first 20% to skip transient)
    start_idx = len(s_h1_arr) // 5
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
        msd_series=msd_series,
        temp_series=temp_series
    )


if __name__ == "__main__":
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("LENNARD-JONES 2D - V8.0 CORRECTED METHODOLOGY")
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
    n_null = N_TRIALS // 3
    print(f"\n2. Running {n_null} null controls (constant T={T_HIGH})...")
    results_null = []

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

    # Null control validation
    phases_null = [r.final_phase for r in results_null]
    n_crystal_null = sum(1 for p in phases_null if p == "CRYSTAL")
    print(f"\nNull Controls (Constant T={T_HIGH}):")
    print(f"  Crystallization: {n_crystal_null}/{n_null}", end="")
    if n_crystal_null == 0:
        print(" ✓ VALID (no false positives)")
    else:
        print(f" ⚠ WARNING: {100*n_crystal_null/n_null:.0f}% crystallized at high T")

    # Precursor analysis
    crystal_trials = [r for r in results_quench if r.final_phase == "CRYSTAL"]
    gaps = []

    if len(crystal_trials) > 0:
        gaps = [r.precursor_gap for r in crystal_trials if r.precursor_gap is not None]
        n_precursor = len(gaps)
        n_no_precursor = len(crystal_trials) - n_precursor

        print(f"\nPrecursor Gap Analysis ({len(crystal_trials)} crystal trials):")
        print(f"  S_H1 peak BEFORE crystal: {n_precursor} trials")
        print(f"  S_H1 peak AFTER crystal:  {n_no_precursor} trials")

        if n_precursor > 0:
            mean_gap = np.mean(gaps)
            std_gap = np.std(gaps)
            print(f"  Mean precursor gap: {mean_gap:.1f} ± {std_gap:.1f} steps")

            # Statistical test
            fraction_precursor = n_precursor / len(crystal_trials)
            if fraction_precursor >= 0.7:
                print(f"\n  ** HYPOTHESIS SUPPORTED: {100*fraction_precursor:.0f}% show precursor **")
            elif fraction_precursor >= 0.5:
                print(f"\n  ** WEAK SUPPORT: {100*fraction_precursor:.0f}% show precursor **")
            else:
                print(f"\n  ** HYPOTHESIS NOT SUPPORTED: only {100*fraction_precursor:.0f}% show precursor **")
        else:
            print(f"\n  ** HYPOTHESIS REFUTED: No trials showed S_H1 precursor **")
    else:
        print("\n⚠ No crystallization observed - cannot test hypothesis")

    # --- Generate Figures ---
    print("\n3. Generating figures...")

    # Figure 1: Ensemble time series
    fig, axs = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    if len(crystal_trials) > 0:
        ref_times = crystal_trials[0].t_series
        psi6_ensemble = np.mean([r.psi6_series for r in crystal_trials], axis=0)
        s_h1_ensemble = np.mean([r.s_h1_series for r in crystal_trials], axis=0)
        msd_ensemble = np.mean([r.msd_series for r in crystal_trials], axis=0)
        temp_ensemble = np.mean([r.temp_series for r in crystal_trials], axis=0)
        psi6_std = np.std([r.psi6_series for r in crystal_trials], axis=0)
        s_h1_std = np.std([r.s_h1_series for r in crystal_trials], axis=0)
        title_suffix = f"(N={len(crystal_trials)} crystal trials)"
    else:
        ref_times = results_quench[0].t_series
        psi6_ensemble = np.mean([r.psi6_series for r in results_quench], axis=0)
        s_h1_ensemble = np.mean([r.s_h1_series for r in results_quench], axis=0)
        msd_ensemble = np.mean([r.msd_series for r in results_quench], axis=0)
        temp_ensemble = np.mean([r.temp_series for r in results_quench], axis=0)
        psi6_std = np.std([r.psi6_series for r in results_quench], axis=0)
        s_h1_std = np.std([r.s_h1_series for r in results_quench], axis=0)
        title_suffix = f"(all {N_TRIALS} trials)"

    # Panel 1: Temperature
    axs[0].plot(ref_times, temp_ensemble, 'r-', linewidth=2)
    axs[0].set_ylabel('Temperature')
    axs[0].set_title(f'V8 Quench Protocol - ρ={DENSITY} {title_suffix}', fontweight='bold', fontsize=12)

    # Panel 2: Hexatic order
    axs[1].plot(ref_times, psi6_ensemble, 'm-', linewidth=2, label='|ψ₆| mean')
    axs[1].fill_between(ref_times, psi6_ensemble - psi6_std, psi6_ensemble + psi6_std,
                        alpha=0.3, color='m')
    axs[1].axhline(0.5, ls='--', color='gray', label='Crystal threshold')
    axs[1].set_ylabel('Hexatic Order |ψ₆|')
    axs[1].legend(loc='upper left')

    # Panel 3: Topological entropy
    axs[2].plot(ref_times, s_h1_ensemble, 'b-', linewidth=2, label='S_H1 mean')
    axs[2].fill_between(ref_times, s_h1_ensemble - s_h1_std, s_h1_ensemble + s_h1_std,
                        alpha=0.3, color='b')
    axs[2].set_ylabel('H1 Persistence Entropy')
    axs[2].legend(loc='upper left')

    # Panel 4: MSD
    axs[3].plot(ref_times, msd_ensemble, 'g-', linewidth=2)
    axs[3].set_ylabel('MSD')
    axs[3].set_xlabel('Production Step')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/FIG1_ensemble_timeseries.png', dpi=150)
    print(f"   Saved: FIG1_ensemble_timeseries.png")
    plt.close()

    # Figure 2: Phase diagram
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {'CRYSTAL': 'blue', 'GLASS': 'orange', 'LIQUID': 'red'}

    for r in results_quench:
        ax.scatter(r.psi6_final, r.msd_final, c=colors[r.final_phase],
                   alpha=0.7, s=80, edgecolors='black', marker='o')

    for r in results_null:
        ax.scatter(r.psi6_final, r.msd_final, c=colors.get(r.final_phase, 'gray'),
                   alpha=0.5, s=80, edgecolors='black', marker='s')

    for phase, color in colors.items():
        ax.scatter([], [], c=color, label=f'{phase} (quench)', s=80, edgecolors='black', marker='o')
    ax.scatter([], [], c='gray', label='Null control', s=80, edgecolors='black', marker='s')

    ax.axvline(0.5, ls='--', color='gray', alpha=0.5)
    ax.set_xlabel('Final |ψ₆|', fontsize=12)
    ax.set_ylabel('Final MSD', fontsize=12)
    ax.set_title('Phase Diagram - Quench vs Null Control', fontweight='bold', fontsize=12)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/FIG2_phase_diagram.png', dpi=150)
    print(f"   Saved: FIG2_phase_diagram.png")
    plt.close()

    # Figure 3: Precursor gap histogram (if data exists)
    if len(gaps) > 0:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.hist(gaps, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(np.mean(gaps), color='red', linestyle='--', linewidth=2,
                   label=f'Mean = {np.mean(gaps):.0f} steps')
        ax.set_xlabel('Precursor Gap (S_H1 max - Crystal step)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Precursor Gap Distribution (n={len(gaps)} trials)', fontweight='bold', fontsize=12)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/FIG3_precursor_gaps.png', dpi=150)
        print(f"   Saved: FIG3_precursor_gaps.png")
        plt.close()

    # --- Save results ---
    summary = {
        'parameters': {
            'N_PARTICLES': N_PARTICLES,
            'DENSITY': DENSITY,
            'T_HIGH': T_HIGH,
            'T_LOW': T_LOW,
            'N_TRIALS': N_TRIALS
        },
        'quench_results': {
            'n_crystal': n_crystal,
            'n_glass': n_glass,
            'n_liquid': n_liquid,
            'precursor_gaps': gaps,
            'mean_gap': float(np.mean(gaps)) if gaps else None,
            'std_gap': float(np.std(gaps)) if gaps else None,
            'fraction_with_precursor': len(gaps) / len(crystal_trials) if crystal_trials else None
        },
        'null_control_results': {
            'n_null': n_null,
            'n_crystal_null': n_crystal_null,
            'valid': n_crystal_null == 0
        }
    }

    with open(f'{OUTPUT_DIR}/results_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   Saved: results_summary.json")

    # --- Final Report ---
    print("\n" + "=" * 70)
    print("FINAL REPORT - V8.0")
    print("=" * 70)
    print(f"""
PARAMETERS:
  Particles: {N_PARTICLES}
  Density: {DENSITY} (lowered from V7's 0.9)
  Temperature: {T_HIGH} → {T_LOW}
  Trials: {N_TRIALS} quench + {n_null} null

PHASE RESULTS (Quench):
  Crystal: {n_crystal}/{N_TRIALS} ({100*n_crystal/N_TRIALS:.1f}%)
  Glass:   {n_glass}/{N_TRIALS} ({100*n_glass/N_TRIALS:.1f}%)
  Liquid:  {n_liquid}/{N_TRIALS} ({100*n_liquid/N_TRIALS:.1f}%)

NULL CONTROL VALIDATION:
  Crystalized at high T: {n_crystal_null}/{n_null}
  Status: {"✓ VALID" if n_crystal_null == 0 else "⚠ INVALID"}
""")

    if len(crystal_trials) > 0:
        frac = len(gaps) / len(crystal_trials) if crystal_trials else 0
        print(f"""PRECURSOR HYPOTHESIS TEST:
  Trials with S_H1 precursor: {len(gaps)}/{len(crystal_trials)} ({100*frac:.1f}%)
  Mean gap: {np.mean(gaps):.1f if gaps else 'N/A'} steps

  CONCLUSION: {"SUPPORTED" if frac >= 0.7 else "WEAK" if frac >= 0.5 else "NOT SUPPORTED"}
""")
    else:
        print("""PRECURSOR HYPOTHESIS TEST:
  No crystallization - cannot evaluate
""")

    print(f"Output directory: {OUTPUT_DIR}")
