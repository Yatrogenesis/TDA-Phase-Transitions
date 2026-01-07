#!/usr/bin/env python3
"""
Generate Publication-Quality Figures for:
"Stochastic Informational Primacy in 2D Phase Transitions"

Output formats: PDF, EPS, PNG (600dpi), JPG (lossless quality)
Figure names match LaTeX references for Overleaf integration.

Author: Francisco Molina Burgos
Date: 2026-01-06
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from ripser import ripser
from scipy.signal import savgol_filter
import os
import json

# --- PUBLICATION SETTINGS ---
DPI = 600
FORMATS = ['pdf', 'eps', 'png']  # JPG added separately with max quality

# Journal-quality matplotlib settings
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
rcParams['font.size'] = 9
rcParams['axes.labelsize'] = 10
rcParams['axes.titlesize'] = 10
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
rcParams['legend.fontsize'] = 8
rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = DPI
rcParams['axes.linewidth'] = 0.8
rcParams['lines.linewidth'] = 1.2
rcParams['patch.linewidth'] = 0.8
rcParams['text.usetex'] = False  # Set True if LaTeX is available
rcParams['mathtext.fontset'] = 'stix'

OUTPUT_DIR = '/Users/yatrogenesis/Desktop/PAPER_FINAL/figures'

# --- SIMULATION PARAMETERS (must match V9) ---
N_PARTICLES = 144
DENSITY = 0.7
BOX_SIZE = np.sqrt(N_PARTICLES / DENSITY)
DT = 0.002
T_HIGH = 2.0
T_LOW = 0.1
STEPS_EQUIL = 2000
STEPS_PROD = 5000
SAMPLE_INTERVAL = 25
N_TRIALS = 30
THERMOSTAT_TAU = 50
CRYSTAL_PERSISTENCE = 4
SAVGOL_WINDOW = 11
SAVGOL_ORDER = 2


# --- PHYSICS FUNCTIONS ---

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


def detect_crystal_persistent(psi6_series, times, threshold=0.5, persistence=CRYSTAL_PERSISTENCE):
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


def run_trial(seed, constant_T=False):
    """Run a single trial and return data for figures."""
    np.random.seed(seed)

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
    pos_0 = pos.copy()
    pos_unwrapped = pos.copy()
    prev_pos = pos.copy()

    times = []
    psi6_series = []
    s_h1_series = []
    msd_series = []
    temp_series = []

    for step in range(STEPS_PROD):
        if constant_T:
            target_T = T_HIGH
        else:
            progress = step / STEPS_PROD
            target_T = T_HIGH + (T_LOW - T_HIGH) * progress

        pos, vel, forces, _ = velocity_verlet_step(pos, vel, forces, DT, BOX_SIZE)

        if step % THERMOSTAT_TAU == 0:
            vel = apply_thermostat(vel, target_T)

        delta = pos - prev_pos
        delta = delta - BOX_SIZE * np.round(delta / BOX_SIZE)
        pos_unwrapped += delta
        prev_pos = pos.copy()

        if step % SAMPLE_INTERVAL == 0:
            msd = np.mean(np.sum((pos_unwrapped - pos_0)**2, axis=1))
            psi6 = hexatic_order_pbc(pos, BOX_SIZE)
            s_h1 = persistent_entropy_h1(pos, BOX_SIZE)
            T_inst = 0.5 * np.sum(vel**2) / len(vel)

            times.append(step)
            psi6_series.append(psi6)
            s_h1_series.append(s_h1)
            msd_series.append(msd)
            temp_series.append(T_inst)

    # Compute derivative
    s_h1_arr = np.array(s_h1_series)
    if len(s_h1_arr) >= SAVGOL_WINDOW:
        derivative = savgol_filter(s_h1_arr, SAVGOL_WINDOW, SAVGOL_ORDER, deriv=1)
    else:
        derivative = np.zeros_like(s_h1_arr)

    # Detect events
    t_phys = detect_crystal_persistent(psi6_series, times)
    t_topo = detect_topo_cusum(s_h1_series, times)

    # Phase
    psi6_final = np.mean(psi6_series[-5:])
    msd_final = msd_series[-1]

    return {
        'times': times,
        'psi6': psi6_series,
        's_h1': s_h1_series,
        'derivative': list(derivative),
        'msd': msd_series,
        'temp': temp_series,
        't_phys': t_phys,
        't_topo': t_topo,
        'psi6_final': psi6_final,
        'msd_final': msd_final,
        'phase': 'CRYSTAL' if psi6_final > 0.5 else 'LIQUID'
    }


def save_figure(fig, basename):
    """Save figure in all required formats."""
    for fmt in FORMATS:
        filepath = os.path.join(OUTPUT_DIR, f'{basename}.{fmt}')
        fig.savefig(filepath, format=fmt, dpi=DPI, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"   Saved: {basename}.{fmt}")

    # JPG with maximum quality
    filepath_jpg = os.path.join(OUTPUT_DIR, f'{basename}.jpg')
    fig.savefig(filepath_jpg, format='jpg', dpi=DPI, bbox_inches='tight',
               facecolor='white', edgecolor='none', quality=100)
    print(f"   Saved: {basename}.jpg")


# --- MAIN ---

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 70)
    print(f"Output: {OUTPUT_DIR}")
    print(f"DPI: {DPI}")
    print(f"Formats: {FORMATS + ['jpg']}")
    print("=" * 70)

    # Run simulations
    print("\n1. Running simulations...")

    quench_results = []
    for i in range(N_TRIALS):
        seed = 3000 + i
        result = run_trial(seed, constant_T=False)
        quench_results.append(result)
        if (i+1) % 10 == 0:
            print(f"   Quench trial {i+1}/{N_TRIALS}")

    null_results = []
    for i in range(10):
        seed = 4000 + i
        result = run_trial(seed, constant_T=True)
        null_results.append(result)
    print(f"   Null controls: 10/10")

    # Filter crystal trials
    crystal_trials = [r for r in quench_results if r['phase'] == 'CRYSTAL']
    print(f"\n   Crystal trials: {len(crystal_trials)}/{N_TRIALS}")

    # Compute gaps
    gaps = []
    for r in crystal_trials:
        if r['t_phys'] is not None and r['t_topo'] is not None:
            gap = r['t_phys'] - r['t_topo']
            gaps.append(gap)

    precursor_count = sum(1 for g in gaps if g > 0)
    print(f"   Precursors: {precursor_count}/{len(gaps)} ({100*precursor_count/len(gaps):.1f}%)")

    # --- FIGURE 1: Ensemble Dynamics ---
    print("\n2. Generating Figure 1 (ensemble dynamics)...")

    fig, axs = plt.subplots(4, 1, figsize=(3.4, 6), sharex=True)
    plt.subplots_adjust(hspace=0.1)

    ref_times = np.array(crystal_trials[0]['times'])

    # Ensemble averages
    psi6_ens = np.mean([r['psi6'] for r in crystal_trials], axis=0)
    s_h1_ens = np.mean([r['s_h1'] for r in crystal_trials], axis=0)
    deriv_ens = np.mean([r['derivative'] for r in crystal_trials], axis=0)
    temp_ens = np.mean([r['temp'] for r in crystal_trials], axis=0)

    psi6_std = np.std([r['psi6'] for r in crystal_trials], axis=0)
    s_h1_std = np.std([r['s_h1'] for r in crystal_trials], axis=0)
    deriv_std = np.std([r['derivative'] for r in crystal_trials], axis=0)

    # (a) Temperature
    axs[0].plot(ref_times, temp_ens, 'r-', lw=1.2)
    axs[0].set_ylabel(r'$T$')
    axs[0].text(0.02, 0.85, '(a)', transform=axs[0].transAxes, fontweight='bold')
    axs[0].set_ylim(0, 2.5)

    # (b) Hexatic order
    axs[1].plot(ref_times, psi6_ens, 'm-', lw=1.2)
    axs[1].fill_between(ref_times, psi6_ens - psi6_std, psi6_ens + psi6_std,
                        alpha=0.3, color='m', lw=0)
    axs[1].axhline(0.5, ls='--', color='gray', lw=0.8, alpha=0.7)
    axs[1].set_ylabel(r'$|\psi_6|$')
    axs[1].text(0.02, 0.85, '(b)', transform=axs[1].transAxes, fontweight='bold')
    axs[1].set_ylim(0, 0.8)

    # (c) S_H1
    axs[2].plot(ref_times, s_h1_ens, 'b-', lw=1.2)
    axs[2].fill_between(ref_times, s_h1_ens - s_h1_std, s_h1_ens + s_h1_std,
                        alpha=0.3, color='b', lw=0)
    axs[2].set_ylabel(r'$S_{H1}$')
    axs[2].text(0.02, 0.85, '(c)', transform=axs[2].transAxes, fontweight='bold')

    # (d) Derivative
    axs[3].plot(ref_times, deriv_ens, 'g-', lw=1.2)
    axs[3].fill_between(ref_times, deriv_ens - deriv_std, deriv_ens + deriv_std,
                        alpha=0.3, color='g', lw=0)
    axs[3].axhline(0, ls='--', color='gray', lw=0.8, alpha=0.7)
    axs[3].set_ylabel(r'$dS_{H1}/dt$')
    axs[3].set_xlabel('Integration step')
    axs[3].text(0.02, 0.85, '(d)', transform=axs[3].transAxes, fontweight='bold')

    for ax in axs:
        ax.tick_params(direction='in', top=True, right=True)

    save_figure(fig, 'fig1_ensemble_dynamics')
    plt.close()

    # --- FIGURE 2: Validation and Gaps ---
    print("\n3. Generating Figure 2 (validation + gaps)...")

    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.8))
    plt.subplots_adjust(wspace=0.35)

    # (a) Phase diagram
    for r in quench_results:
        color = 'blue' if r['phase'] == 'CRYSTAL' else 'red'
        axs[0].scatter(r['psi6_final'], r['msd_final'], c=color, s=30,
                      alpha=0.7, edgecolors='black', linewidths=0.5, marker='o')

    for r in null_results:
        axs[0].scatter(r['psi6_final'], r['msd_final'], c='gray', s=30,
                      alpha=0.5, edgecolors='black', linewidths=0.5, marker='s')

    axs[0].axvline(0.5, ls='--', color='gray', lw=0.8, alpha=0.7)
    axs[0].set_xlabel(r'Final $|\psi_6|$')
    axs[0].set_ylabel('Final MSD')
    axs[0].text(0.02, 0.92, '(a)', transform=axs[0].transAxes, fontweight='bold')

    # Legend
    axs[0].scatter([], [], c='blue', s=30, edgecolors='black', linewidths=0.5,
                  marker='o', label='Quench (crystal)')
    axs[0].scatter([], [], c='gray', s=30, edgecolors='black', linewidths=0.5,
                  marker='s', label='Null control')
    axs[0].legend(loc='upper left', framealpha=0.9)

    # (b) Gap histogram
    axs[1].hist(gaps, bins=12, edgecolor='black', linewidth=0.8,
               alpha=0.7, color='steelblue')
    axs[1].axvline(0, color='red', ls='--', lw=1.5, label='Synchrony')
    axs[1].axvline(np.median(gaps), color='orange', ls='-', lw=1.5,
                  label=f'Median = {np.median(gaps):.0f}')
    axs[1].set_xlabel(r'Gap $\Delta t = t_{phys} - t_{topo}$')
    axs[1].set_ylabel('Count')
    axs[1].text(0.02, 0.92, '(b)', transform=axs[1].transAxes, fontweight='bold')
    axs[1].legend(loc='upper right', framealpha=0.9)

    for ax in axs:
        ax.tick_params(direction='in', top=True, right=True)

    save_figure(fig, 'fig2_validation_gaps')
    plt.close()

    # --- FIGURE 3: Single Trial Mechanism ---
    print("\n4. Generating Figure 3 (mechanism)...")

    # Find a good example with clear precursor
    example = None
    for r in crystal_trials:
        if r['t_phys'] is not None and r['t_topo'] is not None:
            gap = r['t_phys'] - r['t_topo']
            if 500 < gap < 1500:  # Nice visible gap
                example = r
                break

    if example is None:
        example = crystal_trials[0]

    fig, axs = plt.subplots(2, 1, figsize=(3.4, 4), sharex=True)
    plt.subplots_adjust(hspace=0.1)

    times = np.array(example['times'])

    # Top panel: psi6
    axs[0].plot(times, example['psi6'], 'm-', lw=1.2, label=r'$|\psi_6|$')
    axs[0].axhline(0.5, ls='--', color='gray', lw=0.8, alpha=0.7)
    if example['t_phys']:
        axs[0].axvline(example['t_phys'], color='red', ls='-', lw=1.5,
                      label=r'$t_{phys}$')
    axs[0].set_ylabel(r'$|\psi_6|$')
    axs[0].legend(loc='upper left', framealpha=0.9)
    axs[0].text(0.02, 0.88, '(a)', transform=axs[0].transAxes, fontweight='bold')

    # Bottom panel: S_H1
    axs[1].plot(times, example['s_h1'], 'b-', lw=1.2, label=r'$S_{H1}$')
    if example['t_topo']:
        axs[1].axvline(example['t_topo'], color='orange', ls='-', lw=1.5,
                      label=r'$t_{topo}$ (CUSUM)')
    if example['t_phys']:
        axs[1].axvline(example['t_phys'], color='red', ls='--', lw=1.0, alpha=0.7)
    axs[1].set_ylabel(r'$S_{H1}$')
    axs[1].set_xlabel('Integration step')
    axs[1].legend(loc='upper right', framealpha=0.9)
    axs[1].text(0.02, 0.88, '(b)', transform=axs[1].transAxes, fontweight='bold')

    for ax in axs:
        ax.tick_params(direction='in', top=True, right=True)

    # Add gap annotation
    if example['t_phys'] and example['t_topo']:
        gap_val = example['t_phys'] - example['t_topo']
        mid = (example['t_phys'] + example['t_topo']) / 2
        axs[1].annotate('', xy=(example['t_phys'], 2.0), xytext=(example['t_topo'], 2.0),
                       arrowprops=dict(arrowstyle='<->', color='black', lw=1))
        axs[1].text(mid, 2.1, rf'$\Delta t = {gap_val:.0f}$', ha='center', fontsize=8)

    save_figure(fig, 'fig3_mechanism')
    plt.close()

    # --- Summary ---
    print("\n" + "=" * 70)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nFiles generated:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"   {f}: {size/1024:.1f} KB")

    # Save summary
    summary = {
        'n_trials': N_TRIALS,
        'n_crystal': len(crystal_trials),
        'n_precursor': precursor_count,
        'precursor_rate': precursor_count / len(gaps),
        'mean_gap': float(np.mean(gaps)),
        'median_gap': float(np.median(gaps)),
        'std_gap': float(np.std(gaps))
    }

    with open(os.path.join(OUTPUT_DIR, 'figure_data.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print("\n   figure_data.json: statistical summary")
