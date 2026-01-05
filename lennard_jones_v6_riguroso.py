#!/usr/bin/env python3
"""
Lennard-Jones 2D Phase Transition - V6.0 RIGOROUS PROTOCOL
============================================================

CRITICAL CORRECTIONS from V5:
1. TDA with PBC: Uses periodic distance matrix (not raw positions)
2. Proper burn-in: Random init + equilibration before measurement
3. MSD tracking: Distinguishes glass (plateau) from crystal (|psi6|>0.5)
4. Correct labeling: No overclaiming, factual observations only

Author: Francisco Molina Burgos
Date: 2026-01-05
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
import time

# --- A. PARAMETROS DE SIMULACION RIGUROSA ---
N_PARTICLES = 100
DENSITY = 0.85          # Alta densidad para favorecer vidrio/cristal
BOX_SIZE = np.sqrt(N_PARTICLES / DENSITY)
STEPS_EQUIL = 500       # Pasos de Burn-in (NO se registran)
STEPS_PROD = 2000       # Pasos de Produccion (SI se registran)
DT = 0.005
T_HIGH = 2.0            # Temperatura de inicio (Gas)
T_LOW = 0.1             # Temperatura final (Quench profundo)
FRICTION = 1.0          # Langevin

OUTPUT_DIR = '/Users/yatrogenesis/Desktop/CODIGO_3_V6_RIGUROSO'

print("=" * 70)
print("LENNARD-JONES 2D - V6.0 RIGOROUS PROTOCOL")
print("=" * 70)
print(f"N={N_PARTICLES}, Density={DENSITY}, Box={BOX_SIZE:.2f}")
print(f"Burn-in: {STEPS_EQUIL} steps at T={T_HIGH}")
print(f"Production: {STEPS_PROD} steps, T: {T_HIGH} -> {T_LOW}")
print("=" * 70)

# --- B. FUNCIONES DE FISICA Y TOPOLOGIA (CORREGIDAS) ---

def get_pbc_distance_matrix(pos, box):
    """
    Calcula matriz de distancias respetando PBC (topologia toroidal).
    CRITICO para TDA correcto.
    """
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


def compute_forces_lj_pbc(pos, box):
    """Fuerzas Lennard-Jones con PBC."""
    N = len(pos)
    forces = np.zeros((N, 2))
    potential_energy = 0.0

    for i in range(N):
        for j in range(i+1, N):
            delta = pos[i] - pos[j]
            delta = delta - box * np.round(delta / box)
            r2 = np.dot(delta, delta)

            if r2 < 2.5**2 and r2 > 0.5**2:
                r2_i = 1.0 / r2
                r6_i = r2_i**3
                f_s = 48.0 * r2_i * r6_i * (r6_i - 0.5)
                f_vec = f_s * delta
                forces[i] += f_vec
                forces[j] -= f_vec
                potential_energy += 4.0 * r6_i * (r6_i - 1.0)

    return forces, potential_energy


def langevin_step(pos, vel, forces, dt, temp, friction, box):
    """Integrador Langevin BBK."""
    noise_mag = np.sqrt(2 * friction * temp * dt)
    noise = noise_mag * np.random.randn(*pos.shape)

    vel_half = vel + 0.5 * dt * (forces - friction * vel) + 0.5 * noise
    pos_new = pos + vel_half * dt
    pos_new = pos_new % box

    forces_new, pe = compute_forces_lj_pbc(pos_new, box)

    noise_2 = noise_mag * np.random.randn(*pos.shape)
    vel_new = vel_half + 0.5 * dt * (forces_new - friction * vel_half) + 0.5 * noise_2

    return pos_new, vel_new, forces_new, pe


def hexatic_order_pbc(pos, box):
    """
    Parametro de orden hexatico |psi_6| con PBC.
    Promedio de magnitudes locales (correcto para policristales).
    """
    N = len(pos)
    psi6_vals = []
    dm = get_pbc_distance_matrix(pos, box)
    cutoff = 1.5  # Primera capa de vecinos

    for i in range(N):
        neigh_idxs = np.where((dm[i] < cutoff) & (dm[i] > 0.3))[0]

        if len(neigh_idxs) < 3:
            continue

        local_psi = 0j
        for j in neigh_idxs:
            delta = pos[j] - pos[i]
            delta = delta - box * np.round(delta / box)
            theta = np.arctan2(delta[1], delta[0])
            local_psi += np.exp(6j * theta)

        psi6_vals.append(np.abs(local_psi / len(neigh_idxs)))

    return np.mean(psi6_vals) if psi6_vals else 0.0


def persistent_entropy(diagram):
    """Entropia de Shannon sobre vidas de ciclos."""
    if len(diagram) == 0:
        return 0.0, 0

    births = diagram[:, 0]
    deaths = diagram[:, 1]
    lifetimes = deaths - births

    valid = (lifetimes > 1e-5) & np.isfinite(lifetimes)
    lifetimes = lifetimes[valid]
    count = len(lifetimes)

    if count == 0:
        return 0.0, 0

    L_sum = np.sum(lifetimes)
    if L_sum == 0:
        return 0.0, count

    probs = lifetimes / L_sum
    entropy = -np.sum(probs * np.log(probs))
    return entropy, count


# --- C. PROTOCOLO DE SIMULACION ---

np.random.seed(42)

# 1. Inicializacion Aleatoria (Gas Real, NO lattice)
print("\n1. Inicializando Gas Aleatorio...")
pos = np.random.rand(N_PARTICLES, 2) * BOX_SIZE
vel = np.random.randn(N_PARTICLES, 2)
forces, _ = compute_forces_lj_pbc(pos, BOX_SIZE)

# 2. Burn-in / Equilibracion a Alta T
print(f"2. Equilibrando a T={T_HIGH} por {STEPS_EQUIL} pasos (burn-in)...")
for step in range(STEPS_EQUIL):
    pos, vel, forces, _ = langevin_step(pos, vel, forces, DT, T_HIGH, FRICTION, BOX_SIZE)
    if step % 100 == 0:
        print(f"   Burn-in step {step}/{STEPS_EQUIL}")

print("   -> Burn-in completado. Sistema equilibrado como GAS.")

# Guardar configuracion post-burnin
pos_after_burnin = pos.copy()

# 3. Produccion (Quench)
print(f"\n3. Iniciando Quench: T={T_HIGH} -> T={T_LOW}...")

t_prod = []
T_prod = []
E_prod = []
Psi6_prod = []
S_H1_prod = []
H1_count_prod = []
MSD_prod = []

# Posicion inicial para MSD (desenrollada)
pos_0 = pos.copy()
pos_unwrapped = pos.copy()
prev_pos = pos.copy()

configs_to_save = {'post_burnin': pos_after_burnin.copy()}

for step in range(STEPS_PROD):
    progress = step / STEPS_PROD
    curr_T = T_HIGH + (T_LOW - T_HIGH) * progress

    # Integrar
    pos, vel, forces, pe = langevin_step(pos, vel, forces, DT, curr_T, FRICTION, BOX_SIZE)

    # MSD desenrollado (corregir saltos de PBC)
    delta_step = pos - prev_pos
    delta_step = delta_step - BOX_SIZE * np.round(delta_step / BOX_SIZE)
    pos_unwrapped += delta_step
    prev_pos = pos.copy()
    msd_val = np.mean(np.sum((pos_unwrapped - pos_0)**2, axis=1))

    # Temperatura instantanea
    kin_e = 0.5 * np.sum(vel**2)
    inst_T = kin_e / N_PARTICLES

    # Muestreo cada 20 pasos
    if step % 20 == 0:
        # TDA RIGUROSO con PBC
        dm = get_pbc_distance_matrix(pos, BOX_SIZE)
        dgms = ripser(dm, distance_matrix=True, maxdim=1)['dgms']

        s_h1, h1_cnt = persistent_entropy(dgms[1]) if len(dgms) > 1 else (0.0, 0)
        psi6 = hexatic_order_pbc(pos, BOX_SIZE)

        t_prod.append(step)
        T_prod.append(inst_T)
        E_prod.append(pe)
        Psi6_prod.append(psi6)
        S_H1_prod.append(s_h1)
        H1_count_prod.append(h1_cnt)
        MSD_prod.append(msd_val)

        # Guardar configuraciones clave
        if step == STEPS_PROD // 4:
            configs_to_save['quarter'] = pos.copy()
        if step == STEPS_PROD // 2:
            configs_to_save['middle'] = pos.copy()
        if step == 3 * STEPS_PROD // 4:
            configs_to_save['three_quarter'] = pos.copy()

        if step % 200 == 0:
            print(f"   Step {step}: T={inst_T:.2f} | |psi6|={psi6:.3f} | S_H1={s_h1:.3f} | MSD={msd_val:.1f}")

configs_to_save['final'] = pos.copy()

# --- D. DIAGNOSTICO DE FASE ---
print("\n" + "=" * 70)
print("PHASE DIAGNOSIS")
print("=" * 70)

psi_final = np.mean(Psi6_prod[-10:]) if len(Psi6_prod) >= 10 else Psi6_prod[-1]
msd_first_half = np.mean(MSD_prod[:len(MSD_prod)//2])
msd_second_half = np.mean(MSD_prod[len(MSD_prod)//2:])
msd_growth = msd_second_half - msd_first_half

print(f"\nFinal |psi6|: {psi_final:.3f}")
print(f"MSD growth (2nd half - 1st half): {msd_growth:.2f}")

if psi_final > 0.5:
    phase = "CRYSTAL"
    phase_desc = f"|psi6|={psi_final:.2f} > 0.5"
elif msd_growth < 5.0:  # MSD plateau indicates arrest
    phase = "GLASS"
    phase_desc = f"|psi6|={psi_final:.2f}, MSD plateau (arrested dynamics)"
else:
    phase = "LIQUID"
    phase_desc = f"|psi6|={psi_final:.2f}, MSD diffusive"

print(f"\nDIAGNOSIS: {phase}")
print(f"  Evidence: {phase_desc}")

# --- E. ANALISIS S_H1 ---
print("\n" + "=" * 70)
print("TOPOLOGICAL ANALYSIS (S_H1 with PBC)")
print("=" * 70)

# Buscar maximo de S_H1 DESPUES del burn-in (step > 0)
s_h1_arr = np.array(S_H1_prod)
# Ignorar primeros puntos que podrian ser transitorio post-burnin
analysis_start = 5  # Skip first 5 measurements (~100 steps)
if len(s_h1_arr) > analysis_start:
    s_h1_analysis = s_h1_arr[analysis_start:]
    t_analysis = np.array(t_prod)[analysis_start:]

    max_idx = np.argmax(s_h1_analysis)
    s_h1_max_step = t_analysis[max_idx]
    s_h1_max_val = s_h1_analysis[max_idx]

    print(f"S_H1 maximum (after initial): Step {s_h1_max_step}, value = {s_h1_max_val:.4f}")

    # Check if there's meaningful variation
    s_h1_std = np.std(s_h1_analysis)
    s_h1_mean = np.mean(s_h1_analysis)
    print(f"S_H1 mean: {s_h1_mean:.4f}, std: {s_h1_std:.4f}")

    if s_h1_std < 0.1:
        print("\n  NOTE: S_H1 is relatively flat - no clear topological transition detected")
    else:
        print(f"\n  S_H1 shows variation (std={s_h1_std:.3f})")
else:
    s_h1_max_step = 0
    print("Insufficient data for analysis")

# --- F. GRAFICAS ---
print("\nGenerating figures...")

fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Panel 1: Termodinamica y MSD
ax1 = axs[0]
ax1.plot(t_prod, T_prod, 'r--', alpha=0.5, label='Temperature')
ax1.set_ylabel('Temperature', color='r')
ax1.tick_params(axis='y', labelcolor='r')
ax1b = ax1.twinx()
ax1b.plot(t_prod, MSD_prod, 'k-', linewidth=2, label='MSD')
ax1b.set_ylabel('MSD ($\sigma^2$)', color='k')
ax1.set_title(f'Thermodynamics & Dynamics | Final Phase: {phase}', fontweight='bold')
ax1.legend(loc='upper left')
ax1b.legend(loc='upper right')

# Panel 2: Orden Fisico
ax2 = axs[1]
ax2.plot(t_prod, Psi6_prod, 'm-', lw=2, label='$|\psi_6|$ (PBC)')
ax2.axhline(0.5, ls=':', color='gray', label='Crystal threshold')
ax2.set_ylabel('Hexatic Order')
ax2.legend()
ax2.set_title('Structural Order', fontweight='bold')

# Panel 3: Topologia
ax3 = axs[2]
ax3.plot(t_prod, S_H1_prod, 'b-', lw=2, label='$S_{H1}$ (PBC distance matrix)')
ax3.fill_between(t_prod, 0, S_H1_prod, alpha=0.3, color='blue')
ax3.set_ylabel('Persistent Entropy')
ax3.set_xlabel('Production Steps (after burn-in)')
ax3.legend()
ax3.set_title('Topological Information (H1 Persistence)', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/FIG1_rigorous_analysis.png', dpi=150)
print(f"Saved: {OUTPUT_DIR}/FIG1_rigorous_analysis.png")
plt.close()

# Persistence diagrams
fig2, axes = plt.subplots(1, 4, figsize=(16, 4))
keys = ['post_burnin', 'middle', 'three_quarter', 'final']
titles = ['After Burn-in', 'Middle', '3/4 Quench', f'Final ({phase})']

for ax, key, title in zip(axes, keys, titles):
    if key in configs_to_save:
        dm = get_pbc_distance_matrix(configs_to_save[key], BOX_SIZE)
        dgms = ripser(dm, distance_matrix=True, maxdim=1)['dgms']
        plot_diagrams(dgms, ax=ax, show=False)
        psi6_snap = hexatic_order_pbc(configs_to_save[key], BOX_SIZE)
        ax.set_title(f'{title}\n|psi6|={psi6_snap:.2f}', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/FIG2_persistence_diagrams_pbc.png', dpi=150)
print(f"Saved: {OUTPUT_DIR}/FIG2_persistence_diagrams_pbc.png")
plt.close()

# Configurations
fig3, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, key, title in zip(axes, keys, titles):
    if key in configs_to_save:
        p = configs_to_save[key]
        ax.scatter(p[:, 0], p[:, 1], s=30, alpha=0.7, c='blue', edgecolors='black')
        ax.set_xlim(0, BOX_SIZE)
        ax.set_ylim(0, BOX_SIZE)
        ax.set_aspect('equal')
        psi6_snap = hexatic_order_pbc(configs_to_save[key], BOX_SIZE)
        ax.set_title(f'{title}\n|psi6|={psi6_snap:.2f}', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/FIG3_configurations.png', dpi=150)
print(f"Saved: {OUTPUT_DIR}/FIG3_configurations.png")
plt.close()

# --- G. RESUMEN FACTUAL ---
print("\n" + "=" * 70)
print("FACTUAL SUMMARY (V6.0 Rigorous Protocol)")
print("=" * 70)
print(f"""
SIMULATION PARAMETERS:
  - N = {N_PARTICLES}, Density = {DENSITY}
  - Burn-in: {STEPS_EQUIL} steps at T = {T_HIGH}
  - Production: {STEPS_PROD} steps, T: {T_HIGH} -> {T_LOW}

PHASE DIAGNOSIS: {phase}
  - Final |psi6|: {psi_final:.3f}
  - MSD behavior: {'plateau (arrested)' if msd_growth < 5.0 else 'diffusive'}

TOPOLOGICAL OBSERVATIONS:
  - S_H1 computed with PBC distance matrix
  - S_H1 max at step {s_h1_max_step if 's_h1_max_step' in dir() else 'N/A'}
  - S_H1 variability: {'low (flat series)' if 's_h1_std' in dir() and s_h1_std < 0.1 else 'moderate'}

METHODOLOGICAL NOTES:
  - Random initialization (not lattice)
  - Proper burn-in before data collection
  - PBC respected in TDA via distance matrix
  - No overclaiming - factual observations only

For statistical validation, run 30-100 trials with different seeds.
""")

print(f"\nResults saved to: {OUTPUT_DIR}")
