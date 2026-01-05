#!/usr/bin/env python3
"""
Lennard-Jones 2D Phase Transition - Langevin Dynamics
======================================================

Version 2: Improved stability with Langevin thermostat
Author: Francisco Molina Burgos
Date: 2026-01-05
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams

# --- A. CONFIGURACION FISICA ESTABLE ---
N_PARTICLES = 144  # Numero cuadrado para inicializacion en red (12x12)
DENSITY = 0.8      # Densidad tipica para ver cristalizacion en LJ
BOX_SIZE = np.sqrt(N_PARTICLES / DENSITY)
STEPS = 2000       # Mas pasos para enfriamiento suave
DT = 0.005         # Paso de tiempo conservador
T_START = 2.0      # Liquido caliente
T_END = 0.1        # Solido frio
FRICTION = 1.0     # Para termostato Langevin

OUTPUT_DIR = '/Users/yatrogenesis/Desktop/CODIGO_2_LJ_LANGEVIN'

# --- B. FUNCIONES FISICAS ---

def init_positions_grid(n, box):
    # Inicializa en reticula cuadrada para evitar explosiones
    side = int(np.ceil(np.sqrt(n)))
    spacing = box / side
    pos = []
    for i in range(side):
        for j in range(side):
            if len(pos) < n:
                pos.append([i * spacing + spacing/2, j * spacing + spacing/2])
    return np.array(pos)

def compute_forces_lj(pos, box):
    N = len(pos)
    forces = np.zeros((N, 2))
    potential_energy = 0.0
    virial = 0.0

    # Bucle N^2 (lento pero seguro para N pequeno)
    for i in range(N):
        for j in range(i + 1, N):
            delta = pos[i] - pos[j]
            delta = delta - box * np.round(delta / box) # MIC
            r2 = np.dot(delta, delta)

            if r2 < 2.5**2: # Cutoff
                r2_inv = 1.0 / r2
                r6_inv = r2_inv ** 3
                # LJ Force: F = 48 * (1/r^14 - 0.5/r^8) * r_vec
                # V = 4 * (1/r^12 - 1/r^6)
                force_scalar = 48.0 * r2_inv * r6_inv * (r6_inv - 0.5)
                f_vec = force_scalar * delta
                forces[i] += f_vec
                forces[j] -= f_vec
                potential_energy += 4.0 * r6_inv * (r6_inv - 1.0)
                virial += np.dot(f_vec, delta)

    return forces, potential_energy, virial

def langevin_integrator(pos, vel, forces, dt, temp, friction, box):
    # Integrador Langevin (BBK o similar)
    # v(t+0.5)
    vel_half = vel + 0.5 * dt * (forces - friction * vel) + \
               np.sqrt(2 * friction * temp * dt) * np.random.randn(*pos.shape)

    # r(t+1)
    pos_new = pos + vel_half * dt
    pos_new = pos_new % box

    # F(t+1)
    forces_new, pe, virial = compute_forces_lj(pos_new, box)

    # v(t+1)
    vel_new = vel_half + 0.5 * dt * (forces_new - friction * vel_half) + \
              np.sqrt(2 * friction * temp * dt) * np.random.randn(*pos.shape)

    return pos_new, vel_new, forces_new, pe, virial

# --- C. CALCULO DE PARAMETROS DE ORDEN ---

def hexatic_order(pos, box):
    """
    Calcula orden hexatico |psi_6| correctamente (KTHNY).
    psi_6(i) = (1/N_i) * sum_j exp(6i * theta_ij)
    Retorna promedio de |psi_6(i)| sobre todas las particulas.
    """
    N = len(pos)
    cutoff_sq = 1.8**2  # Ajustado para primera capa de vecinos en LJ

    psi6_magnitudes = []

    for i in range(N):
        neighbors = 0
        psi6_local = 0.0 + 0.0j
        for j in range(N):
            if i == j:
                continue
            delta = pos[j] - pos[i]
            delta = delta - box * np.round(delta / box)  # MIC
            r2 = np.dot(delta, delta)

            if r2 < cutoff_sq and r2 > 0.5**2:  # Evitar auto-vecinos
                theta = np.arctan2(delta[1], delta[0])
                psi6_local += np.exp(6j * theta)
                neighbors += 1

        if neighbors > 0:
            # Promedio LOCAL del orden hexatico
            psi6_local /= neighbors
            psi6_magnitudes.append(np.abs(psi6_local))

    # Promedio de MAGNITUDES (no de numeros complejos)
    return np.mean(psi6_magnitudes) if psi6_magnitudes else 0.0


def compute_periodic_distance_matrix(pos, box):
    """
    Calcula matriz de distancias con condiciones de frontera periodicas.
    Necesario para TDA correcto en sistemas con PBC.
    """
    N = len(pos)
    dist_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(i + 1, N):
            delta = pos[i] - pos[j]
            delta = delta - box * np.round(delta / box)  # MIC
            d = np.sqrt(np.dot(delta, delta))
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    return dist_matrix

# --- D. SIMULACION PRINCIPAL ---

print("=" * 70)
print("LENNARD-JONES 2D - LANGEVIN DYNAMICS - TOPOLOGICAL ANALYSIS")
print("=" * 70)
print(f"N={N_PARTICLES}, Rho={DENSITY}, Box={BOX_SIZE:.2f}")
print(f"T: {T_START} -> {T_END}, Steps: {STEPS}")
print("=" * 70)

np.random.seed(42)

# Inicializacion
positions = init_positions_grid(N_PARTICLES, BOX_SIZE)
velocities = np.random.randn(N_PARTICLES, 2)
# Equilibrar fuerzas iniciales
forces, _, _ = compute_forces_lj(positions, BOX_SIZE)

# Historiales
times = []
temps = []
energies = []
psi6_vals = []
topo_entropy = []
h1_counts = []

# Guardar configuraciones para diagramas de persistencia
configs_to_save = {}

for step in range(STEPS):
    # Rampa de enfriamiento
    current_temp = T_START + (T_END - T_START) * (step / STEPS)

    # Integracion
    positions, velocities, forces, pe, virial = langevin_integrator(
        positions, velocities, forces, DT, current_temp, FRICTION, BOX_SIZE
    )

    # Medir T instantanea (Cinetica)
    kin_energy = 0.5 * np.sum(velocities**2)
    inst_temp = kin_energy / N_PARTICLES

    # Analisis (cada 20 pasos)
    if step % 20 == 0:
        # 1. Orden Hexatico (Fisica estandar)
        psi6 = hexatic_order(positions, BOX_SIZE)

        # 2. Entropia Topologica con PBC (matriz de distancias periodicas)
        # Usar distance_matrix=True para respetar topologia toroidal
        dist_matrix = compute_periodic_distance_matrix(positions, BOX_SIZE)
        diagrams = ripser(dist_matrix, maxdim=1, distance_matrix=True)['dgms']
        if len(diagrams[1]) > 0:
            lifetimes = diagrams[1][:, 1] - diagrams[1][:, 0]
            lifetimes = lifetimes[np.isfinite(lifetimes)]
            h1_cnt = len(lifetimes)
            if len(lifetimes) > 0:
                p = lifetimes / np.sum(lifetimes)
                S_topo = -np.sum(p * np.log(p))
            else:
                S_topo = 0
        else:
            S_topo = 0
            h1_cnt = 0

        times.append(step)
        temps.append(inst_temp)
        energies.append(pe)
        psi6_vals.append(psi6)
        topo_entropy.append(S_topo)
        h1_counts.append(h1_cnt)

        # Guardar configuraciones especiales
        if step == 0:
            configs_to_save['initial'] = positions.copy()
        if step == STEPS // 2:
            configs_to_save['middle'] = positions.copy()

        if step % 200 == 0:
            print(f"Step {step}: T={inst_temp:.2f}, Psi6={psi6:.3f}, S_topo={S_topo:.3f}, H1={h1_cnt}")

configs_to_save['final'] = positions.copy()

# --- E. ANALISIS DE TRANSICIONES ---
print("\n" + "=" * 70)
print("TRANSITION ANALYSIS")
print("=" * 70)

# Encontrar pico de entropia topologica
topo_arr = np.array(topo_entropy)
topo_smooth = np.convolve(topo_arr, np.ones(3)/3, mode='same')
topo_peak_idx = np.argmax(topo_smooth[5:-5]) + 5
topo_peak_step = times[topo_peak_idx]
topo_peak_temp = temps[topo_peak_idx]

# Encontrar transicion cristalina (psi6 > 0.5)
psi6_arr = np.array(psi6_vals)
crystal_indices = np.where(psi6_arr > 0.5)[0]
if len(crystal_indices) > 0:
    crystal_idx = crystal_indices[0]
    crystal_step = times[crystal_idx]
    crystal_temp = temps[crystal_idx]
else:
    crystal_step = STEPS
    crystal_temp = T_END

print(f"S_H1 maximum: Step {topo_peak_step}, T = {topo_peak_temp:.3f}")
print(f"  (Note: Early peak may be equilibration transient)")
print(f"Crystal transition (|psi6| > 0.5): Step {crystal_step}, T = {crystal_temp:.3f}")

if crystal_step >= STEPS:
    print("\n  WARNING: System did not crystallize during simulation window.")
    print("  Cannot confirm precursor hypothesis without crystal transition.")
elif topo_peak_step < 100:
    print(f"\n  WARNING: S_H1 peak at step {topo_peak_step} likely equilibration transient.")
    print("  Need longer burn-in or different analysis window.")
else:
    print(f"\nTemporal gap: {crystal_step - topo_peak_step} steps")
    print("  (Requires statistical validation across multiple runs)")

# --- F. VISUALIZACION ---

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Panel 1: Termodinamica vs Orden
ax1.plot(times, temps, 'r-', alpha=0.5, label='Temperature (T)')
ax1.set_ylabel('Temperature', color='r')
ax1.tick_params(axis='y', labelcolor='r')

ax1b = ax1.twinx()
ax1b.plot(times, psi6_vals, 'k-', linewidth=2, label='Hexatic Order ($|\psi_6|$)')
ax1b.set_ylabel('Crystal Order ($|\psi_6|$)', color='k')
ax1b.axhline(0.5, color='gray', linestyle='--', alpha=0.7, label='Crystal threshold')
ax1.set_title('THERMODYNAMICS vs PHYSICAL ORDER', fontweight='bold', fontsize=12)

# Marcar transiciones con etiquetas correctas
# Nota: El pico temprano es transitorio de equilibracion, no necesariamente precursor
ax1.axvline(topo_peak_step, color='blue', linestyle=':', alpha=0.7,
            label=f'S_H1 max @ {topo_peak_step} (equilibration?)')
if crystal_step < STEPS:
    ax1.axvline(crystal_step, color='green', linestyle=':', alpha=0.7,
                label=f'|ψ₆|>0.5 @ {crystal_step}')
ax1.legend(loc='upper right', fontsize=8)

# Panel 2: La Prediccion Topologica
ax2.plot(times, topo_entropy, 'b-', linewidth=2, label='Topological Entropy ($S_{H1}$)')
ax2.fill_between(times, 0, topo_entropy, alpha=0.3, color='blue')
ax2.set_ylabel('Persistence Entropy (PBC)', color='b', fontweight='bold')
ax2.set_xlabel('Simulation Step')
ax2.grid(True, alpha=0.3)
ax2.set_title('TOPOLOGICAL INFORMATION (H1 Persistence with PBC)', fontweight='bold', fontsize=12)

ax2.axvline(topo_peak_step, color='blue', linestyle=':', alpha=0.7)
if crystal_step < STEPS:
    ax2.axvline(crystal_step, color='green', linestyle=':', alpha=0.7)

# NO sombrear "PRECURSOR REGION" sin evidencia robusta
# Solo anotar si hay cambio significativo en la serie (no transitorio inicial)
ax2.legend()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/FIG1_topological_vs_thermodynamic.png', dpi=150)
print(f"\nSaved: {OUTPUT_DIR}/FIG1_topological_vs_thermodynamic.png")
plt.close()

# --- G. DIAGRAMAS DE PERSISTENCIA (con PBC) ---
fig2, axes = plt.subplots(1, 3, figsize=(15, 5))

# Etiquetas correctas segun estado real
final_psi6 = psi6_vals[-1] if psi6_vals else 0
final_state = 'Crystal' if final_psi6 > 0.5 else ('Liquid/Glass' if final_psi6 > 0.3 else 'Disordered')
titles = ['Initial (Lattice)', 'Middle', f'Final ({final_state})']
keys = ['initial', 'middle', 'final']

for ax, key, title in zip(axes, keys, titles):
    if key in configs_to_save:
        # Usar matriz de distancias periodicas para TDA
        dist_matrix = compute_periodic_distance_matrix(configs_to_save[key], BOX_SIZE)
        diagrams = ripser(dist_matrix, maxdim=1, distance_matrix=True)['dgms']
        plot_diagrams(diagrams, ax=ax, show=False)
        ax.set_title(title, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/FIG2_persistence_diagrams.png', dpi=150)
print(f"Saved: {OUTPUT_DIR}/FIG2_persistence_diagrams.png")
plt.close()

# --- H. CONFIGURACIONES ESPACIALES ---
fig3, axes = plt.subplots(1, 3, figsize=(15, 5))

# Calcular psi6 para cada configuracion y etiquetar correctamente
config_titles = []
for key in keys:
    if key in configs_to_save:
        psi6_config = hexatic_order(configs_to_save[key], BOX_SIZE)
        if key == 'initial':
            label = f'Initial (Lattice, |ψ₆|={psi6_config:.2f})'
        elif key == 'middle':
            label = f'Middle (|ψ₆|={psi6_config:.2f})'
        else:
            state = 'Crystal' if psi6_config > 0.5 else 'Disordered'
            label = f'Final ({state}, |ψ₆|={psi6_config:.2f})'
        config_titles.append(label)
    else:
        config_titles.append('')

for ax, key, title in zip(axes, keys, config_titles):
    if key in configs_to_save:
        pos = configs_to_save[key]
        ax.scatter(pos[:, 0], pos[:, 1], s=30, alpha=0.7, c='blue', edgecolors='black')
        ax.set_xlim(0, BOX_SIZE)
        ax.set_ylim(0, BOX_SIZE)
        ax.set_aspect('equal')
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/FIG3_configurations.png', dpi=150)
print(f"Saved: {OUTPUT_DIR}/FIG3_configurations.png")
plt.close()

# --- I. RESUMEN ---
print("\n" + "=" * 70)
print("FACTUAL SUMMARY")
print("=" * 70)
print(f"""
S_H1 Maximum:
  - Step: {topo_peak_step}
  - Temperature: {topo_peak_temp:.4f}
  - H1 entropy: {topo_entropy[topo_peak_idx]:.4f}
  - Status: {'Likely equilibration transient' if topo_peak_step < 100 else 'Needs validation'}

Thermodynamic Transition (|psi6| > 0.5):
  - Step: {crystal_step}
  - Temperature: {crystal_temp:.4f}
  - Order parameter: {psi6_vals[crystal_idx]:.4f if len(crystal_indices) > 0 else 'N/A (no crystallization)'}

OBSERVATIONS:
- S_H1 shows initial transient and then stabilizes
- |psi6| {'crosses threshold at step ' + str(crystal_step) if len(crystal_indices) > 0 else 'does not reach crystal threshold'}

NOTES FOR VALIDATION:
- Need multiple runs (30-100) with different seeds for statistical analysis
- Need null controls (constant T, randomized positions)
- Need to separate burn-in from actual dynamics
- PBC distance matrix now used for TDA (corrected)
""")

print("\nFiles saved to:", OUTPUT_DIR)
print("phi > 0")
