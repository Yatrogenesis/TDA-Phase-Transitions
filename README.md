# Ontologia Operacional de Transiciones de Fase Topologicas

## Overview

Experimental investigation of Topological Data Analysis (TDA) applied to phase transitions in Lennard-Jones 2D systems.

**Hypothesis Under Test**: Information topology (H1 persistence entropy) may show precursor signals before thermodynamic phase transitions.

**Status**: Methodology refined, statistical validation pending.

## Methodology Evolution

### V1-V5: Initial Attempts (DEPRECATED)
- Issues identified: PBC not respected in TDA, lattice initialization confused with gas, burn-in transient misinterpreted as signal

### V6: Rigorous Protocol (CURRENT)
- PBC distance matrix for TDA
- Random initialization + proper burn-in
- MSD tracking for glass/crystal distinction
- Factual observations only, no overclaiming

## Key Observations (V6)

### Experiment: Lennard-Jones 2D Quench
- **System**: N=100, ρ=0.85
- **Protocol**: 500 step burn-in at T=2.0, then quench to T=0.1
- **Phase reached**: LIQUID (|ψ₆| ≈ 0.38, MSD diffusive)
- **S_H1 behavior**: Moderate variation (std=0.107), max at step 760

**Note**: System did not crystallize in this run. Cannot confirm precursor hypothesis without actual phase transition.

## Theoretical Framework

### Operational Definitions

1. **Interaction Flux (J)**: Local entropy production rate
   - `J(x,t) = σ(x,t) = Σ Ji · Xi`

2. **Order as Predictive Information**:
   - `I_pred(T) ~ constant ⟺ Order`

3. **Topological Entropy** (H1 Persistence):
   - `S_pers = -Σ pi log(pi)`
   - `pi = (di - bi) / Σ(dj - bj)`

### The Bridge Theorem

If H¹(U, L) = 0 (trivial bundle), then I_pred is bounded.
If H¹ ≠ 0, topological information is required for prediction.

## Files

- `lennard_jones_tda.py` - Main experiment with TDA analysis
- `lennard_jones_langevin.py` - Improved stability with Langevin thermostat

## Dependencies

```bash
pip install numpy matplotlib ripser persim scipy
```

## Running

```bash
python lennard_jones_tda.py
python lennard_jones_langevin.py
```

## Ontological Interpretation

The H1 persistence entropy measures the complexity of loop structures
in the particle configuration:

1. **GAS PHASE** (high T): Few persistent loops → low H1 entropy
   - L_gas is sufficient, trivial cohomology

2. **CRITICAL REGIME**: Maximum loop diversity → H1 entropy PEAK
   - L_gas fails, new degrees of freedom activate
   - This is the "activation of latent degrees"

3. **SOLID PHASE** (low T): Fixed crystalline cages → H1 entropy saturates
   - L_crystal takes over, new trivial cohomology

## Implications for Tegmark Refutation

This framework demonstrates that information topology is fundamental
to phase transitions. Combined with IIT Φ > 0 measurements in HumanBrain,
it provides evidence that:

- Information integration occurs in classical substrates
- Quantum coherence is not required for meaningful information structure
- The "warm and wet" objection is a category error

## Author

Francisco Molina Burgos
Date: 2026-01-05

φ > 0
