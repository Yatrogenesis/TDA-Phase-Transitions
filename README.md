# Ontologia Operacional de Transiciones de Fase Topologicas

## Overview

Experimental validation of the Ontological Framework for Topological Phase Transitions.

**Core Prediction**: Information topology (the "software" of a system) changes BEFORE thermodynamic observables (the "hardware").

## Key Results

### Experiment 1: Lennard-Jones 2D Gas (Velocity Rescaling)
- **Topological transition**: Step 115, T = 1.438
- **Thermodynamic transition**: Step 275, T = 1.307
- **Precursor gap**: 160 steps
- **Result**: CONFIRMED

### Experiment 2: Lennard-Jones 2D Gas (Langevin Dynamics)
- **Topological transition**: Step 100, T = 2.004
- **Thermodynamic transition**: Not reached (glass formation)
- **Precursor gap**: >1900 steps
- **Result**: CONFIRMED

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
