# Ontologia Operacional de Transiciones de Fase Topologicas

## Overview

Experimental investigation of Topological Data Analysis (TDA) applied to phase transitions in Lennard-Jones 2D systems.

**Hypothesis Under Test**: Information topology (H1 persistence entropy) shows precursor signals BEFORE thermodynamic phase transitions (crystallization).

**Status**: Statistical validation complete. **WEAK SUPPORT** for hypothesis.

## Summary of Results (V8)

### Statistical Validation (N=30 trials + null controls)

| Metric | Value |
|--------|-------|
| Crystallization rate | 100% (30/30) |
| Trials with S_H1 precursor | 53% (16/30) |
| Trials without precursor | 47% (14/30) |
| Mean precursor gap | 987.5 ± 704.8 steps |
| Null control validation | ✓ (0/10 crystallized at high T) |

**Conclusion**: The precursor hypothesis receives **WEAK SUPPORT**. Only ~53% of trials showed S_H1 peak before crystallization, and the variability is high. This suggests that while topological information changes ARE associated with the phase transition, they are not consistently predictive as a precursor signal.

## Methodology Evolution

### V1-V5: Initial Attempts (DEPRECATED)
- Issues identified: PBC not respected in TDA, lattice initialization confused with gas, burn-in transient misinterpreted as signal

### V6: Rigorous Protocol
- PBC distance matrix for TDA
- Random initialization + proper burn-in
- MSD tracking for glass/crystal distinction
- Issue: Numerical instability in Langevin integrator

### V7: Statistical Framework
- 30 trials with different seeds
- Null controls (constant T)
- Stable velocity Verlet integrator
- Issue: Density too high (0.9) - null controls crystallized

### V8: Corrected Methodology (CURRENT)
- Lower density (0.7) - proper liquid-solid transition
- Null controls validated (0% false positive)
- 100% crystallization in quench trials
- Honest statistical assessment

## Key Observations (V8)

### Experiment: Lennard-Jones 2D Quench
- **System**: N=100, ρ=0.7
- **Protocol**: 2000 step equilibration at T=2.0, then quench to T=0.1 over 5000 steps
- **Phase reached**: 100% CRYSTAL (|ψ₆| ≈ 0.57-0.65)
- **S_H1 behavior**: Variable - 53% showed peak before crystallization

### Interpretation
The data does NOT strongly support the hypothesis that S_H1 is a reliable precursor of phase transitions. The signal is present in ~half the trials, suggesting:

1. S_H1 changes are correlated with, but not predictive of, crystallization
2. The timing relationship is not deterministic
3. Further refinement of the hypothesis may be needed

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

| File | Description |
|------|-------------|
| `lennard_jones_tda.py` | V5 - Initial TDA experiment |
| `lennard_jones_langevin.py` | V5 - Langevin thermostat version |
| `lennard_jones_v6_riguroso.py` | V6 - Rigorous protocol |
| `lennard_jones_v7_statistical.py` | V7 - Statistical framework |
| `lennard_jones_v8_corrected.py` | V8 - **Final version** |

## Dependencies

```bash
pip install numpy matplotlib ripser persim scipy
```

## Running

```bash
# Run V8 (recommended - full statistical validation)
python3 lennard_jones_v8_corrected.py

# Output: ~/Desktop/CODIGO_5_V8_CORRECTED/
```

## Output Files (V8)

- `FIG1_ensemble_timeseries.png` - Temperature, |ψ₆|, S_H1, MSD vs time
- `FIG2_phase_diagram.png` - Phase diagram (quench vs null controls)
- `FIG3_precursor_gaps.png` - Histogram of precursor gaps
- `results_summary.json` - Statistical summary

## Ontological Interpretation

The H1 persistence entropy measures the complexity of loop structures in the particle configuration:

1. **GAS PHASE** (high T): Diverse but short-lived loops → moderate H1 entropy
2. **TRANSITION**: Loop structure reorganizes → S_H1 changes (but not deterministically precursor)
3. **CRYSTAL PHASE** (low T): Fixed crystalline cages → S_H1 settles to new value

**Revised conclusion**: The topological information DOES change during phase transitions, but the timing relationship to thermodynamic observables (|ψ₆|) is not consistent enough to serve as a reliable precursor signal.

## Honest Assessment

This experiment demonstrates the importance of:
1. Statistical validation (single runs can mislead)
2. Null controls (validate methodology)
3. Honest reporting (53% is not "prediction confirmed")

The hypothesis is not refuted, but it is not strongly supported either. Further work could explore:
- Longer timescales
- Different cooling rates
- Alternative topological measures (H0, H2)
- Larger systems

## Author

Francisco Molina Burgos
Date: 2026-01-05

φ > 0
