//! Lennard-Jones 2D Phase Transition - RUST VERSION
//! =================================================
//! High-performance implementation using Rayon for parallelism.
//! Author: Francisco Molina Burgos
//! Date: 2026-01-10

use ndarray::Array2;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::time::Instant;

// === PARAMETERS ===
const DT: f64 = 0.002;
const T_HIGH: f64 = 2.0;
const T_LOW: f64 = 0.1;
const STEPS_EQUIL: usize = 2000;
const STEPS_PROD: usize = 5000;
const SAMPLE_INTERVAL: usize = 25;
const THERMOSTAT_TAU: usize = 50;
const R_CUT: f64 = 2.5;
const R_MIN: f64 = 0.7;
const DENSITY: f64 = 0.7;

#[derive(Clone, Serialize, Deserialize)]
struct TrialResult {
    seed: u64,
    phase: String,
    psi6_final: f64,
    t_phys: Option<usize>,
    t_topo: Option<usize>,
    gap: Option<i64>,
}

#[derive(Serialize)]
struct ValidationSummary {
    n_particles: usize,
    n_trials: usize,
    total_time_sec: f64,
    time_per_trial_sec: f64,
    n_crystal: usize,
    n_detected: usize,
    n_precursor: usize,
    precursor_rate: f64,
    mean_gap: f64,
    std_gap: f64,
    gaps: Vec<i64>,
    trials: Vec<TrialResult>,
}

/// Compute PBC distance between two points
#[inline]
fn pbc_distance(p1: &[f64], p2: &[f64], box_size: f64) -> f64 {
    let mut dx = p1[0] - p2[0];
    let mut dy = p1[1] - p2[1];
    dx -= box_size * (dx / box_size).round();
    dy -= box_size * (dy / box_size).round();
    (dx * dx + dy * dy).sqrt()
}

/// Compute full PBC distance matrix (parallel)
fn compute_distance_matrix(pos: &Array2<f64>, box_size: f64) -> Array2<f64> {
    let n = pos.nrows();
    let mut dm = Array2::<f64>::zeros((n, n));

    // Parallel computation of upper triangle
    let results: Vec<(usize, usize, f64)> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            (i + 1..n)
                .map(|j| {
                    let pi = pos.row(i);
                    let pj = pos.row(j);
                    let d = pbc_distance(pi.as_slice().unwrap(), pj.as_slice().unwrap(), box_size);
                    (i, j, d)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    for (i, j, d) in results {
        dm[[i, j]] = d;
        dm[[j, i]] = d;
    }

    dm
}

/// Compute LJ forces (parallel)
fn compute_forces(pos: &Array2<f64>, box_size: f64) -> (Array2<f64>, f64) {
    let n = pos.nrows();
    let r_cut2 = R_CUT * R_CUT;
    let r_min2 = R_MIN * R_MIN;

    let results: Vec<(usize, f64, f64, f64)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut fx = 0.0;
            let mut fy = 0.0;
            let mut pe_i = 0.0;
            let pi = pos.row(i);

            for j in 0..n {
                if i == j {
                    continue;
                }
                let pj = pos.row(j);

                let mut dx = pi[0] - pj[0];
                let mut dy = pi[1] - pj[1];
                dx -= box_size * (dx / box_size).round();
                dy -= box_size * (dy / box_size).round();

                let mut r2 = dx * dx + dy * dy;
                if r2 < r_cut2 {
                    if r2 < r_min2 {
                        r2 = r_min2;
                    }
                    let r2_inv = 1.0 / r2;
                    let r6_inv = r2_inv * r2_inv * r2_inv;
                    let mut f_mag = 48.0 * r2_inv * r6_inv * (r6_inv - 0.5);
                    f_mag = f_mag.clamp(-50.0, 50.0);

                    fx += f_mag * dx;
                    fy += f_mag * dy;

                    if i < j {
                        pe_i += 4.0 * r6_inv * (r6_inv - 1.0);
                    }
                }
            }
            (i, fx, fy, pe_i)
        })
        .collect();

    let mut forces = Array2::<f64>::zeros((n, 2));
    let mut pe = 0.0;

    for (i, fx, fy, pe_i) in results {
        forces[[i, 0]] = fx;
        forces[[i, 1]] = fy;
        pe += pe_i;
    }

    (forces, pe)
}

/// Velocity Verlet step
fn velocity_verlet_step(
    pos: &mut Array2<f64>,
    vel: &mut Array2<f64>,
    forces: &mut Array2<f64>,
    box_size: f64,
) {
    let n = pos.nrows();

    // Half-step velocity
    for i in 0..n {
        vel[[i, 0]] += 0.5 * DT * forces[[i, 0]];
        vel[[i, 1]] += 0.5 * DT * forces[[i, 1]];
    }

    // Update position with PBC
    for i in 0..n {
        pos[[i, 0]] = (pos[[i, 0]] + DT * vel[[i, 0]]).rem_euclid(box_size);
        pos[[i, 1]] = (pos[[i, 1]] + DT * vel[[i, 1]]).rem_euclid(box_size);
    }

    // New forces
    let (new_forces, _) = compute_forces(pos, box_size);
    *forces = new_forces;

    // Complete velocity step
    for i in 0..n {
        vel[[i, 0]] += 0.5 * DT * forces[[i, 0]];
        vel[[i, 1]] += 0.5 * DT * forces[[i, 1]];
    }
}

/// Apply Berendsen thermostat
fn apply_thermostat(vel: &mut Array2<f64>, target_t: f64) {
    let n = vel.nrows() as f64;
    let ke: f64 = vel.iter().map(|v| 0.5 * v * v).sum();
    let current_t = ke / n;

    if current_t > 1e-6 {
        let scale = (target_t / current_t).sqrt().clamp(0.9, 1.1);
        vel.mapv_inplace(|v| v * scale);
    }
}

/// Compute hexatic order parameter |ψ6| (parallel)
fn hexatic_order(pos: &Array2<f64>, dm: &Array2<f64>, box_size: f64) -> f64 {
    let n = pos.nrows();
    let cutoff = 1.8;
    let min_dist = 0.3;

    let psi6_mags: Vec<f64> = (0..n)
        .into_par_iter()
        .filter_map(|i| {
            let pi = pos.row(i);
            let mut psi_r = 0.0;
            let mut psi_i = 0.0;
            let mut n_neigh = 0;

            for j in 0..n {
                if i == j {
                    continue;
                }
                let d = dm[[i, j]];
                if d > min_dist && d < cutoff {
                    let pj = pos.row(j);
                    let mut dx = pj[0] - pi[0];
                    let mut dy = pj[1] - pi[1];
                    dx -= box_size * (dx / box_size).round();
                    dy -= box_size * (dy / box_size).round();

                    let theta = dy.atan2(dx);
                    psi_r += (6.0 * theta).cos();
                    psi_i += (6.0 * theta).sin();
                    n_neigh += 1;
                }
            }

            if n_neigh >= 3 {
                let mag = ((psi_r / n_neigh as f64).powi(2) + (psi_i / n_neigh as f64).powi(2)).sqrt();
                Some(mag)
            } else {
                None
            }
        })
        .collect();

    if psi6_mags.is_empty() {
        0.0
    } else {
        psi6_mags.iter().sum::<f64>() / psi6_mags.len() as f64
    }
}

/// Simplified H1 persistence entropy calculation
/// Uses a fast approximation based on Rips complex edge statistics
fn persistence_entropy_h1(dm: &Array2<f64>) -> f64 {
    let n = dm.nrows();
    if n < 10 {
        return 0.0;
    }

    // Collect all non-zero distances
    let mut edges: Vec<f64> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in i + 1..n {
            let d = dm[[i, j]];
            if d > 0.0 && d < 3.0 {
                edges.push(d);
            }
        }
    }

    if edges.len() < 10 {
        return 0.0;
    }

    edges.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Approximate H1 lifetimes using edge length distribution
    // This captures topological structure without full Rips computation
    let n_bins = 20;
    let max_d = edges.last().copied().unwrap_or(1.0);
    let bin_size = max_d / n_bins as f64;

    let mut hist = vec![0usize; n_bins];
    for &e in &edges {
        let bin = ((e / bin_size) as usize).min(n_bins - 1);
        hist[bin] += 1;
    }

    // Compute derivative (birth/death rate proxy)
    let mut lifetimes: Vec<f64> = Vec::new();
    for i in 1..n_bins {
        let diff = (hist[i] as i64 - hist[i - 1] as i64).abs() as f64;
        if diff > 0.0 {
            lifetimes.push(diff * bin_size);
        }
    }

    if lifetimes.is_empty() {
        return 0.0;
    }

    // Normalize to get entropy
    let total: f64 = lifetimes.iter().sum();
    if total < 1e-10 {
        return 0.0;
    }

    let mut entropy = 0.0;
    for l in &lifetimes {
        let p = l / total;
        if p > 1e-10 {
            entropy -= p * p.ln();
        }
    }

    entropy
}

/// Detect crystallization with persistence
fn detect_crystal(psi6_series: &[f64], times: &[usize], threshold: f64, persistence: usize) -> Option<usize> {
    let above: Vec<bool> = psi6_series.iter().map(|&p| p > threshold).collect();

    for i in 0..above.len().saturating_sub(persistence) {
        if above[i..i + persistence].iter().all(|&b| b) {
            return Some(times[i]);
        }
    }
    None
}

/// CUSUM change point detection
fn detect_cusum(s_h1_series: &[f64], times: &[usize], baseline_fraction: f64) -> Option<usize> {
    let baseline_end = (s_h1_series.len() as f64 * baseline_fraction) as usize;
    if baseline_end < 5 {
        return None;
    }

    let baseline = &s_h1_series[..baseline_end];
    let mu: f64 = baseline.iter().sum::<f64>() / baseline.len() as f64;
    let sigma: f64 = (baseline.iter().map(|&x| (x - mu).powi(2)).sum::<f64>() / baseline.len() as f64).sqrt();

    if sigma < 1e-6 {
        return None;
    }

    let threshold = 3.0 * sigma;
    let mut cusum = 0.0;

    for i in baseline_end..s_h1_series.len() {
        cusum = (cusum + (mu - s_h1_series[i]) - 0.5 * sigma).max(0.0);
        if cusum > threshold {
            return Some(times[i]);
        }
    }
    None
}

/// Run single trial
fn run_trial(n_particles: usize, seed: u64) -> (TrialResult, Vec<f64>, Vec<f64>) {
    let box_size = (n_particles as f64 / DENSITY).sqrt();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Initialize positions
    let mut pos = Array2::<f64>::zeros((n_particles, 2));
    for i in 0..n_particles {
        pos[[i, 0]] = rng.random::<f64>() * box_size;
        pos[[i, 1]] = rng.random::<f64>() * box_size;
    }

    // Initialize velocities
    let mut vel = Array2::<f64>::zeros((n_particles, 2));
    for i in 0..n_particles {
        vel[[i, 0]] = rng.random::<f64>() * 2.0 - 1.0;
        vel[[i, 1]] = rng.random::<f64>() * 2.0 - 1.0;
    }
    // Remove COM velocity
    let com_vx: f64 = vel.column(0).sum() / n_particles as f64;
    let com_vy: f64 = vel.column(1).sum() / n_particles as f64;
    for i in 0..n_particles {
        vel[[i, 0]] -= com_vx;
        vel[[i, 1]] -= com_vy;
    }
    // Scale to target T
    apply_thermostat(&mut vel, T_HIGH);

    let (mut forces, _) = compute_forces(&pos, box_size);

    // Equilibration
    for step in 0..STEPS_EQUIL {
        velocity_verlet_step(&mut pos, &mut vel, &mut forces, box_size);
        if step % THERMOSTAT_TAU == 0 {
            apply_thermostat(&mut vel, T_HIGH);
        }
    }

    // Production
    let mut times = Vec::new();
    let mut psi6_series = Vec::new();
    let mut s_h1_series = Vec::new();

    for step in 0..STEPS_PROD {
        let progress = step as f64 / STEPS_PROD as f64;
        let target_t = T_HIGH + (T_LOW - T_HIGH) * progress;

        velocity_verlet_step(&mut pos, &mut vel, &mut forces, box_size);

        if step % THERMOSTAT_TAU == 0 {
            apply_thermostat(&mut vel, target_t);
        }

        if step % SAMPLE_INTERVAL == 0 {
            let dm = compute_distance_matrix(&pos, box_size);
            let psi6 = hexatic_order(&pos, &dm, box_size);
            let s_h1 = persistence_entropy_h1(&dm);

            times.push(step);
            psi6_series.push(psi6);
            s_h1_series.push(s_h1);
        }
    }

    // Detection
    let t_phys = detect_crystal(&psi6_series, &times, 0.5, 4);
    let t_topo = detect_cusum(&s_h1_series, &times, 0.3);

    let gap = match (t_phys, t_topo) {
        (Some(tp), Some(tt)) => Some(tp as i64 - tt as i64),
        _ => None,
    };

    let psi6_final: f64 = psi6_series[psi6_series.len().saturating_sub(5)..].iter().sum::<f64>()
        / 5.0f64.min(psi6_series.len() as f64);

    let phase = if psi6_final > 0.5 { "CRYSTAL" } else { "LIQUID/GLASS" };

    let result = TrialResult {
        seed,
        phase: phase.to_string(),
        psi6_final,
        t_phys,
        t_topo,
        gap,
    };

    (result, psi6_series, s_h1_series)
}

/// Run validation with custom seed range
fn run_validation_custom(n_particles: usize, n_trials: usize, seed_start: u64) -> ValidationSummary {
    println!("\n{}", "=".repeat(70));
    println!("RUST VALIDATION: N={}, {} trials (seeds {}..{})",
             n_particles, n_trials, seed_start, seed_start + n_trials as u64 - 1);
    println!("{}", "=".repeat(70));

    let total_start = Instant::now();
    let mut results = Vec::new();

    for i in 0..n_trials {
        let seed = seed_start + i as u64;
        let trial_start = Instant::now();
        let (result, _, _) = run_trial(n_particles, seed);
        let elapsed = trial_start.elapsed().as_secs_f64();

        println!(
            "  Trial {}/{}: {}, |ψ6|={:.3}, gap={:?} ({:.1}s)",
            i + 1,
            n_trials,
            result.phase,
            result.psi6_final,
            result.gap,
            elapsed
        );

        results.push(result);
    }

    let total_time = total_start.elapsed().as_secs_f64();

    // Analysis
    let crystal: Vec<&TrialResult> = results.iter().filter(|r| r.phase == "CRYSTAL").collect();
    let n_crystal = crystal.len();

    let gaps: Vec<i64> = crystal.iter().filter_map(|r| r.gap).collect();
    let n_detected = gaps.len();
    let n_precursor = gaps.iter().filter(|&&g| g > 0).count();

    let precursor_rate = if n_detected > 0 {
        n_precursor as f64 / n_detected as f64
    } else {
        0.0
    };

    let mean_gap = if !gaps.is_empty() {
        gaps.iter().sum::<i64>() as f64 / gaps.len() as f64
    } else {
        0.0
    };

    let std_gap = if gaps.len() > 1 {
        let variance = gaps.iter().map(|&g| (g as f64 - mean_gap).powi(2)).sum::<f64>() / gaps.len() as f64;
        variance.sqrt()
    } else {
        0.0
    };

    println!("\n--- RESULTS N={} ---", n_particles);
    println!("Crystallization: {}/{}", n_crystal, n_trials);
    println!(
        "Precursor rate: {}/{} ({:.1}%)",
        n_precursor,
        n_detected,
        100.0 * precursor_rate
    );
    println!("Mean gap: {:.1} ± {:.1}", mean_gap, std_gap);
    println!("Total time: {:.1}s ({:.1} min)", total_time, total_time / 60.0);
    println!("Time/trial: {:.1}s", total_time / n_trials as f64);

    ValidationSummary {
        n_particles,
        n_trials,
        total_time_sec: total_time,
        time_per_trial_sec: total_time / n_trials as f64,
        n_crystal,
        n_detected,
        n_precursor,
        precursor_rate,
        mean_gap,
        std_gap,
        gaps,
        trials: results,
    }
}

fn main() {
    println!("{}", "=".repeat(70));
    println!("TDA-CUSUM EXTENDED VALIDATION - n=30 per system size");
    println!("Additional trials to reach statistical significance");
    println!("{}", "=".repeat(70));

    // Create output directories
    std::fs::create_dir_all("../results_N400").ok();
    std::fs::create_dir_all("../results_N900").ok();
    std::fs::create_dir_all("../results_N1600").ok();

    // N=400: 20 additional trials (existing: 10, need 30 total)
    // Seeds: 400020-400039 (avoid collision with any existing)
    let res_400 = run_validation_custom(400, 20, 400020);
    let json_400 = serde_json::to_string_pretty(&res_400).unwrap();
    let mut file_400 = File::create("../results_N400/validation_N400_additional.json").unwrap();
    file_400.write_all(json_400.as_bytes()).unwrap();

    // N=900: 25 additional trials (existing: 5, need 30 total)
    // Seeds: 900005-900029
    let res_900 = run_validation_custom(900, 25, 900005);
    let json_900 = serde_json::to_string_pretty(&res_900).unwrap();
    let mut file_900 = File::create("../results_N900/validation_N900_additional.json").unwrap();
    file_900.write_all(json_900.as_bytes()).unwrap();

    // N=1600: 27 additional trials (existing: 3, need 30 total)
    // Seeds: 1600003-1600029
    let res_1600 = run_validation_custom(1600, 27, 1600003);
    let json_1600 = serde_json::to_string_pretty(&res_1600).unwrap();
    let mut file_1600 = File::create("../results_N1600/validation_N1600_additional.json").unwrap();
    file_1600.write_all(json_1600.as_bytes()).unwrap();

    // Summary
    println!("\n{}", "=".repeat(70));
    println!("EXTENDED VALIDATION COMPLETE");
    println!("{}", "=".repeat(70));
    println!(
        "N=400:  {:.1}% precursor ({}/{}), {:.1}s/trial",
        res_400.precursor_rate * 100.0,
        res_400.n_precursor,
        res_400.n_detected,
        res_400.time_per_trial_sec
    );
    println!(
        "N=900:  {:.1}% precursor ({}/{}), {:.1}s/trial",
        res_900.precursor_rate * 100.0,
        res_900.n_precursor,
        res_900.n_detected,
        res_900.time_per_trial_sec
    );
    println!(
        "N=1600: {:.1}% precursor ({}/{}), {:.1}s/trial",
        res_1600.precursor_rate * 100.0,
        res_1600.n_precursor,
        res_1600.n_detected,
        res_1600.time_per_trial_sec
    );
    println!("{}", "=".repeat(70));
}
