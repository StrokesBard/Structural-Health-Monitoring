# Structural Health Monitoring: State Estimation & System Identification

Vibration-based damage detection for multi-storey structures, approached from two directions: estimating hidden states when the physical model is known, and identifying the model when it isn't.

Coursework for **CE725 — Vibration-Based Structural Health Monitoring**, IIT Kanpur (Spring 2026).

## Methods

**State estimation.** Continuous and discrete state-space modelling (ZOH, matrix exponential, Ralston RK2), linear Kalman filtering under displacement-, velocity-, and acceleration-only measurement sets. An 8-state augmented Extended Kalman Filter with analytically derived Jacobians performs joint state–parameter estimation; a derivative-free Unscented Kalman Filter (17 sigma points, Cholesky factorization) serves as a cross-check.

**System identification.** Markov parameters recovered from input–output data via Toeplitz least-squares, then realized into a discrete state-space model using the Eigensystem Realization Algorithm with SVD-based order selection. Modal frequencies, damping ratios, and mode shapes are extracted and validated against ground truth using percentage error and the Modal Assurance Criterion.

**Parametric modelling.** MIMO ARX estimation with train–test validation, AIC/BIC sweeps across 100 candidate model orders, and recursive least squares (rank-1 matrix-inversion-lemma update) for online model updating.

## Selected results

- ERA recovered modal parameters within **0.1%** of ground truth (MAC = 1.00), degrading to **< 1%** error and MAC ≥ 0.995 under measurement noise.
- Only 3 of 5 modes proved identifiable — the single excitation point sat near a nodal location of the two missing modes.
- Damage detection separated three days of monitoring data cleanly: healthy day at 0% frequency shift and MAC = 1.00, damaged day at **12.4% and 19.3%** frequency drops.
- The augmented EKF tracked a **25% first-storey stiffness loss** from noisy acceleration measurements alone, within a few timesteps of the event.
- With zero process noise on the parameter states, the Kalman gain collapses and the filter becomes **completely blind to the damage** — detectability is as much a tuning property as a modelling one.
- ARX reached **6.5%** residual and **7.4%** prediction error energy; BIC selected order 6 against a prediction-optimal order near 36.

## Requirements

Python 3.9+ with NumPy, SciPy, pandas, and Matplotlib. MATLAB R2021b or later for the filtering code.

## Notes

The Kalman filter assignments were completed as a group of four; both end-semester components (ERA and ARX) are individual work. Datasets are course-provided and are included where redistribution is permitted.
