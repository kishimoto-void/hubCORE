#!/usr/bin/env python3
"""
hubAXIOM-hubKernel (Ultra-Fast Numba Core v0.2 - Complete)
========================================================
Thorough Numba optimization + minimal allocation + analytic projection.
100% complete, fully pure functional-compatible, and executable single-file core.

Version: 0.2
"""

import math, cmath, time
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from numba import njit, float64, int64

# 定数定義
TWO_PI = 2.0 * math.pi
FADE = 0.05
COMPRESS_RATE = 0.20
LAYER_LAG = 3

# ------------------------------------------------------------------
# Numba Low-Level Kernels (Zero-Allocation Numerics)
# ------------------------------------------------------------------
@njit(fastmath=True, cache=True)
def project_conservation_fast(v: np.ndarray, target_sum: float) -> np.ndarray:
    """O(N) Analytic projection onto sum-conservation constraint."""
    return v + (target_sum - np.sum(v)) / v.size

@njit(fastmath=True, cache=True)
def project_tangent_fast(v: np.ndarray) -> np.ndarray:
    """O(N) Analytic tangent projection (zero-mean operator)."""
    return v - np.mean(v)

@njit(fastmath=True, cache=True)
def resistance_numba(s_v: np.ndarray, residue_v: np.ndarray, recovery: float, phase: float) -> np.ndarray:
    rn = np.linalg.norm(residue_v) + 1e-8
    rf = 1.0 / max(recovery, 0.05)
    pf = 0.16 * (0.5 + 0.5 * np.cos(phase + 0.55))
    res = 0.64 + 0.10 * np.abs(s_v) + 0.29 * np.abs(residue_v) + 0.05 * rn + 0.46 * rf + pf
    return np.maximum(0.12, res)

@njit(fastmath=True, cache=True)
def mobility_numba(g: np.ndarray, recovery: float) -> np.ndarray:
    raw = 1.0 / g
    m_mean = np.mean(raw) + 1e-12
    s = 0.47 * (0.25 + 0.75 * recovery)
    factor = np.clip(1.0 + s * (raw / m_mean - 1.0), 0.27, 2.05)
    return raw * factor

@njit(fastmath=True, cache=True)
def velocity_kernel(s_v: np.ndarray, residue_v: np.ndarray, recovery: float, phase: float, 
                    dt: float, adapt: float, W: np.ndarray) -> np.ndarray:
    g = resistance_numba(s_v, residue_v, recovery, phase)
    m = mobility_numba(g, recovery)
    interaction = W @ s_v
    total = s_v + 0.15 * interaction
    t = project_tangent_fast(total)
    return -m * t * dt * adapt

@njit(fastmath=True, cache=True)
def rk4_step_kernel(s_v: np.ndarray, residue_v: np.ndarray, recovery: float, phase: float, 
                    dt: float, adapt: float, W: np.ndarray, target_sum: float) -> np.ndarray:
    k1 = velocity_kernel(s_v, residue_v, recovery, phase, dt, adapt, W)
    tmp1 = project_conservation_fast(s_v + 0.5 * k1, target_sum)
    k2 = velocity_kernel(tmp1, residue_v, recovery, phase, dt, adapt, W)
    tmp2 = project_conservation_fast(s_v + 0.5 * k2, target_sum)
    k3 = velocity_kernel(tmp2, residue_v, recovery, phase, dt, adapt, W)
    tmp3 = project_conservation_fast(s_v + k3, target_sum)
    k4 = velocity_kernel(tmp3, residue_v, recovery, phase, dt, adapt, W)
    
    next_v = s_v + (1.0 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return project_conservation_fast(next_v, target_sum)

@njit(fastmath=True, cache=True)
def decide_mode_numba(mode_idx: int, anom: float, rec: float) -> int:
    # 0:Normal, 1:Caution, 2:Recovery, 3:Emergency
    if mode_idx == 0:
        if anom > 0.22 or rec < 0.19: return 3
        if anom > 0.11 or rec < 0.37: return 2
        if anom > 0.054: return 1
        return 0
    elif mode_idx == 1:
        if anom > 0.22 or rec < 0.16: return 3
        if anom > 0.11 or rec < 0.34: return 2
        if anom < 0.034 and rec > 0.76: return 0
        return 1
    elif mode_idx == 2:
        if anom > 0.22 or rec < 0.12: return 3
        if anom < 0.034 and rec > 0.86: return 0
        if anom < 0.075 and rec > 0.73: return 1
        return 2
    else:
        if anom < 0.034 and rec > 0.83: return 1
        if anom < 0.075 and rec > 0.64: return 2
        return 3

# ------------------------------------------------------------------
# Main Fast Kernel (Full Inlined Step Logic)
# ------------------------------------------------------------------
@njit(fastmath=True, cache=True)
def step_kernel_fast(world_v: np.ndarray, residue_v: np.ndarray, essence_v: np.ndarray, echo_v: np.ndarray,
                     clock_re: float, clock_im: float, cycle: int, mode_idx: int, step: int,
                     target_sum: float, W: np.ndarray, dt: float, scale: float,
                     pending_apply_at: np.ndarray, pending_cycles: np.ndarray, pending_offsets: np.ndarray, pending_count: int):
    
    # 1. Clock Tick
    old_p = np.arctan2(clock_im, clock_re) % TWO_PI
    r = np.hypot(clock_re, clock_im)
    
    # Clock parameters per mode (0:Normal, 1:Caution, 2:Recovery, 3:Emergency)
    p_dtheta = np.array([0.118, 0.148, 0.198, 0.072])[mode_idx]
    p_dr = np.array([0.014, 0.033, 0.068, 0.016])[mode_idx]
    
    dr = p_dr * (1.0 - r) - (0.006 if mode_idx in (0, 1) else 0.0)
    new_r = max(0.04, min(1.05, r + dr))
    new_t = (old_p + p_dtheta) % TWO_PI
    
    new_clock_re = new_r * np.cos(new_t)
    new_clock_im = new_r * np.sin(new_t)
    new_cycle = cycle
    
    has_event = False
    event_cycle = 0
    event_offset = 0.0
    
    if (old_p > math.pi) and (new_t < math.pi) and new_r > 0.80:
        new_cycle += 1
        d_theta = (TWO_PI - old_p) + new_t
        alpha = (TWO_PI - old_p) / max(d_theta, 1e-12) if d_theta > 0 else 0.5
        has_event = True
        event_cycle = new_cycle
        event_offset = round(alpha, 4)
        
        # Enqueue pending
        if pending_count < len(pending_apply_at):
            pending_apply_at[pending_count] = step + LAYER_LAG
            pending_cycles[pending_count] = event_cycle
            pending_offsets[pending_count] = event_offset
            pending_count += 1

    # 2. Process Pending Queue
    applied_count = 0
    applied_cycle = 0
    applied_offset = 0.0
    
    new_res = residue_v.copy()
    new_ess = essence_v.copy()
    
    read_idx = 0
    write_idx = 0
    while read_idx < pending_count:
        if pending_apply_at[read_idx] <= step:
            new_ess = (1.0 - COMPRESS_RATE) * new_ess + COMPRESS_RATE * new_res
            new_res = FADE * new_res
            applied_count += 1
            applied_cycle = pending_cycles[read_idx]
            applied_offset = pending_offsets[read_idx]
        else:
            pending_apply_at[write_idx] = pending_apply_at[read_idx]
            pending_cycles[write_idx] = pending_cycles[read_idx]
            pending_offsets[write_idx] = pending_offsets[read_idx]
            write_idx += 1
        read_idx += 1
    pending_count = write_idx

    # 3. Integrate RK4
    adapt_base = np.array([0.54, 0.32, 0.12, 0.045])[mode_idx]
    adapt_slope = np.array([0.46, 0.24, 0.07, 0.02])[mode_idx]
    adapt = adapt_base + adapt_slope * new_r
    
    predicted_v = rk4_step_kernel(world_v, new_res, new_r, new_t, dt, adapt, W, target_sum)

    # 4. Observation & Noise Injection
    obs_prob = np.array([0.68, 0.5, 0.24, 0.1])[mode_idx] * (0.5 + 0.5 * np.cos(new_t))
    do_obs = (applied_count > 0) or (np.random.random() < obs_prob)
    
    delta_v = np.zeros_like(world_v)
    if do_obs:
        sc = scale * np.array([1.0, 0.42, 0.14, 0.03])[mode_idx]
        xi = np.random.normal(0.0, sc, size=world_v.shape)
        delta_v = project_tangent_fast(xi) + 0.06 * new_res
        new_res = 0.925 * new_res + 0.075 * delta_v

    # 5. Anomaly & Mode Decision
    new_echo_v = 0.952 * echo_v + 0.048 * world_v
    
    pos_anom = max(0.0, float(np.linalg.norm(world_v - new_echo_v) - 0.53 * np.linalg.norm(predicted_v - new_echo_v)))
    vel_residual_norm = float(np.linalg.norm((predicted_v - world_v) - delta_v))
    anom = pos_anom + 0.20 * vel_residual_norm

    new_mode_idx = decide_mode_numba(mode_idx, anom, new_r)
    new_world_v = project_conservation_fast(predicted_v + delta_v, target_sum)

    return (new_world_v, new_res, new_ess, new_echo_v, new_clock_re, new_clock_im, new_cycle, 
            new_mode_idx, pending_count, new_r, new_t, anom, applied_count, applied_cycle, applied_offset)

# ------------------------------------------------------------------
# High-Level Engine Wrapper
# ------------------------------------------------------------------
@dataclass
class EngineState:
    world_v: np.ndarray
    residue: np.ndarray
    essence: np.ndarray
    echo: np.ndarray
    clock_z: complex
    cycle: int
    mode_idx: int
    step: int
    pending_apply_at: np.ndarray
    pending_cycles: np.ndarray
    pending_offsets: np.ndarray
    pending_count: int

def create_engine_ultra(initial_v: np.ndarray, W: Optional[np.ndarray] = None, lag: int = LAYER_LAG):
    N, D = initial_v.shape if initial_v.ndim == 2 else (initial_v.size, 1)
    initial_v = np.asarray(initial_v, dtype=np.float64).reshape(N, D)
    
    if W is None:
        W = np.ones((N, N), dtype=np.float64) - np.eye(N)
        if N > 1: W /= (N - 1)

    target_sum = float(np.sum(initial_v))
    zero_v = np.zeros_like(initial_v)

    def run(steps: int = 560, print_every: int = 56, seed: int = 20260720):
        np.random.seed(seed)
        
        # Static arrays for pending queue (max capacity 64)
        pending_apply_at = np.zeros(64, dtype=np.int64)
        pending_cycles = np.zeros(64, dtype=np.int64)
        pending_offsets = np.zeros(64, dtype=np.float64)
        
        state = EngineState(
            world_v=project_conservation_fast(initial_v, target_sum),
            residue=zero_v.copy(),
            essence=zero_v.copy(),
            echo=initial_v.copy(),
            clock_z=1.0 + 0.0j,
            cycle=0,
            mode_idx=0,
            step=0,
            pending_apply_at=pending_apply_at,
            pending_cycles=pending_cycles,
            pending_offsets=pending_offsets,
            pending_count=0
        )

        print("╔═══════════════════════════════════════════════════════════════════════════════╗\n║  hubAXIOM-hubKernel — Ultra Numba Optimized Core v0.2 (Complete)               ║\n╚═══════════════════════════════════════════════════════════════════════════════╝\n")

        # Warm-up JIT Compilation
        _ = step_kernel_fast(state.world_v, state.residue, state.essence, state.echo,
                             state.clock_z.real, state.clock_z.imag, state.cycle, state.mode_idx, 0,
                             target_sum, W, 0.80, 0.017,
                             state.pending_apply_at, state.pending_cycles, state.pending_offsets, 0)

        start_time = time.perf_counter()
        mode_counts = [0] * 4
        max_anom = 0.0
        total_applied = 0
        modes_str = ["Normal", "Caution", "Recovery", "Emergency"]

        for t in range(1, steps + 1):
            (state.world_v, state.residue, state.essence, state.echo,
             clock_re, clock_im, state.cycle, state.mode_idx, state.pending_count,
             rec, ph, anom, applied_count, app_cyc, app_off) = step_kernel_fast(
                state.world_v, state.residue, state.essence, state.echo,
                state.clock_z.real, state.clock_z.imag, state.cycle, state.mode_idx, t,
                target_sum, W, 0.80, 0.017,
                state.pending_apply_at, state.pending_cycles, state.pending_offsets, state.pending_count
            )
            state.clock_z = complex(clock_re, clock_im)
            state.step = t

            mode_counts[state.mode_idx] += 1
            max_anom = max(max_anom, anom)
            total_applied += applied_count

            if (t - 1) % print_every == 0 or t == steps or applied_count > 0:
                coords = ", ".join(f"{x:6.3f}" for x in state.world_v.ravel()[:4])
                flag = f"  << ZERO #{app_cyc} (offset={app_off:0.4f}) >>" if applied_count > 0 else ""
                print(f"Step {t:03d} | [{coords}...] | rec={rec:5.3f} | φ={ph*180/math.pi:5.1f}° | cyc={state.cycle} | {modes_str[state.mode_idx]}{flag}")

        elapsed = (time.perf_counter() - start_time) * 1000.0
        mode_dist = {modes_str[i]: mode_counts[i] for i in range(4)}

        print(f"\nExecution Time      : {elapsed:.2f} ms ({elapsed / steps:.3f} ms/step)")
        print(f"Invariant preserved : {abs(np.sum(state.world_v) - target_sum) < 1e-6} (Max error: {abs(np.sum(state.world_v) - target_sum):.2e})")
        print(f"Final Recovery      : {abs(state.clock_z):.4f}\nFinal Cycle         : {state.cycle}\nZero applied        : {total_applied}\nMax Anomaly         : {max_anom:.4f}\nMode distribution   : {mode_dist}")

        return state.world_v, state.residue, state.essence, abs(state.clock_z)

    return run

if __name__ == "__main__":
    init_particles = np.array([
        [1.31, -0.72],
        [0.86, -1.42],
        [0.50, -0.32]
    ], dtype=np.float64)

    W_interaction = np.array([
        [ 0.0,  0.5, -0.5],
        [-0.5,  0.0,  0.5],
        [ 0.5, -0.5,  0.0]
    ], dtype=np.float64)

    engine = create_engine_ultra(init_particles, W=W_interaction, lag=3)
    final_v, res_v, ess_v, rec = engine(steps=560)
    print("\nFinal State (Multi-Bubble):\n", np.round(final_v, 5))
    print("Final Recovery:", round(rec, 4))
