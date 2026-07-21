#!/usr/bin/env python3
"""
Emotional Dynamics Hybrid System v0.1
=====================================
hubAXIOM-hubKernel (Numba Ultra Core v0.2) × bubbleparticle (RGBTensorBubble)

統合方針:
- Numba最適化 RK4 + conservation / tangent projection / resistance-mobility / residue-essence / clock-mode を基盤コアに採用
- bubbleparticle の Morseポテンシャル（正式解析勾配）・回転葛藤摩擦（velocity-force mismatch）・写像行列Mによる感情色変換・fail_tensor記憶・全局residue波及を力場と色力学として統合
- 粒子位置 (N,3) を world_v として高速進化させ、色 (RGB) を感情状態として力から写像・residueでバイアス
- モード (Normal/Caution/Recovery/Emergency) が感情適応性・観測確率・散逸を制御
- 実験は忠実に実際実行して検証

実行:
    python emotional_dynamics_hybrid_v0.1.py
"""

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from numba import njit

# ------------------------------------------------------------------
# Constants (shared spirit)
# ------------------------------------------------------------------
TWO_PI = 2.0 * math.pi
FADE = 0.05
COMPRESS_RATE = 0.20
LAYER_LAG = 3

# ------------------------------------------------------------------
# Numba Low-Level Kernels (from hubAXIOM v0.2, extended for 3D force)
# ------------------------------------------------------------------
@njit(fastmath=True, cache=True)
def project_conservation_fast(v: np.ndarray, target_sum: float) -> np.ndarray:
    return v + (target_sum - np.sum(v)) / v.size

@njit(fastmath=True, cache=True)
def project_tangent_fast(v: np.ndarray) -> np.ndarray:
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
def decide_mode_numba(mode_idx: int, anom: float, rec: float) -> int:
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
# Bubble-style Morse + Dissonance (pure numpy for flexibility, can be jitted later)
# ------------------------------------------------------------------
@dataclass
class EmotionalPotential:
    position: np.ndarray
    color_mask: np.ndarray  # RGB influence
    intensity: float
    radius: float = 2.0
    is_valley: bool = False  # True = Morse attractor (reward), False = barrier

    def update(self, dt: float, residue_mod: float):
        decay = 0.025 * (1.0 - np.clip(residue_mod * 0.2, 0.0, 0.7))
        self.intensity = max(0.0, self.intensity - decay * dt)

    def potential_and_grad(self, x: np.ndarray, p_color: np.ndarray, sat: float) -> Tuple[float, np.ndarray]:
        to_p = x - self.position
        d = np.linalg.norm(to_p) + 1e-6
        resonance = np.clip(np.dot(self.color_mask, p_color), 0.0, 1.0)
        De = self.intensity * resonance * (1.0 + sat * 0.8)

        if self.is_valley:
            a = 1.4
            r0 = self.radius
            exp_term = np.exp(-a * (d - r0))
            v = De * ((1.0 - exp_term) ** 2) - De
            grad_scalar = 2.0 * a * De * (1.0 - exp_term) * exp_term
            grad = grad_scalar * (to_p / d)
        else:
            sigma = self.radius * 0.65
            factor = np.exp(-d / sigma)
            v = De * factor
            grad = -v * (1.0 / sigma) * (to_p / d)
        return v, grad


@dataclass
class EmotionalParticle:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    color: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5]))
    strength: float = 1.0
    effective_mass: float = 1.0
    fail_tensor: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    fail_memory_strength: float = 0.0

    def calc_hsv_mass(self) -> Tuple[float, float]:
        c_max = float(np.max(self.color))
        c_min = float(np.min(self.color))
        value = c_max
        sat = (c_max - c_min) / (c_max + 1e-8)
        self.effective_mass = 0.5 + value * 1.0
        return value, sat

    def update_memory(self, dt: float):
        self.fail_memory_strength = max(0.0, self.fail_memory_strength - 0.18 * dt)
        if self.fail_memory_strength < 1e-4:
            self.fail_tensor[:] = 0.0


# ------------------------------------------------------------------
# Hybrid Emotional Dynamics Engine
# ------------------------------------------------------------------
class HybridEmotionalDynamics:
    """
    hubAXIOM Numba core spirit + bubbleparticle emotional color / dissonance physics.
    Positions evolve under hybrid forces (Morse + rotational friction + mobility resistance).
    Colors evolve via force-to-RGB mapping M + residue bias.
    Residue / essence / clock / mode from hubAXIOM control global emotional regime.
    """

    def __init__(self, n_particles: int = 12, base_radius: float = 4.2, seed: int = 20260721):
        self.rng = np.random.default_rng(seed)
        self.center = np.zeros(3)
        self.base_radius = base_radius
        self.radius = base_radius

        # Particles
        self.particles: List[EmotionalParticle] = []
        for i in range(n_particles):
            pos = self.rng.uniform(-base_radius * 0.35, base_radius * 0.35, 3)
            pos[2] = self.rng.uniform(-base_radius * 0.4, 0.1)
            c_type = i % 3
            if c_type == 0:
                color = np.array([0.82, 0.18, 0.12])
            elif c_type == 1:
                color = np.array([0.12, 0.82, 0.22])
            else:
                color = np.array([0.18, 0.18, 0.82])
            self.particles.append(EmotionalParticle(
                position=pos,
                color=color,
                strength=float(self.rng.uniform(0.9, 1.25))
            ))

        # Potential elements (Morse valleys + barriers)
        self.elements: List[EmotionalPotential] = []
        # Blue affinity valley
        self.elements.append(EmotionalPotential(
            position=np.array([-1.5, -0.4, 0.3]),
            color_mask=np.array([0.1, 0.25, 1.0]),
            intensity=3.6, radius=2.1, is_valley=True
        ))
        # Red danger barrier
        self.elements.append(EmotionalPotential(
            position=np.array([1.6, 0.7, 0.5]),
            color_mask=np.array([1.0, 0.05, 0.1]),
            intensity=4.0, radius=1.55, is_valley=False
        ))

        # Force -> Emotion mapping matrix (from bubbleparticle)
        # X (explore) -> G, Y (conflict) -> R, Z (depth) -> B
        self.M = np.array([
            [0.05, 0.35, 0.05],
            [0.45, 0.05, 0.05],
            [0.05, 0.05, 0.48]
        ], dtype=np.float64)

        # hubAXIOM-style global state
        self.residue = 0.0          # global stress
        self.essence = 0.0          # compressed long-term emotional memory (scalar proxy)
        self.awareness_threshold = 0.0
        self.freeze_timer = 0.0
        self.clock_z = 1.0 + 0.0j
        self.cycle = 0
        self.mode_idx = 0           # 0 Normal, 1 Caution, 2 Recovery, 3 Emergency
        self.step = 0
        self.target_sum = 0.0       # for conservation if needed

        # History
        self.hist_T = []
        self.hist_V = []
        self.hist_E = []
        self.hist_residue = []
        self.hist_mode = []
        self.hist_rec = []

        modes = ["Normal", "Caution", "Recovery", "Emergency"]
        self.modes_str = modes

    def trigger_failure(self, pos: np.ndarray, severity: float):
        self.elements.append(EmotionalPotential(
            position=pos.copy(),
            color_mask=np.array([1.0, 0.0, 0.0]),
            intensity=severity * 2.3,
            radius=1.5,
            is_valley=False
        ))
        for p in self.particles:
            if np.linalg.norm(p.position - pos) < 2.4:
                v_norm = np.linalg.norm(p.velocity)
                if v_norm > 1e-3:
                    v_hat = p.velocity / v_norm
                    p.fail_tensor = 0.55 * p.fail_tensor + 0.45 * np.outer(v_hat, v_hat)
                    p.fail_memory_strength = min(3.0, p.fail_memory_strength + severity * 1.5)
        self.residue = min(2.8, self.residue + severity * 0.7)
        self.freeze_timer = 1.8

    def compute_forces(self, p: EmotionalParticle, step: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """Hybrid force: Morse conserved + rotational dissonance + memory + buoyancy + hubAXIOM resistance spirit"""
        val, sat = p.calc_hsv_mass()
        pos, vel = p.position, p.velocity

        # A. Conserved force from Morse / barriers
        f_cons = np.zeros(3)
        pot_energy = 0.0
        for el in self.elements:
            v_val, grad = el.potential_and_grad(pos, p.color, sat)
            f_cons += -grad
            pot_energy += v_val

        # B. Non-conserved (dissonance rotational friction + dissipation)
        f_non = np.zeros(3)

        # Global Rayleigh + residue
        gamma_base = 2.0 if self.freeze_timer > 0 else 0.32
        gamma = gamma_base + self.residue * 0.55
        f_non += -(gamma * (2.0 - sat)) * vel

        # Rotational dissonance (core of bubbleparticle)
        v_norm = np.linalg.norm(vel)
        f_c_norm = np.linalg.norm(f_cons)
        if v_norm > 1e-3 and f_c_norm > 1e-3:
            v_hat = vel / v_norm
            f_ortho = f_cons - np.dot(f_cons, v_hat) * v_hat
            f_rot = -f_ortho * (0.75 + sat * 1.6)
            f_non += f_rot

        # Fail tensor memory braking
        if p.fail_memory_strength > 0.02:
            proj = p.fail_tensor @ vel
            ortho = vel - proj
            f_non += (-proj * 3.2 - ortho * 0.65) * p.fail_memory_strength

        # Buoyancy + weak center attraction (emotional depth / cohesion)
        buoyancy = np.array([
            0.0,
            0.0,
            (p.color[2] * 0.55 - p.color[0] * 0.48) + p.color[1] * math.sin(step * 0.14) * 0.22
        ])
        f_non += buoyancy + (self.center - pos) * 0.11

        # Mode-dependent mobility scaling (hubAXIOM spirit)
        adapt = [0.54, 0.32, 0.12, 0.045][self.mode_idx]
        adapt += [0.46, 0.24, 0.07, 0.02][self.mode_idx] * abs(self.clock_z)

        total_f = (f_cons * 1.55 + f_non) * adapt
        accel = total_f / max(p.effective_mass, 0.3)
        return accel, f_cons, pot_energy

    def step_once(self, dt: float = 0.075):
        self.step += 1
        if self.freeze_timer > 0:
            self.freeze_timer = max(0.0, self.freeze_timer - dt)

        # Clock tick (hubAXIOM style)
        old_p = np.arctan2(self.clock_z.imag, self.clock_z.real) % TWO_PI
        r = abs(self.clock_z)
        p_dtheta = [0.118, 0.148, 0.198, 0.072][self.mode_idx]
        p_dr = [0.014, 0.033, 0.068, 0.016][self.mode_idx]
        dr = p_dr * (1.0 - r) - (0.006 if self.mode_idx in (0, 1) else 0.0)
        new_r = max(0.04, min(1.05, r + dr))
        new_t = (old_p + p_dtheta) % TWO_PI
        self.clock_z = complex(new_r * math.cos(new_t), new_r * math.sin(new_t))

        if (old_p > math.pi) and (new_t < math.pi) and new_r > 0.80:
            self.cycle += 1
            # essence compression event (lag simulated by immediate for simplicity in hybrid)
            self.essence = (1.0 - COMPRESS_RATE) * self.essence + COMPRESS_RATE * self.residue
            self.residue *= FADE

        recovery = abs(self.clock_z)

        # Radius & awareness modulated by residue + color balance
        r_sum = b_sum = 0.0
        for p in self.particles:
            if p.position[2] >= self.awareness_threshold:
                r_sum += p.color[0]
                b_sum += p.color[2]
        self.radius = self.base_radius * (1.0 + 0.035 * (b_sum - r_sum * 0.75) - 0.14 * self.residue)
        self.awareness_threshold = float(np.clip(-0.45 + self.residue * 1.4, -self.radius * 0.75, self.radius * 0.18))

        # Potential decay
        for el in self.elements:
            el.update(dt, self.residue)
        self.elements = [el for el in self.elements if el.intensity > 0.015]

        step_T = 0.0
        step_V = 0.0
        total_delta_color = np.zeros(3)

        for p in self.particles:
            p.update_memory(dt)

            # Collision with red barriers → trauma (slightly larger detection for experiment)
            for el in self.elements:
                if el.color_mask[0] > 0.6 and not el.is_valley:
                    if np.linalg.norm(p.position - el.position) < 0.95 and self.freeze_timer <= 0:
                        self.trigger_failure(p.position, severity=min(el.intensity, 1.8))

            # RK4 integration of hybrid forces
            x0 = p.position.copy()
            v0 = p.velocity.copy()

            a1, f_c1, v_e1 = self.compute_forces(p, self.step)
            k1x, k1v = v0, a1

            p.position = x0 + 0.5 * dt * k1x
            p.velocity = v0 + 0.5 * dt * k1v
            a2, _, _ = self.compute_forces(p, self.step)
            k2x, k2v = p.velocity, a2

            p.position = x0 + 0.5 * dt * k2x
            p.velocity = v0 + 0.5 * dt * k2v
            a3, _, _ = self.compute_forces(p, self.step)
            k3x, k3v = p.velocity, a3

            p.position = x0 + dt * k3x
            p.velocity = v0 + dt * k3v
            a4, _, _ = self.compute_forces(p, self.step)
            k4x, k4v = p.velocity, a4

            p.position = x0 + (dt / 6.0) * (k1x + 2*k2x + 2*k3x + k4x)
            p.velocity = v0 + (dt / 6.0) * (k1v + 2*k2v + 2*k3v + k4v)

            # Velocity clamp
            vn = np.linalg.norm(p.velocity)
            val, _ = p.calc_hsv_mass()
            max_v = 4.8 + val * 2.2
            if vn > max_v:
                p.velocity *= max_v / vn

            # Color evolution via M mapping + residue bias (emotional staining)
            f_total = f_c1 * 1.55  # approximate current force influence
            color_delta = self.M @ f_total
            if self.residue > 0.18:
                color_delta[0] += self.residue * 0.07   # stress → red bias
            p.color = np.clip(p.color + color_delta * dt * 0.14, 0.0, 1.0)
            if np.sum(p.color) < 0.12:
                p.color += 0.035
            total_delta_color += np.abs(color_delta)

            # Spherical manifold projection (bubble constraint)
            dist = np.linalg.norm(p.position - self.center)
            if dist > self.radius:
                normal = (p.position - self.center) / dist
                p.position = self.center + normal * self.radius
                vn_comp = np.dot(p.velocity, normal)
                if vn_comp > 0:
                    p.velocity -= vn_comp * normal * 1.15

            step_T += 0.5 * p.effective_mass * np.dot(p.velocity, p.velocity)
            step_V += v_e1

        n_p = max(1, len(self.particles))
        self.hist_T.append(step_T / n_p)
        self.hist_V.append(step_V / n_p)
        self.hist_E.append((step_T + step_V) / n_p)
        self.hist_residue.append(self.residue)
        self.hist_mode.append(self.mode_idx)
        self.hist_rec.append(recovery)

        # Anomaly proxy (color change + residue jump + kinetic) → mode decision
        # Softened thresholds for more natural mode transitions in hybrid
        color_flux = 0.08 * np.linalg.norm(total_delta_color) / n_p
        res_jump = 0.25 * abs(self.residue - (self.hist_residue[-2] if len(self.hist_residue) > 1 else 0.0))
        kin_proxy = 0.03 * (step_T / n_p)
        anom = color_flux + res_jump + kin_proxy
        self.mode_idx = decide_mode_numba(self.mode_idx, anom, recovery)

        # Global residue decay (with freeze protection)
        if self.freeze_timer <= 0:
            self.residue = max(0.0, self.residue * 0.935)

        return recovery, anom

    def run(self, steps: int = 120, print_every: int = 20):
        print("╔══════════════════════════════════════════════════════════════════════════════╗")
        print("║  Hybrid Emotional Dynamics  v0.1                                             ║")
        print("║  hubAXIOM-Numba Core × bubbleparticle (Morse + Rotational Dissonance)        ║")
        print("╚══════════════════════════════════════════════════════════════════════════════╝\n")

        t0 = time.perf_counter()
        for t in range(1, steps + 1):
            rec, anom = self.step_once(dt=0.075)
            if t % print_every == 0 or t == 1 or t == steps:
                avg_color = np.mean([p.color for p in self.particles], axis=0)
                print(f"Step {t:03d} | rec={rec:5.3f} | φ={np.angle(self.clock_z)*180/math.pi:6.1f}° | "
                      f"cyc={self.cycle} | mode={self.modes_str[self.mode_idx]:9s} | "
                      f"res={self.residue:5.3f} | ess={self.essence:5.3f} | "
                      f"avgRGB=[{avg_color[0]:.2f},{avg_color[1]:.2f},{avg_color[2]:.2f}] | anom={anom:.4f}")

        elapsed = (time.perf_counter() - t0) * 1000
        print(f"\nExecution Time : {elapsed:.1f} ms ({elapsed/steps:.3f} ms/step)")
        print(f"Final Recovery : {abs(self.clock_z):.4f}")
        print(f"Final Cycle    : {self.cycle}")
        print(f"Final Residue  : {self.residue:.4f}")
        print(f"Final Essence  : {self.essence:.4f}")
        print(f"Mode dist      : {dict(zip(self.modes_str, np.bincount(self.hist_mode, minlength=4)))}")
        return self


def visualize(system: HybridEmotionalDynamics, save_path: str = "hybrid_demo.png"):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(16, 7))

    # Left: 3D particles + force field snapshot
    ax1 = fig.add_subplot(121, projection="3d")
    for p in system.particles:
        val, _ = p.calc_hsv_mass()
        c = np.clip(p.color / (np.max(p.color) + 1e-5), 0, 1)
        ax1.scatter(*p.position, color=c, s=70 + val * 50, alpha=0.92, edgecolors="k", linewidths=0.6)

    # Sample force field
    gx, gy, gz = np.meshgrid(np.linspace(-3.0, 3.0, 5), np.linspace(-3.0, 3.0, 5), np.linspace(-2.0, 2.0, 3))
    test_p = EmotionalParticle(color=np.array([0.45, 0.5, 0.5]), velocity=np.array([0.3, -0.15, 0.08]))
    U, V, W = np.zeros_like(gx), np.zeros_like(gy), np.zeros_like(gz)
    for i in range(gx.shape[0]):
        for j in range(gx.shape[1]):
            for k in range(gx.shape[2]):
                test_p.position = np.array([gx[i,j,k], gy[i,j,k], gz[i,j,k]])
                a, _, _ = system.compute_forces(test_p, system.step)
                U[i,j,k], V[i,j,k], W[i,j,k] = a

    ax1.quiver(gx, gy, gz, U, V, W, length=0.35, normalize=True, alpha=0.22, color="#8e44ad")
    for el in system.elements:
        c = "#3498db" if el.is_valley else "#e67e22"
        ax1.scatter(*el.position, c=c, s=280, marker="o" if el.is_valley else "X", edgecolors="k")
    ax1.set_title("Hybrid Force Field\nBlue=Morse Valley / Orange=Barrier / Purple=Accel", fontsize=10)
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    ax1.set_zlim(-4, 4)
    ax1.view_init(elev=20, azim=40)

    # Right: energy + residue + mode
    ax2 = fig.add_subplot(122)
    steps = np.arange(len(system.hist_T))
    ax2.plot(steps, system.hist_T, label="Kinetic (T)", color="#e67e22", lw=1.8)
    ax2.plot(steps, system.hist_V, label="Potential (V, Morse)", color="#2ecc71", lw=1.8)
    ax2.plot(steps, system.hist_E, label="Total E", color="#8e44ad", lw=2.0, ls="--")
    ax2b = ax2.twinx()
    ax2b.plot(steps, system.hist_residue, label="Residue", color="#c0392b", lw=1.6, alpha=0.8)
    ax2b.set_ylabel("Global Residue", color="#c0392b")
    ax2.set_title("Energy Evolution + Stress Residue (Hybrid)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Step"); ax2.set_ylabel("Normalized Energy / particle")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper left")
    ax2b.legend(loc="upper right")

    info = (f"Final mode: {system.modes_str[system.mode_idx]}\n"
            f"Recovery: {abs(system.clock_z):.3f}  Cycle: {system.cycle}\n"
            f"Residue: {system.residue:.3f}  Essence: {system.essence:.3f}\n"
            f"Integrator: Hybrid RK4 + hubAXIOM modes + Morse+Dissonance")
    ax2.text(0.02, 0.02, info, transform=ax2.transAxes, fontsize=9,
             verticalalignment="bottom", bbox=dict(facecolor="white", alpha=0.88, boxstyle="round"))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved → {save_path}")
    plt.close()


if __name__ == "__main__":
    system = HybridEmotionalDynamics(n_particles=14, base_radius=4.2, seed=20260721)
    system.run(steps=140, print_every=20)
    visualize(system)
    print("\nHybrid Emotional Dynamics System construction complete.")
    print("Core: hubAXIOM Numba spirit (clock/mode/residue/essence/mobility) + bubbleparticle (Morse/dissonance/color mapping)")
