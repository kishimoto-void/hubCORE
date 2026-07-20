#!/usr/bin/env python3
"""
hubAXIOM-purecore
=================
Fully pure functional core.

Everything is now:
  State (data) + Pure Functions + Delayed Events

Core itself is pure:
  step_once(CoreState, world_state) → (new_CoreState, new_world_state)

Axioms remain:
  A0 Constraint
  A1 Clock          (pure)
  A2 Time Delay
  A3 Memory
  A4 Metric
  A5 Dynamics
  A6 Observation
  A7 Mode
"""

import random
import math
import cmath
from dataclasses import dataclass, replace
from typing import List, Tuple, Callable, Optional, Sequence

# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
@dataclass(frozen=True)
class State:
    coordinates: Tuple[float, ...]

@dataclass(frozen=True)
class VectorField:
    values: Tuple[float, ...]

@dataclass(frozen=True)
class ClockState:
    z: complex
    cycle: int

@dataclass(frozen=True)
class ZeroClockEvent:
    cycle: int
    phase: float
    radius: float
    emit_step: int

@dataclass(frozen=True)
class CoreState:
    clock: ClockState
    residue: State
    essence: State
    echo: State
    mode: str
    pending: Tuple[Tuple[int, ZeroClockEvent], ...]   # immutable queue
    step: int

Constraint = Callable[[State], float]
EnergyGrad = Callable[[State], VectorField]

MODES = ("Normal", "Caution", "Recovery", "Emergency")
TWO_PI = 2.0 * math.pi
FADE = 0.05
COMPRESS_RATE = 0.20
LAYER_LAG = 3
SCALE_FACTORS = {"Normal": 1.0, "Caution": 0.42, "Recovery": 0.14, "Emergency": 0.03}

# ------------------------------------------------------------------
# A0 Constraint
# ------------------------------------------------------------------
def create_invariant(initial: State) -> Constraint:
    target = sum(initial.coordinates)
    def c(s: State) -> float:
        return sum(s.coordinates) - target
    return c

def normal(s: State) -> Tuple[float, ...]:
    return tuple(1.0 for _ in s.coordinates)

def project_tangent(vec: Tuple[float, ...], n: Tuple[float, ...]) -> Tuple[float, ...]:
    dv = sum(v * ni for v, ni in zip(vec, n))
    nn = sum(ni * ni for ni in n)
    f = dv / max(nn, 1e-12)
    return tuple(v - f * ni for v, ni in zip(vec, n))

def project(s: State, constraints: Sequence[Constraint]) -> State:
    cur = s
    for cfn in constraints:
        viol = cfn(cur)
        if abs(viol) > 1e-10:
            n = normal(cur)
            nn = sum(x * x for x in n)
            corr = tuple((viol / nn) * x for x in n)
            cur = State(tuple(a - b for a, b in zip(cur.coordinates, corr)))
    return cur

def energy_grad(s: State) -> VectorField:
    return VectorField(s.coordinates)

# ------------------------------------------------------------------
# A1 Pure Clock
# ------------------------------------------------------------------
def clock_tick(cs: ClockState, mode: str, step: int) -> Tuple[ClockState, Optional[ZeroClockEvent]]:
    old_phase = cmath.phase(cs.z) % TWO_PI
    r = abs(cs.z)
    if mode == "Recovery":
        omega, radial = 0.198, 0.068
    elif mode == "Emergency":
        omega, radial = 0.072, 0.016
    elif mode == "Caution":
        omega, radial = 0.148, 0.033
    else:
        omega, radial = 0.118, 0.014
    new_r = max(0.04, min(1.05, r + radial * (1.0 - r) - (0.006 if mode in ("Normal", "Caution") else 0.0)))
    new_theta = (cmath.phase(cs.z) + omega) % TWO_PI
    new_z = cmath.rect(new_r, new_theta)
    event = None
    new_cycle = cs.cycle
    if (old_phase > math.pi) and (new_theta < math.pi) and new_r > 0.80:
        new_cycle = cs.cycle + 1
        event = ZeroClockEvent(new_cycle, new_theta, new_r, step)
    return ClockState(new_z, new_cycle), event

def clock_radius(cs: ClockState) -> float:
    return abs(cs.z)

def clock_phase(cs: ClockState) -> float:
    return cmath.phase(cs.z) % TWO_PI

# ------------------------------------------------------------------
# A3 Memory
# ------------------------------------------------------------------
def compress(residue: State, essence: State) -> State:
    return State(tuple((1 - COMPRESS_RATE) * e + COMPRESS_RATE * r
                       for e, r in zip(essence.coordinates, residue.coordinates)))

def fade(residue: State) -> State:
    return State(tuple(FADE * r for r in residue.coordinates))

# ------------------------------------------------------------------
# A4–A5 Metric + Dynamics
# ------------------------------------------------------------------
def resistance(s: State, residue: State, recovery: float, phase: float) -> Tuple[float, ...]:
    n = len(s.coordinates)
    rn = math.sqrt(sum(r * r for r in residue.coordinates)) + 1e-8
    rf = 1.0 / max(recovery, 0.05)
    pf = 0.16 * (0.5 + 0.5 * math.cos(phase + 0.55))
    return tuple(max(0.12, 0.64 + 0.10 * abs(s.coordinates[i])
                     + 0.29 * abs(residue.coordinates[i]) + 0.05 * rn
                     + 0.46 * rf + pf) for i in range(n))

def mobility(g: Tuple[float, ...], recovery: float) -> Tuple[float, ...]:
    raw = [1.0 / x for x in g]
    mean = sum(raw) / len(raw) + 1e-12
    s = 0.47 * (0.25 + 0.75 * recovery)
    return tuple(m * max(0.27, min(2.05, 1.0 + s * (m / mean - 1.0))) for m in raw)

def velocity(s: State, residue: State, recovery: float, phase: float,
             dt: float, mode: str) -> VectorField:
    g = resistance(s, residue, recovery, phase)
    m = mobility(g, recovery)
    t = project_tangent(energy_grad(s).values, normal(s))
    adapt = {"Normal": 0.54 + 0.46 * recovery,
             "Caution": 0.32 + 0.24 * recovery,
             "Recovery": 0.12 + 0.07 * recovery,
             "Emergency": 0.045 + 0.02 * recovery}[mode]
    return VectorField(tuple(-mi * ti * dt * adapt for mi, ti in zip(m, t)))

def integrate(s: State, vel: VectorField, dt: float) -> State:
    return State(tuple(x + dt * v for x, v in zip(s.coordinates, vel.values)))

def obs_prob(phase: float, mode: str) -> float:
    b = 0.5 + 0.5 * math.cos(phase)
    return {"Emergency": 0.10, "Recovery": 0.24, "Caution": 0.50, "Normal": 0.68}[mode] * b

def difference(s: State, residue: State, scale: float, rng: random.Random,
               mode: str, observe: bool) -> Tuple[float, ...]:
    n = len(s.coordinates)
    if not observe:
        return tuple(0.0 for _ in range(n))
    sc = scale * SCALE_FACTORS[mode]
    xi = [rng.gauss(0.0, sc) for _ in range(n)]
    d = list(project_tangent(xi, normal(s)))
    for i in range(n):
        d[i] += 0.06 * residue.coordinates[i]
    return tuple(d)

def update_residue(residue: State, delta: Tuple[float, ...]) -> State:
    return State(tuple(0.925 * r + 0.075 * d for r, d in zip(residue.coordinates, delta)))

def update_echo(echo: State, current: State) -> State:
    return State(tuple(0.952 * e + 0.048 * c for e, c in zip(echo.coordinates, current.coordinates)))

def anomaly(current: State, echo: State, predicted: State) -> float:
    res = math.sqrt(sum((c - e)**2 for c, e in zip(current.coordinates, echo.coordinates)))
    exp = math.sqrt(sum((p - e)**2 for p, e in zip(predicted.coordinates, echo.coordinates)))
    return max(0.0, res - 0.53 * exp)

def decide_mode(mode: str, anom: float, recovery: float) -> str:
    if mode == "Normal":
        if anom > 0.220 or recovery < 0.19: return "Emergency"
        if anom > 0.110 or recovery < 0.37: return "Recovery"
        if anom > 0.054: return "Caution"
        return "Normal"
    if mode == "Caution":
        if anom > 0.220 or recovery < 0.16: return "Emergency"
        if anom > 0.110 or recovery < 0.34: return "Recovery"
        if anom < 0.034 and recovery > 0.76: return "Normal"
        return "Caution"
    if mode == "Recovery":
        if anom > 0.220 or recovery < 0.12: return "Emergency"
        if anom < 0.075 and recovery > 0.73: return "Caution"
        if anom < 0.034 and recovery > 0.86: return "Normal"
        return "Recovery"
    if anom < 0.034 and recovery > 0.83: return "Caution"
    if anom < 0.075 and recovery > 0.64: return "Recovery"
    return "Emergency"

# ------------------------------------------------------------------
# Pure step_once
# ------------------------------------------------------------------
def step_once(
    core: CoreState,
    world: State,
    constraints: Sequence[Constraint],
    rng: random.Random,
    dt: float = 0.80,
    scale: float = 0.017,
    lag: int = LAYER_LAG,
) -> Tuple[CoreState, State, float, float, list]:
    """
    Pure transition:
      (CoreState, world) → (new_CoreState, new_world, recovery, anomaly, applied_events)
    """
    step = core.step + 1

    # A1 Clock
    new_clock, event = clock_tick(core.clock, core.mode, step)
    recovery = clock_radius(new_clock)
    phase = clock_phase(new_clock)

    # A2 Queue with lag
    pending = list(core.pending)
    if event is not None:
        pending.append((step + lag, event))

    # A3 Apply due events
    applied = []
    new_residue = core.residue
    new_essence = core.essence
    still_pending = []
    for apply_at, ev in pending:
        if apply_at <= step:
            new_essence = compress(new_residue, new_essence)
            new_residue = fade(new_residue)
            applied.append(ev)
        else:
            still_pending.append((apply_at, ev))
    force_obs = len(applied) > 0

    # A5 Dynamics
    vel = velocity(world, new_residue, recovery, phase, dt, core.mode)
    predicted = project(integrate(world, vel, dt), constraints)

    # A6 Observation
    do_obs = force_obs or (rng.random() < obs_prob(phase, core.mode))
    delta = difference(predicted, new_residue, scale, rng, core.mode, do_obs)
    perturbed = State(tuple(x + d for x, d in zip(predicted.coordinates, delta)))
    if do_obs:
        new_residue = update_residue(new_residue, delta)

    new_echo = update_echo(core.echo, world)
    anom = anomaly(world, new_echo, predicted)
    new_mode = decide_mode(core.mode, anom, recovery)

    new_world = project(perturbed, constraints)

    new_core = CoreState(
        clock=new_clock,
        residue=new_residue,
        essence=new_essence,
        echo=new_echo,
        mode=new_mode,
        pending=tuple(still_pending),
        step=step,
    )
    return new_core, new_world, recovery, anom, applied

# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------
def create_engine(initial: State, lag: int = LAYER_LAG):
    inv = create_invariant(initial)
    constraints = [inv]
    dim = len(initial.coordinates)
    zero = State(tuple(0.0 for _ in range(dim)))

    def run(steps: int = 560, print_every: int = 56, seed: int = 20260720):
        rng = random.Random(seed)
        world = project(initial, constraints)
        core = CoreState(
            clock=ClockState(1.0 + 0.0j, 0),
            residue=zero,
            essence=zero,
            echo=world,
            mode="Normal",
            pending=(),
            step=0,
        )

        print("╔═══════════════════════════════════════════════════════════════════════════════╗")
        print("║  hubAXIOM-purecore  —  Fully pure functional Core                         ║")
        print("║  step_once(CoreState, world) → (CoreState, world)                         ║")
        print("╚═══════════════════════════════════════════════════════════════════════════════╝\n")

        mode_counts = {m: 0 for m in MODES}
        max_anom = 0.0
        applied_count = 0

        for t in range(steps):
            core, world, rec, anom, applied = step_once(
                core, world, constraints, rng, lag=lag
            )
            mode_counts[core.mode] += 1
            max_anom = max(max_anom, anom)
            applied_count += len(applied)

            if t % print_every == 0 or t == steps - 1 or applied:
                coords = ", ".join(f"{x:6.3f}" for x in world.coordinates[:4])
                ph = clock_phase(core.clock) * 180 / math.pi
                flag = f"  << APPLY ZERO #{applied[0].cycle} >>" if applied else ""
                print(f"Step {t:03d} | [{coords}...] | rec={rec:5.3f} | φ={ph:5.1f}° | cyc={core.clock.cycle} | {core.mode}{flag}")

        print("\nInvariant preserved :", abs(constraints[0](world)) < 1e-6)
        print(f"Final Recovery      : {clock_radius(core.clock):.4f}")
        print(f"Final Cycle         : {core.clock.cycle}")
        print(f"Zero applied        : {applied_count}")
        print(f"Max Anomaly         : {max_anom:.4f}")
        print("Mode distribution   :", mode_counts)
        return world, core.residue, core.essence, clock_radius(core.clock)

    return run

if __name__ == "__main__":
    init = State((1.31, -0.72, 0.86, -1.42, 0.50, -0.32))
    engine = create_engine(init, lag=3)
    final, res, ess, rec = engine(steps=560)
    print("\nFinal State   :", [round(x, 5) for x in final.coordinates])
    print("Final Residue :", [round(x, 5) for x in res.coordinates])
    print("Final Essence :", [round(x, 5) for x in ess.coordinates])
    print("Final Recovery:", round(rec, 4))
    print("\nCore is now fully pure: step_once(CoreState, world) → (CoreState, world)")
