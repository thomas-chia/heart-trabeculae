"""Run one Tier 2 ventricle for several cardiac cycles.

    python -m ventricle.run --kind trab --cycles 3

Writes results/tier2/<kind>.npz (time series and last cycle snapshots).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from squeeze_film.reynolds import MU_BLOOD, RHO_BLOOD

from .geometry import Ventricle
from .solver import PenalizedNS, advect_scalar

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "tier2"

# box and grid: 75 mm by 125 mm at 0.25 mm
NX, NY, DX = 300, 500, 0.25e-3
X0, Y0 = -37.5e-3, -103e-3
SPONGE = 1e-4  # sponge mask level (penalization rate SPONGE / eta)
H_RES = 0.75e-3  # resolved apical crest gap, 3 cells at the production grid
SNAP_PHASES = {"mid_ejection": 0.12, "early_e": 0.415, "peak_e": 0.45, "diastasis": 0.60, "end_diastole": 0.795}


def run(kind: str, cycles: int = 3, dt_max: float = 1e-4, cfl: float = 0.3,
        sample_every: float = 5e-4, n_proj: int = 2, nx: int = NX, ny: int = NY,
        dx: float = DX, verbose: bool = True):
    nu = MU_BLOOD / RHO_BLOOD
    ns = PenalizedNS(nx, ny, nx * dx, ny * dx, X0, Y0, nu, n_proj=n_proj)
    v = Ventricle(kind=kind, h_res=H_RES, eps=dx)
    X, Y = ns.X, ns.Y
    period = v.cycle.period
    t_end = cycles * period

    f = v.fields(X, Y, 0.0)
    c = f["cav"].astype(float)  # tracer: 1 marks blood present at end diastole
    c0 = c.sum()

    # weak sponge at the far ends of the atrial and aortic channels and in the
    # outer box, so 2D vortices shed into the reservoirs do not recirculate for
    # whole cycles; the damping time is eta / SPONGE, about 10 ms at dt_max
    sponge = (np.clip((Y - (v.up_height - 4e-3)) / 4e-3, 0, 1)
              + np.clip((-(v.L_ed + v.th_ed + 3e-3) - Y) / 3e-3, 0, 1)
              + np.clip((np.abs(X) - 34e-3) / 2e-3, 0, 1))
    sponge = SPONGE * np.clip(sponge, 0, 1)
    in_sponge = sponge > 0

    # sampling regions (fixed in space)
    ref = (X > -18e-3) & (X < 2e-3) & (Y > 6e-3) & (Y < 12e-3)  # mitral channel
    base_box = (X > -12e-3) & (X < 0) & (Y > -8e-3) & (Y < -4e-3)
    row = np.argmin(np.abs(Y[0] + 2.5e-3))  # flux row just below the valve plane
    in_base = np.abs(X[:, row]) < v.a_base

    rec = {k: [] for k in (
        "t", "area", "q_row", "e_row", "p_base", "p_apex", "p_min", "power_wall",
        "dissipation", "ke", "residual", "residual_layer", "speed_layer",
        "speed_cav", "umax", "div_cav")}
    snaps = {}
    next_sample, t, nstep = 0.0, 0.0, 0
    wall_clock = time.time()
    while t < t_end - 1e-12:
        speed = max(ns.max_speed(), 0.15)
        dt = min(dt_max, cfl * dx / speed, t_end - t)
        f = v.fields(X, Y, t + dt)
        weak = in_sponge & (f["chi"] < SPONGE)
        chi = np.where(weak, sponge, f["chi"])
        us = np.where(weak, 0.0, f["us"])
        vs = np.where(weak, 0.0, f["vs"])
        ns.step(dt, chi, us, vs)
        t += dt
        nstep += 1
        c = advect_scalar(c, ns.u, ns.v, dt, dx, dx)
        c[~f["lv"]] = 0.0  # fresh blood outside the ventricle

        if t >= next_sample - 1e-12:
            next_sample += sample_every
            cav = f["cav"]
            fluid = cav & (f["chi"] < 0.05)
            p = ns.pressure() * RHO_BLOOD
            p = p - p[ref].mean()
            L, _ = v.L(t)
            apex_box = cav & (Y < -0.65 * L) & (Y > -0.75 * L)
            om = ns.vorticity()
            sp = np.hypot(ns.u, ns.v)
            moving = f["chi"] > 0.5
            rec["t"].append(t)
            rec["area"].append(cav.sum() * dx * dx)
            rec["q_row"].append(-(ns.v[in_base, row]).sum() * dx)
            # mechanical energy flux into the cavity through the valve plane
            prow, urow, vrow = p[in_base, row], ns.u[in_base, row], ns.v[in_base, row]
            rec["e_row"].append(((prow + 0.5 * RHO_BLOOD * (urow**2 + vrow**2)) * -vrow).sum() * dx)
            rec["p_base"].append(p[base_box & cav].mean() if (base_box & cav).any() else np.nan)
            rec["p_apex"].append(p[apex_box].mean() if apex_box.any() else np.nan)
            rec["p_min"].append(p[fluid].min())
            rec["power_wall"].append(RHO_BLOOD * ((ns.fx * f["us"] + ns.fy * f["vs"])[moving]).sum() * dx * dx)
            rec["dissipation"].append(MU_BLOOD * (om[fluid] ** 2).sum() * dx * dx)
            rec["ke"].append(0.5 * RHO_BLOOD * (sp[fluid] ** 2).sum() * dx * dx)
            rec["residual"].append(c[cav].sum() / c0)
            lay = f["layer"]
            rec["residual_layer"].append(c[lay].sum() / max(lay.sum(), 1))
            rec["speed_layer"].append(sp[lay].mean() if lay.any() else np.nan)
            rec["speed_cav"].append(sp[fluid].mean())
            rec["umax"].append(sp.max())
            rec["div_cav"].append(np.abs(ns.divergence()[fluid]).max())

        tau = t % period
        if t > (cycles - 1) * period:
            for name, ph in SNAP_PHASES.items():
                if name not in snaps and tau >= ph:
                    p = ns.pressure() * RHO_BLOOD
                    snaps[name] = dict(
                        t=t, chi=f["chi"].astype(np.float32), omega=ns.vorticity().astype(np.float32),
                        c=c.astype(np.float32), p=(p - p[ref].mean()).astype(np.float32),
                        u=ns.u.astype(np.float32), v=ns.v.astype(np.float32))
        if verbose and nstep % 2000 == 0:
            el = time.time() - wall_clock
            print(f"[{kind}] t={t:.4f} s step={nstep} dt={dt:.2e} umax={ns.max_speed():.3f} "
                  f"residual={rec['residual'][-1]:.3f} wall={el:.0f}s", flush=True)
        if not np.isfinite(ns.u).all():
            raise FloatingPointError(f"{kind}: blew up at t={t}")

    out = {k: np.array(vv) for k, vv in rec.items()}
    out["exact_area"] = np.array([v.cavity_area(tt) for tt in out["t"]])
    out["meta"] = np.array([kind, cycles, dx, nstep, time.time() - wall_clock], dtype=object)
    return out, snaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True)
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--dt-max", type=float, default=1e-4)
    ap.add_argument("--dx", type=float, default=DX)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    nx = int(round(NX * DX / a.dx))
    ny = int(round(NY * DX / a.dx))
    out, snaps = run(a.kind, a.cycles, dt_max=a.dt_max, nx=nx, ny=ny, dx=a.dx)
    OUT.mkdir(parents=True, exist_ok=True)
    flat = dict(out)
    for name, s in snaps.items():
        for k, val in s.items():
            flat[f"snap_{name}_{k}"] = val
    np.savez_compressed(OUT / f"{a.kind}{a.tag}.npz", **flat)
    print("saved", OUT / f"{a.kind}{a.tag}.npz")


if __name__ == "__main__":
    main()
