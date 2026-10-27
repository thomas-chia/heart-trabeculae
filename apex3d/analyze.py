"""Tier 3 analysis: calibrate the reduced film model on the resolved 3D runs,
validate it against the two way coupled runs, then open physiological apical
walls with it.

    python -m apex3d.analyze
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from squeeze_film.reynolds import MMHG, MU_BLOOD, RHO_BLOOD
from ventricle.geometry import Ventricle

from .reduced import FilmModel, open_walls, recoil, time_to_gap

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "tier3"
FIG = ROOT / "figures"
INK, MUTED, GRID_INK = "#0b0b0b", "#52514e", "#e4e3df"
C_SMOOTH, C_MESH, C_VOL = "#eb6834", "#2a78d6", "#1baf7a"
SURF_COLOR = {"smooth": C_SMOOTH, "mesh": C_MESH}
SURF_LABEL = {"smooth": "Smooth", "mesh": "Groove mesh"}
SPEEDS = (8, 25, 80)  # mm/s, kinematic runs at the production grid
SIGMAS = (0.5, 2)  # mmHg, coupled runs
DX = 0.1e-3

# physiological application
R_PATCH = 0.5 * (1 - 0.75) * 70e-3  # apposed apical slit, as in the Tier 2 closure
M_WALL = 10.6  # kg / m^2, about 10 mm of myocardium per wall
TAU = 20e-3  # recoil rise time
H0S = (0.1e-3, 0.2e-3, 0.5e-3)
SIGMA_PHYS = (1.0, 2.0, 5.0)
H_OPEN = 2e-3


def style(ax, xlabel=None, ylabel=None):
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID_INK)
    ax.grid(True, color=GRID_INK, lw=0.6)
    ax.set_axisbelow(True)


def load(name):
    p = RUNS / f"{name}.npz"
    return dict(np.load(p, allow_pickle=True)) if p.exists() else None


def windows(d, dx, t_min):
    """Average over windows of two cells of wall travel (4 dx of gap), the
    period of the residual immersed boundary force oscillation seen in the
    runs (alternate one cell windows swing high and low)."""
    m = d["t"] > t_min
    h = d["h"][m]
    C = -d["F_lub"][m] / np.where(d["hdot"][m] == 0, np.nan, d["hdot"][m])
    out = []
    edges = np.arange(h.min(), h.max() + 1e-12, 4 * dx)
    for a, b in zip(edges[:-1], edges[1:]):
        w = (h >= a) & (h < b)
        if w.sum() < 3:
            continue
        visc = -(C[w] * d["hdot_eff"][m][w]).mean()
        inert = (d["M_a"][m][w] * d["hddot"][m][w]).mean()
        hd = d["hdot_eff"][m][w].mean()
        hm = h[w].mean()
        out.append(dict(h=hm, hdot=hd, re=RHO_BLOOD * abs(hd) * hm / MU_BLOOD,
                        ratio=(d["F_film"][m][w].mean() + inert) / visc))
    return out


def calibrate(surface):
    pts = []
    for v in SPEEDS:
        d = load(f"{surface}_kin_{v}mms")
        if d is None:
            continue
        t_ramp = float(d["config"][6])
        pts += [dict(p, speed=v) for p in windows(d, DX, 1.5 * t_ramp)]
    re = np.array([p["re"] for p in pts])
    ratio = np.array([p["ratio"] for p in pts])
    A = np.vstack([np.ones_like(re), re]).T
    (r0, kappa), *_ = np.linalg.lstsq(A, ratio, rcond=None)
    resid = ratio - (r0 + kappa * re)
    lo = re < 5
    return dict(r0=float(r0), kappa=float(kappa), rms=float(np.sqrt(np.mean(resid**2))),
                n=len(pts), re_range=[float(re.min()), float(re.max())],
                low_re_ratio=float(ratio[lo].mean()) if lo.any() else None), pts


def convergence(surface):
    out = {}
    for tag, dx in (("", 0.1e-3), ("_dx150", 0.15e-3), ("_dx200", 0.2e-3)):
        d = load(f"{surface}_kin_25mms{tag}")
        if d is None:
            continue
        w = windows(d, dx, 1.5 * float(d["config"][6]))
        out[f"{dx*1e3:g}mm"] = float(np.mean([p["ratio"] for p in w]))
    return out


def dns_patch_model(surface, cal):
    return FilmModel.build(shape="square", R=6e-3, surface=surface, depth=1.5e-3, n=120,
                           h=np.geomspace(0.2e-3, 8e-3, 36), kappa=cal["kappa"], r0=cal["r0"],
                           re_max=cal["re_range"][1])


def validate_fsi(surface, cal):
    res = {}
    model = dns_patch_model(surface, cal)
    for s in SIGMAS:
        d = load(f"{surface}_fsi_{s:g}mmHg")
        if d is None:
            continue
        sig = recoil(s * MMHG, float(d["config"][7]))
        mw = float(d["config"][8])
        h0, h1 = float(d["h"][0]), float(d["h"][-1])
        variants = {
            "calibrated": open_walls(model, sig, mw, h0, h1, t_max=0.5),
            "tier1_with_inertia": open_walls(model, sig, mw, h0, h1, t_max=0.5, kappa=0.0, r0=1.0),
            "quasi_static_tier1": open_walls(model, sig, mw, h0, h1, t_max=0.5, inertia=False, r0=1.0),
        }
        targets = [h0 + f * (h1 - h0) for f in (0.5, 0.9)]
        dns_t = [time_to_gap(dict(t=d["t"], h=d["h"]), g) for g in targets]
        row = {"dns_ms": [1e3 * t for t in dns_t], "targets_mm": [1e3 * g for g in targets]}
        for k, r in variants.items():
            mt = [time_to_gap(r, g) for g in targets]
            row[k + "_ms"] = [1e3 * t for t in mt]
            row[k + "_err"] = [float(m / t - 1) for m, t in zip(mt, dns_t)]
        res[f"{s:g}mmHg"] = row
        res[f"{s:g}mmHg_curves"] = dict(dns=(d["t"], d["h"]),
                                        **{k: (r["t"], r["h"]) for k, r in variants.items()})
    return res


def tier2_gap_time(h0, gap):
    """Time after mitral opening at which the Tier 2 prescribed kinematics
    open the apical slit to `gap` (film gap convention of the closure)."""
    v = Ventricle()
    t = np.linspace(v.cycle.t_e0, v.cycle.t_e1, 4001)
    g = np.array([h0 + v.apposition(tt)[0] - v.h_res for tt in t])
    i = np.nonzero(g >= gap)[0]
    return float(t[i[0]] - v.cycle.t_e0) if len(i) else np.nan, t - v.cycle.t_e0, g


def physiology(cals):
    models = {
        "trab": FilmModel.build(surface="mesh", R=R_PATCH, depth=2e-3, **cals["mesh"]),
        "smooth_crest": FilmModel.build(surface="smooth", R=R_PATCH, **cals["smooth"]),
        "smooth_vol": FilmModel.build(surface="smooth", R=R_PATCH, extra_gap=2e-3, **cals["smooth"]),
    }
    free = FilmModel.build(surface="smooth", R=R_PATCH, h=np.geomspace(1e-3, 1.0, 20))
    free.C, free.Ma = free.C * 1e-12, free.Ma * 1e-12
    table, curves = {}, {}
    for s in SIGMA_PHYS:
        sig = recoil(s * MMHG, TAU)
        for h0 in H0S:
            key = f"{s:g}mmHg_{h0*1e3:g}mm"
            t_free = time_to_gap(open_walls(free, sig, M_WALL, h0, H_OPEN, t_max=1.0), H_OPEN)
            t_kin, _, _ = tier2_gap_time(h0, H_OPEN)
            row = {"no_film_ms": 1e3 * t_free, "tier2_kinematics_ms": 1e3 * t_kin}
            for name, m in models.items():
                r = open_walls(m, sig, M_WALL, h0, H_OPEN, t_max=1.0)
                r_qs = open_walls(m, sig, M_WALL, h0, H_OPEN, t_max=1.0, inertia=False, r0=1.0)
                t_open = time_to_gap(r, H_OPEN)
                p_peak = float(np.max(m.peak_pressure(r["h"], np.maximum(r["hdot"], 0))))
                row[name] = dict(
                    open_ms=1e3 * t_open,
                    open_quasi_static_ms=1e3 * time_to_gap(r_qs, H_OPEN),
                    film_delay_ms=1e3 * (t_open - t_free),
                    peak_mean_suction_mmHg=float(np.max(-r["F"]) / m.area / MMHG),
                    peak_suction_mmHg=p_peak / MMHG,
                    gap_at_peak_e_mm=1e3 * float(np.interp(0.085, r["t"], r["h"], right=H_OPEN)),
                )
                curves[(name, s, h0)] = (r["t"], r["h"])
            table[key] = row
    return table, curves


# ------------------------------------------------------------------ figures

def fig_calibration(cals, pts, conv, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=150, sharey=True)
    markers = {8: "o", 25: "s", 80: "^"}
    for ax, surface in zip(axes, ("smooth", "mesh")):
        c = SURF_COLOR[surface]
        for v in SPEEDS:
            p = [q for q in pts[surface] if q["speed"] == v]
            if p:
                ax.plot([q["re"] for q in p], [q["ratio"] for q in p], markers[v], color=c, ms=5,
                        alpha=0.85, label=f"{v} mm/s")
        re = np.linspace(0, max(q["re"] for q in pts[surface]) * 1.05, 50)
        cal = cals[surface]
        ax.plot(re, cal["r0"] + cal["kappa"] * re, color=MUTED, lw=1.2,
                label=f"fit: {cal['r0']:.2f} + {cal['kappa']:.3f} Re")
        ax.axhline(1.0, color=GRID_INK, lw=1)
        style(ax, r"Squeeze Reynolds number $\rho \dot h h / \mu$",
              "Resolved / Tier 1 viscous force" if surface == "smooth" else None)
        ax.set_title(f"{SURF_LABEL[surface]} (added mass removed)", fontsize=10, color=INK, loc="left")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_fsi(val, path):
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2), dpi=150)
    styles = {"dns": (INK, "-", 2.2, "3D resolved, two way coupled"),
              "calibrated": ("#2a78d6", "--", 1.6, "reduced model, calibrated"),
              "tier1_with_inertia": ("#1baf7a", ":", 1.6, "Tier 1 + wall and film inertia"),
              "quasi_static_tier1": ("#eb6834", "-.", 1.4, "quasi static Tier 1")}
    for i, surface in enumerate(("smooth", "mesh")):
        for j, s in enumerate(SIGMAS):
            ax = axes[i, j]
            cur = val[surface].get(f"{s:g}mmHg_curves")
            if cur is None:
                ax.axis("off")
                continue
            for k, (col, ls, lw, lab) in styles.items():
                t, h = cur[k]
                ax.plot(np.asarray(t) * 1e3, np.asarray(h) * 1e3, color=col, ls=ls, lw=lw, label=lab)
            style(ax, "Time (ms)" if i == 1 else None, "Crest gap (mm)" if j == 0 else None)
            ax.set_title(f"{SURF_LABEL[surface]}, recoil {s:g} mmHg", fontsize=10, color=INK, loc="left")
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_physiology(curves, path, s=2.0, h0=0.2e-3):
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    lab = {"trab": ("Trabeculated (groove mesh)", C_MESH),
           "smooth_crest": ("Smooth at crest", C_SMOOTH),
           "smooth_vol": ("Smooth, volume matched", C_VOL)}
    for name, (l, c) in lab.items():
        t, h = curves[(name, s, h0)]
        ax.plot(np.asarray(t) * 1e3, np.asarray(h) * 1e3, color=c, lw=2, label=l)
    _, tk, gk = tier2_gap_time(h0, H_OPEN)
    m = gk <= H_OPEN * 1.02
    ax.plot(tk[m] * 1e3, gk[m] * 1e3, color=MUTED, lw=1.2, ls=":", label="Tier 2 prescribed kinematics")
    ax.axhline(H_OPEN * 1e3, color=GRID_INK, lw=1)
    style(ax, "Time after mitral opening (ms)", "Apical crest gap (mm)")
    ax.set_title(f"Pressure driven apical opening: recoil {s:g} mmHg, end systolic gap {h0*1e3:g} mm",
                 fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    cals, pts = {}, {}
    for surface in ("smooth", "mesh"):
        cals[surface], pts[surface] = calibrate(surface)
    conv = {s: convergence(s) for s in ("smooth", "mesh")}
    val = {s: validate_fsi(s, cals[s]) for s in ("smooth", "mesh")}
    phys_cal = {s: dict(kappa=cals[s]["kappa"], r0=cals[s]["r0"], re_max=cals[s]["re_range"][1])
                for s in cals}
    table, curves = physiology(phys_cal)
    runs = {}
    for f in sorted(RUNS.glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        runs[f.stem] = dict(steps=int(d["steps"]), wall_s=float(d["wall_clock"]),
                            wall_speed_ratio=float(np.nanmean(d["hdot_eff"][len(d["t"]) // 4:]
                                                              / np.where(d["hdot"][len(d["t"]) // 4:] == 0, np.nan,
                                                                         d["hdot"][len(d["t"]) // 4:]))),
                            seepage=float(np.nanmean(d["leak"][len(d["t"]) // 4:])),
                            cg_iters=float(np.mean(d["cg"])))
    res = dict(calibration=cals, convergence_ratio_at_25mms=conv,
               fsi_validation={s: {k: v for k, v in val[s].items() if not k.endswith("curves")}
                               for s in val},
               physiology=table, runs=runs,
               physiology_params=dict(R_patch_mm=R_PATCH * 1e3, m_wall=M_WALL, tau_ms=TAU * 1e3,
                                      h_open_mm=H_OPEN * 1e3))
    (ROOT / "results" / "tier3_results.json").write_text(json.dumps(res, indent=2, default=float))
    fig_calibration(cals, pts, conv, FIG / "fig12_tier3_inertia_calibration.png")
    fig_fsi(val, FIG / "fig13_tier3_fsi_validation.png")
    fig_physiology(curves, FIG / "fig14_tier3_apical_opening.png")
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
