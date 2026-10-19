"""Tier 1 experiments: does trabecular grooving relieve squeeze film suction?

Run with `python -m squeeze_film.experiments`. Writes figures to figures/ and
the numbers quoted in RESULTS.md to results/results.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import reynolds as rf

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
OUT = ROOT / "results"

# baseline physiology (order of magnitude, see RESULTS.md for the rationale)
R = 10e-3  # apposed apical patch radius
HDOT = 0.05  # wall separation speed, m/s (twice a ~2.5 cm/s radial endocardial velocity)
N = 400  # grid cells across the patch (dx = 50 um)
KEY_GAPS = np.array([0.1e-3, 0.2e-3, 0.5e-3, 1e-3, 2e-3])
H_GRID = np.unique(np.concatenate([np.geomspace(0.05e-3, 3e-3, 28), KEY_GAPS]))
DIASTOLIC_IVPG = 2.5 * rf.MMHG  # typical early diastolic base to apex gradient
FILLING_WINDOW = 0.10  # s, early rapid filling

# trabecular pattern: 2 mm pitch, 1 mm grooves, 2 mm deep recesses
PITCH, WIDTH, DEPTH = 2e-3, 1e-3, 2e-3

# reference categorical palette, fixed order
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID_INK = "#0b0b0b", "#52514e", "#e4e3df"

GEOMETRIES = {
    "smooth": (lambda g: rf.recess_smooth(g), "Smooth wall"),
    "parallel": (lambda g: rf.recess_parallel(g, PITCH, WIDTH, DEPTH), "Parallel grooves"),
    "mesh": (lambda g: rf.recess_mesh(g, PITCH, WIDTH, DEPTH), "Groove mesh"),
    "fractal": (lambda g: rf.recess_fractal(g, 2 * PITCH, WIDTH, DEPTH, levels=3), "Fractal mesh (3 levels)"),
}


def style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID_INK)
    ax.grid(True, which="major", color=GRID_INK, lw=0.6)
    ax.set_axisbelow(True)


def resistance_curves(grid):
    """C(h) = suction force per unit separation speed, and peak suction per unit speed."""
    curves = {}
    for key, (fn, _) in GEOMETRIES.items():
        surf = fn(grid)
        sols = [rf.solve(grid, surf, h, 1.0) for h in H_GRID]
        curves[key] = {
            "C": np.array([-s.force for s in sols]),
            "pmin": np.array([-s.p_min for s in sols]),
            "pmean": np.array([-s.p_mean for s in sols]),
        }
    # volume matched smooth control: same blood volume in the gap as the mesh
    mesh_mean_depth = float(GEOMETRIES["mesh"][0](grid).depth[grid.inside].mean())
    smooth = rf.recess_smooth(grid)
    sols = [rf.solve(grid, smooth, h + mesh_mean_depth, 1.0) for h in H_GRID]
    curves["smooth_vol"] = {
        "C": np.array([-s.force for s in sols]),
        "pmin": np.array([-s.p_min for s in sols]),
        "pmean": np.array([-s.p_mean for s in sols]),
    }
    return curves, mesh_mean_depth


def sep_time(Ccurve, h_a, h_b, stress, area):
    m = (H_GRID >= h_a * 0.999) & (H_GRID <= h_b * 1.001)
    hs, Cs = H_GRID[m], Ccurve[m]
    return float(np.trapezoid(Cs * hs, np.log(hs)) / (stress * area))


def main():
    FIG.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    grid = rf.Grid(R=R, n=N)
    area = float(grid.inside.sum() * grid.dx**2)
    curves, mesh_mean_depth = resistance_curves(grid)
    labels = {k: v[1] for k, v in GEOMETRIES.items()}
    labels["smooth_vol"] = "Smooth, volume matched to mesh"
    order = ["smooth", "smooth_vol", "parallel", "mesh", "fractal"]
    res = {"params": dict(R_mm=R * 1e3, hdot_m_s=HDOT, n=N, pitch_mm=PITCH * 1e3,
                          width_mm=WIDTH * 1e3, depth_mm=DEPTH * 1e3,
                          mu_mPa_s=rf.MU_BLOOD * 1e3,
                          mesh_mean_recess_mm=mesh_mean_depth * 1e3)}

    # ---- figure 1: peak suction vs crest gap at baseline hdot
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    for i, k in enumerate(order):
        y = curves[k]["pmin"] * HDOT / rf.MMHG
        ax.loglog(H_GRID * 1e3, y, color=C[i], lw=2, label=labels[k],
                  ls="--" if k == "smooth_vol" else "-")
    ax.axhline(DIASTOLIC_IVPG / rf.MMHG, color=MUTED, lw=1, ls=":")
    ax.text(H_GRID[-1] * 1e3, DIASTOLIC_IVPG / rf.MMHG * 1.15,
            "typical early diastolic IVPG, 2.5 mmHg", ha="right", fontsize=8, color=MUTED)
    style(ax, "Crest gap between apposed walls $h_0$ (mm)", "Peak suction (mmHg)")
    ax.set_title(f"Peak film suction at wall separation speed {HDOT*100:.0f} cm/s, patch R = {R*1e3:.0f} mm",
                 fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_peak_suction_vs_gap.png")
    plt.close(fig)

    # tabulate at the key gaps
    table = {}
    for h_mm in (0.1, 0.2, 0.5, 1.0):
        j = int(np.argmin(np.abs(H_GRID - h_mm * 1e-3)))
        row = {}
        for k in order:
            row[k] = dict(
                h_mm=float(H_GRID[j] * 1e3),
                peak_mmHg=float(curves[k]["pmin"][j] * HDOT / rf.MMHG),
                mean_mmHg=float(curves[k]["pmean"][j] * HDOT / rf.MMHG),
                force_reduction_vs_smooth=float(curves["smooth"]["C"][j] / curves[k]["C"][j]),
            )
        table[f"{h_mm:g}mm"] = row
    res["suction_table"] = table
    res["squeeze_reynolds"] = {
        f"h={h}mm,hdot={v}m/s": rf.squeeze_reynolds(h * 1e-3, v)
        for h in (0.1, 0.2, 0.5) for v in (0.02, 0.05, 0.1)
    }

    # ---- figure 2: reduction factor vs groove pitch (mesh, 50% groove fraction)
    h0 = 0.2e-3
    smooth_C = -rf.solve(grid, rf.recess_smooth(grid), h0, 1.0).force
    pitches = np.array([0.5, 1, 2, 3, 4, 6, 8, 12]) * 1e-3
    red_mesh, red_par = [], []
    for lam in pitches:
        red_mesh.append(smooth_C / -rf.solve(grid, rf.recess_mesh(grid, lam, lam / 2, DEPTH), h0, 1.0).force)
        red_par.append(smooth_C / -rf.solve(grid, rf.recess_parallel(grid, lam, lam / 2, DEPTH), h0, 1.0).force)
    res["pitch_sweep_h0_0.2mm"] = dict(pitch_mm=(pitches * 1e3).tolist(),
                                       reduction_mesh=red_mesh, reduction_parallel=red_par)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    ax.loglog(pitches * 1e3, red_par, color=C[2], lw=2, marker="o", ms=5, label="Parallel grooves")
    ax.loglog(pitches * 1e3, red_mesh, color=C[3], lw=2, marker="o", ms=5, label="Groove mesh")
    ax.loglog(pitches * 1e3, (R / pitches) ** 2, color=MUTED, lw=1, ls=":", label=r"$(R/\lambda)^2$ estimate")
    ax.axvspan(1, 3, color=GRID_INK, alpha=0.5, lw=0)
    ax.text(1.05, 1.3, "typical\ntrabecular\npitch", fontsize=8, color=MUTED)
    style(ax, r"Groove pitch $\lambda$ (mm), groove width $\lambda/2$, depth 2 mm",
          "Suction force reduction vs smooth (x)")
    ax.set_title("How much grooving relieves the separation force at $h_0$ = 0.2 mm",
                 fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_reduction_vs_pitch.png")
    plt.close(fig)

    # ---- figure 3: reduction vs groove depth (mesh at baseline pitch)
    depths = np.geomspace(0.02e-3, 5e-3, 14)
    red_depth = [smooth_C / -rf.solve(grid, rf.recess_mesh(grid, PITCH, WIDTH, d), h0, 1.0).force
                 for d in depths]
    red_depth_nocap = [smooth_C / -rf.solve(grid, rf.recess_mesh(grid, PITCH, WIDTH, d), h0, 1.0,
                                            duct_correction=False).force for d in depths]
    res["depth_sweep_h0_0.2mm"] = dict(depth_mm=(depths * 1e3).tolist(),
                                       reduction_mesh=red_depth,
                                       reduction_mesh_no_duct_cap=red_depth_nocap)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    ax.loglog(depths / h0, red_depth, color=C[3], lw=2, marker="o", ms=5, label="Groove mesh, duct corrected")
    ax.loglog(depths / h0, red_depth_nocap, color=C[3], lw=1.5, ls="--", label="Groove mesh, wide slot assumption")
    style(ax, r"Recess depth / crest gap $D/h_0$", "Suction force reduction vs smooth (x)")
    ax.set_title("Recess depth needed before grooves act as drains (pitch 2 mm, $h_0$ = 0.2 mm)",
                 fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_reduction_vs_depth.png")
    plt.close(fig)

    # ---- figure 4: opening time under a constant recoil stress
    stresses = np.array([0.5, 1, 2, 5]) * rf.MMHG
    starts = [0.1e-3, 0.2e-3, 0.5e-3]
    h_end = 2e-3
    times = {}
    for k in order:
        times[k] = {f"{hs*1e3:g}mm": {f"{s/rf.MMHG:g}mmHg": sep_time(curves[k]["C"], hs, h_end, s, area)
                                      for s in stresses} for hs in starts}
    res["opening_time_s_to_2mm"] = times
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    s_line = np.geomspace(0.3, 10, 40) * rf.MMHG
    for i, k in enumerate(order):
        t = [sep_time(curves[k]["C"], 0.2e-3, h_end, s, area) * 1e3 for s in s_line]
        ax.loglog(s_line / rf.MMHG, t, color=C[i], lw=2, label=labels[k],
                  ls="--" if k == "smooth_vol" else "-")
    ax.axhline(FILLING_WINDOW * 1e3, color=MUTED, lw=1, ls=":")
    ax.text(10, FILLING_WINDOW * 1e3 * 1.15, "early rapid filling, ~100 ms", ha="right", fontsize=8, color=MUTED)
    style(ax, "Opening stress available from elastic recoil (mmHg)", "Time to open from 0.2 mm to 2 mm (ms)")
    ax.set_title("Viscous delay to pull apposed walls apart", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_opening_time.png")
    plt.close(fig)

    # ---- figure 5: pressure maps at h0 = 0.2 mm, baseline hdot
    keys = ["smooth", "mesh", "fractal"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), dpi=150)
    sols = {k: rf.solve(grid, GEOMETRIES[k][0](grid), h0, HDOT) for k in keys}
    for ax, k in zip(axes, keys):
        s = sols[k]
        im = ax.imshow((-s.p / rf.MMHG).T, origin="lower", extent=[-R * 1e3, R * 1e3] * 2,
                       cmap="Blues", vmin=0, vmax=-s.p_min / rf.MMHG)
        ax.set_title(f"{labels[k]}\npeak {-s.p_min/rf.MMHG:.1f} mmHg", fontsize=9, color=INK)
        ax.set_xticks([]); ax.set_yticks([])
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
        cb = fig.colorbar(im, ax=ax, shrink=0.8)
        cb.ax.tick_params(labelsize=8, colors=MUTED)
        cb.set_label("Suction (mmHg)", color=MUTED, fontsize=8)
    fig.suptitle(f"Film suction at $h_0$ = 0.2 mm, separation {HDOT*100:.0f} cm/s "
                 "(each panel on its own scale; note the 50x range)",
                 fontsize=10, color=INK, x=0.02, ha="left")
    fig.savefig(FIG / "fig5_pressure_maps.png", bbox_inches="tight")
    plt.close(fig)

    # grid convergence on the finest geometry
    conv = {}
    for n in (200, 400, 600):
        g = rf.Grid(R=R, n=n)
        conv[n] = -rf.solve(g, GEOMETRIES["fractal"][0](g), h0, 1.0).force
    res["fractal_grid_convergence_C"] = {str(k): v for k, v in conv.items()}

    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
