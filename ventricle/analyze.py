"""Tier 2 analysis: metrics, film closure and figures from the saved runs.

    python -m ventricle.analyze
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from squeeze_film.reynolds import MMHG

from . import closure
from .geometry import KINDS, Ventricle
from .run import DX, H_RES, NX, NY, OUT, X0, Y0

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
LABEL = {"trab": "Trabeculated", "smooth_crest": "Smooth at crest surface",
         "smooth_vol": "Smooth, volume matched"}
COLOR = {"trab": "#2a78d6", "smooth_crest": "#eb6834", "smooth_vol": "#1baf7a"}
INK, MUTED, GRID_INK = "#0b0b0b", "#52514e", "#e4e3df"
H0S = (0.1e-3, 0.2e-3, 0.5e-3)


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


def load(kind, tag=""):
    path = OUT / f"{kind}{tag}.npz"
    return dict(np.load(path, allow_pickle=True)) if path.exists() else None


def window(d, t0, t1):
    return (d["t"] >= t0) & (d["t"] < t1)


def integrate(d, key, m):
    return float(np.trapezoid(np.nan_to_num(d[key][m]), d["t"][m]))


def metrics(d, v: Ventricle, cycles: int):
    T = v.cycle.period
    c0 = (cycles - 1) * T  # analyse the last cycle
    cyc = v.cycle
    e = window(d, c0 + cyc.t_e0, c0 + cyc.t_e1)
    dia = window(d, c0 + cyc.t_ej, c0 + T)
    sys = window(d, c0, c0 + cyc.t_ej)
    last = window(d, c0, c0 + T)
    ivpg = d["p_base"] - d["p_apex"]
    dA = np.gradient(d["exact_area"], d["t"])
    leak = np.trapezoid(np.abs(d["q_row"][last] - dA[last]), d["t"][last]) / np.trapezoid(
        np.abs(dA[last]), d["t"][last])
    p_endo = endocardial_power(d)
    out = dict(
        ivpg_peak_e_mmHg=float(np.nanmax(ivpg[e]) / MMHG),
        p_min_e_mmHg=float(np.nanmin(d["p_min"][e]) / MMHG),
        p_apex_min_e_mmHg=float(np.nanmin(d["p_apex"][e]) / MMHG),
        endo_work_diastole_mJ_per_m=1e3 * float(np.trapezoid(p_endo[dia], d["t"][dia])),
        endo_work_e_wave_mJ_per_m=1e3 * float(np.trapezoid(p_endo[e], d["t"][e])),
        endo_work_systole_mJ_per_m=1e3 * float(np.trapezoid(p_endo[sys], d["t"][sys])),
        dissipation_cycle_mJ_per_m=1e3 * integrate(d, "dissipation", last),
        dissipation_diastole_mJ_per_m=1e3 * integrate(d, "dissipation", dia),
        ke_peak_e_mJ_per_m=1e3 * float(np.nanmax(d["ke"][e])),
        flux_mismatch=float(leak),
        speed_cavity_diastole_mm_s=1e3 * float(np.nanmean(d["speed_cav"][dia])),
        speed_layer_diastole_mm_s=(1e3 * float(np.nanmean(d["speed_layer"][dia]))
                                   if np.isfinite(d["speed_layer"][dia]).any() else None),
        u_max_m_s=float(np.nanmax(d["umax"][last])),
    )
    # residual of the original end diastolic blood at each later end diastole
    res = []
    for n in range(1, cycles + 1):
        i = np.argmin(np.abs(d["t"] - (n * T - 1e-3)))
        res.append(float(d["residual"][i]))
    out["residual_at_ed"] = res
    out["layer_old_blood_at_ed"] = float(d["residual_layer"][np.argmin(np.abs(d["t"] - (cycles * T - 1e-3)))])
    ef = 1 - v.cavity_area(v.cycle.t_ej) / v.cavity_area(0.0)
    out["ef"] = float(ef)
    out["mixing_reference"] = [float((1 - ef) ** n) for n in range(1, cycles + 1)]
    return out


def endocardial_power(d):
    """Power the endocardium delivers to the cavity blood (W per m depth).

    Mechanical energy balance of the cavity: dKE/dt + dissipation minus the
    flux of p + rho |u|^2 / 2 entering through the valve plane, with pressure
    measured against the atrial reference. Positive in diastole means the
    wall pays to draw blood in against the atrial reference.
    """
    dke = np.gradient(d["ke"], d["t"])
    return dke + d["dissipation"] - d["e_row"]


def endocardial_length(v: Ventricle, t):
    L, _ = v.L(t)
    xi = np.linspace(0, 1, 4001)
    W, _ = v.W(xi, t)
    y = -xi * L
    return float(2 * np.trapezoid(np.sqrt(1 + np.gradient(W, y) ** 2), -y))


def film_results(cycles: int):
    v0 = Ventricle()
    tables = closure.tables_for(v0)
    t = np.linspace(0, v0.cycle.period, 4001)
    out, hist_all = {}, {}
    for kind in KINDS:
        v = Ventricle(kind=kind, h_res=H_RES)
        tab, extra = tables[kind]
        out[kind] = {}
        for h0 in H0S:
            h = closure.film_history(v, tab, h0, t, extra_gap=extra)
            dia = (t >= v.cycle.t_e0) & (t < v.cycle.t_e1)
            suction = np.where(h["hdot"] > 0, h["p_mean"], 0.0)
            peak = np.where(h["hdot"] > 0, h["p_peak"], 0.0)
            j = int(np.argmax(peak))
            out[kind][f"{h0*1e3:g}mm"] = dict(
                film_mean_suction_peak_mmHg=float(suction.max() / MMHG),
                film_peak_suction_mmHg=float(peak.max() / MMHG),
                time_of_peak_ms_after_mitral_opening=1e3 * float(t[j] - v.cycle.t_e0),
                gap_at_peak_mm=1e3 * float(h["h"][j]),
                film_work_mJ_per_m2=1e3 * float(np.trapezoid(np.where(dia, h["power"], 0), t)),
            )
            hist_all[(kind, h0)] = (t, h)
    return out, hist_all


def fig_geometry(path):
    x = X0 + DX * np.arange(NX)
    y = Y0 + DX * np.arange(NY)
    X, Y = np.meshgrid(x, y, indexing="ij")
    fig, axes = plt.subplots(2, 3, figsize=(9, 8.4), dpi=150)
    ext = [x[0] * 1e3, x[-1] * 1e3, y[0] * 1e3, y[-1] * 1e3]
    for j, kind in enumerate(KINDS):
        v = Ventricle(kind=kind, h_res=H_RES, eps=DX)
        for i, (t, name) in enumerate(((0.0, "end diastole"), (0.33, "end systole"))):
            ax = axes[i, j]
            ax.imshow(v.fields(X, Y, t)["chi"].T, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1.6)
            ax.set_xlim(-36, 36)
            ax.set_ylim(-100, 20)
            ax.set_title(f"{LABEL[kind]}\n{name}, cavity {v.cavity_area(t)*1e6:.0f} mm$^2$",
                         fontsize=9, color=INK)
            ax.tick_params(colors=MUTED, labelsize=7)
            for s in ax.spines.values():
                s.set_visible(False)
    fig.suptitle("Tier 2 ventricles: shared crest motion, three endocardia (mm)",
                 fontsize=10, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_snapshots(runs, path, field="omega"):
    names = ["early_e", "peak_e", "diastasis"]
    titles = {"early_e": "early E wave", "peak_e": "peak E wave", "diastasis": "diastasis"}
    fig, axes = plt.subplots(3, 3, figsize=(9.6, 10.5), dpi=150)
    x = X0 + DX * np.arange(NX)
    y = Y0 + DX * np.arange(NY)
    ext = [x[0] * 1e3, x[-1] * 1e3, y[0] * 1e3, y[-1] * 1e3]
    lim = 400.0
    for j, kind in enumerate(KINDS):
        d = runs[kind]
        for i, n in enumerate(names):
            ax = axes[i, j]
            key = f"snap_{n}_{field}"
            if key not in d:
                ax.axis("off")
                continue
            chi = d[f"snap_{n}_chi"]
            if field == "omega":
                img = np.where(chi > 0.5, np.nan, d[key])
                im = ax.imshow(img.T, origin="lower", extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim)
            else:
                img = np.where(chi > 0.5, np.nan, d[key])
                im = ax.imshow(img.T, origin="lower", extent=ext, cmap="Blues", vmin=0, vmax=1)
            ax.imshow(np.where(chi > 0.5, 1.0, np.nan).T, origin="lower", extent=ext,
                      cmap="Greys", vmin=0, vmax=2.5)
            ax.set_xlim(-34, 34)
            ax.set_ylim(-95, 15)
            ax.set_title(f"{LABEL[kind]}, {titles[n]}", fontsize=8, color=INK)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.5, location="bottom", pad=0.02)
    cb.set_label("Vorticity (1/s)" if field == "omega" else "Old blood fraction", color=MUTED, fontsize=9)
    cb.ax.tick_params(labelsize=8, colors=MUTED)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_traces(runs, cycles, path):
    v = Ventricle()
    T = v.cycle.period
    c0 = (cycles - 1) * T
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.6), dpi=150, sharex=True)
    for kind in KINDS:
        d = runs[kind]
        m = window(d, c0, c0 + T)
        tt = (d["t"][m] - c0) * 1e3
        dA = -np.gradient(d["exact_area"], d["t"])[m]
        axes[0].plot(tt, dA * 1e4, color=COLOR[kind], lw=1.6, label=LABEL[kind])
        ivpg = (d["p_base"] - d["p_apex"])[m] / MMHG
        axes[1].plot(tt, ivpg, color=COLOR[kind], lw=1.6)
        axes[2].plot(tt, endocardial_power(d)[m], color=COLOR[kind], lw=1.6)
    style(axes[0], ylabel="Outflow, $-dA/dt$ (cm$^2$/s)")
    style(axes[1], ylabel="Base minus apex pressure (mmHg)")
    style(axes[2], "Time in cycle (ms)", "Endocardial power into cavity blood (W/m)")
    for ax in axes:
        for tb in (v.cycle.t_ej, v.cycle.t_e0, v.cycle.t_e1, v.cycle.t_a0, v.cycle.t_a1):
            ax.axvline(tb * 1e3, color=GRID_INK, lw=0.8)
    axes[1].set_ylim(-4, 4)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title(f"Last simulated cycle ({cycles} of {cycles}); phase boundaries in grey",
                      fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_washout(runs, mets, path):
    v = Ventricle()
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    for kind in KINDS:
        d = runs[kind]
        ax.plot(d["t"] * 1e3, d["residual"], color=COLOR[kind], lw=1.8, label=LABEL[kind])
    ef = mets["trab"]["ef"]
    n = len(mets["trab"]["residual_at_ed"])
    tt = np.arange(0, n + 1) * v.cycle.period * 1e3
    ax.step(tt, [1] + [(1 - ef) ** k for k in range(1, n + 1)], where="post", color=MUTED,
            lw=1, ls=":", label=f"perfect mixing, EF {ef:.2f}")
    style(ax, "Time (ms)", "Fraction of the initial blood still in the cavity")
    ax.set_title("Washout of the end diastolic blood over three beats", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_film(hist, mets, path):
    v = Ventricle()
    h0 = 0.2e-3
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    for kind in KINDS:
        t, h = hist[(kind, h0)]
        m = (t >= v.cycle.t_e0 - 0.01) & (t < v.cycle.t_e0 + 0.12)
        peak = np.where(h["hdot"] > 0, h["p_peak"], 0.0) / MMHG
        ax.semilogy((t[m] - v.cycle.t_e0) * 1e3, np.maximum(peak[m], 1e-3), color=COLOR[kind],
                    lw=1.8, label=LABEL[kind])
    ivpg = mets["trab"]["ivpg_peak_e_mmHg"]
    ax.axhline(ivpg, color=MUTED, lw=1, ls=":")
    ax.text(118, ivpg * 1.2, f"resolved peak E wave IVPG, {ivpg:.1f} mmHg", ha="right", fontsize=8, color=MUTED)
    ax.set_ylim(1e-2, 1e4)
    style(ax, "Time after mitral opening (ms)", "Peak apical film suction (mmHg)")
    ax.set_title("Subgrid film suction at the apical slit, end systolic crest gap 0.2 mm",
                 fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    runs = {k: load(k) for k in KINDS}
    coarse = {k: load(k, "_coarse") for k in KINDS}
    cycles = int(runs["trab"]["meta"][1])
    res = {"grid": dict(dx_mm=DX * 1e3, nx=NX, ny=NY, h_res_mm=H_RES * 1e3, cycles=cycles)}
    mets = {k: metrics(runs[k], Ventricle(kind=k), cycles) for k in KINDS}
    res["resolved"] = mets
    if all(coarse.values()):
        res["resolved_coarse_0.5mm"] = {k: metrics(coarse[k], Ventricle(kind=k), cycles) for k in KINDS}
    for k in KINDS:
        m = runs[k]["meta"]
        res["resolved"][k]["steps"] = int(m[3])
        res["resolved"][k]["wall_clock_s"] = float(m[4])
    v = Ventricle()
    res["endocardial_length_mid_diastole_mm"] = 1e3 * endocardial_length(v, 0.47)
    film, hist = film_results(cycles)
    res["film"] = film
    (OUT.parent / "tier2_results.json").write_text(json.dumps(res, indent=2))
    fig_geometry(FIG / "fig6_tier2_geometry.png")
    fig_snapshots(runs, FIG / "fig7_tier2_vorticity.png", "omega")
    fig_snapshots(runs, FIG / "fig8_tier2_old_blood.png", "c")
    fig_traces(runs, cycles, FIG / "fig9_tier2_traces.png")
    fig_washout(runs, mets, FIG / "fig10_tier2_washout.png")
    fig_film(hist, mets, FIG / "fig11_tier2_film_closure.png")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
