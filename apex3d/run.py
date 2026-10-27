"""Run one 3D apposed patch case.

    python -m apex3d.run --surface mesh --mode kinematic --hdot 0.04
    python -m apex3d.run --surface smooth --mode fsi --sigma 2

Writes results/tier3/<surface>_<mode>_<value>[tag].npz.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from squeeze_film.reynolds import MMHG

from .patch import Drive, PatchConfig, run

OUT = Path(__file__).resolve().parent.parent / "results" / "tier3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surface", default="smooth")
    ap.add_argument("--mode", default="kinematic")
    ap.add_argument("--hdot", type=float, default=0.04)
    ap.add_argument("--sigma", type=float, default=2.0, help="recoil stress, mmHg")
    ap.add_argument("--dx", type=float, default=0.1e-3)
    ap.add_argument("--dt-max", type=float, default=2e-4)
    ap.add_argument("--h-end", type=float, default=1.8e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cfg = PatchConfig(surface=a.surface, dx=a.dx, dt_max=a.dt_max, workers=a.workers)
    if a.mode == "kinematic":
        drv = Drive(mode="kinematic", hdot0=a.hdot, t_ramp=min(5e-3, 0.15e-3 / a.hdot), h_end=a.h_end)
        name = f"{a.surface}_kin_{a.hdot*1e3:g}mms{a.tag}"
    else:
        drv = Drive(mode="fsi", sigma0=a.sigma * MMHG, h_end=a.h_end)
        name = f"{a.surface}_fsi_{a.sigma:g}mmHg{a.tag}"
    out = run(cfg, drv)
    out["config"] = np.array([a.surface, a.mode, a.hdot, a.sigma, a.dx, a.dt_max, drv.t_ramp,
                              drv.tau, drv.m_wall, cfg.a, cfg.pitch, cfg.width, cfg.depth], dtype=object)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{name}.npz", **out)
    print("saved", OUT / f"{name}.npz", "steps", out["steps"], "wall", round(out["wall_clock"]))


if __name__ == "__main__":
    main()
