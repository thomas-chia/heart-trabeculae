"""Subgrid squeeze film closure for the apical slit, built on Tier 1.

The Tier 2 grid cannot resolve apical crest gaps of 0.1 to 0.5 mm, so the
resolved geometry keeps a minimum gap h_res of a few cells. The film physics
below that is supplied by the Tier 1 Reynolds solver: the resolved kinematics
give the gap opening rate hdot(t) in the middle of the apposed slit, the
physiological crest gap is h_film(t) = h0 + (gap_res(t) - h_res), and the film
pressure follows from the Tier 1 resistance C(h) of a circular apposed patch
whose radius matches the slit. The closure is one way: film pressures are
reported alongside the resolved flow, not fed back into it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from squeeze_film import reynolds as rf

from .geometry import Ventricle


@dataclass
class FilmTable:
    """Tier 1 film resistance of one surface, tabulated against crest gap."""

    h: np.ndarray
    C: np.ndarray  # suction force per unit separation speed, N s / m
    pmin_per_hdot: np.ndarray  # peak suction per unit separation speed, Pa s / m
    area: float

    def at(self, h):
        lh = np.log(np.clip(h, self.h[0], self.h[-1]))
        C = np.exp(np.interp(lh, np.log(self.h), np.log(self.C)))
        P = np.exp(np.interp(lh, np.log(self.h), np.log(self.pmin_per_hdot)))
        return C, P


def film_table(surface: str, R: float, depth: float, width: float, pitch: float,
               n: int = 200, h=None) -> FilmTable:
    h = np.geomspace(0.03e-3, 8e-3, 36) if h is None else h
    grid = rf.Grid(R=R, n=n)
    if surface == "smooth":
        surf = rf.recess_smooth(grid)
    elif surface == "mesh":
        surf = rf.recess_mesh(grid, pitch, width, depth)
    else:
        raise ValueError(surface)
    sols = [rf.solve(grid, surf, hh, 1.0) for hh in h]
    area = float(grid.inside.sum() * grid.dx**2)
    return FilmTable(h=h, C=np.array([-s.force for s in sols]),
                     pmin_per_hdot=np.array([-s.p_min for s in sols]), area=area)


def patch_radius(v: Ventricle) -> float:
    """Half the length of the apposed apical slit at end systole."""
    return 0.5 * (1 - v.xi_slit) * v.L_es


def film_history(v: Ventricle, table: FilmTable, h0: float, t: np.ndarray,
                 extra_gap: float = 0.0):
    """Film gap, opening rate, mean and peak suction (Pa) and power per area."""
    gap_res, hdot = np.array([v.apposition(tt) for tt in t]).T
    h = h0 + (gap_res - v.h_res) + extra_gap
    C, P = table.at(h)
    # only an opening film sucks; a closing film pushes fluid out (positive p)
    p_mean = C * hdot / table.area
    p_peak = P * hdot
    power = p_mean * hdot  # W per m^2 of apposed patch
    return dict(h=h, hdot=hdot, p_mean=p_mean, p_peak=p_peak, power=power)


def tables_for(v: Ventricle, n: int = 200):
    """Film tables for each endocardium kind (smooth_vol sits further apart)."""
    R = patch_radius(v)
    smooth = film_table("smooth", R, v.depth, v.width, v.pitch, n=n)
    mesh = film_table("mesh", R, v.depth, v.width, v.pitch, n=n)
    return {
        "trab": (mesh, 0.0),
        "smooth_crest": (smooth, 0.0),
        "smooth_vol": (smooth, 2 * v.depth * (1 - v.width / v.pitch)),
    }
