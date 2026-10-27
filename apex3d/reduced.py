"""Reduced film model with inertia and wall dynamics, calibrated on the 3D runs.

The film force on an apposed patch opening at rate hdot with acceleration
hddot is modelled as

    F = -C(h) hdot (r0 + kappa Re_h) - M_a(h) hddot,   Re_h = rho |hdot| h / mu,

where C(h) is the Tier 1 viscous resistance (Reynolds equation with the duct
correction for deep grooves), M_a(h) is the film's added mass from the
inertial form of the same equation, div(h grad p) = rho hddot, and kappa
collects convective inertia (the deceleration of the inflow over the lands
and the entrance loss at the rim), fitted to the resolved 3D runs. r0 is the
resolved to Reynolds ratio left at zero Reynolds number, which measures how
well the Tier 1 drainage model (and the duct correction) captures the real 3D
flow through the grooves; it is 1 when Tier 1 is exact.

Wall dynamics: two walls of mass m_w per unit area are pulled apart by a
recoil stress sigma(t). Each wall obeys m_w A z'' = sigma A + F with h = h0 +
2 z, so

    (m_w A / 2 + M_a) hddot = sigma A - C hdot (r0 + kappa Re_h).

This is the fluid structure interaction the resolved runs test, and the
model then carries it to the physiological gaps and patch sizes no grid here
can resolve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from squeeze_film import reynolds as rf

MU, RHO = rf.MU_BLOOD, rf.RHO_BLOOD


@dataclass
class FilmModel:
    h: np.ndarray
    C: np.ndarray
    Ma: np.ndarray
    pmin_per_hdot: np.ndarray
    area: float
    kappa: float = 0.0
    r0: float = 1.0
    re_max: float = np.inf  # do not extrapolate the inertia term past its calibration
    extra_gap: float = 0.0  # a wall sitting further back (volume matched control)

    @classmethod
    def build(cls, shape="disk", R=8.75e-3, surface="smooth", pitch=2e-3, width=1e-3,
              depth=2e-3, n=160, h=None, kappa=0.0, r0=1.0, re_max=np.inf, extra_gap=0.0):
        h = np.geomspace(0.03e-3, 12e-3, 48) if h is None else np.asarray(h)
        g = rf.Grid(R=R, n=n, shape=shape)
        if surface == "smooth":
            surf = rf.recess_smooth(g)
        elif surface == "mesh":
            surf = rf.recess_mesh(g, pitch, width, depth)
        elif surface == "parallel":
            surf = rf.recess_parallel(g, pitch, width, depth)
        else:
            raise ValueError(surface)
        visc = [rf.solve(g, surf, hh, 1.0) for hh in h]
        C = np.array([-s.force for s in visc])
        P = np.array([-s.p_min for s in visc])
        Ma = np.array([-rf.solve(g, surf, hh, RHO, kernel="inertial").force for hh in h])
        area = float(g.inside.sum() * g.dx**2)
        return cls(h=h, C=C, Ma=Ma, pmin_per_hdot=P, area=area, kappa=kappa, r0=r0,
                   re_max=re_max, extra_gap=extra_gap)

    def _interp(self, arr, h):
        x = np.log(np.clip(np.asarray(h, dtype=float) + self.extra_gap, self.h[0], self.h[-1]))
        return np.exp(np.interp(x, np.log(self.h), np.log(arr)))

    def reynolds(self, h, hdot):
        hg = np.asarray(h) + self.extra_gap
        return np.minimum(RHO * np.abs(hdot) * hg / MU, self.re_max)

    def resistance(self, h):
        return self._interp(self.C, h)

    def added_mass(self, h):
        return self._interp(self.Ma, h)

    def force(self, h, hdot, hddot=0.0, kappa=None, r0=None):
        k = self.kappa if kappa is None else kappa
        r = self.r0 if r0 is None else r0
        return -self.resistance(h) * hdot * (r + k * self.reynolds(h, hdot)) - self.added_mass(h) * hddot

    def peak_pressure(self, h, hdot, kappa=None):
        """Peak film suction (Pa), viscous peak scaled by the inertial factor."""
        k = self.kappa if kappa is None else kappa
        return self._interp(self.pmin_per_hdot, h) * hdot * (self.r0 + k * self.reynolds(h, hdot))


def recoil(sigma0, tau):
    return lambda t: sigma0 * (1.0 - np.exp(-t / tau)) if tau > 0 else sigma0


def open_walls(model: FilmModel, sigma, m_wall, h0, h_end, t_max=1.0, kappa=None,
               r0=None, inertia=True, n_out=2000):
    """Integrate the wall opening from rest at gap h0 until h_end or t_max.

    inertia=False drops wall mass, added mass and kappa, which recovers the
    quasi static Tier 1 estimate sigma A = C(h) hdot.
    """
    A = model.area
    k = model.kappa if kappa is None else kappa
    r = model.r0 if r0 is None else r0

    def rhs(t, y):
        h, hd = y
        s = sigma(t)
        if not inertia:
            return [s * A / (r * model.resistance(h)), 0.0]
        visc = model.resistance(h) * hd * (r + k * model.reynolds(h, hd))
        hdd = (s * A - visc) / (0.5 * m_wall * A + model.added_mass(h))
        return [hd, hdd]

    def reached(t, y):
        return y[0] - h_end
    reached.terminal = True
    reached.direction = 1

    sol = solve_ivp(rhs, (0.0, t_max), [h0, 0.0], events=reached, max_step=t_max / 400,
                    rtol=1e-7, atol=[1e-10, 1e-8], dense_output=True)
    t_end = sol.t_events[0][0] if len(sol.t_events[0]) else sol.t[-1]
    t = np.linspace(0, t_end, n_out)
    y = sol.sol(t)
    h, hd = y[0], (y[1] if inertia else np.array([rhs(tt, [hh, 0])[0] for tt, hh in zip(t, y[0])]))
    hdd = np.gradient(hd, t) if len(t) > 2 else np.zeros_like(t)
    F = model.force(h, hd, hdd if inertia else 0.0, kappa=k if inertia else 0.0, r0=r)
    return dict(t=t, h=h, hdot=hd, hddot=hdd, F=F, reached=bool(len(sol.t_events[0])), t_end=float(t_end))


def time_to_gap(res, gap):
    """First time the opening trajectory reaches `gap` (nan if it never does)."""
    idx = np.nonzero(res["h"] >= gap * (1 - 1e-9))[0]
    if not len(idx):
        return np.nan
    i = idx[0]
    if i == 0:
        return 0.0
    t0, t1, h0, h1 = res["t"][i - 1], res["t"][i], res["h"][i - 1], res["h"][i]
    return float(t0 + (gap - h0) * (t1 - t0) / (h1 - h0))
