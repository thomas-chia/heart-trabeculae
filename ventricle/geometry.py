"""Prescribed kinematics of a 2D long axis left ventricle.

The cavity is described in material coordinates: xi = -y / L(t) runs from the
base (xi = 0, y = 0) to the apex (xi = 1), and the lateral offset
d = |x| - W(xi, t) is measured from the crest surface W, the envelope that the
trabecular tips (or a smooth endocardium) sit on. W morphs between an
elliptical end diastolic shape and an end systolic shape whose apical quarter
collapses into a slit of crest gap h_res, so opposing walls come into near
apposition exactly as in the Tier 1 problem. The cycle runs through ejection,
isovolumic relaxation, early filling (E wave), diastasis, the atrial kick
(A wave) and isovolumic contraction, with mitral and aortic valves in the base
that open and close on schedule.

Three endocardia share the same crest surface motion:

* ``trab``: a compact wall sitting D behind the crest surface, with trabecular
  ridges of width w at pitch lam reaching out to the crest surface;
* ``smooth_crest``: a smooth wall on the crest surface (same apposition gap,
  smaller cavity);
* ``smooth_vol``: a smooth wall D (1 - w / lam) behind the crest surface, so
  its cavity area (end diastolic, end systolic and stroke) matches ``trab``.

Compact wall thickness th(t) follows from conserving myocardial area, so the
myocardium is incompressible to leading order. All quantities are SI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

KINDS = ("trab", "smooth_crest", "smooth_vol")


@dataclass
class Cycle:
    """Cavity volume fraction s(t): 1 at end diastole, 0 at end systole."""

    period: float = 0.8
    t_ej: float = 0.30  # ejection ends
    t_e0: float = 0.38  # isovolumic relaxation ends, E wave starts
    t_e1: float = 0.55  # E wave ends
    t_a0: float = 0.66  # atrial kick starts
    t_a1: float = 0.78  # atrial kick ends; isovolumic contraction to period
    s_e: float = 0.75  # fraction of filling done by the end of the E wave
    s_d: float = 0.80  # fraction done by the end of diastasis
    ramp: float = 0.005  # valve opening and closing time

    def _segments(self):
        return [
            (0.0, self.t_ej, 1.0, 0.0),
            (self.t_ej, self.t_e0, 0.0, 0.0),
            (self.t_e0, self.t_e1, 0.0, self.s_e),
            (self.t_e1, self.t_a0, self.s_e, self.s_d),
            (self.t_a0, self.t_a1, self.s_d, 1.0),
            (self.t_a1, self.period, 1.0, 1.0),
        ]

    def s(self, t: float) -> tuple[float, float]:
        """Return (s, ds/dt) with cosine ramps inside each phase."""
        tau = t % self.period
        for a, b, s0, s1 in self._segments():
            if a <= tau < b:
                phase = (tau - a) / (b - a)
                s = s0 + (s1 - s0) * 0.5 * (1 - np.cos(np.pi * phase))
                ds = (s1 - s0) * 0.5 * np.pi / (b - a) * np.sin(np.pi * phase)
                return float(s), float(ds)
        return 1.0, 0.0

    def valve_closed(self, t: float) -> tuple[float, float]:
        """Closed fraction (0 open, 1 shut) of the (mitral, aortic) valves."""
        tau = t % self.period
        T, r = self.period, self.ramp

        def open_frac(t_open, t_close):
            # opens over [t_open - r, t_open], closes over [t_close, t_close + r]
            x = (tau - (t_open - r)) % T
            span = (t_close - t_open + r) % T
            if x < r:
                return x / r
            if x < span:
                return 1.0
            if x < span + r:
                return 1.0 - (x - span) / r
            return 0.0

        mitral = open_frac(self.t_e0, self.t_a1)
        aortic = open_frac(0.0, self.t_ej)
        return 1.0 - mitral, 1.0 - aortic

    def phase(self, t: float) -> str:
        tau = t % self.period
        if tau < self.t_ej:
            return "ejection"
        if tau < self.t_e0:
            return "ivr"
        if tau < self.t_e1:
            return "e_wave"
        if tau < self.t_a0:
            return "diastasis"
        if tau < self.t_a1:
            return "a_wave"
        return "ivc"


@dataclass
class Ventricle:
    kind: str = "trab"
    h_res: float = 0.75e-3  # resolved crest gap in the apical slit at end systole
    a_base: float = 22e-3  # base half width (annulus)
    L_ed: float = 85e-3
    L_es: float = 70e-3
    xi_slit: float = 0.75  # the apical quarter collapses to a slit at end systole
    es_power: float = 0.8
    th_ed: float = 9e-3  # compact wall lateral thickness at end diastole
    depth: float = 2e-3  # trabecular recess depth D
    width: float = 1e-3  # trabecular ridge width w (material, at end diastole)
    pitch: float = 2e-3  # trabecular pitch lam (material, at end diastole)
    xi_trab: float = 0.40  # trabecular zone starts here (full depth)
    taper: float = 0.10  # and tapers to nothing over this distance basally
    # base: mitral orifice on the left, aortic on the right, fibrous partition
    x_part: float = 6e-3
    w_part: float = 3e-3
    valve_t: float = 1e-3
    up_height: float = 20e-3  # atrial and aortic channels above the base
    up_wall: float = 4e-3
    eps: float = 0.25e-3  # interface ramp width (set to the grid spacing)
    cycle: Cycle = field(default_factory=Cycle)

    def __post_init__(self):
        assert self.kind in KINDS
        self.frac = self.width / self.pitch
        self._A0 = None
        self._A0 = self.wall_area(0.0, self.th_ed)

    # ------------------------------------------------------------- shapes
    def W_ed(self, xi):
        return np.maximum(self.a_base * np.sqrt(np.clip(1 - xi**2, 0, 1)), self.h_res / 2)

    def W_es(self, xi):
        h2 = self.h_res / 2
        return h2 + (self.a_base - h2) * np.clip(1 - xi / self.xi_slit, 0, 1) ** self.es_power

    def L(self, t):
        s, ds = self.cycle.s(t)
        return self.L_es + s * (self.L_ed - self.L_es), ds * (self.L_ed - self.L_es)

    def W(self, xi, t):
        """Crest half width and its rate at fixed material xi."""
        s, ds = self.cycle.s(t)
        we, wd = self.W_es(xi), self.W_ed(xi)
        return we + s * (wd - we), ds * (wd - we)

    def taper_fn(self, xi):
        z = np.clip((xi - (self.xi_trab - self.taper)) / self.taper, 0, 1)
        return z * z * (3 - 2 * z)

    def recess(self, xi):
        """Offset of the compact endocardium behind the crest surface."""
        if self.kind == "trab":
            return self.depth * self.taper_fn(xi)
        if self.kind == "smooth_vol":
            return self.depth * (1 - self.frac) * self.taper_fn(xi)
        return np.zeros_like(np.asarray(xi, dtype=float))

    def band_phi(self, xi, L):
        """Signed distance (m) to the edge of the nearest trabecular ridge."""
        m = xi * self.L_ed
        ph = np.mod(m, self.pitch)
        w = self.width
        inside = ph < w
        d_in = np.minimum(ph, w - ph)
        d_out = -np.minimum(ph - w, self.pitch - ph)
        return np.where(inside, d_in, d_out) * L / self.L_ed

    # --------------------------------------------- myocardial area balance
    def _trab_area(self, L):
        if self.kind != "trab":
            return 0.0
        xi = np.linspace(0, 1, 20001)
        ridge = (np.mod(xi * self.L_ed, self.pitch) < self.width).astype(float)
        return 2 * L * np.trapezoid(self.recess(xi) * ridge, xi)

    def wall_area(self, t, th=None):
        """Myocardial area: compact walls, trabeculae and apical cap."""
        L, _ = self.L(t)
        if th is None:
            th = self.th(t)
        w1, _ = self.W(1.0, t)
        cap_half = w1 + self.recess(1.0) + th
        return 2 * th * L + self._trab_area(L) + 2 * th * cap_half

    def th(self, t):
        """Compact thickness that conserves myocardial area (solves a quadratic)."""
        if self._A0 is None:
            return self.th_ed
        L, _ = self.L(t)
        w1, _ = self.W(1.0, t)
        a = 2.0
        b = 2 * L + 2 * (w1 + float(self.recess(1.0)))
        c = self._trab_area(L) - self._A0
        return (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)

    def th_rate(self, t, dt=1e-5):
        return (self.th(t + dt) - self.th(t - dt)) / (2 * dt)

    # -------------------------------------------------------------- fields
    def _H(self, phi):
        return np.clip(phi / self.eps + 0.5, 0.0, 1.0)

    def fields(self, X, Y, t):
        """Solid mask chi, solid velocity (us, vs), cavity mask, LV region mask
        (cavity plus myocardium) and the near wall layer 0 < d < D."""
        H = self._H
        L, dL = self.L(t)
        th, dth = self.th(t), self.th_rate(t)
        xi = -Y / L
        xic = np.clip(xi, 0.0, 1.0)
        W, dW = self.W(xic, t)
        D = self.recess(xic)
        ax = np.abs(X)
        d = ax - W
        in_len = np.minimum(H(-Y), H(Y + L))  # 0 >= y >= -L

        compact = np.minimum(H(d - D), H(D + th - d))
        if self.kind == "trab":
            ridge = np.minimum.reduce([H(d + self.eps), H(D + self.eps - d), H(self.band_phi(xic, L))])
            ridge = ridge * (D > 0.5 * self.eps)
            wall = np.maximum(compact, ridge)
        else:
            wall = compact
        wall = wall * in_len

        w1, _ = self.W(1.0, t)
        d1 = float(self.recess(1.0))
        cap = np.minimum(np.minimum(H(Y + L + th), H(-L + self.eps - Y)),
                         H(w1 + d1 + th - ax))
        myo = np.maximum(wall, cap)

        # myocardial velocity in material coordinates
        sgn = np.sign(X)
        us = sgn * (dW + np.maximum(d - D, 0.0) / th * dth)
        vs = -xi * dL
        # the cap rides with the apex; its lateral motion follows the wall below
        us = np.where(xi > 1, sgn * np.maximum(ax - w1 - d1, 0) / th * dth, us)

        # base: annulus walls, fibrous partition and the two valve leaflets
        up = np.minimum(H(Y + self.valve_t), H(self.up_height - Y))
        ann = up * np.minimum(H(ax - self.a_base), H(self.a_base + self.up_wall - ax))
        part = up * H(self.w_part / 2 - np.abs(X - self.x_part))
        mit_c, aor_c = self.cycle.valve_closed(t)
        leaf = np.minimum(H(Y + self.valve_t), H(-Y))
        x_in = H(self.a_base + self.eps - ax)
        mitral = leaf * x_in * H(self.x_part - X) * mit_c
        aortic = leaf * x_in * H(X - self.x_part) * aor_c
        fixed = np.maximum.reduce([ann, part, mitral, aortic])

        chi = np.maximum(myo, fixed)
        us = np.where(fixed > myo, 0.0, us)
        vs = np.where(fixed > myo, 0.0, vs)

        # cavity: fluid between the walls below the valve plane
        below = (Y < -self.valve_t) & (Y > -L)
        cav = (d < D) & below & (chi < 0.5)
        layer = cav & (d > 0)
        lv = (Y < 0) & (Y > -L - th) & (d < D + th)
        return dict(chi=chi, us=us, vs=vs, cav=cav, lv=lv, layer=layer)

    # ------------------------------------------------------------ helpers
    def cavity_area(self, t, n=40001):
        """Exact cavity area from the shape functions (no grid)."""
        L, _ = self.L(t)
        xi = np.linspace(0, 1, n)
        W, _ = self.W(xi, t)
        D = self.recess(xi)
        lumen = 2 * W
        if self.kind == "trab":
            ridge = (np.mod(xi * self.L_ed, self.pitch) < self.width)
            lumen = lumen + 2 * D * (~ridge)
        else:
            lumen = lumen + 2 * D
        return L * np.trapezoid(lumen, xi)

    def apposition(self, t, xi=0.875):
        """Crest gap and its opening rate at the middle of the apical slit."""
        W, dW = self.W(xi, t)
        return 2 * W, 2 * dW
