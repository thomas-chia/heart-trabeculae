"""3D apposed apical patch: two walls pulled apart, resolved directly.

Geometry: two plates of half side `a` sit in a triply periodic box with open
blood around them, standing in for the cavity that surrounds the apposed
apical walls. The bottom plate is smooth. The top plate is smooth or carries
an orthogonal groove network (pitch, width, depth) cut into its face, the 3D
trabecular network that Tier 1 models and that a 2D slice cannot represent.
The crest gap h is measured from the bottom plate to the top plate's crests,
and the plates move symmetrically, each at hdot / 2.

Two drives:

* kinematic: hdot ramps smoothly to a constant, so the measured force can be
  compared with Tier 1 lubrication at the same instantaneous gap;
* fsi: each plate is a rigid body of mass m_w per unit area pulled outward by
  a recoil stress sigma(t) and held back by the fluid force computed from the
  penalization field (two way coupling). The fluid's added mass is lagged by
  one step, so the plate equation carries an added mass correction
  (M + M_a) a_new = F_drive + F_fluid + M_a a_old, which leaves the converged
  dynamics unchanged and keeps the partitioned scheme stable when the film's
  added mass rivals the wall's.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from squeeze_film import reynolds as rf

from .solver3d import PenalizedNS3D

MU, RHO = rf.MU_BLOOD, rf.RHO_BLOOD


@dataclass
class PatchConfig:
    surface: str = "smooth"  # "smooth" or "mesh"
    a: float = 6e-3  # plate half side
    pitch: float = 2e-3
    width: float = 1e-3
    depth: float = 1.5e-3
    t_bottom: float = 1.5e-3
    t_top: float = 2.5e-3  # includes the groove depth
    h0: float = 0.6e-3
    dx: float = 0.1e-3
    box_xy: float = 16e-3
    box_z: float = 9.6e-3
    z_shift: float = 0.5e-3  # box offset so both plates keep fluid around them
    n_proj: int = 3
    dt_max: float = 2e-4
    move_cfl: float = 0.05  # max wall travel per step, in cells
    cfl: float = 0.3
    workers: int = 2
    ramp_cells: float = 3.0
    ramp_shift_cells: float = 0.39
    profile: str = "brinkman"
    coupled: bool = True

    @property
    def n(self):
        return (int(round(self.box_xy / self.dx)),) * 2 + (int(round(self.box_z / self.dx)),)


def _S(phi, eps, shift=0.0):
    """Solid indicator ramping over eps across the surface (phi is the signed
    distance into the solid), centred `shift` toward the fluid.

    The penalized wall acts where the penalty first dominates viscous
    diffusion, not at the geometric surface. With the viscous scaled
    Brinkman profile and a 3 cell ramp that point sits a steady 0.39 cells
    inside the solid at any sub cell position (spread about 0.2 cells) and
    any time step, measured on a penalized channel; centring the ramp that
    far toward the fluid puts the effective wall on the surface.
    """
    return np.clip((phi + shift) / eps + 0.5, 0.0, 1.0)


def _band_phi(coord, pitch, width):
    """Signed distance into the nearest groove band (positive inside)."""
    phase = np.mod(coord + 0.5 * pitch, pitch) - 0.5 * pitch
    return 0.5 * width - np.abs(phase)


class Patch:
    def __init__(self, cfg: PatchConfig):
        self.cfg = cfg
        c = cfg
        n = c.n
        L = (n[0] * c.dx, n[1] * c.dx, n[2] * c.dx)
        origin = (-L[0] / 2, -L[1] / 2, -L[2] / 2 + c.z_shift)
        self.ns = PenalizedNS3D(n, L, origin, MU / RHO, n_proj=c.n_proj, workers=c.workers,
                                profile=c.profile, coupled=c.coupled, cg_tol=3e-3)
        X, Y, Z = self.ns.X, self.ns.Y, self.ns.Z
        self.eps = eps = c.ramp_cells * c.dx
        self.shift = sh = c.ramp_shift_cells * c.dx
        self.lat = np.minimum(_S(c.a - np.abs(X), eps, sh), _S(c.a - np.abs(Y), eps, sh))
        if c.surface == "mesh":
            # solid (not groove) indicator in the plane of the top plate's face
            self.land = np.minimum(_S(-_band_phi(X, c.pitch, c.width), eps, sh),
                                   _S(-_band_phi(Y, c.pitch, c.width), eps, sh))
        else:
            self.land = np.ones_like(X)
        self.dV = c.dx**3
        # probes: a land centre for the pressure, open blood for the reference
        self.i_probe = np.unravel_index(np.argmin(np.abs(X - c.pitch / 2) + np.abs(Y - c.pitch / 2)
                                                  + 0 * Z), X.shape)[:2]
        self.far = ((np.abs(X) > c.a + 1e-3) | (np.abs(Y) > c.a + 1e-3)) & (np.abs(Z) < 1e-3)
        self.up = Z > 0
        # mid crest plane over the patch footprint, where the film pressure is read
        self.k_mid = int(np.argmin(np.abs(Z[0, 0, :])))
        self.foot = (np.abs(X[:, :, 0]) < c.a) & (np.abs(Y[:, :, 0]) < c.a)

    def masks(self, h):
        """Solid mask and the top and bottom plate masks for crest gap h."""
        c, eps, sh, Z = self.cfg, self.eps, self.shift, self.ns.Z
        top_slab = np.minimum.reduce([self.lat, _S(Z - h / 2, eps, sh),
                                      _S(h / 2 + c.t_top - Z, eps, sh)])
        not_recess = np.maximum(self.land, _S(Z - h / 2 - c.depth, eps, sh))
        top = np.minimum(top_slab, not_recess)
        bot = np.minimum.reduce([self.lat, _S(-h / 2 - Z, eps, sh), _S(Z + h / 2 + c.t_bottom, eps, sh)])
        return np.maximum(top, bot), top, bot

    def lubrication_table(self, h_lo, h_hi, n_h=40, n=120):
        """Tier 1 viscous resistance C(h) and film added mass M_a(h) for this
        square patch, tabulated so the time loop only interpolates."""
        g = rf.Grid(R=self.cfg.a, n=n, shape="square")
        surf = (rf.recess_mesh(g, self.cfg.pitch, self.cfg.width, self.cfg.depth)
                if self.cfg.surface == "mesh" else rf.recess_smooth(g))
        hs = np.geomspace(h_lo, h_hi, n_h)
        C = np.array([-rf.solve(g, surf, h, 1.0).force for h in hs])
        Ma = np.array([-rf.solve(g, surf, h, RHO, kernel="inertial").force for h in hs])
        lh = np.log(hs)

        def lookup(h):
            x = np.log(np.clip(h, h_lo, h_hi))
            return float(np.exp(np.interp(x, lh, np.log(C)))), float(np.exp(np.interp(x, lh, np.log(Ma))))
        return lookup


@dataclass
class Drive:
    mode: str = "kinematic"
    hdot0: float = 0.02  # kinematic target opening speed
    t_ramp: float = 5e-3
    h_end: float = 1.8e-3
    sigma0: float = 2 * rf.MMHG  # fsi recoil stress
    tau: float = 3e-3  # fsi recoil rise time
    m_wall: float = 10.6  # kg / m^2 per wall (about 10 mm of myocardium)
    t_max: float = 0.5
    record: dict = field(default_factory=dict)


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def run(cfg: PatchConfig, drv: Drive, verbose=True, sample_every=2):
    P = Patch(cfg)
    ns = P.ns
    A = (2 * cfg.a) ** 2
    h, hdot, hddot, t = cfg.h0, 0.0, 0.0, 0.0
    rec = {k: [] for k in ("t", "h", "hdot", "hdot_eff", "hddot", "F_top", "F_film", "p_probe",
                           "p_gap_mean", "leak", "umax", "sigma", "F_lub", "M_a", "cg")}
    lub = P.lubrication_table(0.5 * cfg.h0, 1.5 * drv.h_end)

    step = 0
    wall = time.time()
    F_prev, a_prev = 0.0, 0.0
    while h < drv.h_end and t < drv.t_max:
        speed = max(ns.max_speed(), abs(hdot), 1e-3)
        dt = min(cfg.dt_max, cfg.cfl * cfg.dx / speed,
                 cfg.move_cfl * cfg.dx / max(abs(hdot) / 2, 1e-6))
        if drv.mode == "kinematic":
            hdot_new = drv.hdot0 * smoothstep((t + dt) / drv.t_ramp)
            hddot = (hdot_new - hdot) / dt
            hdot = hdot_new
        else:
            # each plate: m_w A z'' = sigma A + F_fluid; gap h = h0 + 2 (z - z0)
            sigma = drv.sigma0 * (1 - np.exp(-(t + dt) / drv.tau))
            M = drv.m_wall * A
            # the film force on a plate is -M_film hddot = -2 M_film z'', so the
            # plate sees twice the film added mass; 1.5x margin for 3D effects
            Ma = 3.0 * lub(h)[1]
            a_new = (sigma * A + F_prev + Ma * a_prev) / (M + Ma)
            a_prev = a_new
            hddot = 2 * a_new
            hdot = hdot + dt * hddot
        h = h + dt * hdot
        chi, top, bot = P.masks(h)
        uz = hdot / 2 * top - hdot / 2 * bot
        zero = np.zeros(ns.shape)
        ns.step(dt, chi, [zero, zero, uz])
        t += dt
        step += 1
        # net fluid force on the top plate (positive pushes it open), and the
        # film force alone: gap pressure over the footprint against the
        # surrounding blood, which leaves out the artificial load from blood
        # pushed around the plates' backs (tissue, in a heart)
        F_top = -RHO * float((ns.f[2] * P.up).sum()) * P.dV
        p = ns.pressure() * RHO
        p_far = float(p[P.far].mean())
        F_film = float((p[:, :, P.k_mid][P.foot] - p_far).sum()) * cfg.dx**2
        F_prev = F_film

        if step % sample_every == 0:
            p = p - p_far
            in_top, in_bot = top > 0.99, bot > 0.99
            hdot_eff = float(ns.u[2][in_top].mean() - ns.u[2][in_bot].mean())
            i, j = P.i_probe
            zc = np.argmin(np.abs(ns.Z[i, j, :]))
            gapz = (np.abs(ns.Z) < h / 2 - cfg.dx) & (P.lat > 0.99)
            # net seepage into the gap through each plate's mid plane, as a
            # fraction of the film's volume growth hdot * A
            z1d = ns.Z[0, 0, :]
            kt = np.argmin(np.abs(z1d - (h / 2 + (cfg.depth + cfg.t_top) / 2)))
            kb = np.argmin(np.abs(z1d + h / 2 + cfg.t_bottom / 2))
            inner = (np.abs(ns.X[:, :, 0]) < cfg.a - P.eps) & (np.abs(ns.Y[:, :, 0]) < cfg.a - P.eps)
            q_top = -float((ns.u[2][:, :, kt] - hdot / 2)[inner].sum())
            q_bot = float((ns.u[2][:, :, kb] + hdot / 2)[inner].sum())
            leak = (q_top + q_bot) * cfg.dx**2 / max(abs(hdot) * A, 1e-12)
            C, Ma = lub(h)
            rec["t"].append(t)
            rec["h"].append(h)
            rec["hdot"].append(hdot)
            rec["hdot_eff"].append(hdot_eff)
            rec["F_film"].append(F_film)
            rec["cg"].append(ns.cg_iters)
            rec["hddot"].append(hddot)
            rec["F_top"].append(F_top)
            rec["p_probe"].append(float(p[i, j, zc]))
            rec["p_gap_mean"].append(float(p[gapz].mean()) if gapz.any() else np.nan)
            rec["leak"].append(leak)
            rec["umax"].append(ns.max_speed())
            rec["sigma"].append(drv.sigma0 * (1 - np.exp(-t / drv.tau)) if drv.mode == "fsi" else 0.0)
            rec["F_lub"].append(-C * hdot)
            rec["M_a"].append(Ma)
        if verbose and step % 200 == 0:
            print(f"[{cfg.surface} {drv.mode}] t={t*1e3:.2f} ms h={h*1e3:.3f} mm hdot={hdot*1e3:.1f} mm/s "
                  f"F={F_top*1e3:.3f} mN wall={time.time()-wall:.0f}s", flush=True)
        if not np.isfinite(ns.u[0]).all():
            raise FloatingPointError("blew up")
    out = {k: np.array(v) for k, v in rec.items()}
    out["wall_clock"] = time.time() - wall
    out["steps"] = step
    out["area"] = A
    return out
