import numpy as np
import pytest

from apex3d import patch as P
from apex3d.reduced import FilmModel, open_walls, recoil, time_to_gap
from apex3d.solver3d import PenalizedNS3D
from squeeze_film import reynolds as rf


def test_abc_beltrami_flow_decays_exactly():
    n, nu = 16, 0.05
    s = PenalizedNS3D((n, n, n), (2 * np.pi,) * 3, (0, 0, 0), nu, filter_order=0)
    X, Y, Z = s.X, s.Y, s.Z
    u0 = [np.sin(Z) + 0.6 * np.cos(Y), 0.8 * np.sin(X) + np.cos(Z), 0.6 * np.sin(Y) + 0.8 * np.cos(X)]
    s.set_velocity(u0)
    for _ in range(50):
        s.step(0.02)
    decay = np.exp(-nu * 1.0)
    assert max(np.abs(a - b * decay).max() for a, b in zip(s.u, u0)) < 1e-10
    assert np.abs(s.divergence()).max() < 1e-10


@pytest.mark.parametrize("shift", [0.0, 0.25, 0.5, 0.75])
def test_brinkman_wall_sits_on_the_surface_at_any_subcell_position(shift):
    """Penalized channel: with the viscous scaled Brinkman profile and the
    patch's recentred 3 cell ramp, the effective no slip wall lands on the
    geometric surface wherever it falls between grid nodes."""
    n, nu, g = (8, 8, 64), 0.2, 1.0
    s = PenalizedNS3D(n, (0.5, 0.5, 2.0), (0, 0, -1.0), nu, profile="brinkman")
    dz = s.d[2]
    h = 1.0 + 2 * shift * dz
    chi = P._S(np.abs(s.Z) - h / 2, P.PatchConfig.ramp_cells * dz, P.PatchConfig.ramp_shift_cells * dz)
    zero = [np.zeros(s.shape)] * 3
    dt = 0.04 * dz**2 / nu
    for _ in range(int(2.5 / dt)):
        s.step(dt, chi, zero, body=(g, 0, 0))
    uc = s.u[0][0, 0, np.argmin(np.abs(s.Z[0, 0]))]
    offset_cells = (2 * np.sqrt(2 * nu * uc / g) - h) / 2 / dz
    assert abs(offset_cells) < 0.15


def test_coupled_solve_enforces_rigid_body_motion():
    """A slab translating through a periodic box must move at its prescribed
    speed; plain alternating penalize and project passes let it lag."""
    n = (16, 16, 48)
    lag = {}
    for coupled in (False, True):
        s = PenalizedNS3D(n, (4e-3, 4e-3, 12e-3), (-2e-3, -2e-3, -6e-3), 3.3e-6,
                          coupled=coupled, profile="brinkman", n_proj=3, cg_tol=3e-3)
        chi = np.minimum(np.clip((1.5e-3 - np.abs(s.Z)) / 5e-4, 0, 1),
                         np.clip((1.5e-3 - np.abs(s.X)) / 5e-4, 0, 1))
        v = 0.01
        us = [np.zeros(s.shape), np.zeros(s.shape), v * (chi > 0)]
        for _ in range(40):
            s.step(1e-4, chi, us)
        lag[coupled] = 1 - s.u[2][chi > 0.99].mean() / v
    assert lag[True] < 0.01
    assert lag[True] < lag[False]


def test_square_patch_lubrication_matches_series():
    R, h = 6e-3, 1e-3
    g = rf.Grid(R=R, n=120, shape="square")
    F = rf.solve(g, rf.recess_smooth(g), h, 1.0).force
    assert F == pytest.approx(rf.square_force(2 * R, h, 1.0), rel=2e-3)


def test_inertial_kernel_gives_disk_added_mass():
    R, h = 8e-3, 0.5e-3
    g = rf.Grid(R=R, n=160)
    Ma = -rf.solve(g, rf.recess_smooth(g), h, rf.RHO_BLOOD, kernel="inertial").force
    assert Ma == pytest.approx(rf.disk_added_mass(R, h), rel=5e-3)


def test_grooves_cut_added_mass_less_than_viscous_resistance():
    # drainage helps viscous flow (h^3) far more than inertial flow (h)
    g = rf.Grid(R=6e-3, n=120, shape="square")
    sm, me = rf.recess_smooth(g), rf.recess_mesh(g, 2e-3, 1e-3, 1.5e-3)
    h = 0.3e-3
    c_ratio = rf.solve(g, me, h, 1.0).force / rf.solve(g, sm, h, 1.0).force
    m_ratio = (rf.solve(g, me, h, 1.0, kernel="inertial").force
               / rf.solve(g, sm, h, 1.0, kernel="inertial").force)
    assert c_ratio < m_ratio < 1


def test_reduced_model_quasi_static_limit_is_tier1():
    m = FilmModel.build(surface="smooth", n=100)
    s = 2 * rf.MMHG
    r = open_walls(m, recoil(s, 0), 10.6, 0.1e-3, 1e-3, inertia=False, t_max=2)
    R = 8.75e-3
    exact = 0.75 * rf.MU_BLOOD * R**2 / s * (1 / 0.1e-3**2 - 1 / 1e-3**2)
    assert r["t_end"] == pytest.approx(exact, rel=0.01)


def test_reduced_model_ballistic_without_film():
    # with a vanishing film (huge gap) the walls follow (m/2) hddot = sigma
    m = FilmModel.build(surface="smooth", n=60, h=np.geomspace(1e-3, 0.5, 20))
    m.C = m.C * 1e-9
    m.Ma = m.Ma * 1e-9
    s, mw = 2 * rf.MMHG, 10.6
    r = open_walls(m, recoil(s, 0), mw, 0.1, 0.101, t_max=1.0)
    t_exact = np.sqrt(2 * 1e-3 / (2 * s / mw))
    assert r["t_end"] == pytest.approx(t_exact, rel=1e-3)
    assert time_to_gap(r, 0.1005) < r["t_end"]


def test_patch_masks_place_walls_at_the_crest_gap():
    """The chi = 0.5 contour sits ramp_shift toward the fluid from each
    surface, so the chi < 0.5 column spans the crest gap less two shifts on a
    land, and the groove depth more under a groove."""
    cfg = P.PatchConfig(surface="mesh", dx=0.2e-3, workers=1)
    pt = P.Patch(cfg)
    h = 0.8e-3
    chi, top, bot = pt.masks(h)
    X, Y, Z = pt.ns.X, pt.ns.Y, pt.ns.Z
    sh = cfg.ramp_shift_cells * cfg.dx
    i = np.argmin(np.abs(X[:, 0, 0] - 1e-3))  # land centre
    j = np.argmin(np.abs(Y[0, :, 0] - 1e-3))
    near = np.abs(Z[i, j]) < 2e-3
    assert (chi[i, j][near] < 0.5).sum() * cfg.dx == pytest.approx(h - 2 * sh, abs=cfg.dx)
    ig = np.argmin(np.abs(X[:, 0, 0]))  # groove centre
    col = (chi[ig, j] < 0.5) & (Z[ig, j] > -h / 2 - 0.5e-3) & (Z[ig, j] < h / 2 + cfg.depth + 0.5e-3)
    assert col.sum() * cfg.dx == pytest.approx(h + cfg.depth - 2 * sh, abs=2 * cfg.dx)


@pytest.mark.slow
def test_resolved_3d_patch_matches_lubrication_at_low_reynolds():
    """Cross tier check: separating smooth plates resolved in 3D at low
    squeeze Reynolds number reproduce the Tier 1 square patch force."""
    cfg = P.PatchConfig(surface="smooth", dx=0.2e-3, workers=2)
    out = P.run(cfg, P.Drive(mode="kinematic", hdot0=0.006, t_ramp=10e-3, h_end=1.0e-3),
                verbose=False)
    m = out["t"] > 15e-3
    C = -out["F_lub"][m] / out["hdot"][m]
    ratio = out["F_film"][m].mean() / (-(C * out["hdot_eff"][m]).mean())
    assert ratio == pytest.approx(1.0, abs=0.12)
    assert np.mean(out["hdot_eff"][m] / out["hdot"][m]) > 0.99
