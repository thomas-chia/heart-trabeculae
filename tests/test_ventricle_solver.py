import numpy as np
import pytest

from squeeze_film import reynolds as rf
from ventricle.solver import PenalizedNS, advect_scalar


def smooth_box(y, lo, hi, eps):
    """Linear ramp indicator of lo < y < hi with ramp width eps."""
    return np.clip(np.minimum(y - lo, hi - y) / eps + 0.5, 0, 1)


def test_taylor_green_decay_is_exact():
    n, nu = 64, 0.05
    s = PenalizedNS(n, n, 2 * np.pi, 2 * np.pi, 0, 0, nu, filter_order=0)
    u0 = np.sin(s.X) * np.cos(s.Y)
    v0 = -np.cos(s.X) * np.sin(s.Y)
    s.set_velocity(u0, v0)
    dt, T = 0.01, 2.0
    for _ in range(int(round(T / dt))):
        s.step(dt)
    decay = np.exp(-2 * nu * T)
    err = np.abs(s.u - u0 * decay).max()
    assert err < 1e-6
    assert np.abs(s.divergence()).max() < 1e-8
    # pressure of this Taylor Green field: p = (cos 2x + cos 2y) / 4 * decay^2 (kinematic)
    p = s.pressure()
    p_exact = (np.cos(2 * s.X) + np.cos(2 * s.Y)) / 4 * decay**2
    assert np.abs((p - p.mean()) - p_exact).max() < 2e-3


def test_penalized_poiseuille_channel():
    # fluid for |y| < h/2, solid walls elsewhere, driven by a body force
    nx, ny, h, nu, g = 16, 128, 1.0, 0.2, 1.0
    s = PenalizedNS(nx, ny, 0.5, 2.0, 0, -1.0, nu)
    dy = s.dy
    chi = 1 - smooth_box(s.Y, -h / 2, h / 2, dy)
    zero = np.zeros(s.shape)
    # nu dt / dy^2 = 0.4 here; the splitting error is first order in dt and
    # far smaller in the ventricle runs, where nu dt / dx^2 is about 1e-5
    dt = 0.0005
    for _ in range(16000):
        s.step(dt, chi, zero, zero, body=(g, 0.0))
    y = s.Y[0]
    fluid = np.abs(y) < h / 2 - 2 * dy
    exact = g / (2 * nu) * (h**2 / 4 - y**2)
    err = np.abs(s.u[0, fluid] - exact[fluid]).max() / exact.max()
    assert err < 0.015
    assert np.abs(s.u[0, np.abs(y) > h / 2 + 2 * dy]).max() < 0.01 * exact.max()


def test_moving_penalized_wall_drives_couette_flow():
    nx, ny, h, nu, U = 16, 128, 1.0, 0.5, 0.3
    s = PenalizedNS(nx, ny, 0.5, 2.0, 0, -1.0, nu)
    dy = s.dy
    chi = 1 - smooth_box(s.Y, -h / 2, h / 2, dy)
    us = np.where(s.Y > 0, U, 0.0)
    zero = np.zeros(s.shape)
    for _ in range(3000):
        s.step(0.002, chi, us, zero)
    y = s.Y[0]
    fluid = np.abs(y) < h / 2 - 2 * dy
    exact = U * (y + h / 2) / h
    assert np.abs(s.u[0, fluid] - exact[fluid]).max() < 0.04 * U


def test_semi_lagrangian_is_bounded_and_translates():
    n = 64
    c = np.zeros((n, n))
    c[20:30, 20:30] = 1.0
    u = np.full_like(c, 1.0)
    v = np.zeros_like(c)
    out = c.copy()
    for _ in range(10):
        out = advect_scalar(out, u, v, 1.0, 1.0, 1.0)
    assert out.min() >= 0 and out.max() <= 1
    assert out.sum() == pytest.approx(c.sum(), rel=1e-12)
    assert np.allclose(out, np.roll(c, 10, axis=0))


@pytest.mark.slow
def test_squeeze_film_between_resolved_plates_matches_lubrication():
    """Cross tier check: separating plates in the Navier Stokes solver
    reproduce the 1D strip lubrication pressure used in Tier 1.

    The plates are 9 gaps long so the lubrication limit applies, and dt is
    small so the penalized plates stay nearly impermeable (see solver doc)."""
    mu, rho = rf.MU_BLOOD, rf.RHO_BLOOD
    nu = mu / rho
    half_len, gap, thick, hdot = 10e-3, 1e-3, 2e-3, 2e-4
    s = PenalizedNS(512, 256, 40e-3, 20e-3, -20e-3, -10e-3, nu, n_proj=3)
    e = s.dx
    t, dt = 0.0, 2e-4
    for _ in range(2500):
        g = gap + hdot * t
        plate_x = smooth_box(s.X, -half_len, half_len, e)
        top = plate_x * smooth_box(s.Y, g / 2, g / 2 + thick, e)
        bot = plate_x * smooth_box(s.Y, -g / 2 - thick, -g / 2, e)
        chi = np.maximum(top, bot)
        vs = hdot / 2 * top - hdot / 2 * bot
        s.step(dt, chi, np.zeros(s.shape), vs)
        t += dt
    g = gap + hdot * t
    p = s.pressure() * rho
    far = (np.abs(s.X) > half_len + 5e-3)
    p = p - p[far].mean()
    i0 = np.argmin(np.abs(s.X[:, 0]))
    j0 = np.argmin(np.abs(s.Y[0]))
    expected = -6 * mu * hdot * half_len**2 / g**3  # 1D strip, two walls moving apart
    assert p[i0, j0] == pytest.approx(expected, rel=0.15)
