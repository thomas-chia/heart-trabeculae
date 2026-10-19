import numpy as np
import pytest

from squeeze_film import reynolds as rf

R = 10e-3
H = 0.2e-3
HDOT = 0.05


@pytest.fixture(scope="module")
def grid():
    return rf.Grid(R=R, n=200)


def test_smooth_disk_matches_stefan_solution(grid):
    s = rf.solve(grid, rf.recess_smooth(grid), H, HDOT)
    assert s.p_min == pytest.approx(rf.disk_center_pressure(R, H, HDOT), rel=0.01)
    assert s.force == pytest.approx(rf.disk_force(R, H, HDOT), rel=0.02)
    r = grid.r[grid.inside]
    exact = 3 * rf.MU_BLOOD * HDOT * (r**2 - R**2) / H**3
    err = np.abs(s.p[grid.inside] - exact).max() / np.abs(exact).max()
    assert err < 0.01


def test_pressure_is_suction_when_walls_separate(grid):
    s = rf.solve(grid, rf.recess_smooth(grid), H, HDOT)
    assert np.nanmax(s.p) < 0
    s_close = rf.solve(grid, rf.recess_smooth(grid), H, -HDOT)
    assert np.nanmin(s_close.p) > 0


def test_linear_in_hdot_and_mu(grid):
    rec = rf.recess_mesh(grid, 2e-3, 1e-3, 1e-3)
    base = rf.solve(grid, rec, H, HDOT).force
    assert rf.solve(grid, rec, H, 3 * HDOT).force == pytest.approx(3 * base, rel=1e-9)
    assert rf.solve(grid, rec, H, HDOT, mu=2 * rf.MU_BLOOD).force == pytest.approx(2 * base, rel=1e-9)


def test_inverse_cube_gap_scaling(grid):
    f1 = rf.solve(grid, rf.recess_smooth(grid), H, HDOT).force
    f2 = rf.solve(grid, rf.recess_smooth(grid), 2 * H, HDOT).force
    assert f1 / f2 == pytest.approx(8.0, rel=1e-9)


def test_deep_parallel_grooves_approach_strip_limit():
    # land width a between grooves much deeper than the gap behaves like a 1D strip
    g = rf.Grid(R=R, n=400)
    spacing, width = 2e-3, 1e-3
    # very deep, no duct cap: the grooves are effectively at cavity pressure
    rec = rf.recess_parallel(g, spacing, width, depth=5e-3)
    s = rf.solve(g, rec, H, HDOT, duct_correction=False)
    expected = rf.strip_center_pressure(spacing - width, H, HDOT)
    assert s.p_min == pytest.approx(expected, rel=0.05)


def test_grid_convergence_for_mesh_grooves():
    vals = []
    for n in (100, 200, 400):
        g = rf.Grid(R=R, n=n)
        vals.append(rf.solve(g, rf.recess_mesh(g, 2e-3, 1e-3, 1e-3), H, HDOT).force)
    assert abs(vals[2] - vals[1]) < abs(vals[1] - vals[0])
    assert abs(vals[2] - vals[1]) / abs(vals[2]) < 0.03


@pytest.mark.parametrize("h0", [H, 1e-3, 3e-3])
def test_grooves_never_increase_suction(grid, h0):
    smooth = rf.solve(grid, rf.recess_smooth(grid), h0, HDOT).force
    for rec in (
        rf.recess_parallel(grid, 2e-3, 1e-3, 1e-3),
        rf.recess_mesh(grid, 2e-3, 1e-3, 1e-3),
        rf.recess_fractal(grid, 4e-3, 1e-3, 2e-3),
    ):
        assert abs(rf.solve(grid, rec, h0, HDOT).force) <= abs(smooth) * (1 + 1e-9)


def test_separation_time_matches_analytic_for_smooth_disk():
    g = rf.Grid(R=R, n=160)
    stress = 2 * rf.MMHG
    h_a, h_b = 0.1e-3, 1e-3
    t, _, _ = rf.separation_time(g, rf.recess_smooth, h_a, h_b, stress, n_h=60)
    exact = 0.75 * rf.MU_BLOOD * R**2 / stress * (1 / h_a**2 - 1 / h_b**2)
    assert t == pytest.approx(exact, rel=0.03)


def test_duct_correction_limits():
    h = np.array([1e-4, 1e-3, 5e-3])
    # wide, shallow gaps are untouched (up to the small side wall term)
    assert rf.effective_h3(h, np.full(3, np.inf), 1e-4) == pytest.approx(h**3)
    # a groove much deeper than wide conducts like h w^2, not h^3
    w = 1e-4
    out = rf.effective_h3(np.array([1e-2]), np.array([w]), 1e-5)[0]
    assert out == pytest.approx(1e-2 * w**2 * (1 - 0.63 * w / 1e-2))
    assert out < 1e-2**3
    # but a narrow groove never conducts less than the crest film above it
    h0 = 1e-3
    assert rf.effective_h3(np.array([h0 + 5e-4]), np.array([2.5e-4]), h0)[0] == pytest.approx(h0**3)


def test_duct_correction_raises_suction(grid):
    rec = rf.recess_mesh(grid, 2e-3, 1e-3, 2e-3)
    capped = rf.solve(grid, rec, H, HDOT).force
    uncapped = rf.solve(grid, rec, H, HDOT, duct_correction=False).force
    assert abs(capped) > abs(uncapped)
