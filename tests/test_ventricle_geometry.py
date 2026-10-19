import numpy as np
import pytest

from squeeze_film import reynolds as rf
from ventricle import closure
from ventricle.geometry import KINDS, Cycle, Ventricle

DX = 0.25e-3


@pytest.fixture(scope="module")
def grid():
    x = -37.5e-3 + DX * np.arange(300)
    y = -103e-3 + DX * np.arange(500)
    return np.meshgrid(x, y, indexing="ij")


def test_cycle_is_continuous_and_derivative_consistent():
    c = Cycle()
    t = np.linspace(0, c.period, 20001)
    s = np.array([c.s(tt)[0] for tt in t])
    ds = np.array([c.s(tt)[1] for tt in t])
    assert np.abs(np.diff(s)).max() < 1e-3
    assert np.allclose(np.gradient(s, t)[1:-1], ds[1:-1], atol=0.05)
    assert s[0] == pytest.approx(1.0) and c.s(c.t_ej + 0.01)[0] == pytest.approx(0.0)


def test_valves_never_open_together_and_follow_flow():
    c = Cycle()
    for t in np.linspace(0, c.period, 4001):
        mit, aor = c.valve_closed(t)
        assert mit + aor >= 1.0 - 1e-9
        s, ds = c.s(t)
        if ds < -1e-6:  # ejecting: aortic open
            assert aor < 1e-9
        if ds > 1e-6:  # filling: mitral open
            assert mit < 1e-9


@pytest.mark.parametrize("kind", KINDS)
def test_myocardial_area_is_conserved(kind):
    v = Ventricle(kind=kind)
    areas = [v.wall_area(t) for t in np.linspace(0, 0.8, 41)]
    assert np.ptp(areas) / np.mean(areas) < 1e-10


def test_volume_matched_control_matches_trabeculated_cavity():
    trab, crest, vol = (Ventricle(kind=k) for k in KINDS)
    for t in (0.0, 0.3, 0.45):
        assert vol.cavity_area(t) == pytest.approx(trab.cavity_area(t), rel=2e-3)
        assert crest.cavity_area(t) < trab.cavity_area(t)
    # stroke area is shared to within the shortening of the recess layer itself
    sv = [v.cavity_area(0.0) - v.cavity_area(0.3) for v in (trab, vol, crest)]
    assert sv[0] == pytest.approx(sv[1], rel=1e-3)
    assert np.ptp(sv) / np.mean(sv) < 0.015
    # and the crest gap at end systole is shared by trab and smooth_crest
    assert trab.apposition(0.3)[0] == pytest.approx(crest.apposition(0.3)[0])


@pytest.mark.parametrize("kind", KINDS)
def test_grid_cavity_area_tracks_exact_area(grid, kind):
    X, Y = grid
    v = Ventricle(kind=kind, eps=DX)
    for t in (0.0, 0.3):
        cav = v.fields(X, Y, t)["cav"]
        # the grid cavity excludes the 1 mm valve plane strip across the base
        strip = 2 * v.a_base * v.valve_t
        assert cav.sum() * DX**2 == pytest.approx(v.cavity_area(t) - strip, rel=0.02)


@pytest.mark.parametrize("kind", KINDS)
def test_wall_velocity_is_consistent_with_mask_motion(grid, kind):
    """dA/dt of the cavity equals the flux of wall velocity through its boundary."""
    X, Y = grid
    v = Ventricle(kind=kind, eps=DX)
    for t in (0.15, 0.45):
        f = v.fields(X, Y, t)
        chi = f["chi"]
        gx, gy = np.gradient(chi, DX, DX)
        # endocardial side only: inside the inner half of the wall
        L, _ = v.L(t)
        xi = np.clip(-Y / L, 0, 1)
        W, _ = v.W(xi, t)
        inner = (np.abs(X) - W < v.recess(xi) + 0.5 * v.th(t)) & (Y < -2e-3) & (Y > -L + 1e-3)
        rate = ((f["us"] * gx + f["vs"] * gy) * inner).sum() * DX**2
        h = 1e-4
        exact = (v.cavity_area(t + h) - v.cavity_area(t - h)) / (2 * h)
        assert rate == pytest.approx(exact, rel=0.08)


def test_film_closure_reproduces_tier1_smooth_disk():
    v = Ventricle()
    R = closure.patch_radius(v)
    tab = closure.film_table("smooth", R, v.depth, v.width, v.pitch, n=120,
                             h=np.geomspace(0.1e-3, 2e-3, 8))
    h = 0.3e-3
    C, P = tab.at(np.array([h]))
    assert C[0] == pytest.approx(-rf.disk_force(R, h, 1.0), rel=0.03)
    assert P[0] == pytest.approx(-rf.disk_center_pressure(R, h, 1.0), rel=0.03)


def test_film_history_only_sucks_while_opening():
    v = Ventricle()
    R = closure.patch_radius(v)
    tab = closure.film_table("smooth", R, v.depth, v.width, v.pitch, n=80,
                             h=np.geomspace(0.05e-3, 8e-3, 12))
    t = np.linspace(0, 0.8, 401)
    hist = closure.film_history(v, tab, 0.2e-3, t)
    opening = hist["hdot"] > 0
    assert (hist["p_mean"][opening] >= 0).all()  # suction magnitude while opening
    assert (hist["p_mean"][hist["hdot"] < 0] <= 0).all()
    assert hist["h"].min() == pytest.approx(0.2e-3, rel=1e-6)
