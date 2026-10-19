"""Tier 1 squeeze film model for apposed endocardial walls.

Two rigid walls separate at a uniform normal velocity hdot. The lower wall is
flat; the upper wall carries a recess pattern D(x, y) (trabecular grooves), so
the local gap is h(x, y) = h0 + D(x, y), where h0 is the crest (land) gap. In
the thin gap limit the pressure obeys the Reynolds lubrication equation

    div( h^3 / (12 mu) grad p ) = dh/dt = hdot

on a circular contact patch of radius R, with p = 0 (cavity pressure) on the
rim. Grooves deeper than they are wide are treated as rectangular ducts (see
effective_h3) rather than as wide slots. A positive hdot (walls pulling apart) gives p < 0 inside the patch, i.e.
suction that resists separation. For a smooth disk the exact solution is

    p(r) = 3 mu hdot (r^2 - R^2) / h^3,    F = -3 pi mu R^4 hdot / (2 h^3)

(Stefan adhesion). All quantities are SI unless a name says otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

MMHG = 133.322  # Pa per mmHg
MU_BLOOD = 3.5e-3  # Pa s, high shear Newtonian approximation
RHO_BLOOD = 1060.0  # kg / m^3


@dataclass
class Grid:
    """Cell centred square grid covering the disk of radius R."""

    R: float
    n: int

    def __post_init__(self) -> None:
        self.dx = 2.0 * self.R / self.n
        c = -self.R + self.dx * (np.arange(self.n) + 0.5)
        self.x, self.y = np.meshgrid(c, c, indexing="ij")
        self.r = np.hypot(self.x, self.y)
        self.inside = self.r < self.R


@dataclass
class Solution:
    p: np.ndarray  # Pa, NaN outside the patch
    h: np.ndarray  # m, local gap
    grid: Grid
    hdot: float

    @property
    def force(self) -> float:
        """Net normal force on the wall (N); negative means suction."""
        return float(np.nansum(self.p) * self.grid.dx**2)

    @property
    def area(self) -> float:
        return float(self.grid.inside.sum() * self.grid.dx**2)

    @property
    def p_min(self) -> float:
        return float(np.nanmin(self.p))

    @property
    def p_mean(self) -> float:
        """Area averaged pressure (Pa), i.e. force per unit patch area."""
        return self.force / self.area


# ---------------------------------------------------------------- geometries
#
# Each geometry returns a Surface: the recess depth D(x, y) below the crest and
# the local groove width (inf on the lands). The width lets the solver cap the
# conductance of grooves that are deeper than they are wide.


@dataclass
class Surface:
    depth: np.ndarray
    width: np.ndarray


def recess_smooth(grid: Grid) -> Surface:
    return Surface(np.zeros_like(grid.x), np.full_like(grid.x, np.inf))


def _bands(coord: np.ndarray, spacing: float, width: float) -> np.ndarray:
    """True inside grooves of the given width centred every `spacing`."""
    phase = np.mod(coord + 0.5 * spacing, spacing) - 0.5 * spacing
    return np.abs(phase) < 0.5 * width


def _surface(mask: np.ndarray, width: float, depth: float) -> Surface:
    return Surface(np.where(mask, depth, 0.0), np.where(mask, width, np.inf))


def recess_parallel(grid: Grid, spacing: float, width: float, depth: float) -> Surface:
    """Straight grooves along y, spaced `spacing` apart in x."""
    return _surface(_bands(grid.x, spacing, width), width, depth)


def recess_mesh(grid: Grid, spacing: float, width: float, depth: float) -> Surface:
    """Orthogonal groove network, the crudest stand in for a trabecular mesh."""
    g = _bands(grid.x, spacing, width) | _bands(grid.y, spacing, width)
    return _surface(g, width, depth)


def recess_fractal(
    grid: Grid, spacing: float, width: float, depth: float, levels: int = 3
) -> Surface:
    """Self similar mesh: each level halves spacing, width and depth.

    Level 0 is the coarse trabecular network; finer levels mimic the smaller
    secondary trabeculations that branch between the primary ones. Where levels
    overlap the deepest (coarsest) groove wins.
    """
    out = recess_smooth(grid)
    for k in reversed(range(levels)):
        s = 2.0**-k
        lvl = recess_mesh(grid, spacing * s, width * s, depth * s)
        m = lvl.depth >= out.depth
        m &= lvl.depth > 0
        out.depth[m] = lvl.depth[m]
        out.width[m] = lvl.width[m]
    return out


def effective_h3(h: np.ndarray, width: np.ndarray, h0: float) -> np.ndarray:
    """Per unit width flow conductance (times 12 mu) of the local gap.

    A wide thin gap conducts like h^3. A groove of total height h and width w
    is a rectangular duct, whose Poiseuille conductance with sides a >= b is
    about a b^3 (1 - 0.63 b / a) / 12 mu; spreading it over the groove width
    gives an equivalent h^3. This stops a deep, narrow recess from being
    treated as an unrealistically wide slot. A groove can never conduct less
    than the crest film h0 that still runs over it, so the result is bounded
    below by h0^3 and above by h^3.
    """
    out = h**3
    g = np.isfinite(width)
    hg, w = h[g], width[g]
    a, b = np.maximum(hg, w), np.minimum(hg, w)
    duct = a * b**3 * (1.0 - 0.63 * b / a) / w
    out[g] = np.clip(duct, min(h0**3, 1.0), hg**3)
    return out


# -------------------------------------------------------------------- solver

def solve(
    grid: Grid,
    surface: Surface,
    h0: float,
    hdot: float,
    mu: float = MU_BLOOD,
    duct_correction: bool = True,
) -> Solution:
    """Finite volume solve of the Reynolds squeeze film equation.

    Face conductances use the harmonic mean of h^3/(12 mu), which keeps flux
    continuous across the sharp land/groove steps. The rim uses a cut cell
    distance to the true circle so the curved Dirichlet boundary is not
    staircased.
    """
    h = h0 + surface.depth
    h3 = effective_h3(h, surface.width, h0) if duct_correction else h**3
    k = h3 / (12.0 * mu)
    inside = grid.inside
    idx = -np.ones(inside.shape, dtype=np.int64)
    idx[inside] = np.arange(inside.sum())
    n_unk = int(inside.sum())
    n, dx, R = grid.n, grid.dx, grid.R

    rows, cols, vals = [], [], []
    diag = np.zeros(n_unk)
    I, J = np.nonzero(inside)
    me = idx[I, J]
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        In, Jn = I + di, J + dj
        valid = (In >= 0) & (In < n) & (Jn >= 0) & (Jn < n)
        nb_in = np.zeros_like(valid)
        nb_in[valid] = inside[In[valid], Jn[valid]]

        # interior faces
        a = nb_in
        kf = 2.0 * k[I[a], J[a]] * k[In[a], Jn[a]] / (k[I[a], J[a]] + k[In[a], Jn[a]])
        rows.append(me[a])
        cols.append(idx[In[a], Jn[a]])
        vals.append(kf)
        np.add.at(diag, me[a], -kf)

        # rim faces: distance from the cell centre to the circle along the axis
        b = ~nb_in
        xc, yc = grid.x[I[b], J[b]], grid.y[I[b], J[b]]
        along, across, sign = (xc, yc, di) if di else (yc, xc, dj)
        edge = np.sqrt(np.maximum(R**2 - across**2, 0.0))
        theta = np.clip((sign * edge - along) * sign / dx, 0.05, 1.5)
        np.add.at(diag, me[b], -k[I[b], J[b]] / theta)

    rows.append(np.arange(n_unk))
    cols.append(np.arange(n_unk))
    vals.append(diag)
    A = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_unk, n_unk),
    )
    rhs = np.full(n_unk, hdot * dx**2)
    p_vec = spla.spsolve(A.tocsc(), rhs)

    p = np.full(inside.shape, np.nan)
    p[inside] = p_vec
    return Solution(p=p, h=h, grid=grid, hdot=hdot)


# ------------------------------------------------------------------ analytics

def disk_center_pressure(R: float, h: float, hdot: float, mu: float = MU_BLOOD) -> float:
    return -3.0 * mu * R**2 * hdot / h**3


def disk_force(R: float, h: float, hdot: float, mu: float = MU_BLOOD) -> float:
    return -1.5 * np.pi * mu * R**4 * hdot / h**3


def strip_center_pressure(a: float, h: float, hdot: float, mu: float = MU_BLOOD) -> float:
    """1D land of width a between two perfectly draining grooves."""
    return -1.5 * mu * a**2 * hdot / h**3


def squeeze_reynolds(h: float, hdot: float, mu: float = MU_BLOOD, rho: float = RHO_BLOOD) -> float:
    """Squeeze Reynolds number rho hdot h / mu; lubrication needs this << 1."""
    return rho * abs(hdot) * h / mu


# ------------------------------------------------------- force driven opening

def separation_time(
    grid: Grid,
    recess_fn,
    h_start: float,
    h_end: float,
    stress: float,
    mu: float = MU_BLOOD,
    n_h: int = 40,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Time for a constant opening stress to pull the walls from h_start to h_end.

    The film is linear in hdot, so F = -C(h) hdot with C(h) = -F(hdot=1). Under a
    constant applied stress sigma over the patch, hdot = sigma A / C(h) and
    t = int C(h) / (sigma A) dh. Returns (t, h samples, C samples).
    """
    hs = np.geomspace(h_start, h_end, n_h)
    surface = recess_fn(grid)
    C = np.array([-solve(grid, surface, h, 1.0, mu).force for h in hs])
    area = float(grid.inside.sum() * grid.dx**2)
    # integrate in log h, where C is close to a power law
    integrand = C * hs / (stress * area)
    t = float(np.trapezoid(integrand, np.log(hs)))
    return t, hs, C
