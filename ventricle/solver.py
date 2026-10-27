"""2D incompressible Navier Stokes with Brinkman volume penalization.

Pseudo spectral on a doubly periodic box. Solid bodies (myocardium, valves)
enter through a mask chi in [0, 1] and a prescribed solid velocity u_s; the
momentum equation gains a term -chi (u - u_s) / eta that drives the fluid
velocity toward the solid velocity inside bodies (Angot et al. 1999,
Kolomenskiy and Schneider 2009). Each step:

1. advection by second order Adams Bashforth (2/3 rule dealiasing),
2. exact viscous decay via an integrating factor,
3. implicit pointwise penalization,
4. exact spectral projection onto divergence free fields,

with steps 3 and 4 repeated n_proj times. A single penalize then project pass
lets the projection push fluid through thin bodies at a speed of order
dt |grad p| / rho (the bodies act porous); alternating the two operators
converges toward the field that is both divergence free and equal to u_s in
the solid, and cutting dt shrinks the leak linearly. The summed projection
potentials are the pressure (divided by density), so pressure costs no extra
solve. Quantities are SI; p is kinematic (Pa / rho).
"""

from __future__ import annotations

import numpy as np
import scipy.fft as sfft
from scipy.ndimage import map_coordinates

WORKERS = 1


def _fft(a):
    return sfft.rfft2(a, workers=WORKERS)


def _ifft(a, shape):
    return sfft.irfft2(a, s=shape, workers=WORKERS)


class PenalizedNS:
    def __init__(self, nx: int, ny: int, lx: float, ly: float, x0: float, y0: float,
                 nu: float, eta_factor: float = 1e-2, filter_order: int = 36,
                 n_proj: int = 2):
        """Grid of nx by ny cells on [x0, x0 + lx) by [y0, y0 + ly), indexed [i, j] = (x, y)."""
        self.nx, self.ny, self.lx, self.ly, self.nu = nx, ny, lx, ly, nu
        self.dx, self.dy = lx / nx, ly / ny
        x = x0 + self.dx * np.arange(nx)
        y = y0 + self.dy * np.arange(ny)
        self.X, self.Y = np.meshgrid(x, y, indexing="ij")
        self.shape = (nx, ny)
        kx = 2 * np.pi * sfft.fftfreq(nx, self.dx)
        ky = 2 * np.pi * sfft.rfftfreq(ny, self.dy)
        self.KX, self.KY = np.meshgrid(kx, ky, indexing="ij")
        self.K2 = self.KX**2 + self.KY**2
        self.K2inv = np.where(self.K2 > 0, 1.0 / np.where(self.K2 > 0, self.K2, 1.0), 0.0)
        kxmax, kymax = np.abs(kx).max(), np.abs(ky).max()
        self.dealias = (np.abs(self.KX) < 2 / 3 * kxmax) & (np.abs(self.KY) < 2 / 3 * kymax)
        # weak exponential filter on the top of the spectrum keeps the sharp
        # penalized interfaces from seeding grid scale noise
        kn = np.sqrt((self.KX / kxmax) ** 2 + (self.KY / kymax) ** 2) / np.sqrt(2)
        self.filt = np.exp(-36.0 * kn**filter_order) if filter_order else np.ones_like(kn)
        self.eta_factor = eta_factor
        self.n_proj = n_proj
        self.u = np.zeros(self.shape)
        self.v = np.zeros(self.shape)
        self.uh = _fft(self.u)
        self.vh = _fft(self.v)
        self.p = np.zeros(self.shape)
        self.fx = np.zeros(self.shape)  # last penalization force per unit mass
        self.fy = np.zeros(self.shape)
        self._nl_prev = None
        self._dt_prev = None
        self.t = 0.0

    # ------------------------------------------------------------ operators
    def ddx(self, ah):
        return 1j * self.KX * ah

    def ddy(self, ah):
        return 1j * self.KY * ah

    def vorticity(self):
        return _ifft(self.ddx(self.vh) - self.ddy(self.uh), self.shape)

    def divergence(self):
        return _ifft(self.ddx(self.uh) + self.ddy(self.vh), self.shape)

    def _nonlinear(self):
        d = self.dealias
        uh, vh = self.uh * d, self.vh * d
        u, v = _ifft(uh, self.shape), _ifft(vh, self.shape)
        ux, uy = _ifft(self.ddx(uh), self.shape), _ifft(self.ddy(uh), self.shape)
        vx, vy = _ifft(self.ddx(vh), self.shape), _ifft(self.ddy(vh), self.shape)
        nx = _fft(-(u * ux + v * uy)) * d
        ny = _fft(-(u * vx + v * vy)) * d
        return nx, ny

    # ------------------------------------------------------------- stepping
    def step(self, dt: float, chi=None, us=None, vs=None, body=None):
        """Advance one step. body is an optional constant (fx, fy) acceleration."""
        nlx, nly = self._nonlinear()
        if body is not None:
            nlx = nlx.copy()
            nly = nly.copy()
            nlx[0, 0] += body[0] * self.nx * self.ny
            nly[0, 0] += body[1] * self.nx * self.ny
        if self._nl_prev is None:
            ax, ay = nlx, nly
        else:
            r = dt / self._dt_prev
            ax = (1 + r / 2) * nlx - (r / 2) * self._nl_prev[0]
            ay = (1 + r / 2) * nly - (r / 2) * self._nl_prev[1]
        self._nl_prev, self._dt_prev = (nlx, nly), dt

        ef = np.exp(-self.nu * self.K2 * dt)
        uh = (self.uh + dt * ax) * ef
        vh = (self.vh + dt * ay) * ef

        ph_tot = 0.0
        u0 = v0 = None
        for _ in range(self.n_proj if chi is not None else 1):
            if chi is not None:
                u, v = _ifft(uh, self.shape), _ifft(vh, self.shape)
                if u0 is None:
                    u0, v0 = u, v
                lam = chi / self.eta_factor  # dt / eta, eta = eta_factor * dt
                uh = _fft((u + lam * us) / (1 + lam))
                vh = _fft((v + lam * vs) / (1 + lam))
            # projection; the potential is the kinematic pressure
            div = self.ddx(uh) + self.ddy(vh)
            ph = -div * self.K2inv / dt
            uh = uh - dt * self.ddx(ph)
            vh = vh - dt * self.ddy(ph)
            ph_tot = ph_tot + ph
        uh, vh, ph = uh * self.filt, vh * self.filt, ph_tot
        self.uh, self.vh, self._ph = uh, vh, ph
        self.u, self.v = _ifft(uh, self.shape), _ifft(vh, self.shape)
        if chi is not None:
            # net force per unit mass the bodies exerted on the fluid this step
            self.fx, self.fy = (self.u - u0) / dt, (self.v - v0) / dt
            self.fx += _ifft(self.ddx(ph), self.shape)
            self.fy += _ifft(self.ddy(ph), self.shape)
        self.t += dt

    def pressure(self):
        return _ifft(self._ph, self.shape)

    def set_velocity(self, u, v):
        self.uh, self.vh = _fft(u), _fft(v)
        self.u, self.v = _ifft(self.uh, self.shape), _ifft(self.vh, self.shape)
        self._nl_prev = None

    def max_speed(self):
        return float(np.sqrt(self.u**2 + self.v**2).max())


def advect_scalar(c: np.ndarray, u: np.ndarray, v: np.ndarray, dt: float, dx: float, dy: float):
    """Semi Lagrangian advection with linear interpolation (bounded, monotone)."""
    nx, ny = c.shape
    i = np.arange(nx)[:, None] - u * dt / dx
    j = np.arange(ny)[None, :] - v * dt / dy
    return map_coordinates(c, [i, j], order=1, mode="grid-wrap")
