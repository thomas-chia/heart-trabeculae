"""3D incompressible Navier Stokes with Brinkman volume penalization.

The 3D counterpart of ventricle/solver.py, with the same scheme: pseudo
spectral on a triply periodic box, second order Adams Bashforth advection
with 2/3 rule dealiasing, an exact viscous integrating factor, implicit
pointwise penalization toward a prescribed solid velocity, and exact spectral
projection, with penalization and projection alternated n_proj times so thin
bodies stay nearly impermeable. The summed projection potentials are the
kinematic pressure. Arrays are indexed [i, j, k] = (x, y, z). SI units.
"""

from __future__ import annotations

import numpy as np
import scipy.fft as sfft


class PenalizedNS3D:
    def __init__(self, n, L, origin, nu, eta_factor=1e-2, filter_order=36,
                 n_proj=2, workers=1, profile="linear", coupled=False,
                 cg_tol=1e-6, cg_maxit=200, brinkman_K=1e4):
        """n = (nx, ny, nz) cells on a box of size L = (lx, ly, lz) starting at origin."""
        self.n = tuple(int(v) for v in n)
        self.L = tuple(float(v) for v in L)
        self.nu, self.eta_factor, self.n_proj, self.workers = nu, eta_factor, n_proj, workers
        self.profile = profile
        self.brinkman_K = brinkman_K
        self.coupled, self.cg_tol, self.cg_maxit = coupled, cg_tol, cg_maxit
        self.cg_iters = 0
        self._phi = None
        self.d = tuple(l / m for l, m in zip(self.L, self.n))
        axes = [o + h * np.arange(m) for o, h, m in zip(origin, self.d, self.n)]
        self.X, self.Y, self.Z = np.meshgrid(*axes, indexing="ij")
        self.shape = self.n
        kx = 2 * np.pi * sfft.fftfreq(self.n[0], self.d[0])
        ky = 2 * np.pi * sfft.fftfreq(self.n[1], self.d[1])
        kz = 2 * np.pi * sfft.rfftfreq(self.n[2], self.d[2])
        # Parseval weights for inner products over the half spectrum
        w = np.full(kz.shape, 2.0)
        w[0] = 1.0
        if self.n[2] % 2 == 0:
            w[-1] = 1.0
        self._w = w[None, None, :]
        self.K = np.meshgrid(kx, ky, kz, indexing="ij")
        K2 = sum(k**2 for k in self.K)
        self.K2 = K2
        self.K2inv = np.where(K2 > 0, 1.0 / np.where(K2 > 0, K2, 1.0), 0.0)
        kmax = [np.abs(k).max() for k in (kx, ky, kz)]
        self.dealias = np.ones(K2.shape, dtype=bool)
        kn2 = np.zeros(K2.shape)
        for k, km in zip(self.K, kmax):
            self.dealias &= np.abs(k) < 2 / 3 * km
            kn2 = kn2 + (k / km) ** 2
        kn = np.sqrt(kn2 / 3)
        self.filt = np.exp(-36.0 * kn**filter_order) if filter_order else np.ones_like(kn)
        self.uh = [self._fft(np.zeros(self.shape)) for _ in range(3)]
        self.u = [np.zeros(self.shape) for _ in range(3)]
        self.f = [np.zeros(self.shape) for _ in range(3)]
        self._ph = np.zeros_like(self.uh[0])
        self._nl_prev = None
        self._dt_prev = None
        self.t = 0.0

    def _dot(self, a, b):
        return float((self._w * (a.real * b.real + a.imag * b.imag)).sum())

    def _coupled_step(self, dt, uh, chi, us):
        """Implicit penalization and projection solved together.

        With a = 1 / (1 + lambda) and b = a (u* + lambda u_s), the velocity
        u = b - a grad(phi) is divergence free when div(a grad phi) = div b.
        That variable coefficient Poisson problem is solved by conjugate
        gradients on the half spectrum, preconditioned by the constant
        coefficient FFT Laplacian and warm started from the last step, so the
        solid velocity is enforced to O(1 / lambda) instead of leaking
        through the projection. A final exact projection removes the CG
        residual. phi / dt is the kinematic pressure.
        """
        ustar = [self._ifft(c) for c in uh]
        a = 1.0 / (1.0 + self.penalty(chi, dt))
        b = [a * (u + (1.0 / a - 1.0) * v) for u, v in zip(ustar, us)]
        bh = [self._fft(c) for c in b]
        rhs = -sum(1j * k * c for k, c in zip(self.K, bh))  # -div b

        def op(ph):  # -div(a grad phi), SPD on mean free fields
            g = [self._fft(a * self._ifft(1j * k * ph)) for k in self.K]
            return -sum(1j * k * c for k, c in zip(self.K, g))

        x = (self._phi * (dt / self._dt_prev) if self._phi is not None and self._dt_prev
             else rhs * self.K2inv)
        x[0, 0, 0] = 0.0
        r = rhs - op(x)
        r[0, 0, 0] = 0.0
        z = r * self.K2inv
        pdir = z.copy()
        rz = self._dot(r, z)
        r0 = np.sqrt(self._dot(rhs, rhs * self.K2inv)) + 1e-300
        it = 0
        while it < self.cg_maxit and np.sqrt(abs(rz)) > self.cg_tol * r0:
            Ap = op(pdir)
            alpha = rz / self._dot(pdir, Ap)
            x = x + alpha * pdir
            r = r - alpha * Ap
            z = r * self.K2inv
            rz_new = self._dot(r, z)
            pdir = z + (rz_new / rz) * pdir
            rz = rz_new
            it += 1
        self.cg_iters = it
        self._phi = x
        uh = [bc - self._fft(a * self._ifft(1j * k * x)) for bc, k in zip(bh, self.K)]
        # remove the remaining divergence exactly
        div = sum(1j * k * c for k, c in zip(self.K, uh))
        corr = -div * self.K2inv
        uh = [c - 1j * k * corr for c, k in zip(uh, self.K)]
        ph = (x + corr) / dt
        self.uh = [c * self.filt for c in uh]
        self._ph = ph
        self.u = [self._ifft(c) for c in self.uh]
        gp = [self._ifft(g) for g in self._grad(ph)]
        self.f = [(u1 - u0) / dt + g for u1, u0, g in zip(self.u, ustar, gp)]
        self.t += dt

    def penalty(self, chi, dt=None):
        """dt / eta as a function of the mask.

        "linear": chi / eta_factor. "exp": graded geometrically,
        (1 + 1/eta_factor)^chi - 1. "brinkman": scaled by the viscous rate
        per step, lambda_v = nu dt / dx^2, as lambda_v chi K^(2 chi - 1).

        A penalized wall acts as no slip at the first grid node where
        lambda is large against lambda_v, so with the first two profiles the
        effective wall snaps from node to node as a surface moves and sits up
        to a cell away from the geometric one. Scaling with lambda_v makes the
        ramp a graded Brinkman layer: the penalty passes through lambda_v at
        chi = 0.5, so a ramp centred on the surface puts the effective wall on
        it and moves it continuously with sub cell position, while the solid
        core still sees K lambda_v.
        """
        if self.profile == "brinkman":
            lam_v = self.nu * dt / min(self.d) ** 2
            return lam_v * chi * self.brinkman_K ** (2 * chi - 1)
        lmax = 1.0 / self.eta_factor
        if self.profile == "exp":
            return np.expm1(chi * np.log1p(lmax))
        return chi * lmax

    def _fft(self, a):
        return sfft.rfftn(a, workers=self.workers)

    def _ifft(self, a):
        return sfft.irfftn(a, s=self.shape, workers=self.workers)

    def _grad(self, ah):
        return [1j * k * ah for k in self.K]

    def divergence(self):
        return self._ifft(sum(1j * k * a for k, a in zip(self.K, self.uh)))

    def pressure(self):
        return self._ifft(self._ph)

    def max_speed(self):
        return float(np.sqrt(sum(c**2 for c in self.u)).max())

    def set_velocity(self, u):
        self.uh = [self._fft(c) for c in u]
        self.u = [self._ifft(c) for c in self.uh]
        self._nl_prev = None

    def _nonlinear(self):
        d = self.dealias
        uh = [c * d for c in self.uh]
        u = [self._ifft(c) for c in uh]
        out = []
        for c in uh:
            g = [self._ifft(gc) for gc in self._grad(c)]
            out.append(self._fft(-(u[0] * g[0] + u[1] * g[1] + u[2] * g[2])) * d)
        return out

    def step(self, dt, chi=None, us=None, body=None):
        """Advance one step; us is a list of three solid velocity components."""
        nl = self._nonlinear()
        if body is not None:
            for c in range(3):
                nl[c] = nl[c].copy()
                nl[c][0, 0, 0] += body[c] * np.prod(self.n)
        if self._nl_prev is None:
            adv = nl
        else:
            r = dt / self._dt_prev
            adv = [(1 + r / 2) * a - (r / 2) * b for a, b in zip(nl, self._nl_prev)]
        self._nl_prev, self._dt_prev = nl, dt

        ef = np.exp(-self.nu * self.K2 * dt)
        uh = [(a + dt * b) * ef for a, b in zip(self.uh, adv)]
        if chi is not None and self.coupled:
            self._coupled_step(dt, uh, chi, us)
            return
        ph_tot = 0.0
        u0 = None
        lam = None if chi is None else self.penalty(chi, dt)
        for _ in range(self.n_proj if chi is not None else 1):
            if chi is not None:
                u = [self._ifft(c) for c in uh]
                if u0 is None:
                    u0 = u
                uh = [self._fft((a + lam * b) / (1 + lam)) for a, b in zip(u, us)]
            div = sum(1j * k * a for k, a in zip(self.K, uh))
            ph = -div * self.K2inv / dt
            uh = [a - dt * 1j * k * ph for a, k in zip(uh, self.K)]
            ph_tot = ph_tot + ph
        self.uh = [a * self.filt for a in uh]
        self._ph = ph_tot
        self.u = [self._ifft(a) for a in self.uh]
        if chi is not None:
            # force per unit mass the bodies exerted on the fluid this step
            gp = [self._ifft(g) for g in self._grad(ph_tot)]
            self.f = [(a - b) / dt + g for a, b, g in zip(self.u, u0, gp)]
        self.t += dt
