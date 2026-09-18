"""
Gaussian-process (RBF-kernel) random boundary displacement for the inner
cylindrical surface (r = r1) of the quarter tube.

Design notes
------------
The inner surface is 2-D, parametrised by (theta, z) with theta in [0, pi/2]
and z in [0, h].  We keep the DeepONet branch input 1-D by sampling three
independent 1-D GPs along the arc-length coordinate

        s = r1 * theta,      s in [0, L],   L = r1 * pi/2

and fixing the z-profile analytically.  This mirrors the original analytic BC
(u_x, u_y uniform in z; u_z quadratic in z).

Boundary-condition COMPATIBILITY (this is the part that a raw GP sample breaks)
------------------------------------------------------------------------------
The quarter model carries three homogeneous constraints that the prescribed
inner displacement MUST respect, otherwise the Dirichlet sets conflict along
the shared edges and DOLFINx silently lets the last-applied bc win:

    x = 0 plane  (tag SYM_Y) locks u_x  ->  u_x must vanish at theta = pi/2
    y = 0 plane  (tag SYM_X) locks u_y  ->  u_y must vanish at theta = 0
    z = 0, z = h (rollers)   lock u_z   ->  u_z must vanish at z = 0 and z = h

The original expression satisfied all three by construction
(0.01*x^2 = x*(0.01*x) vanishes at x=0, etc.).  We reproduce that here with
multiplicative tapers:

    u_x(theta, z) = cos(theta)      * g_x(s)
    u_y(theta, z) = sin(theta)      * g_y(s)
    u_z(theta, z) = 4 z (h - z)/h^2 * g_z(s)

where g_x, g_y, g_z ~ GP(0, k_RBF).  The tapers are exactly 1 at the "free"
end of their range, so the GP output_scale is directly the peak displacement
amplitude.

If you ever move to the FULL 360-degree tube, swap RBF -> RBF_periodic below so
that g(0) = g(2*pi*r1) and the field closes on itself.
"""

import numpy as np

__all__ = ["RBF", "RBF_periodic", "gp_cholesky", "InnerBCSampler"]


# ---------------------------------------------------------------------------
# 1.  kernels
# ---------------------------------------------------------------------------
def RBF(x1, x2, params):
    """Squared-exponential kernel.  x1: (n,1), x2: (m,1) -> (n,m).

    (Identical in 1-D to the `abs(sum(diffs))` form used in the ADR code:
     summing over a length-1 axis returns the signed difference, and squaring
     it afterwards removes the abs.  Written explicitly here so it also works
     for multi-dimensional inputs.)
    """
    output_scale, lengthscale = params
    diffs = np.expand_dims(x1 / lengthscale, 1) - np.expand_dims(x2 / lengthscale, 0)
    r2 = np.sum(diffs ** 2, axis=2)
    return output_scale ** 2 * np.exp(-0.5 * r2)


def RBF_periodic(x1, x2, params, period):
    """Periodic (exp-sine-squared) kernel -- use for the full 360-deg tube so
    that the sampled field satisfies g(s) = g(s + period)."""
    output_scale, lengthscale = params
    diffs = np.expand_dims(x1, 1) - np.expand_dims(x2, 0)
    d = np.sqrt(np.sum(diffs ** 2, axis=2))
    return output_scale ** 2 * np.exp(-2.0 * np.sin(np.pi * d / period) ** 2 / lengthscale ** 2)


# ---------------------------------------------------------------------------
# 2.  factorisation
# ---------------------------------------------------------------------------
def gp_cholesky(K, jitter=1e-12):
    """Return a matrix L with L @ L.T = K.

    Cholesky is tried first; RBF Gram matrices with a long lengthscale are
    numerically rank-deficient, so we fall back to the symmetric eigen-
    decomposition with clipped negative eigenvalues (this is what the ADR code
    does unconditionally).
    """
    n = K.shape[0]
    Kj = K + jitter * np.eye(n)
    try:
        return np.linalg.cholesky(Kj)
    except np.linalg.LinAlgError:
        D, V = np.linalg.eigh(Kj)
        D = np.maximum(D, 0.0)
        return V @ np.diag(np.sqrt(D))


# ---------------------------------------------------------------------------
# 3.  sampler
# ---------------------------------------------------------------------------
class InnerBCSampler:
    """Draws random inner-surface displacement fields and evaluates them at
    arbitrary mesh points.

    Parameters
    ----------
    r1, h        : inner radius and tube height (must match the .geo)
    m            : number of DeepONet sensor points along the arc
    N            : GP discretisation (>= m; the sample is drawn on this grid
                   and linearly interpolated, exactly as in the ADR code)
    amp          : (amp_x, amp_y, amp_z) GP output_scale = peak amplitude
    lengthscale  : correlation length in ARC-LENGTH units (L = r1*pi/2 ~ 1.571).
                   Scalar, or one value per component.

    Matching the analytic ground-truth BC
    -------------------------------------
    Writing 0.01*x^2, 0.01*y^2, 0.005*z(z-h) in the taper form above gives

        g_x(s) = 0.01 cos(s)      peak 0.01,  effective lengthscale 1.000
        g_y(s) = 0.01 sin(s)      peak 0.01,  effective lengthscale 1.000
        g_z(s) = -0.02            peak 0.02,  effective lengthscale infinite

    (effective lengthscale = sqrt(int g^2 / int g'^2), which is 1/ell for a
    squared-exponential GP).  So amp = (0.01, 0.01, 0.02) and an in-plane
    lengthscale of 1.0 put the ground-truth profile squarely inside the prior.
    The axial component of the ground truth is CONSTANT along the arc, so a
    deliberately larger lengthscale is used for it -- a big ell approximates
    "one random number per sample" rather than a wiggly profile.
    """

    def __init__(self, r1=1.0, h=4.0, m=101, N=512,
                 amp=(0.010, 0.010, 0.020),
                 lengthscale=(1.0, 1.0, 3.0), jitter=1e-12):
        self.r1, self.h, self.m = r1, h, m
        self.L = r1 * np.pi / 2.0                     # arc length of the quarter
        self.S = np.linspace(0.0, self.L, N)[:, None]  # GP grid (arc length)
        self.s_sensor = np.linspace(0.0, self.L, m)    # branch-net sensor grid
        self.amp = np.asarray(amp, dtype=float)
        self.N = N

        ls = np.broadcast_to(np.asarray(lengthscale, dtype=float), (3,))
        self.lengthscale = np.array(ls)
        # One factor per distinct lengthscale (unit amplitude; rescaled later).
        self._chol = []
        cache = {}
        for l in ls:
            if l not in cache:
                cache[l] = gp_cholesky(RBF(self.S, self.S, (1.0, l)), jitter)
            self._chol.append(cache[l])

    # -- draw one realisation ------------------------------------------------
    def sample(self, seed):
        """Return g = (3, N) GP samples on self.S for (u_x, u_y, u_z)."""
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((self.N, 3))
        g = np.stack([self._chol[i] @ z[:, i] for i in range(3)])  # (3, N)
        return g * self.amp[:, None]

    # -- sensor values fed to the branch net --------------------------------
    def sensors(self, g):
        """Evaluate a realisation at the m sensor locations -> (3, m)."""
        Sf = self.S.ravel()
        return np.stack([np.interp(self.s_sensor, Sf, g[i]) for i in range(3)])

    # -- z-profile used for u_z ---------------------------------------------
    def z_taper(self, z):
        """4 z (h-z) / h^2 : equals 1 at mid-height, 0 at z = 0 and z = h."""
        return 4.0 * z * (self.h - z) / self.h ** 2

    # -- callable handed to dolfinx Function.interpolate ---------------------
    def as_dolfinx_callable(self, g):
        """Build f(x) -> (3, npoints) for `u_inner.interpolate(f)`.

        `x` is (3, npoints) in Cartesian coordinates.  Because interpolate()
        is evaluated on the whole function space (the bc later selects only
        the inner-surface dofs), the expression must be well defined off the
        r = r1 surface too -- it is, since it only uses theta and z.
        """
        Sf = self.S.ravel()

        def f(x):
            theta = np.arctan2(x[1], x[0])
            theta = np.clip(theta, 0.0, np.pi / 2.0)   # guard round-off at the edges
            s = self.r1 * theta
            gx = np.interp(s, Sf, g[0])
            gy = np.interp(s, Sf, g[1])
            gz = np.interp(s, Sf, g[2])
            return np.vstack([
                np.cos(theta) * gx,            # -> 0 at theta = pi/2  (x = 0 plane)
                np.sin(theta) * gy,            # -> 0 at theta = 0     (y = 0 plane)
                self.z_taper(x[2]) * gz,       # -> 0 at z = 0 and z = h (rollers)
            ])

        return f
