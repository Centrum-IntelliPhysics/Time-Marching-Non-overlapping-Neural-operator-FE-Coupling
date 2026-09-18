# -*- coding: utf-8 -*-
"""
=============================================================================
PointNet-DeepONet for the COUPLEABLE Omega_I operator (r = 1 .. 2).

The branch input is the INTERFACE field at r = 2, not the load at r = 1.  That
is the change that makes the operator usable in a Schwarz iteration: r = 2 is
what Omega_II hands over and what varies from one coupling step to the next,
while r = 1 carries a fixed standard load that is the same in every sample.
The operator is therefore the interface Dirichlet-to-Neumann map

    u|_{r=2}  ->  Omega_I field,  in particular sigma.n on r = 2

exactly the object the 2-D script learns.

Branch architecture taken from
coupling_example/NO_multigrid_single_AM_5_full.py -- 1x1 convolutions applied
per point, average pooling along the point axis, a max pool to a global
feature, then dense layers.  It is permutation invariant and its parameter
count does not depend on the number of points, which is what makes the full
loaded-surface field usable as input.

Exact correspondence with the 2-D script
----------------------------------------
The 2-D branch takes the boundary field at ALL of its CG2 dofs, not a
subsample:

    num_points = 100 line segments on the circle
    CG2 on a 1-D loop  ->  100 vertices + 100 edge midpoints = 200 dofs
    m = 200,  branch input 2*m = 400 = (u, v) at every dof

Here the interface r = 2 is transfinite 50 x 80:

    4000 vertices + 11741 edge midpoints = 15741 CG2 dofs
    branch input 3 * 15741 = 47223 = (u_x, u_y, u_z) at every dof

Same object, one dimension up.  A dense first layer would be 47223 x 128 =
6.0e6 weights; the PointNet branch replaces it with 1x1 convolutions whose
size is independent of the point count.

What the operator maps
----------------------
    branch : the displacement field on r = 2, all 15741 CG2 dofs,
             fed as (x, y, z, u_x, u_y, u_z) per point
    trunk  : (x, y, z) anywhere in Omega_I
    output : u, and strain by autodiff from it
    data   : the six faces of Omega_I -- the interface strain is what the
             coupling consumes
    physics: the 3-D Navier residual at interior collocation points

The residual is NOT a regulariser here.  There is no interior data in the
training loss, so the residual is the ONLY thing that determines the solution
inside Omega_I.  The `volume/` block of the .h5 (first N_VOL_SAMPLES samples)
is held out and used to measure that interior, never to train it.

Units: physical throughout, as in the 2-D script.  The trunk coordinates are
shifted (not scaled) so that a tanh first layer is not saturated by z up to 4;
the shift has unit scale, so every derivative taken outside the network is a
physical derivative and the strain needs no correction.
=============================================================================
"""

import os
import time
import pickle
import itertools
from functools import partial

import numpy as npo
import h5py
import jax
import jax.numpy as np
from jax import random, grad, jit, vmap
from jax.example_libraries import optimizers
from jax.nn import relu
import jax.nn as jnn
from jax.lax import conv_general_dilated as conv_lax
import flax.linen as fnn
from jax import config
from tqdm import trange

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ===========================================================================
# 0.  configuration
# ===========================================================================
# region hyper paras
HERE = os.path.dirname(os.path.abspath(__file__))
H5_PATH = os.path.join(HERE, "results", "FE_full_static_dataset_r_2_N_1000_new",
                       "dataset_omega1_r_2.h5")
OUT_DIR = os.path.join(HERE, "results", "DeepONet_3D_tube_r_2_new_2_edge_scale_500")

# trunk points drawn from each face per step (the faces are stored in full;
# only the per-step sample is subsetted, the BRANCH always sees everything)
FACE_BATCH = {"inner": 500, "interface": 5000, "bottom": 500,
              "top": 500, "sym_x": 500, "sym_y": 500}

# --- the 12 edges of Omega_I ----------------------------------------------
# The strain error piles up on the edges: a uniform draw inside a face almost
# never lands on one, and the strain is a derivative, so with no support on
# either side the network is extrapolating exactly there.  Two fixes, both
# cheap:
#   EDGE_FRAC  reserves part of each face's per-step budget for dofs that sit
#              on an edge, so the DATA loss actually sees them.
#   Q_EDGE     adds residual collocation in a thin tube around the edges.
#              The momentum balance holds up to the boundary, so these need no
#              FE data at all -- free supervision where the face data is
#              thinnest.
EDGE_FRAC = 0.25           # share of FACE_BATCH reserved for edge dofs
EDGE_TOL = 1e-6            # geometric tolerance for "on a bounding surface"
Q_EDGE = 512               # residual points in the edge tube (0 disables)
EDGE_TUBE = 0.08           # how far the tube reaches into the domain

W_STRAIN = 1e-1            # same weights as the 2-D script
W_RES = 1e3
USE_RESIDUAL = True
Q_COLLOC = 2048

BATCH = 16                 # PointNet holds (B, 15741, C) activations; 16 is a
                           # safe start, raise it if the GPU has room
N_ITER = 1_000_000
LR = 1e-3
DECAY_STEPS, DECAY_RATE = 50_000, 0.9
LOG_EVERY = 1000

SCALE = 500 # physical units are small, so scale up to activate the Neural Network


# the first N_VOL samples carry interior data in the .h5 -> use exactly those
# as the held-out split, so the interior error is measured on unseen inputs
TRUNK_LAYERS_HIDDEN = [128, 128, 128, 128]
P_LATENT = 512

SEED = 1234
XC, YC, ZC = 1.0, 1.0, 2.0      # trunk shift, scale factor 1
R1, R2, H = 1.0, 2.0, 4.0

FACES = ["inner", "interface", "bottom", "top", "sym_x", "sym_y"]
STRAIN_COMPS = ["xx", "yy", "zz", "xy", "xz", "yz"]


def createFolder(folder_name):
    try:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    except OSError:
        print("Error: Creating folder. " + folder_name)


# ===========================================================================
# 1.  networks
# ===========================================================================
# region MLP
def MLP(layers, activation=relu):
    """Vanilla MLP (trunk)."""
    def init(rng_key):
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
            W = glorot_stddev * random.normal(k1, (d_in, d_out))
            b = np.zeros(d_out)
            return W, b
        key, *keys = random.split(rng_key, len(layers))
        return list(map(init_layer, keys, layers[:-1], layers[1:]))

    def apply(params, inputs):
        for W, b in params[:-1]:
            inputs = activation(np.dot(inputs, W) + b)
        W, b = params[-1]
        return np.dot(inputs, W) + b
    return init, apply

# region PointNet
def conv1d(x, w, b):
    """1x1 convolution along the point axis: a per-point shared linear map.
    x (N, C_in) -> (N, C_out)."""
    out = conv_lax(lhs=x[None, ..., None], rhs=w[..., None, :],
                   window_strides=(1, 1), padding="VALID",
                   dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return out[0, :, 0, :] + b


def avg_pool_1d(x, window=8, stride=8):
    """(N, C) -> (N//stride, C)."""
    x = x[None, :, None, :]
    x = fnn.avg_pool(x, window_shape=(window, 1), strides=(stride, 1),
                     padding="VALID")
    return x[0, :, 0, :]


def init_PointNet_params(p, in_channels, key=random.PRNGKey(0)):
    """Same layout as NO_multigrid_single_AM_5_full.init_SimplePointNet_params,
    with the input channel count opened up (6 here: x,y,z,u_x,u_y,u_z)."""
    k = random.split(key, 6)
    c1, c2, c3 = 12, 24, 48
    return [
        (random.normal(k[0], (1, in_channels, c1)) * 0.1,
         random.normal(k[1], (c1,)) * 0.1),
        (random.normal(k[1], (1, c1, c2)) * 0.1,
         random.normal(k[2], (c2,)) * 0.1),
        (random.normal(k[2], (1, c2, c3)) * 0.1,
         random.normal(k[3], (c3,)) * 0.1),
        (random.normal(k[3], (c3, 32)) * np.sqrt(2.0 / c3), np.zeros(32)),
        (random.normal(k[4], (32, 24)) * np.sqrt(2.0 / 128), np.zeros(24)),
        (random.normal(k[5], (24, p)) * np.sqrt(2.0 / 64), np.zeros(p)),
    ]


def BranchNet_PointNet(params, x):
    """x: (B, N, C) -> (B, p).

    Per-point 1x1 convs, average pooling along the point axis, a max pool to a
    global descriptor, then dense layers.  Permutation invariant and
    independent of N, which is the whole reason the full 15741-dof field can be
    used as the branch input.
    """
    def single_forward(params, x):
        (c1w, c1b), (c2w, c2b), (c3w, c3b), \
            (d1w, d1b), (d2w, d2b), (d3w, d3b) = params
        x = np.tanh(conv1d(x, c1w, c1b))
        x = avg_pool_1d(x, 8, 8)
        x = np.tanh(conv1d(x, c2w, c2b))
        x = avg_pool_1d(x, 8, 8)
        x = np.tanh(conv1d(x, c3w, c3b))
        x = np.max(x, axis=0)                    # global feature (C,)
        x = jnn.silu(np.dot(x, d1w) + d1b)
        x = jnn.silu(np.dot(x, d2w) + d2b)
        return np.dot(x, d3w) + d3b
    return vmap(partial(single_forward, params))(x)


# ===========================================================================
# 2.  the model
# ===========================================================================
class PI_DeepONet:

    def __init__(self, trunk_layers, in_channels, **ela_model):
        self.trunk_init, self.trunk_apply = MLP(trunk_layers, activation=np.tanh)
        self.branch_apply = BranchNet_PointNet

        self.E = ela_model["E"]
        self.nu = ela_model["nu"]
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lmbda = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.shift = np.asarray([XC, YC, ZC])
        self.p = trunk_layers[-1]

        k = random.split(random.PRNGKey(SEED), 4)
        # ONE PointNet emitting 3*p, split into three blocks of p -- one per
        # displacement component.  Three separate PointNets would triple the
        # cost for no gain: the input field is the same for all three.
        branch_params_u = init_PointNet_params(self.p, in_channels, key=k[0])
        branch_params_v = init_PointNet_params(self.p, in_channels, key=k[1])
        branch_params_w = init_PointNet_params(self.p, in_channels, key=k[2])

        trunk_params = self.trunk_init(rng_key=k[1])
        params = (branch_params_u, branch_params_v, branch_params_w, trunk_params)

        self.opt_init, self.opt_update, self.get_params = optimizers.adam(
            optimizers.exponential_decay(LR, decay_steps=DECAY_STEPS,
                                         decay_rate=DECAY_RATE))
        self.opt_state = self.opt_init(params)
        self.itercount = itertools.count()

        self.loss_log, self.loss_bcs_log = [], []
        self.loss_res_log, self.loss_bcs_strain_log = [], []
        self.loss_bcs_test_log, self.loss_bcs_strain_test_log = [], []
        self.loss_res_test_log, self.loss_vol_log = [], []
        self.face_log = []

    # -- architecture -------------------------------------------------------
    '''def _uvw(self, params, f, x, y, z):
        """f: (B, N, C) branch field;  x,y,z: (P,)  ->  three (B, P) arrays."""
        branch_params, trunk_params = params
        Bn = self.branch_apply(branch_params, f)                 # (B, 3p)
        h = np.stack([x, y, z], axis=-1) - self.shift            # (P, 3)
        T = self.trunk_apply(trunk_params, h)                    # (P, p)
        Bn = Bn.reshape(Bn.shape[0], 3, self.p)                  # (B, 3, p)
        out = np.einsum("bcm,pm->bcp", Bn, T)                    # (B, 3, P)
        return out[:, 0], out[:, 1], out[:, 2]

    def operator_net_u(self, params, f, x, y, z):
        return self._uvw(params, f, x, y, z)[0]

    def operator_net_v(self, params, f, x, y, z):
        return self._uvw(params, f, x, y, z)[1]

    def operator_net_w(self, params, f, x, y, z):
        return self._uvw(params, f, x, y, z)[2]'''
    # region operator_net
    def operator_net_u(self, params, f, x, y, z):
        """f: (B, N, C) branch field;  x,y,z: (P,)  ->  three (B, P) arrays."""
        branch_params_u, _, _, trunk_params = params
        Bn = self.branch_apply(branch_params_u, f)                 # (B, p)
        h = np.stack([x, y, z], axis=-1) - self.shift            # (P, 3)
        T = self.trunk_apply(trunk_params, h)                    # (P, p)
        out = np.einsum("bm,pm->bp", Bn, T)                    # (B, P)
        return out

    def operator_net_v(self, params, f, x, y, z):
        """f: (B, N, C) branch field;  x,y,z: (P,)  ->  three (B, P) arrays."""
        _, branch_params_v, _, trunk_params = params
        Bn = self.branch_apply(branch_params_v, f)                 # (B, p)
        h = np.stack([x, y, z], axis=-1) - self.shift            # (P, 3)
        T = self.trunk_apply(trunk_params, h)                    # (P, p)
        out = np.einsum("bm,pm->bp", Bn, T)                    # (B, P)
        return out

    def operator_net_w(self, params, f, x, y, z):
        """f: (B, N, C) branch field;  x,y,z: (P,)  ->  three (B, P) arrays."""
        _, _, branch_params_w, trunk_params = params
        Bn = self.branch_apply(branch_params_w, f)                 # (B, p)
        h = np.stack([x, y, z], axis=-1) - self.shift            # (P, 3)
        T = self.trunk_apply(trunk_params, h)                    # (P, p)
        out = np.einsum("bm,pm->bp", Bn, T)                    # (B, P)
        return out
    
    # -- derivatives --------------------------------------------------------
    def _d(self, fn, x, y, z, axis):
        if axis == 0:
            return jax.jvp(lambda a: fn(a, y, z), (x,), (np.ones_like(x),))[1]
        if axis == 1:
            return jax.jvp(lambda a: fn(x, a, z), (y,), (np.ones_like(y),))[1]
        return jax.jvp(lambda a: fn(x, y, a), (z,), (np.ones_like(z),))[1]

    def _dd(self, fn, x, y, z, a1, a2):
        inner = lambda xx, yy, zz: self._d(fn, xx, yy, zz, a2)
        return self._d(inner, x, y, z, a1)

    # -- strain -------------------------------------------------------------
    def strain(self, params, f, x, y, z):
        fu = lambda a, b, c: self.operator_net_u(params, f, a, b, c)
        fv = lambda a, b, c: self.operator_net_v(params, f, a, b, c)
        fw = lambda a, b, c: self.operator_net_w(params, f, a, b, c)
        s_u_x = self._d(fu, x, y, z, 0)
        s_u_y = self._d(fu, x, y, z, 1)
        s_u_z = self._d(fu, x, y, z, 2)
        s_v_x = self._d(fv, x, y, z, 0)
        s_v_y = self._d(fv, x, y, z, 1)
        s_v_z = self._d(fv, x, y, z, 2)
        s_w_x = self._d(fw, x, y, z, 0)
        s_w_y = self._d(fw, x, y, z, 1)
        s_w_z = self._d(fw, x, y, z, 2)
        return (s_u_x, s_v_y, s_w_z,
                (s_u_y + s_v_x) / 2, (s_u_z + s_w_x) / 2, (s_v_z + s_w_y) / 2)

    # -- residual -----------------------------------------------------------
    def residual_net(self, params, f, x, y, z):
        """(lambda + mu) d_i(div u) + mu laplacian(u_i) = 0, physical units.
        15 second derivatives: u_yz, v_xz and w_xy never appear."""
        fu = lambda a, b, c: self.operator_net_u(params, f, a, b, c)
        fv = lambda a, b, c: self.operator_net_v(params, f, a, b, c)
        fw = lambda a, b, c: self.operator_net_w(params, f, a, b, c)
        u_xx = self._dd(fu, x, y, z, 0, 0); u_yy = self._dd(fu, x, y, z, 1, 1)
        u_zz = self._dd(fu, x, y, z, 2, 2); u_xy = self._dd(fu, x, y, z, 0, 1)
        u_xz = self._dd(fu, x, y, z, 0, 2)
        v_xx = self._dd(fv, x, y, z, 0, 0); v_yy = self._dd(fv, x, y, z, 1, 1)
        v_zz = self._dd(fv, x, y, z, 2, 2); v_xy = self._dd(fv, x, y, z, 0, 1)
        v_yz = self._dd(fv, x, y, z, 1, 2)
        w_xx = self._dd(fw, x, y, z, 0, 0); w_yy = self._dd(fw, x, y, z, 1, 1)
        w_zz = self._dd(fw, x, y, z, 2, 2); w_xz = self._dd(fw, x, y, z, 0, 2)
        w_yz = self._dd(fw, x, y, z, 1, 2)
        lm, mu = self.lmbda, self.mu
        res0 = (lm + mu) * (u_xx + v_xy + w_xz) + mu * (u_xx + u_yy + u_zz)
        res1 = (lm + mu) * (u_xy + v_yy + w_yz) + mu * (v_xx + v_yy + v_zz)
        res2 = (lm + mu) * (u_xz + v_yz + w_zz) + mu * (w_xx + w_yy + w_zz)
        return res0, res1, res2

    # -- losses -------------------------------------------------------------
    def loss_bcs(self, params, batch):
        inputs, outputs = batch
        f, h = inputs
        x, y, z = h[:, 0], h[:, 1], h[:, 2]
        
        pu = self.operator_net_u(params, f, x, y, z)
        pv = self.operator_net_v(params, f, x, y, z)
        pw = self.operator_net_w(params, f, x, y, z)

        return (np.mean((outputs[0] - pu) ** 2)
                + np.mean((outputs[1] - pv) ** 2)
                + np.mean((outputs[2] - pw) ** 2))

    def loss_bcs_strain(self, params, batch):
        inputs, outputs = batch
        f, h = inputs
        pred = self.strain(params, f, h[:, 0], h[:, 1], h[:, 2])
        return W_STRAIN * sum(np.mean((outputs[3 + i] - pred[i]) ** 2)
                              for i in range(6))

    def loss_res(self, params, batch):
        inputs, outputs = batch
        f, h = inputs
        r0, r1, r2 = self.residual_net(params, f, h[:, 0], h[:, 1], h[:, 2])
        return W_RES * (np.mean((outputs[0] - r0) ** 2)
                        + np.mean((outputs[1] - r1) ** 2)
                        + np.mean((outputs[2] - r2) ** 2))

    def loss(self, params, bcs_batch, res_batch):
        total = self.loss_bcs(params, bcs_batch) \
            + self.loss_bcs_strain(params, bcs_batch)
        if USE_RESIDUAL:
            total = total + self.loss_res(params, res_batch)
        return total

    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, bcs_batch, res_batch):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, bcs_batch, res_batch)
        return self.opt_update(i, g, opt_state)

    @partial(jit, static_argnums=(0,))
    def eval_all(self, params, bcs_batch, res_batch):
        return (self.loss_bcs(params, bcs_batch),
                self.loss_bcs_strain(params, bcs_batch),
                self.loss_res(params, res_batch) if USE_RESIDUAL else 0.0)

    # -- prediction ---------------------------------------------------------
    @partial(jit, static_argnums=(0,))
    def predict_s(self, params, F_star, Y_star):
        """F_star (B, N, C);  Y_star (P, 3) physical.  -> u (B,P,3), eps (B,P,6)."""
        x, y, z = Y_star[:, 0], Y_star[:, 1], Y_star[:, 2]
        u = np.stack([self.operator_net_u(params, F_star, x, y, z), 
                      self.operator_net_v(params, F_star, x, y, z),
                      self.operator_net_w(params, F_star, x, y, z)], axis=-1) / SCALE
        
        e = np.stack(self.strain(params, F_star, x, y, z), axis=-1) / SCALE
        return u, e

    # -- training -----------------------------------------------------------
    def train(self, data, nIter=N_ITER):
        ev = data.fixed_eval()
        pbar = trange(nIter)
        for it in pbar:
            bcs_batch, res_batch = data.batch()
            self.opt_state = self.step(next(self.itercount), self.opt_state,
                                       bcs_batch, res_batch)
            if it % LOG_EVERY == 0:
                p = self.get_params(self.opt_state)
                lb, ls, lr_ = [float(v) for v in
                               self.eval_all(p, ev["tr"], ev["tr_res"])]
                tb, ts, tr_ = [float(v) for v in
                               self.eval_all(p, ev["te"], ev["te_res"])]
                vol = float(data.volume_error(self, p))

                self.loss_bcs_log.append(lb)
                self.loss_bcs_strain_log.append(ls)
                self.loss_res_log.append(lr_)
                self.loss_log.append(lb + ls + lr_)
                self.loss_bcs_test_log.append(tb)
                self.loss_bcs_strain_test_log.append(ts)
                self.loss_res_test_log.append(tr_)
                self.loss_vol_log.append(vol)
                self.face_log.append(data.per_face_loss(self, p))

                pbar.set_postfix({"bcs": f"{lb:.2e}", "strain": f"{ls:.2e}",
                                  "res": f"{lr_:.2e}",
                                  "T_bcs": f"{tb:.2e}", "T_strain": f"{ts:.2e}",
                                  "VOL": f"{vol:.2e}"})


def stress_from_strain(eps, lmbda, mu):
    """(...,6) strain [xx yy zz xy xz yz] -> (...,6) stress."""
    tr = eps[..., 0] + eps[..., 1] + eps[..., 2]
    out = [lmbda * tr + 2.0 * mu * eps[..., i] if i < 3 else 2.0 * mu * eps[..., i]
           for i in range(6)]
    return npo.stack(out, -1) if isinstance(eps, npo.ndarray) else np.stack(out, -1)


# ===========================================================================
# 3.  data
# ===========================================================================
# region DataGenerator
class DataGenerator:
    """Branch = the FULL r=1 CG2 field.  Trunk points are subsetted per step;
    the branch never is."""

    def __init__(self, h5_path, rng_key=random.PRNGKey(1234)):
        self.key = rng_key
        with h5py.File(h5_path, "r") as f:
            self.N = int(f.attrs["n_samples"])
            self.E, self.nu = float(f.attrs["E"]), float(f.attrs["nu"])
            self.mu, self.lmbda = float(f.attrs["mu"]), float(f.attrs["lmbda"])

            # --- branch: every CG2 dof of the INTERFACE r = 2 ---------------
            # stored explicitly by the dataset as `sensors_iface`, so nothing
            # has to be reconstructed here
            self.xyz_in = npo.asarray(f["sensors_iface_xyz"])     # (N_in, 3)
            u_in = npo.asarray(f["sensors_iface"]) * SCALE                # (N, N_in, 3)
            self.n_in = self.xyz_in.shape[0]
            # (B, N_in, 6) = (x, y, z, u_x, u_y, u_z) per point
            self.Fb = npo.concatenate(
                [npo.broadcast_to(self.xyz_in, (self.N,) + self.xyz_in.shape),
                 u_in], axis=2).astype("f4")

            # --- targets on the six faces ----------------------------------
            XYZ, U, EPS, self.slices, self.loc = [], [], [], {}, {}
            start = 0
            for name in FACES:
                g = f[f"faces/{name}"]
                n = g["xyz"].shape[0]
                XYZ.append(npo.asarray(g["xyz"]))
                U.append(npo.asarray(g["u"])* SCALE) 
                EPS.append(npo.asarray(g["strain"])* SCALE) 
                self.slices[name] = (start, start + n)
                loc = g.attrs.get("location", "?")
                self.loc[name] = loc if isinstance(loc, str) else loc.decode()
                start += n
                print(f"  {name:10s} {self.loc[name]:10s} dofs {n:6d}")

            # --- held-out interior, validation only -------------------------
            # Which SAMPLES carry interior data is read from the file, not
            # assumed.  FE_full_static_dataset.py writes `sample_index`; if the
            # stored subset ever changes, the split here follows it instead of
            # silently testing on the wrong samples.
            gv = f["volume"]
            self.n_vol = int(gv.attrs["n_samples"])
            vi = gv.attrs.get("sample_index", None)
            self.vol_index = (npo.arange(self.n_vol) if vi is None
                              else npo.asarray(vi).ravel().astype(int))
            assert self.vol_index.size == self.n_vol, \
                "volume/sample_index does not match volume/n_samples"
            self.xyz_vol = np.asarray(npo.asarray(gv["xyz"]), np.float32)
            self.u_vol = np.asarray(npo.asarray(gv["u"]), np.float32)
            print(f"  volume     interior   dofs {self.xyz_vol.shape[0]:6d}"
                  f"  samples {list(self.vol_index)} (VALIDATION ONLY)")

        self.XYZ = np.asarray(npo.concatenate(XYZ, 0), np.float32)
        self.U = np.asarray(npo.concatenate(U, 1), np.float32)
        self.EPS = np.asarray(npo.concatenate(EPS, 1), np.float32)
        self.Fb = np.asarray(self.Fb, np.float32)

        self._build_edge_index()

        # The samples that carry interior data ARE the held-out set -- that is
        # the whole point: the interior can only be checked on inputs the
        # operator never trained on.  u_vol row i corresponds to sample
        # vol_index[i], so _te must be in that same order.
        self._te = np.asarray(self.vol_index)
        tr = npo.setdiff1d(npo.arange(self.N), self.vol_index)
        self._tr = np.asarray(tr)
        self.B_tr = min(BATCH, tr.size)
        print(f"N={self.N}  train {tr.size}  test {self.n_vol} "
              f"(indices {list(self.vol_index)})"
              f"  branch {self.n_in} pts x 6 ch  (dense would be "
              f"{3*self.n_in} inputs)")

    # -- which stored dofs sit on an edge of Omega_I --------------------------
    def _build_edge_index(self):
        """Split each face's dof range into {on an edge, interior to the face}.

        A point is on an edge when it lies on at least two of the six bounding
        surfaces at once (r=R1, r=R2, z=0, z=H, y=0, x=0) -- those intersections
        are exactly the 12 edges of Omega_I.

        The printout matters: if a face reports 0 edge dofs then the dataset
        stored only face-interior points and no amount of reweighting will put
        data on that edge -- regenerate the .h5, or lean on Q_EDGE instead.
        """
        xyz = npo.asarray(self.XYZ)
        r = npo.hypot(xyz[:, 0], xyz[:, 1])
        on = npo.stack([npo.abs(r - R1) < 1e-4, npo.abs(r - R2) < 1e-4,
                        npo.abs(xyz[:, 2]) < EDGE_TOL,
                        npo.abs(xyz[:, 2] - H) < EDGE_TOL,
                        npo.abs(xyz[:, 1]) < EDGE_TOL,
                        npo.abs(xyz[:, 0]) < EDGE_TOL])
        is_edge = on.sum(0) >= 2

        self.edge_idx, self.bulk_idx, self.n_edge = {}, {}, {}
        print("  edge dofs (on >=2 bounding surfaces):")
        for name in FACES:
            a, b = self.slices[name]
            loc = npo.arange(a, b)
            e, k = loc[is_edge[a:b]], loc[~is_edge[a:b]]
            take = min(int(round(FACE_BATCH[name] * EDGE_FRAC)), e.size)
            if e.size == 0:
                print(f"    {name:10s} {0:6d} / {b - a:6d}   !! none stored -- "
                      f"only the residual can reach this face's edges")
            else:
                print(f"    {name:10s} {e.size:6d} / {b - a:6d}   "
                      f"{take:4d}/{FACE_BATCH[name]} per step reserved")
            # keep a valid array even when empty so the jitted gather is happy
            self.edge_idx[name] = np.asarray(e if e.size else loc[:1])
            self.bulk_idx[name] = np.asarray(k if k.size else loc)
            self.n_edge[name] = take

    def _trunk_points(self, key):
        """Per-face draw, stratified so a fixed share of every face's budget
        lands on the edges instead of leaving them unsupervised."""
        pts = []
        for j, name in enumerate(FACES):
            kb, ke = random.split(random.fold_in(key, j))
            n_e = self.n_edge[name]
            n_b = min(FACE_BATCH[name] - n_e, self.bulk_idx[name].size)
            pts.append(random.choice(kb, self.bulk_idx[name], (n_b,),
                                     replace=False))
            if n_e:
                pts.append(random.choice(ke, self.edge_idx[name], (n_e,),
                                         replace=False))
        return np.concatenate(pts)

    def _pack(self, bidx, pts):
        u = self.U[bidx][:, pts, :]
        e = self.EPS[bidx][:, pts, :]
        inputs = (self.Fb[bidx], self.XYZ[pts])
        outputs = tuple(u[..., i] for i in range(3)) \
            + tuple(e[..., i] for i in range(6))
        return inputs, outputs

    def _edge_colloc(self, key, q=Q_EDGE):
        """Residual points in a thin tube around the 12 edges of Omega_I.

        One of (r, th, z) runs free -- that is the direction the edge points
        in -- and the other two are pinned near an end of their range with a
        random inward offset.  Two pinned coordinates puts the point on an
        edge; the offset spreads it into the layer the strain stencil needs.
        The offsets are scaled so the tube is the same physical thickness in
        all three directions.
        """
        k1, k2, k3, k4 = random.split(key, 4)
        lo = np.array([R1, 0.0, 0.0])
        hi = np.array([R2, np.pi / 2, H])
        tube = np.array([EDGE_TUBE, EDGE_TUBE / R2, EDGE_TUBE])

        free = random.randint(k1, (q, 1), 0, 3)              # edge direction
        side = random.bernoulli(k2, 0.5, (q, 3))             # which end pinned
        off = random.uniform(k3, (q, 3)) * tube              # inward depth
        u = random.uniform(k4, (q, 3))                       # free position

        c = np.where(np.arange(3) == free,
                     lo + u * (hi - lo),                     # free coordinate
                     np.where(side, lo + off, hi - off))     # pinned near a face
        r, th, z = c[:, 0], c[:, 1], c[:, 2]
        return np.stack([r * np.cos(th), r * np.sin(th), z], -1)

    def _colloc(self, key, q=Q_COLLOC):
        k1, k2, k3, k4 = random.split(key, 4)
        r = np.sqrt(random.uniform(k1, (q,), minval=R1 ** 2, maxval=R2 ** 2))
        th = random.uniform(k2, (q,), minval=0.0, maxval=np.pi / 2)
        z = random.uniform(k3, (q,), minval=0.0, maxval=H)
        bulk = np.stack([r * np.cos(th), r * np.sin(th), z], -1)
        if not Q_EDGE:
            return bulk
        return np.concatenate([bulk, self._edge_colloc(k4)], 0)

    def _pack_res(self, bidx, h):
        zero = np.zeros((bidx.shape[0], h.shape[0]))
        return (self.Fb[bidx], h), (zero, zero, zero)

    @partial(jit, static_argnums=(0,))
    def _make(self, key):
        k1, k2, k3 = random.split(key, 3)
        bidx = random.choice(k1, self._tr, (self.B_tr,), replace=False)
        return self._pack(bidx, self._trunk_points(k2)), \
            self._pack_res(bidx, self._colloc(k3))

    def __iter__(self):
        return self

    def __next__(self):
        return self.batch()

    def batch(self):
        self.key, sk = random.split(self.key)
        return self._make(sk)

    # -- frozen batches for the logged curves --------------------------------
    def fixed_eval(self):
        pts = self._trunk_points(random.PRNGKey(999))
        cp = self._colloc(random.PRNGKey(998))
        tr_i = self._tr[:self.B_tr]
        return {"tr": self._pack(tr_i, pts), "te": self._pack(self._te, pts),
                "tr_res": self._pack_res(tr_i, cp),
                "te_res": self._pack_res(self._te, cp)}

    # -- the number that says whether the INTERIOR is right ------------------
    def volume_error(self, model, params):
        """Relative L2 of the displacement inside Omega_I, on the held-out
        samples.  Nothing in the training loss sees interior data, so this is
        the only check that the residual is doing its job."""
        u, _ = model.predict_s(params, self.Fb[self._te], self.xyz_vol)
        return np.linalg.norm(u - self.u_vol) / np.maximum(
            np.linalg.norm(self.u_vol), 1e-30)

    def per_face_loss(self, model, params):
        out = {}
        for name in FACES:
            a, b = self.slices[name]
            sel = np.arange(a, b)[::max(1, (b - a) // 400)]
            batch = self._pack(self._te, sel)
            out[name] = (float(model.loss_bcs(params, batch)),
                         float(model.loss_bcs_strain(params, batch)))
        return out


# ===========================================================================
# 4.  main
# ===========================================================================
if __name__ == "__main__":
    config.update("jax_enable_x64", False)
    createFolder(OUT_DIR)

    print("loading", H5_PATH)
    data = DataGenerator(H5_PATH)

    trunk_layers = [3] + TRUNK_LAYERS_HIDDEN + [P_LATENT]
    print("trunk", trunk_layers, "  branch: PointNet, 6 channels")
    print(f"weights: strain x{W_STRAIN:g}, residual x{W_RES:g} "
          f"({'ON' if USE_RESIDUAL else 'OFF'})")

    model = PI_DeepONet(trunk_layers, in_channels=6,
                        E=data.E, nu=data.nu)

    t0 = time.time()
    model.train(data, nIter=N_ITER)
    print(f"training time {time.time()-t0:.0f} s")

    params = model.get_params(model.opt_state)
    with open(os.path.join(OUT_DIR, "DeepONet_3D_tube_r_2.pkl"), "wb") as fh:
        pickle.dump({"params": params,
                     "config": {"shift": (XC, YC, ZC), "trunk_layers": trunk_layers,
                                "in_channels": 6, "p": P_LATENT,
                                "n_in": data.n_in, "xyz_in": data.xyz_in,
                                "branch_face": "interface",
                                "E": data.E, "nu": data.nu,
                                "mu": data.mu, "lmbda": data.lmbda}}, fh)

    LOGS = (("bcs", model.loss_bcs_log, model.loss_bcs_test_log, "C0"),
            ("strain", model.loss_bcs_strain_log,
             model.loss_bcs_strain_test_log, "C1"),
            ("residual", model.loss_res_log, model.loss_res_test_log, "C2"))
    for nm, a, b, _ in LOGS:
        npo.savetxt(os.path.join(OUT_DIR, f"loss_{nm}_train.txt"), a)
        npo.savetxt(os.path.join(OUT_DIR, f"loss_{nm}_test.txt"), b)
    npo.savetxt(os.path.join(OUT_DIR, "volume_rel_l2.txt"), model.loss_vol_log)

    it = npo.arange(len(model.loss_bcs_log)) * LOG_EVERY
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for nm, a, b, c in LOGS:
        ax[0].semilogy(it, npo.maximum(a, 1e-16), "-", color=c, label=f"{nm} (train)")
        ax[0].semilogy(it[:len(b)], npo.maximum(b, 1e-16), "--", color=c,
                       label=f"{nm} (test)")
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel("loss")
    ax[0].set_title("solid = train,  dashed = test")
    ax[0].grid(True, which="both", alpha=0.3); ax[0].legend(fontsize=8, ncol=2)

    ax[1].semilogy(it[:len(model.loss_vol_log)],
                   npo.maximum(model.loss_vol_log, 1e-16), "k-")
    ax[1].set_xlabel("iteration")
    ax[1].set_ylabel(r"interior $\|u_{NO}-u_{FE}\|/\|u_{FE}\|$")
    ax[1].set_title("Omega_I INTERIOR, held-out samples\n"
                    "(no interior data in the loss -- this is the residual's job)")
    ax[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "loss.png"), dpi=200)

    print("\nfinal per-face test loss:")
    print(f"  {'face':10s} {'where':10s} {'L_bcs':>11s} {'L_strain':>11s}")
    for name, (lu, ls) in model.face_log[-1].items():
        print(f"  {name:10s} {data.loc[name]:10s} {lu:11.3e} {ls:11.3e}")
    print(f"\ninterior relative L2 on the held-out samples: "
          f"{model.loss_vol_log[-1]:.3e}")
    print("wrote", OUT_DIR)
