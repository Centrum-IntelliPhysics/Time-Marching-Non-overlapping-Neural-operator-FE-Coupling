"""
=============================================================================
FE-NO non-overlapping coupling: the trained operator replaces Omega_I.

    Omega_I   r in [1, 2]   the NEURAL OPERATOR      (no mesh, no solve)
    Omega_II  r in [2, 4]   FE on Tube_outer.msh

THE D-N ASSIGNMENT IS THE OPPOSITE OF FE_FE_static_coupling_r_2.py
------------------------------------------------------------------
That script gives Omega_I the Neumann datum and Omega_II the Dirichlet datum.
The operator cannot be used that way round: it was trained as the interface
DIRICHLET-to-Neumann map,  u|_{r=2} -> field, so it must RECEIVE displacement
and RETURN traction:

    Omega_I  (NO) :  u_Gamma          ->  sigma_I . n
    Omega_II (FE) :  sigma_I . n_II   ->  u_Gamma          (Neumann problem)

Omega_II with a pure Neumann condition on r = 2 and a free surface at r = 4 is
still well posed: the three roller planes remove all six rigid-body modes
(x = 0 locks u_x, y = 0 locks u_y, z = 0 and z = h lock u_z, and those three
together also kill all three rotations).

WHERE THE RELAXATION ACTS  (--relax)
------------------------------------
    --relax u          u_Gamma <- (1-th) u_Gamma  +  th u_new
                       standard under-relaxation, th weights the NEW value

    --relax traction   t       <- th sigma_II.n|_FE  +  (1-th) sigma_I.n|_NO
                       the 2-D script's flux blend, th weights the FE SIDE

The two conventions are mirror images.  That is deliberate -- each is the one
natural to its own scheme -- and the formula actually in force is printed at
the start of every run, so there is nothing to remember.  Be careful anyway:
th = 0.07 under --relax u is heavy damping, th = 0.07 under --relax traction is
almost none.

The traction form is what FE_DeepONet_static_coupling.py, the working 2-D case,
does.  It is a convex combination of the TWO SUBDOMAINS' estimates of the same
interface flux at the SAME sweep, not a damping across sweeps, which is what
makes it look like a different object.  It reduces to one anyway: Omega_II was
solved with the previous sweep's traction as its Neumann datum, so the stress
it reports back on Gamma is that datum again up to stress-recovery error, and
the update collapses to

    t^(k) = th t^(k-1) + (1-th) N(u_Gamma^(k))

Two consequences worth stating plainly.  Neither form moves the fixed point:
t* = th t* + (1-th) N(u*) gives t* = N(u*), so at convergence the operator
alone decides and th has dropped out.  And for a LINEAR operator the two are
equivalent in rate as well -- the iteration is F.N in u-space and N.F in
t-space, and AB and BA share their nonzero spectrum.

What the flux form buys is the transient.  sigma_II.n is read back from the FE
SOLUTION rather than from the stored datum, so the blend is between two genuine
estimates of a quantity that must agree at the true solution, and the FE side
is the accurate one.  The learned N is also not linear, so the two iterations
differ in stability and in the path they take through the operator's input
distribution.  Both reasons are empirical, which is what the switch is for.

SIGN
----
The Neumann datum for a subdomain is sigma . n with n the OUTWARD normal OF
THAT SUBDOMAIN.  At r = 2, Omega_I's outward normal is +e_r and Omega_II's is
-e_r.  The FE-FE script feeds Omega_I  t = sigma_II . (+e_r)  with no minus; the
mirror image of that, used below, is

    t_for_Omega_II = - ( sigma_I . e_r )

A sign error does not crash, it just diverges, so section 4 checks the sign
against the FE-FE traction before the loop starts.

WHAT TO EXPECT
--------------
The operator's interface strain error propagates into the converged coupled
solution amplified by 1/(1 - rho), with rho ~ 0.40 measured from the FE-FE run.
The script measures the actual coupled error against the FE-FE solution and
compares it with that prediction, so the estimate is checked, not assumed.

Run:  python FE_NO_static_coupling_r_2_original.py
      python FE_NO_static_coupling_r_2_original.py --theta 0.3     # if it diverges
      python FE_NO_static_coupling_r_2_original.py --relax traction
      python FE_NO_static_coupling_r_2_original.py --ckpt ... --data ...
=============================================================================
"""

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import ufl
import os
import time
import h5py
import pickle
import argparse
from scipy.spatial import cKDTree

from dolfinx import fem, default_scalar_type
from dolfinx.io import gmsh as dgmsh
from dolfinx.fem.petsc import (assemble_matrix, assemble_vector,
                               apply_lifting, set_bc, create_vector)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax.numpy as jnp
from utils import createFolder
from prepare_DeepONet_3D_tube_r_2_final_regularization import PI_DeepONet


# ===========================================================================
# 0.  configuration
# ===========================================================================
ap = argparse.ArgumentParser(description="FE-NO non-overlapping coupling")
ap.add_argument('--ckpt', type=str,
                default='results/DeepONet_3D_tube_r_2_new_2_edge_scale_500/DeepONet_3D_tube_r_2.pkl')
ap.add_argument('--data', type=str,
                default='results/FE_full_static_dataset_r_2_N_1000_new/dataset_omega1_r_2.h5')
ap.add_argument('--fefe', type=str,
                default='results/FE_FE_coupling_results/interface_trajectory.h5')
ap.add_argument('--theta', type=float, default=0.9,
                help='relaxation factor')
ap.add_argument('--relax', type=str, default='traction', choices=('u', 'traction'),
                help="'u': u <- (1-th) u_old + th u_new, th on the NEW value.  "
                     "'traction': t <- th sigma_II.n|FE + (1-th) sigma_I.n|NO, "
                     "the 2-D script's flux blend, th on the FE SIDE.  The "
                     "conventions are mirror images; the formula in force is "
                     "printed at the start of the run")
ap.add_argument('--niter', type=int, default=400)
ap.add_argument('--tol', type=float, default=1e-3)
ap.add_argument('--error-list-only', action='store_true',
                help="stop right after writing error_list_FE_NN.npy (the "
                     "convergence curve), before the monolithic solve and the "
                     "per-sweep dataset dump.  Use it when all you want is the "
                     "static_convergence_plot.py data.")
args = ap.parse_args()

THETA, NITER, TOL, RELAX = args.theta, args.niter, args.tol, args.relax

SCALE = 500 # scale factor for the branch input, to match the training data

R1, R2, R3, H = 1.0, 2.0, 4.0, 4.0
BOTTOM, TOP, S_INNER, S_OUTER, SYM_A, SYM_B = 1, 2, 3, 4, 5, 6
#   Omega_II (Tube_outer):  S_INNER = r=2 (interface)   S_OUTER = r=4 (free)

E_MOD, NU = 0.210e-2, 0.3
MU = E_MOD / (2.0 * (1.0 + NU))
LMBDA = E_MOD * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))

RHO_FEFE = 0.40            # Schwarz contraction factor measured from FE-FE

originalDir = os.path.dirname(os.path.abspath(__file__))
createFolder(os.path.join(originalDir, "results"))
out_dir = os.path.join(originalDir, "results", "FE_NO_coupling_results_final_sigma_data")
createFolder(out_dir)
mesh_dir = os.path.join(originalDir, "Mesh_3D_cylinder")


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(originalDir, p)


CKPT, DATA, FEFE = _abs(args.ckpt), _abs(args.data), _abs(args.fefe)
for path, name, flag in ((CKPT, "checkpoint", "--ckpt"),
                         (DATA, "dataset", "--data")):
    if not os.path.exists(path):
        raise SystemExit(f"{name} not found:\n  {path}\nuse {flag} to override")


def epsilon(u):
    return ufl.sym(ufl.grad(u))


def sigma(u):
    return LMBDA * ufl.tr(epsilon(u)) * ufl.Identity(3) + 2.0 * MU * epsilon(u)


def hooke_voigt(EPS):
    """(n, 6) Voigt TENSOR strain [xx yy zz xy xz yz] -> (n, 3, 3) stress."""
    E = np.asarray(EPS, float)
    tr = E[:, 0] + E[:, 1] + E[:, 2]
    out = np.empty((E.shape[0], 3, 3))
    out[:, 0, 0] = LMBDA * tr + 2.0 * MU * E[:, 0]
    out[:, 1, 1] = LMBDA * tr + 2.0 * MU * E[:, 1]
    out[:, 2, 2] = LMBDA * tr + 2.0 * MU * E[:, 2]
    out[:, 0, 1] = out[:, 1, 0] = 2.0 * MU * E[:, 3]
    out[:, 0, 2] = out[:, 2, 0] = 2.0 * MU * E[:, 4]
    out[:, 1, 2] = out[:, 2, 1] = 2.0 * MU * E[:, 5]
    return out


# ===========================================================================
# 1.  Omega_II -- the only FE subdomain left
# ===========================================================================
def detect_roller(V, facet_tags, tag, fdim, tol=1e-9):
    """Which component the flat face `tag` locks, read off the geometry."""
    dofs = fem.locate_dofs_topological(V, fdim, facet_tags.find(tag))
    xc = V.tabulate_dof_coordinates()[dofs]
    for comp in (0, 1, 2):
        if np.ptp(xc[:, comp]) < tol:
            return comp
    raise RuntimeError(f"facet tag {tag} is not a coordinate plane")


class Subdomain:
    """Linear-elastic subdomain with a fixed matrix and a reusable KSP.

    The stiffness matrix and the Dirichlet dof set never change during the
    Schwarz iteration -- only the Neumann traction does -- so A is assembled
    and the KSP built once, and each sweep only rebuilds the right-hand side.
    """

    def __init__(self, msh_file, name):
        self.name = name
        md = dgmsh.read_from_msh(msh_file, MPI.COMM_WORLD, gdim=3)
        self.mesh, self.facet_tags = md.mesh, md.facet_tags
        self.tdim = self.mesh.topology.dim
        self.fdim = self.tdim - 1
        self.mesh.topology.create_connectivity(self.fdim, self.tdim)

        self.V = fem.functionspace(self.mesh, ("Lagrange", 2, (3,)))
        self.uh = fem.Function(self.V, name="displacement")
        self.dof_x = self.V.tabulate_dof_coordinates()
        print(f"[{name}] cells={self.mesh.topology.index_map(self.tdim).size_local}"
              f"  P2 vector dofs={self.V.dofmap.index_map.size_local * 3}")

    def rollers(self):
        """u_n = 0 on the two symmetry planes and on z = 0 / z = h."""
        bcs = []
        for tag in (BOTTOM, TOP, SYM_A, SYM_B):
            comp = detect_roller(self.V, self.facet_tags, tag, self.fdim)
            dofs = fem.locate_dofs_topological(
                self.V.sub(comp), self.fdim, self.facet_tags.find(tag))
            bcs.append(fem.dirichletbc(default_scalar_type(0.0), dofs,
                                       self.V.sub(comp)))
            print(f"[{self.name}]   tag {tag}: locking u_{'xyz'[comp]}")
        return bcs

    def iface_dofs(self, tag):
        d = fem.locate_dofs_topological(self.V, self.fdim,
                                        self.facet_tags.find(tag))
        return d, self.dof_x[d]

    def setup(self, bcs, iface_tag, traction):
        self.bcs = bcs
        u, v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        f = fem.Constant(self.mesh, default_scalar_type((0.0, 0.0, 0.0)))
        self.a_form = fem.form(ufl.inner(sigma(u), epsilon(v)) * ufl.dx)
        ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.facet_tags)
        self.L_form = fem.form(ufl.inner(f, v) * ufl.dx
                               + ufl.inner(traction, v) * ds(iface_tag))
        self.A = assemble_matrix(self.a_form, bcs=bcs)
        self.A.assemble()
        # DOLFINx 0.11: create_vector takes the FUNCTION SPACE, not the form.
        self.b = create_vector(self.V)
        self.ksp = PETSc.KSP().create(self.mesh.comm)
        self.ksp.setOperators(self.A)
        self.ksp.setType(PETSc.KSP.Type.CG)
        self.ksp.getPC().setType(PETSc.PC.Type.GAMG)
        self.ksp.setTolerances(rtol=1e-10)
        self.ksp.setFromOptions()

    def solve(self):
        with self.b.localForm() as loc:
            loc.set(0.0)
        assemble_vector(self.b, self.L_form)
        apply_lifting(self.b, [self.a_form], bcs=[self.bcs])
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b, self.bcs)
        self.ksp.solve(self.b, self.uh.x.petsc_vec)
        self.uh.x.scatter_forward()
        return self.ksp.getIterationNumber()

    def disp_at(self, blocks):
        return self.uh.x.array.reshape(-1, 3)[blocks]

    def set_at(self, fn, blocks, values):
        fn.x.array.reshape(-1, 3)[blocks] = values


os.chdir(mesh_dir)
t0 = time.time()
OII = Subdomain("Tube_outer.msh", "Omega_II")
dII, xII = OII.iface_dofs(S_INNER)
print(f"interface dofs on Omega_II: {dII.size}")

# outward normal of OMEGA_II at r = 2 points towards the axis, so it is -e_r
rr = np.hypot(xII[:, 0], xII[:, 1])
e_r = np.zeros((dII.size, 3))
e_r[:, 0] = xII[:, 0] / rr
e_r[:, 1] = xII[:, 1] / rr


# ===========================================================================
# 2.  the operator
# ===========================================================================
with h5py.File(DATA, "r") as f:
    mu_d, lmbda_d = float(f.attrs["mu"]), float(f.attrs["lmbda"])
assert abs(mu_d - MU) < 1e-12 and abs(lmbda_d - LMBDA) < 1e-12, \
    "material constants differ between this script and the training data"

with open(CKPT, "rb") as fh:
    _ck = pickle.load(fh)
params, cfg = _ck["params"], _ck["config"]

# The checkpoint carries the interface coordinates it was trained on.  Using
# those, rather than the dataset's, means a mismatched dataset shows up in the
# point-matching assert below instead of silently producing a wrong answer.
xyz_if = np.asarray(cfg["xyz_in"], np.float64)
m_s = xyz_if.shape[0]
model = PI_DeepONet(list(cfg["trunk_layers"]),
                    in_channels=int(cfg["in_channels"]),
                    E=cfg["E"], nu=cfg["nu"])
print(f"loaded {CKPT}")
print(f"  trunk {list(cfg['trunk_layers'])}  branch {m_s} pts x "
      f"{cfg['in_channels']} ch  shift {cfg['shift']}")

# This file feeds the branch UNSCALED displacement and reads predict_s back
# unscaled.  That matches a checkpoint trained without the input scaling.  If
# --ckpt points at a run that used SCALE, the branch input needs `* SCALE`
# here and predict_s already divides on the way out -- mixing the two is a
# silent factor of SCALE, not an error, so the guard is explicit.
_ck_scale = cfg.get("scale", None)
assert _ck_scale in (None, 1, 1.0), (
    f"the checkpoint was trained with SCALE={_ck_scale} but this script feeds "
    f"the branch unscaled displacement.  Multiply u by SCALE in "
    f"operator_traction, or use the no_operator.py adapter, which handles it.")

assert m_s == dII.size, \
    f"branch expects {m_s} interface points, Omega_II has {dII.size}"

# The branch input must be assembled in the TRAINING point order, so map
# Omega_II's interface dofs onto it.  Conformal meshes -> exact permutation.
#
# The tolerance is a fraction of the actual point spacing, not an absolute
# number.  xyz_in is stored as float32, whose resolution at |x| ~ 4 is 4.8e-7,
# so any fixed threshold below that can never be met however well the meshes
# agree.  What actually matters is that each point is far closer to its match
# than to any other point, which is what the ratio below measures -- and the
# permutation check is the real guarantee: if two training points had claimed
# the same Omega_II dof, the match would be ambiguous and that assert would
# catch it regardless of the distances.
def _match(src, dst, label):
    tree = cKDTree(dst)
    dist, idx = tree.query(src)
    spacing = tree.query(dst, k=2)[0][:, 1].min()      # closest pair in dst
    ratio = dist.max() / spacing
    assert len(np.unique(idx)) == len(src), \
        f"{label}: not a permutation, two points claim the same dof"
    assert ratio < 1e-2, \
        (f"{label}: point clouds do not match.  max gap {dist.max():.2e} is "
         f"{ratio:.1%} of the {spacing:.2e} point spacing -- different meshes?")
    print(f"  {label}: exact permutation, max gap {dist.max():.2e} "
          f"({ratio:.1e} of the {spacing:.2e} point spacing)")
    return idx


perm = _match(xyz_if, xII, "branch point map")

_xyz_f32 = xyz_if.astype(np.float32)
Y_star = jnp.asarray(xII, jnp.float32)   # evaluate the operator ON Omega_II's dofs


def operator_traction(u_gamma):
    """Interface displacement in Omega_II's dof order  ->  (sigma_I . e_r, eps).

    The branch is fed the field in the TRAINING point order; the trunk is
    evaluated directly at Omega_II's interface coordinates, so nothing has to
    be permuted back on the way out.
    """
    Fb = np.concatenate([_xyz_f32, u_gamma[perm].astype(np.float32) * SCALE], axis=1)
    _, eps = model.predict_s(params, jnp.asarray(Fb[None, :, :]), Y_star)
    eps = np.asarray(eps[0], float)                        # (n, 6)
    sig = hooke_voigt(eps)                                 # (n, 3, 3)
    return np.einsum("nij,nj->ni", sig, e_r), eps


# ===========================================================================
# 3.  Omega_II with the operator's traction as Neumann data
# ===========================================================================
traction = fem.Function(OII.V)
OII.setup(OII.rollers(), S_INNER, traction)

# --- Omega_II's OWN interface traction, for --relax traction ----------------
# The 2-D script blends the operator's sigma.n with the one it recovers from
# the FE solution.  That second estimate has to come from the SOLUTION, not
# from the datum that was fed in -- reading back the datum would make the
# blend a pure iteration lag and throw away the point of it.
#
# Sign: this returns sigma_II . (+e_r), the same convention operator_traction
# uses, so the two are directly comparable.  Omega_II's own outward normal is
# -e_r; that minus is applied once, where the datum is set.
#
# The blocks of the (3,3) CG2 space and the (3,) CG2 space share the scalar
# dofmap, so `dII` indexes both.  Section 8 already relies on that alignment.
VtII_iter = fem.functionspace(OII.mesh, ("Lagrange", 2, (3, 3)))
_ip_iter = VtII_iter.element.interpolation_points
_ip_iter = _ip_iter() if callable(_ip_iter) else _ip_iter
_sig_II_fn = fem.Function(VtII_iter)
_sig_II_expr = fem.Expression(sigma(OII.uh), _ip_iter)


def omega_II_traction():
    """sigma_II . e_r on Gamma, recovered from Omega_II's current FE solution."""
    _sig_II_fn.interpolate(_sig_II_expr)
    S = _sig_II_fn.x.array.reshape(-1, 9)[dII].reshape(-1, 3, 3)
    return np.einsum("nij,nj->ni", S, e_r)


# ===========================================================================
# 4.  check the sign BEFORE iterating
#     Feed the operator the converged FE-FE interface field and compare the
#     traction it returns against the one FE-FE recorded there.  A sign error
#     shows up as a relative difference near 2, not near the operator's error.
# ===========================================================================
u_conv, t_err_fixed = None, None
if os.path.exists(FEFE):
    with h5py.File(FEFE, "r") as f:
        xyz_fe = np.asarray(f["xyz"], np.float64)
        u_fe = np.asarray(f["u"], np.float64)
        t_fe = np.asarray(f["t_I" if "t_I" in f else "t"], np.float64)
    p2 = _match(xII, xyz_fe, "FE-FE point map")
    u_conv = u_fe[-1][p2]                    # converged u_Gamma, Omega_II order
    t_ref = t_fe[-1][p2]                     # sigma_I . (+e_r) from FE
    t_no, _ = operator_traction(u_conv)
    rel_p = np.linalg.norm(t_no - t_ref) / np.linalg.norm(t_ref)
    rel_m = np.linalg.norm(-t_no - t_ref) / np.linalg.norm(t_ref)
    print(f"\nsign check on the converged FE-FE interface field")
    print(f"   || t_NO - t_FE || / || t_FE ||  = {rel_p:.3e}")
    print(f"   || -t_NO - t_FE || / || t_FE || = {rel_m:.3e}")
    if rel_m < rel_p:
        raise SystemExit(
            "The flipped sign fits better -- the traction convention is wrong.\n"
            "Fix it before iterating; the loop would only diverge.")
    t_err_fixed = rel_p
    print(f"   sign OK.  {rel_p:.1%} is the operator's traction error AT the "
          f"fixed point;\n   with rho = {RHO_FEFE} that predicts a coupled "
          f"error near {rel_p / (1 - RHO_FEFE):.1%}.")
else:
    print(f"\n{FEFE} not found -- no sign check and no reference to score "
          f"against.\nRun FE_FE_static_coupling_r_2.py first; without it a sign "
          f"error only shows up as divergence.")


# ===========================================================================
# 5.  alternating Schwarz
# ===========================================================================
print(f"\nSchwarz FE-NO: TOL={TOL:.0e}  max {NITER} iterations")
if RELAX == "traction":
    print(f"  relax on the TRACTION (the 2-D script's flux blend):")
    print(f"      t <- {THETA:.3g} * sigma_II.n|FE  +  {1-THETA:.3g} * "
          f"sigma_I.n|NO")
    print(f"      theta weights the FE SIDE, so larger theta = more damping")
else:
    print(f"  relax on the DISPLACEMENT (standard under-relaxation):")
    print(f"      u_Gamma <- {1-THETA:.3g} * u_old  +  {THETA:.3g} * u_new")
    print(f"      theta weights the NEW value, so smaller theta = more damping")
print("u_Gamma^0 = 0.  That is the GP prior's MEAN, so the operator starts at")
print("the centre of its training distribution, not outside it.\n")

u_gamma = np.zeros((dII.size, 3))
u_prev_full, err0, t_II = None, None, None
hist, iterates = [], []
uII_iterates = []          # full Omega_II (FE) displacement field each sweep,
                           # kept so the per-sweep dataset can be rebuilt below

for it in range(NITER):
    # ---- (a) the OPERATOR plays Omega_I ----------------------------------
    iterates.append(u_gamma.copy())
    t_NO, eps_I = operator_traction(u_gamma)          # T_c_new in the 2-D script

    # ---- (b) blend the TWO SIDES' estimates of the same interface flux ----
    # t_II is Omega_II's own sigma.n, read back from its FE SOLUTION at the end
    # of the previous sweep -- T_c0 in the 2-D script.  Reading the solution
    # rather than the datum that was fed in is the whole point: at the true
    # solution the two sides' fluxes agree, so this is a convex combination of
    # two estimates of one physical quantity, and the FE side is the accurate
    # one.  On the first sweep there is no FE solution yet, so the operator's
    # estimate goes in unblended.
    if RELAX == "traction":
        t_I = t_NO if t_II is None else THETA * t_II + (1.0 - THETA) * t_NO
    else:
        t_I = t_NO

    # ---- (c) Omega_II, whose outward normal at r = 2 is -e_r -------------
    OII.set_at(traction, dII, -t_I)
    nII = OII.solve()
    u_new = OII.disp_at(dII)
    if RELAX == "traction":
        t_II = omega_II_traction()                    # T_c0 for the next sweep

    # ---- (d) relaxation on the DISPLACEMENT ------------------------------
    du = np.linalg.norm(u_new - u_gamma)
    u_gamma = ((1.0 - THETA) * u_gamma + THETA * u_new
               if RELAX == "u" else u_new)

    # ---- (e) convergence --------------------------------------------------
    uII_full = OII.uh.x.array.reshape(-1, 3).copy()
    uII_iterates.append(uII_full)          # paired with iterates[it] above
    err = np.nan if u_prev_full is None else \
        np.linalg.norm(uII_full - u_prev_full) + du
    u_prev_full = uII_full
    rel_du = du / max(np.linalg.norm(u_gamma), 1e-30)
    hist.append([err, rel_du])

    es = "   --    " if np.isnan(err) else f"{err:.3e}"
    print(f"  it {it:3d}  ksp {nII:3d}  error={es}  "
          f"d|u_Gamma|/|u_Gamma|={rel_du:.3e}")

    if not np.isnan(err) and err < TOL:
        print(f"converged after {it + 1} iterations  ({err:.3e} < {TOL:.0e})")
        break

    # Divergence guard, once there is something to compare against.  At it = 0
    # err is the NaN sentinel (no previous field), and np.isfinite(nan) is
    # False -- testing it there declares divergence on the very first sweep.
    # The first rel_du is not informative either: u_Gamma starts at 0, so
    # du = |u_new| while |u_Gamma| = THETA*|u_new|, making rel_du exactly
    # 1/THETA (2.0 at THETA = 0.5) however well the iteration is going.
    # The growth baseline must be the first FINITE increment, not hist[0],
    # which is NaN and would make every comparison against it False.
    if np.isfinite(err):
        if err0 is None:
            err0 = err
        elif err > 1e3 * err0:
            print(f"DIVERGED -- the increment grew from {err0:.2e} to "
                  f"{err:.2e}.  Try a smaller --theta")
            break
    elif it > 0:
        print("DIVERGED -- the increment is not finite.  Try a smaller --theta")
        break
else:
    print(f"WARNING: not converged in {NITER} iterations")

hist = np.array(hist)
print(f"wall time {time.time() - t0:.1f} s")

# ---- the single convergence curve for static_convergence_plot.py ------------
# It loads current/error_list_FE_NN.npy as ONE column, "L2 error vs inner
# iteration".  That is exactly this run's per-sweep convergence measure
# hist[:,0] (here: ||Delta u_II|| + ||Delta u_Gamma||, the quantity the stop
# test compares to TOL -- which is why the reference curves all end at ~TOL).
# The first sweep has no predecessor, so its NaN is dropped and the curve starts
# at inner iteration 1, matching the 2-D reference point counts.
error_list_FE_NN = hist[np.isfinite(hist[:, 0]), 0]
np.save(os.path.join(out_dir, "error_list_FE_NN.npy"), error_list_FE_NN)
print(f"wrote error_list_FE_NN.npy ({error_list_FE_NN.size} inner iterations) "
      f"-> feed to static_convergence_plot.py as current/error_list_FE_NN.npy")

# This is the FIRST file the run produces, so with --error-list-only you can
# stop here and skip the expensive monolithic solve, dataset dump and rendering.
if args.error_list_only:
    print(f"--error-list-only: stopping after error_list_FE_NN.npy "
          f"(in {out_dir}).")
    raise SystemExit(0)


# ===========================================================================
# 6.  how good is it?
# ===========================================================================
print("\n" + "=" * 70)
print("FE-NO vs FE-FE on the interface")
print("=" * 70)
if u_conv is not None:
    e_u = np.linalg.norm(u_gamma - u_conv) / np.linalg.norm(u_conv)
    print(f"  interface displacement, relative L2    {e_u:.3e}")
    if t_err_fixed is not None:
        pred = t_err_fixed / (1 - RHO_FEFE)
        print(f"  predicted from t_err/(1-rho)           {pred:.3e}")
        print(f"  measured / predicted                   {e_u / pred:.2f}")
        print("\n  A ratio near 1 means the operator behaves along the whole")
        print("  iteration path the way it does at the fixed point.  Much")
        print("  larger means the intermediate iterates are outside what it")
        print("  learned -- `iterates` in the .npz has them, feed those to")
        print("  the operator to see which sweep goes wrong.")
    print(f"\n  For scale: FE-FE against the monolithic solution is 7.56e-04,")
    print(f"  so replacing the solver by the operator costs a factor "
          f"{e_u / 7.56e-4:.0f} here.")
else:
    print("  no FE-FE reference available")

# ===========================================================================
# 7.  the real reference: the monolithic solution on Tube_entire.msh
#
#     FE-FE is a useful intermediate, but it is not ground truth -- it carries
#     its own 7.56e-04.  And the interface displacement alone does not say
#     whether the OPERATOR is right: u_Gamma is produced by Omega_II, so a bad
#     operator can still land on a plausible interface while the field it
#     reports INSIDE Omega_I is wrong.  That field is the actual deliverable,
#     so it is compared here too.
# ===========================================================================
E_INNER, E_INTERFACE, E_OUTER, E_BOTTOM, E_TOP, E_SYMX, E_SYMY = 1, 2, 3, 4, 5, 6, 7

print("\n" + "=" * 70)
print("FE-NO vs the MONOLITHIC solution (Tube_entire.msh)")
print("=" * 70)
print("solving the monolithic reference ...")
os.chdir(mesh_dir)
_md = dgmsh.read_from_msh("Tube_entire.msh", MPI.COMM_WORLD, gdim=3)
dom, ftags = _md.mesh, _md.facet_tags
fdim = dom.topology.dim - 1
dom.topology.create_connectivity(fdim, dom.topology.dim)
Vm = fem.functionspace(dom, ("Lagrange", 2, (3,)))

# the SAME load on r = 1 that the dataset used -- that is the whole point
um_load = fem.Function(Vm)
um_load.interpolate(lambda x: np.array([0.01 * x[0] ** 2,
                                        0.01 * x[1] ** 2,
                                        0.005 * (x[2] - H) * x[2]]))
bcs_m = [fem.dirichletbc(um_load, fem.locate_dofs_topological(
    Vm, fdim, ftags.find(E_INNER)))]
for tag in (E_BOTTOM, E_TOP, E_SYMX, E_SYMY):
    comp = detect_roller(Vm, ftags, tag, fdim)
    bcs_m.append(fem.dirichletbc(
        default_scalar_type(0.0),
        fem.locate_dofs_topological(Vm.sub(comp), fdim, ftags.find(tag)),
        Vm.sub(comp)))

u_, v_ = ufl.TrialFunction(Vm), ufl.TestFunction(Vm)
f_ = fem.Constant(dom, default_scalar_type((0.0, 0.0, 0.0)))
am = fem.form(ufl.inner(sigma(u_), epsilon(v_)) * ufl.dx)
Lm = fem.form(ufl.inner(f_, v_) * ufl.dx)
Am = assemble_matrix(am, bcs=bcs_m); Am.assemble()
bm = create_vector(Vm)                       # 0.11: space, not form
assemble_vector(bm, Lm)
apply_lifting(bm, [am], bcs=[bcs_m])
bm.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(bm, bcs_m)
uhm = fem.Function(Vm, name="displacement")
kspm = PETSc.KSP().create(dom.comm); kspm.setOperators(Am)
kspm.setType(PETSc.KSP.Type.CG); kspm.getPC().setType(PETSc.PC.Type.GAMG)
kspm.setTolerances(rtol=1e-10); kspm.solve(bm, uhm.x.petsc_vec)
uhm.x.scatter_forward()

# strain of the monolithic solution, CG2 -- the same recovery the dataset used,
# so the comparison measures the operator and not a post-processing difference
Vtm = fem.functionspace(dom, ("Lagrange", 2, (3, 3)))
_ip = Vtm.element.interpolation_points
_ip = _ip() if callable(_ip) else _ip
epsm = fem.Function(Vtm)
epsm.interpolate(fem.Expression(epsilon(uhm), _ip))
Um = uhm.x.array.reshape(-1, 3)
Em = epsm.x.array.reshape(-1, 9)[:, [0, 4, 8, 1, 2, 5]]
xm = Vm.tabulate_dof_coordinates()


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


rows = []

# ---- (a) the interface -----------------------------------------------------
dm_if = fem.locate_dofs_topological(Vm, fdim, ftags.find(E_INTERFACE))
q = _match(xII, xm[dm_if], "monolithic interface map")
u_ref, e_ref = Um[dm_if][q], Em[dm_if][q]
_, eps_no = operator_traction(u_gamma)
rows.append(("interface  u", rel(u_gamma, u_ref)))
rows.append(("interface  strain", rel(eps_no, e_ref)))

# ---- (b) INSIDE Omega_I -- what the operator actually produces -------------
# every monolithic dof with 1 <= r <= 2.  The trunk takes arbitrary points, so
# the operator is simply evaluated there; no interpolation is involved.
rm = np.hypot(xm[:, 0], xm[:, 1])
inI = np.flatnonzero((rm > R1 - 1e-9) & (rm < R2 + 1e-9))
Fb_conv = np.concatenate([_xyz_f32, u_gamma[perm].astype(np.float32) * SCALE], axis=1)
u_pred, e_pred = model.predict_s(params, jnp.asarray(Fb_conv[None, :, :]),
                                 jnp.asarray(xm[inI], jnp.float32))
rows.append((f"Omega_I    u   ({inI.size} pts)",
             rel(np.asarray(u_pred[0], float), Um[inI])))
rows.append(("Omega_I    strain", rel(np.asarray(e_pred[0], float), Em[inI])))

# ---- (c) Omega_II ----------------------------------------------------------
# Tube_entire is NOT the union of Tube_inner and Tube_outer.  Only the two
# cylindrical surfaces r = 1 and r = 2 are conformal (they share the same
# transfinite 50x80 grid, which is why the interface matched to 1e-15 above);
# the volumes are meshed independently -- 8409 vertices at r >= 2 in
# Tube_entire against 7377 in Tube_outer.  So there is no permutation here and
# Omega_II's field has to be interpolated onto the monolithic points.
#
# That interpolation is linear on a mesh of spacing ~0.05 acting on a field
# whose second derivative is O(1e-2), so its own error is ~1e-4 relative --
# two orders below the operator error being measured, and confined to Omega_II
# which the operator never touches.
inII = np.flatnonzero(rm > R2 - 1e-9)


def interp_to(src_pts, src_vals, dst_pts, label=""):
    """Linear interpolation between non-matching meshes, nearest as fallback
    for the handful of destination points outside the source hull."""
    from scipy.interpolate import LinearNDInterpolator
    print(f"  interpolating {label} ({src_pts.shape[0]} -> "
          f"{dst_pts.shape[0]} pts) ...")
    f = LinearNDInterpolator(src_pts, src_vals)
    out = f(dst_pts)
    bad = ~np.isfinite(out).all(axis=1)
    if bad.any():
        out[bad] = src_vals[cKDTree(src_pts).query(dst_pts[bad])[1]]
        print(f"    {bad.sum()} point(s) outside the hull -> nearest")
    return out


U_II_on_m = interp_to(OII.dof_x, OII.uh.x.array.reshape(-1, 3),
                      xm[inII], "Omega_II displacement")
rows.append((f"Omega_II   u   ({inII.size} pts, interp)",
             rel(U_II_on_m, Um[inII])))

print()
print(f"  {'quantity':32s} {'relative L2':>12s}")
for name, v in rows:
    print(f"  {name:32s} {v:12.3e}")
print(f"\n  FE-FE on the same interface: 7.56e-04")
print("  Omega_I is the operator's own output -- it is the number that says")
print("  whether the coupled run reproduces the field, not just the boundary.")

np.savez_compressed(os.path.join(out_dir, "fe_no_coupling.npz"),
                    monolithic=np.array([v for _, v in rows]),
                    monolithic_names=np.array([n for n, _ in rows], dtype=object),
                    history=hist, theta=THETA, relax=RELAX,
                    xyz_interface=xII.astype(np.float32),
                    u_interface=u_gamma.astype(np.float32),
                    t_interface=t_I.astype(np.float32),
                    eps_interface=eps_I.astype(np.float32),
                    iterates=np.asarray(iterates, np.float32))

# ===========================================================================
# 8.  3-D fields, all three on the MONOLITHIC mesh
#
#     Putting the coupled answer, the reference and their difference on ONE
#     mesh is what makes them comparable: the same dof, the same point, no
#     interpolation anywhere.  Omega_I's half of the coupled field is the
#     OPERATOR evaluated at those dofs -- the trunk takes arbitrary points, so
#     there is nothing to interpolate there either.
# ===========================================================================
print("\nassembling the 3-D fields on the monolithic mesh ...")

# strain of the Omega_II FE solution, CG2, so both halves are recovered the
# same way as the training data
VtII = fem.functionspace(OII.mesh, ("Lagrange", 2, (3, 3)))
_ip2 = VtII.element.interpolation_points
_ip2 = _ip2() if callable(_ip2) else _ip2
epsII = fem.Function(VtII)
epsII.interpolate(fem.Expression(epsilon(OII.uh), _ip2))
E_II = epsII.x.array.reshape(-1, 9)[:, [0, 4, 8, 1, 2, 5]]

# --- displacement -----------------------------------------------------------
# Omega_I's half is the OPERATOR evaluated at those dofs -- exact, no
# interpolation.  Omega_II's half has to be interpolated (non-matching volume
# meshes, see above); it is written second so it wins on r = 2, where the two
# grids do coincide and the value is u_Gamma either way.
U_cpl = np.zeros_like(Um)
U_cpl[inI] = np.asarray(u_pred[0], float)
U_cpl[inII] = U_II_on_m

# --- strain -----------------------------------------------------------------
E_cpl = np.zeros_like(Em)
E_cpl[inI] = np.asarray(e_pred[0], float)
E_cpl[inII] = interp_to(OII.dof_x, E_II, xm[inII], "Omega_II strain")

u_cpl_fn = fem.Function(Vm, name="u_coupled")
u_ref_fn = fem.Function(Vm, name="u_monolithic")
u_err_fn = fem.Function(Vm, name="u_abs_error")
u_cpl_fn.x.array[:] = U_cpl.reshape(-1)
u_ref_fn.x.array[:] = Um.reshape(-1)
u_err_fn.x.array[:] = np.abs(U_cpl - Um).reshape(-1)

e_cpl_fn = fem.Function(Vtm, name="strain_coupled")
e_ref_fn = fem.Function(Vtm, name="strain_monolithic")
e_err_fn = fem.Function(Vtm, name="strain_abs_error")
_back = [0, 3, 4, 3, 1, 5, 4, 5, 2]                # Voigt -> full 3x3, row major
e_cpl_fn.x.array[:] = E_cpl[:, _back].reshape(-1)
e_ref_fn.x.array[:] = Em[:, _back].reshape(-1)
e_err_fn.x.array[:] = np.abs(E_cpl - Em)[:, _back].reshape(-1)

_cwd = os.getcwd()
os.chdir(out_dir)
from dolfinx.io import VTXWriter
for fname, fns in (("u_coupled.bp", [u_cpl_fn]),
                   ("u_monolithic.bp", [u_ref_fn]),
                   ("u_error.bp", [u_err_fn]),
                   ("strain_coupled.bp", [e_cpl_fn]),
                   ("strain_monolithic.bp", [e_ref_fn]),
                   ("strain_error.bp", [e_err_fn])):
    with VTXWriter(dom.comm, fname, fns, engine="BP4") as w:
        w.write(0.0)
print(f"  wrote 6 .bp files -- open them in ParaView")

# ===========================================================================
# 9.  PyVista snapshots
#     A P1 grid FOR DISPLAY ONLY: VTK vertex grids are linear, so the CG2
#     fields are sampled down just to draw them.  Nothing above is affected.
# ===========================================================================
try:
    import pyvista as pv
    from dolfinx import plot
    Vs1 = fem.functionspace(dom, ("Lagrange", 1))
    Vv1 = fem.functionspace(dom, ("Lagrange", 1, (3,)))

    def to_p1_vec(arr):
        f3 = fem.Function(Vm)
        f3.x.array[:] = arr.reshape(-1)
        g = fem.Function(Vv1)
        g.interpolate(f3)
        return g.x.array.reshape(-1, 3)

    def to_p1_scalar(vals):
        """vals is one CG2 scalar per block -> P1 nodal values."""
        f1 = fem.Function(Vm)
        a = np.zeros_like(Um)
        a[:, 0] = vals
        f1.x.array[:] = a.reshape(-1)
        g = fem.Function(Vv1)
        g.interpolate(f1)
        return g.x.array.reshape(-1, 3)[:, 0]

    topo, ct, geo = plot.vtk_mesh(Vs1)
    grid = pv.UnstructuredGrid(topo, ct, geo)

    def snap(name, vals, cmap="seismic", clim=None):
        grid[name] = vals
        p = pv.Plotter(off_screen=True, window_size=[1100, 850])
        p.add_mesh(grid.copy(), scalars=name, cmap=cmap, clim=clim,
                   scalar_bar_args={"title": name, "fmt": "%.2e"})
        p.view_vector((-1, -1, 0.6), viewup=(0, 0, 1))
        p.camera.view_angle = 35
        p.add_axes()
        p.screenshot(f"{name}.png")
        p.close()

    Ucp, Urf = to_p1_vec(U_cpl), to_p1_vec(Um)
    for i, nm in enumerate(("u_x", "u_y", "u_z")):
        # a SHARED colour range for coupled and reference: without it the two
        # pictures autoscale differently and look alike whatever the error is
        lim = (min(Ucp[:, i].min(), Urf[:, i].min()),
               max(Ucp[:, i].max(), Urf[:, i].max()))
        snap(f"{nm}_coupled", Ucp[:, i], clim=lim)
        snap(f"{nm}_monolithic", Urf[:, i], clim=lim)
        snap(f"{nm}_abserr", np.abs(Ucp[:, i] - Urf[:, i]), cmap="viridis")

    for k, c in enumerate(("xx", "yy", "zz", "xy", "xz", "yz")):
        a, b = to_p1_scalar(E_cpl[:, k]), to_p1_scalar(Em[:, k])
        lim = (min(a.min(), b.min()), max(a.max(), b.max()))
        snap(f"E_{c}_coupled", a, clim=lim)
        snap(f"E_{c}_monolithic", b, clim=lim)
        snap(f"E_{c}_abserr", np.abs(a - b), cmap="viridis")

    print(f"  wrote 27 PNGs (coupled / monolithic / abserr for u and strain)")
except Exception as _e:
    print(f"  PyVista skipped: {_e}")
os.chdir(_cwd)

# ===========================================================================
# 10.  PER-SWEEP DATASET  for the 3-D u_x figure (static_ux_figure_3d.py)
#
#      Records, at EVERY Schwarz sweep j, u_x and the strain in BOTH subdomains,
#      laid out exactly like the 2-D static example so the same plotting logic
#      ports over.  The only additions are Z coordinates (3-D) and strain files.
#
#      NAME MAPPING -- the FIGURE follows the 2-D example's convention
#      (Omega_I = the FE subdomain, Omega_II = the NO subdomain), which is the
#      OPPOSITE of this coupling code's internal Omega_I/Omega_II:
#
#          FIGURE Omega_I   (FE)  ==  this code's Omega_II,  r in [2, 4]  OUTER
#          FIGURE Omega_II  (NO)  ==  this code's Omega_I,   r in [1, 2]  INNER
#
#      Everything reuses objects already built above:
#          xm, Um, inI, inII        monolithic mesh, reference field, r-masks (S7)
#          interp_to                non-matching-mesh interpolation           (S7)
#          model, params, perm, _xyz_f32, SCALE, epsilon                      (S2)
#          epsII, VtII, _ip2        CG2 strain recovery on Omega_II           (S8)
#          iterates, uII_iterates   per-sweep interface u and FE field        (S5)
#
#      inI  = monolithic dofs with r in [1, 2]  ->  FIGURE Omega_II (NO, INNER)
#      inII = monolithic dofs with r in [2, 4]  ->  FIGURE Omega_I  (FE, OUTER)
#
#      Files under out_dir/dataset_ux/ :
#          X.txt  Y.txt  Z.txt          FIGURE Omega_I  (FE)  point cloud
#          X1.txt Y1.txt Z1.txt         FIGURE Omega_II (NO)  point cloud
#          groud_truth/{u,X,Y,Z}.txt    monolithic u_x + its own cloud (misspelt
#                                       to match the 2-D example verbatim)
#          FE_NO/u  i = {j} .txt        u_x in FIGURE Omega_I  (FE)  at sweep j
#          FE_NO/u2 i = {j} .txt        u_x in FIGURE Omega_II (NO)  at sweep j
#          FE_NO/eps  i = {j} .txt      strain (6 Voigt cols) Omega_I  at sweep j
#          FE_NO/eps2 i = {j} .txt      strain (6 Voigt cols) Omega_II at sweep j
#
#      Only u_x is written to the scalar u-files (COMP = 0); rerun with COMP = 1
#      or 2 for u_y / u_z.  all_iters_ux.npz carries all three components + both
#      strains so later components need no re-solve, just a re-export.
# ===========================================================================
COMP, COMP_NAME = 0, "ux"                      # 0=u_x  1=u_y  2=u_z

# Only these sweeps are written, not every one.  Any index past the actual number
# of sweeps is dropped, and the TRUE last sweep is always added so the error /
# final column is the converged field regardless of where it landed.
SAVE_INDICES = [0, 12, 26]
nsw = len(iterates)
save_idx = sorted(set([j for j in SAVE_INDICES if j < nsw] + [nsw - 1]))
print(f"\n[dataset] {nsw} sweeps ran  ->  last sweep index = {nsw - 1}")
print(f"[dataset] dumping u_{'xyz'[COMP]} for sweeps {save_idx}  "
      f"(requested {SAVE_INDICES}; dropped any >= {nsw}) ...")
if 26 in SAVE_INDICES and 26 != nsw - 1:
    print(f"[dataset] note: you asked for j=26 as the last step, but the run "
          f"actually ended at j={nsw - 1}.")

ds_dir  = os.path.join(out_dir, f"dataset_{COMP_NAME}")
fe_no_d = os.path.join(ds_dir, "FE_NO")
gt_d    = os.path.join(ds_dir, "groud_truth")
for d in (ds_dir, fe_no_d, gt_d):
    createFolder(d)

# --- point clouds, written once --------------------------------------------
x_fig_I  = xm[inII]                            # FIGURE Omega_I  (FE, outer)
x_fig_II = xm[inI]                             # FIGURE Omega_II (NO, inner)
for arr, stem in ((x_fig_I, ("X",  "Y",  "Z")),
                  (x_fig_II, ("X1", "Y1", "Z1"))):
    for k, nm in enumerate(stem):
        np.savetxt(os.path.join(ds_dir, nm + ".txt"), arr[:, k])

# --- monolithic ground truth (whole domain; the figure interpolates it) -----
np.savetxt(os.path.join(gt_d, "u.txt"), Um)                # u v w  (3 columns)
np.savetxt(os.path.join(gt_d, "eps.txt"), Em)              # strain (6 Voigt cols)
for k, nm in enumerate(("X", "Y", "Z")):
    np.savetxt(os.path.join(gt_d, nm + ".txt"), xm[:, k])

# --- per-sweep fields on both subdomains -----------------------------------
Xt_fig_II = jnp.asarray(x_fig_II, jnp.float32)  # trunk pts, NO side (operator)
U_all_I, U_all_II = [], []                      # (nsweep, npts, 3) for the npz
E_all_I, E_all_II = [], []                      # (nsweep, npts, 6)

# --- P1 DISPLAY MESH for SOLID rendering in the figure ----------------------
# The same linear grid the PyVista section above draws.  Saved once; the coupled
# field of every sweep is interpolated onto its nodes below, so the figure can
# draw solid surfaces (thresholded by radius into Omega_I / Omega_II) instead of
# a scatter.  Best-effort: on failure the txt point dataset is still written and
# the figure falls back to scatter.
_P1_OK = False
try:
    from dolfinx import plot as _dplot
    _Vv1 = fem.functionspace(dom, ("Lagrange", 1, (3,)))
    _topo_p1, _ct_p1, _geo_p1 = _dplot.vtk_mesh(
        fem.functionspace(dom, ("Lagrange", 1)))

    def _to_p1(full3):
        _f = fem.Function(Vm); _f.x.array[:] = np.asarray(full3).reshape(-1)
        _g = fem.Function(_Vv1); _g.interpolate(_f)
        return _g.x.array.reshape(-1, 3)

    # 6-component version for the strain (Voigt xx yy zz xy xz yz)
    _Vv6 = fem.functionspace(dom, ("Lagrange", 1, (6,)))
    _Vt6 = fem.functionspace(dom, ("Lagrange", 2, (6,)))

    def _to_p1_6(full6):
        _f = fem.Function(_Vt6); _f.x.array[:] = np.asarray(full6).reshape(-1)
        _g = fem.Function(_Vv6); _g.interpolate(_f)
        return _g.x.array.reshape(-1, 6)

    np.save(os.path.join(ds_dir, "grid_points.npy"), _geo_p1.astype(np.float32))
    np.save(os.path.join(ds_dir, "grid_cells.npy"), np.asarray(_topo_p1, np.int64))
    np.save(os.path.join(ds_dir, "grid_celltypes.npy"), np.asarray(_ct_p1, np.uint8))
    np.save(os.path.join(gt_d, "u_p1.npy"), _to_p1(Um).astype(np.float32))
    np.save(os.path.join(gt_d, "eps_p1.npy"), _to_p1_6(Em).astype(np.float32))
    _P1_OK = True
    print(f"[dataset] P1 solid mesh: {_geo_p1.shape[0]} nodes -> grid_*.npy")
except Exception as _e:
    print(f"[dataset] P1 solid-mesh export skipped ({_e}); figure will scatter")

for j in save_idx:
    ug = iterates[j]

    # FIGURE Omega_II (NO): the operator evaluated at the inner points.  Same
    # branch assembly as operator_traction, trunk read straight at x_fig_II.
    Fb_j = np.concatenate([_xyz_f32, ug[perm].astype(np.float32) * SCALE], axis=1)
    up_II, ep_II = model.predict_s(params, jnp.asarray(Fb_j[None, :, :]), Xt_fig_II)
    up_II = np.asarray(up_II[0], float)         # (npts_II, 3)
    ep_II = np.asarray(ep_II[0], float)         # (npts_II, 6)

    # FIGURE Omega_I (FE): the stored FE field, its CG2 strain recovered the
    # same way as the training data, both interpolated onto the outer points.
    OII.uh.x.array[:] = uII_iterates[j].reshape(-1)
    epsII.interpolate(fem.Expression(epsilon(OII.uh), _ip2))
    E_II_fe = epsII.x.array.reshape(-1, 9)[:, [0, 4, 8, 1, 2, 5]]
    up_I = interp_to(OII.dof_x, uII_iterates[j], x_fig_I, f"sweep {j:>3d} u  (FE)")
    ep_I = interp_to(OII.dof_x, E_II_fe,         x_fig_I, f"sweep {j:>3d} eps(FE)")

    # u-files now hold ALL THREE displacement components (u v w), 3 columns.
    # (naming kept verbatim from the 2-D example; strains stay 6-column Voigt.)
    np.savetxt(os.path.join(fe_no_d, f"u i = {j} .txt"),  up_I)    # u v w
    np.savetxt(os.path.join(fe_no_d, f"u2 i = {j} .txt"), up_II)   # u v w
    np.savetxt(os.path.join(fe_no_d, f"eps i = {j} .txt"),  ep_I)
    np.savetxt(os.path.join(fe_no_d, f"eps2 i = {j} .txt"), ep_II)
    U_all_I.append(up_I);  U_all_II.append(up_II)
    E_all_I.append(ep_I);  E_all_II.append(ep_II)

    # coupled field on the whole monolithic mesh -> P1 nodes, for solid render
    if _P1_OK:
        _U = np.zeros_like(Um)
        _U[inII] = up_I           # figure Omega_I  (outer, r in [2,4])
        _U[inI]  = up_II          # figure Omega_II (inner, r in [1,2])
        np.save(os.path.join(fe_no_d, f"u_p1 i = {j} .npy"),
                _to_p1(_U).astype(np.float32))
        _E = np.zeros_like(Em)                        # coupled strain, 6 Voigt cols
        _E[inII] = ep_I           # figure Omega_I  (FE strain)
        _E[inI]  = ep_II          # figure Omega_II (operator strain)
        np.save(os.path.join(fe_no_d, f"eps_p1 i = {j} .npy"),
                _to_p1_6(_E).astype(np.float32))

np.savez_compressed(
    os.path.join(ds_dir, "all_iters_ux.npz"),
    xyz_I=x_fig_I.astype(np.float32),  xyz_II=x_fig_II.astype(np.float32),
    xyz_gt=xm.astype(np.float32),      u_gt=Um.astype(np.float32),
    u_I=np.asarray(U_all_I, np.float32),   u_II=np.asarray(U_all_II, np.float32),
    eps_I=np.asarray(E_all_I, np.float32), eps_II=np.asarray(E_all_II, np.float32),
    sweeps=np.asarray(save_idx),           # which sweep index each slice is
    comp=COMP, comp_name=COMP_NAME, method="FE_NO",
    n_sweeps=nsw, theta=THETA, relax=RELAX)
print(f"[dataset] wrote sweeps {save_idx} x 4 txt files -> {fe_no_d}")
print(f"[dataset] coords + ground truth -> {ds_dir}")
print(f"[dataset] all components + strains -> {os.path.join(ds_dir,'all_iters_ux.npz')}")
# restore the converged FE field the plot below expects
OII.uh.x.array[:] = uII_iterates[-1].reshape(-1)


fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].semilogy(hist[:, 0], "o-", label=r"$\|\Delta u_{II}\|+\|\Delta u_\Gamma\|$")
ax[0].axhline(TOL, color="k", ls="--", lw=0.8, label=f"tol {TOL:.0e}")
ax[0].set_xlabel("Schwarz iteration"); ax[0].set_ylabel("increment")
ax[0].set_title(rf"FE-NO alternating Schwarz, $\theta$={THETA}, relax {RELAX}")
ax[0].grid(True, which="both", alpha=0.3); ax[0].legend(fontsize=8)

nrm = [np.linalg.norm(u) for u in iterates]
ax[1].plot(nrm, "o-", label="FE-NO")
if u_conv is not None:
    ax[1].axhline(np.linalg.norm(u_conv), color="r", ls="--",
                  label="FE-FE converged")
ax[1].set_xlabel("Schwarz iteration"); ax[1].set_ylabel(r"$\|u_\Gamma\|$")
ax[1].set_title("does it land on the right fixed point?")
ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "fe_no_convergence.png"), dpi=170)
print(f"\nresults in {out_dir}")