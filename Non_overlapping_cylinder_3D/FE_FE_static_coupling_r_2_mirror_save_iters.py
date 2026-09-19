"""
=============================================================================
FE-FE coupling, MIRRORED onto the FE-NO iteration so the two are comparable.

    *** PORTED TO DOLFINx 0.8.0 ***
    Same COMPAT block as FE_NO_static_coupling_r_2_dolfinx080.py, so this file
    also runs unchanged on 0.9 / 0.10 / 0.11.  The four differences are:

      1. mesh reading      0.8  dolfinx.io.gmshio.read_from_msh -> 3-tuple
                           0.9+ MeshData object (.mesh/.cell_tags/.facet_tags)
      2. create_vector     <=0.10 takes the FORM,  0.11 takes the SPACE
      3. PETSc vector of a Function   Function.vector  vs  Function.x.petsc_vec
      4. element.interpolation_points   method (0.8) vs property (0.9+)

    Nothing else in the script is version dependent.
-----------------------------------------------------------------------------

This is FE_FE_static_coupling_r_2.py with one thing changed: the iteration is
now bit-for-bit the one FE_NO_static_coupling runs, with the neural operator
replaced by an exact FE solve of Omega_I.  Everything else -- meshes, load,
material, rollers, tolerance -- is identical to the original script.

WHY A SEPARATE SCRIPT
---------------------
The original FE-FE gives Omega_I the Neumann datum and Omega_II the Dirichlet
datum.  The operator cannot be used that way round: it was trained as the
interface Dirichlet-to-Neumann map, so in FE-NO the assignment is reversed.

                     Omega_I        Omega_II       theta weights
    FE-FE original   Neumann        Dirichlet      the NEW traction
    FE-NO / 2-D      Dirichlet      Neumann        the FE side (old)
    this script      Dirichlet      Neumann        the FE side (old)

Two consequences of the original mismatch, both of which this script removes:

  * "FE-FE converges in N iterations, FE-NO in M" was never a like-for-like
    statement -- the two were different iterations, not the same iteration
    with a different Omega_I.

  * FE_NO_static_coupling uses RHO_FEFE = 0.40 to predict how the operator's
    interface error is amplified in the converged coupled solution.  That 0.40
    was measured on the ORIGINAL scheme.  The contraction factor is a property
    of the iteration, not of the problem, so it need not carry over.  This
    script measures rho for the iteration FE-NO actually runs, which is the
    number that prediction needs.

THE ITERATION
-------------
    u_Gamma^0 = 0
    repeat
        solve Omega_I   u = u_load on r=1,  u = u_Gamma on Gamma   (Dirichlet)
        t_I   = sigma_I . e_r                      <- the exact DtN map
        t     = th t_II + (1-th) t_I               <- flux blend, th on the FE side
        solve Omega_II  sigma.n = -t on Gamma      (Neumann)
        u_Gamma = u_II|Gamma
        t_II  = sigma_II . e_r                     <- for the next sweep
    until the interface stops moving

Omega_I here plays exactly the role the operator plays in FE-NO: it is handed a
displacement on Gamma and returns a traction.  Omega_II is the same Neumann
problem in both scripts.

STOPPING TEST -- ALIGNED WITH FE-NO
-----------------------------------
    err = || u_II^k - u_II^{k-1} ||  +  || u_Gamma^k - u_Gamma^{k-1} ||

term for term what FE_NO_static_coupling compares to TOL: the Omega_II volume
increment plus the interface increment.  The earlier version of this script
summed the two VOLUME increments instead (Omega_I + Omega_II).  Since these are
unnormalised dof-sum norms, the Omega_I term (~257k points) dwarfed the
interface term FE-NO has in that slot (~thousands), making the same TOL a
strictly harder target here -- so the sweep counts were never comparable.  They
are now.  ||Delta u_I|| is still recorded as a diagnostic in `conv`,
convergence_mirror.txt and the npz; it simply no longer drives the stop.

Only FE-NO's DEFAULT mode (--relax traction) is mirrored here.  FE-NO's
--relax u applies displacement under-relaxation with no flux blend, which this
script has no branch for.

SIGN
----
Both t_I and t_II are reported as sigma . (+e_r), the convention
operator_traction uses in the FE-NO script, so the two are directly
comparable.  Omega_II's own outward normal on Gamma is -e_r; that minus is
applied once, where the Neumann datum is set.

WELL-POSEDNESS
--------------
Omega_I: Dirichlet on r=1 and on Gamma, rollers elsewhere -- fine.
Omega_II: pure Neumann on Gamma, free at r=4, rollers on the four flat faces.
The three roller planes remove all six rigid-body modes (x=0 locks u_x, y=0
locks u_y, z=0 and z=h lock u_z, and those three together kill the rotations),
so it is well posed.  Same argument as in the FE-NO script.

Serial.
=============================================================================
"""

import os
import time
import argparse
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import dolfinx
from dolfinx import fem, default_scalar_type
from dolfinx.io import VTXWriter
from dolfinx.fem.petsc import (assemble_matrix, assemble_vector,
                               apply_lifting, set_bc, create_vector)
from scipy.spatial import cKDTree

from utils import createFolder


# ===========================================================================
# COMPAT -- everything DOLFINx 0.8 and 0.11 disagree about, in one place.
#           Identical to the block in FE_NO_static_coupling_r_2_dolfinx080.py.
# ===========================================================================
print(f"DOLFINx {dolfinx.__version__}")


def read_msh(path, comm=MPI.COMM_WORLD, gdim=3):
    """(mesh, cell_tags, facet_tags).

    0.8   dolfinx.io.gmshio.read_from_msh -> plain 3-tuple
    0.9   dolfinx.io.gmshio.read_from_msh -> MeshData
    0.10+ dolfinx.io.gmsh.read_from_msh   -> MeshData
    """
    try:
        from dolfinx.io import gmshio as _g
    except ImportError:                       # 0.10+ renamed the module
        from dolfinx.io import gmsh as _g
    out = _g.read_from_msh(path, comm, gdim=gdim)
    if hasattr(out, "mesh"):                  # MeshData
        return out.mesh, out.cell_tags, out.facet_tags
    return out                                # 0.8 tuple


def create_rhs(V, L_form):
    """Ghosted PETSc RHS vector.

    fem.petsc.create_vector takes the FORM up to 0.10 and the FUNCTION SPACE
    in 0.11.  Try the 0.8 spelling first, fall back to the 0.11 one.  The
    ghost layout matters: ghostUpdate(ADD, REVERSE) below depends on it.
    """
    try:
        return create_vector(L_form)          # <= 0.10
    except (TypeError, AttributeError):
        return create_vector(V)               # 0.11


def petsc_vec(f):
    """The PETSc Vec behind a Function: f.vector on 0.8, f.x.petsc_vec later."""
    pv = getattr(f.x, "petsc_vec", None)
    return pv if pv is not None else f.vector


def ipoints(V):
    """element.interpolation_points: a METHOD on 0.8, a PROPERTY on 0.9+."""
    ip = V.element.interpolation_points
    return ip() if callable(ip) else ip


# ===========================================================================
# 0.  configuration
# ===========================================================================
ap = argparse.ArgumentParser(
    description="FE-FE coupling mirrored onto the FE-NO iteration")
ap.add_argument('--theta', type=float, default=0.9,
                help='flux blend: t <- theta*sigma_II.n|FE + (1-theta)*'
                     'sigma_I.n|FE.  theta weights the FE (old) side, the same '
                     'convention FE_NO_static_coupling --relax traction uses')
ap.add_argument('--niter', type=int, default=400)
ap.add_argument('--tol', type=float, default=1e-3)
ap.add_argument('--load', type=str, default='analytic',
                choices=('analytic', 'gp'))
ap.add_argument('--gp_seed', type=int, default=0)
ap.add_argument('--out', type=str, default=None,
                help='output folder under results/.  Defaults to one tagged '
                     'with theta, so a sweep does not overwrite itself')
args = ap.parse_args()

THETA, NITER, TOL = args.theta, args.niter, args.tol
LOAD_MODE, GP_SEED = args.load, args.gp_seed

VALIDATE_AGAINST_MONOLITHIC = True

R1, R2, R3, H = 1.0, 2.0, 4.0, 4.0

VOIGT_IDX = [0, 4, 8, 1, 2, 5]                 # xx yy zz xy xz yz
COMPS = ["u_x", "u_y", "u_z",
         "S_xx", "S_yy", "S_zz", "S_xy", "S_xz", "S_yz"]

# --- physical tags of Tube_inner.msh / Tube_outer.msh ----------------------
BOTTOM, TOP, S_INNER, S_OUTER, SYM_A, SYM_B = 1, 2, 3, 4, 5, 6
#   Omega_I  (Tube_inner):  S_INNER = r=1 (load)       S_OUTER = r=2 (Gamma)
#   Omega_II (Tube_outer):  S_INNER = r=2 (Gamma)      S_OUTER = r=4 (free)

# --- physical tags of Tube_entire.msh (different scheme!) ------------------
E_INNER, E_INTERFACE, E_OUTER, E_BOTTOM, E_TOP, E_SYMX, E_SYMY = 1, 2, 3, 4, 5, 6, 7

E_MOD, NU = 0.210e-2, 0.3
MU = E_MOD / (2.0 * (1.0 + NU))
LMBDA = E_MOD * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))

originalDir = os.path.dirname(os.path.abspath(__file__))
createFolder(os.path.join(originalDir, "results"))
# theta goes in the folder name: the whole point of this script right now is a
# theta sweep, and a fixed name means each run silently eats the last one's
# figures.  --out overrides it.
out_dir = os.path.join(originalDir, "results",
                       args.out or f"FE_FE_coupling_results_mirror_theta{THETA:g}_data")
createFolder(out_dir)
print(f"output -> {os.path.basename(out_dir)}")
mesh_dir = os.path.join(originalDir, "Mesh_3D_cylinder")


def epsilon(u):
    return ufl.sym(ufl.grad(u))


def sigma(u):
    return LMBDA * ufl.tr(epsilon(u)) * ufl.Identity(3) + 2.0 * MU * epsilon(u)


# ===========================================================================
# 1.  the applied load on r = 1
# ===========================================================================
if LOAD_MODE == "analytic":
    def load_callable(x):
        return np.array([0.01 * x[0] ** 2,
                         0.01 * x[1] ** 2,
                         0.005 * (x[2] - H) * x[2]])
else:
    from gp_boundary import InnerBCSampler
    _sampler = InnerBCSampler(r1=R1, h=H)
    load_callable = _sampler.as_dolfinx_callable(_sampler.sample(GP_SEED))


# ===========================================================================
# 2.  helpers
# ===========================================================================
def detect_roller(V, facet_tags, tag, fdim, tol=1e-9):
    """Which displacement component does the flat face `tag` lock?

    Determined from the geometry, not the tag name: the two .geo files in this
    project disagree about what "sym_x" means, and a silently swapped roller
    produces a plausible-looking but wrong answer.
    """
    facets = facet_tags.find(tag)
    if facets.size == 0:
        raise RuntimeError(f"facet tag {tag} is empty")
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    xc = V.tabulate_dof_coordinates()[dofs]
    for comp in (0, 1, 2):
        if np.ptp(xc[:, comp]) < tol:
            return comp
    raise RuntimeError(f"facet tag {tag} is not a coordinate plane")


class Subdomain:
    """One linear-elastic subdomain: mesh, space, fixed matrix, reusable KSP."""

    def __init__(self, msh_file, name):
        self.name = name
        self.mesh, _, self.facet_tags = read_msh(msh_file, MPI.COMM_WORLD,
                                                 gdim=3)          # COMPAT
        self.tdim = self.mesh.topology.dim
        self.fdim = self.tdim - 1
        self.mesh.topology.create_connectivity(self.fdim, self.tdim)

        self.V = fem.functionspace(self.mesh, ("Lagrange", 2, (3,)))
        self.uh = fem.Function(self.V, name="displacement")
        self.dof_x = self.V.tabulate_dof_coordinates()

        # CG2 tensor space for sigma; same element => same block dofmap as V,
        # so a block index means the same point in both spaces.
        self.Vsig = fem.functionspace(self.mesh, ("Lagrange", 2, (3, 3)))
        assert np.allclose(self.Vsig.tabulate_dof_coordinates(), self.dof_x), \
            "V and Vsig block layouts differ"
        self.sig = fem.Function(self.Vsig)
        self.sig_expr = fem.Expression(sigma(self.uh),
                                       ipoints(self.Vsig))        # COMPAT

        print(f"[{name}] cells={self.mesh.topology.index_map(self.tdim).size_local}"
              f"  P2 vector dofs={self.V.dofmap.index_map.size_local * 3}")
        print(f"[{name}] CG2 space: nodes={self.V.dofmap.index_map.size_local}  "
              f"dofs={self.V.dofmap.index_map.size_local * self.V.dofmap.index_map_bs}  "
              f"elements={self.mesh.topology.index_map(self.tdim).size_local}")

    def rollers(self):
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

    def setup(self, bcs, iface_tag=None, traction=None):
        self.bcs = bcs
        u, v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        f = fem.Constant(self.mesh, default_scalar_type((0.0, 0.0, 0.0)))
        self.a_form = fem.form(ufl.inner(sigma(u), epsilon(v)) * ufl.dx)
        L = ufl.inner(f, v) * ufl.dx
        if traction is not None:
            ds = ufl.Measure("ds", domain=self.mesh,
                             subdomain_data=self.facet_tags)
            L += ufl.inner(traction, v) * ds(iface_tag)
        self.L_form = fem.form(L)
        self.A = assemble_matrix(self.a_form, bcs=bcs)
        self.A.assemble()
        # COMPAT: 0.8 create_vector(form); 0.11 create_vector(space).
        self.b = create_rhs(self.V, self.L_form)
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
        self.ksp.solve(self.b, petsc_vec(self.uh))                # COMPAT
        self.uh.x.scatter_forward()
        return self.ksp.getIterationNumber()

    def stress_at(self, blocks):
        """sigma (n, 3, 3) at the given blocked dof indices."""
        self.sig.interpolate(self.sig_expr)
        return self.sig.x.array.reshape(-1, 9)[blocks].reshape(-1, 3, 3)

    def disp_at(self, blocks):
        return self.uh.x.array.reshape(-1, 3)[blocks]

    def set_disp_at(self, fn, blocks, values):
        fn.x.array.reshape(-1, 3)[blocks] = values


# ===========================================================================
# 3.  the two subdomains and the exact interface permutation
# ===========================================================================
os.chdir(mesh_dir)
t0 = time.time()

OI = Subdomain("Tube_inner.msh", "Omega_I ")
OII = Subdomain("Tube_outer.msh", "Omega_II")

dI, xI = OI.iface_dofs(S_OUTER)      # r = 2 seen from Omega_I
dII, xII = OII.iface_dofs(S_INNER)   # r = 2 seen from Omega_II
print(f"interface dofs: Omega_I {dI.size}   Omega_II {dII.size}")
assert dI.size == dII.size, "interface dof counts differ -> meshes not conformal"

# All three meshes put the same 50x80 transfinite grid on r=2, so the transfer
# is an exact PERMUTATION -- no interpolation error enters the coupling.
dist, perm = cKDTree(xII).query(xI)
h_iface = R2 * (np.pi / 2) / 49
assert dist.max() < 1e-6 * h_iface, (
    f"interface dofs do not match (max gap {dist.max():.2e}, spacing "
    f"{h_iface:.3e}); the meshes are not conformal on r=2")
assert np.unique(perm).size == perm.size, "interface match is not a bijection"
print(f"interface match: max gap {dist.max():.2e} vs spacing {h_iface:.2e}"
      f"  -> exact permutation")
#   xI[i]  <->  xII[perm[i]]

# +e_r on Gamma, in the Omega_I ordering.  Both tractions below are reported in
# this convention, matching operator_traction in the FE-NO script.
rr = np.hypot(xI[:, 0], xI[:, 1])
e_r = np.zeros((dI.size, 3))
e_r[:, 0] = xI[:, 0] / rr
e_r[:, 1] = xI[:, 1] / rr


# ===========================================================================
# 4.  Omega_I : Dirichlet everywhere -- this is the subdomain the operator
#     replaces, so it must RECEIVE displacement and RETURN traction
# ===========================================================================
print("\n[Omega_I ] boundary conditions  (Dirichlet on Gamma: the NO's role)")
u_load = fem.Function(OI.V)
u_load.interpolate(load_callable)
bc_load = fem.dirichletbc(u_load, fem.locate_dofs_topological(
    OI.V, OI.fdim, OI.facet_tags.find(S_INNER)))

u_gamma_fn = fem.Function(OI.V)          # live Dirichlet datum on Gamma
bc_gamma_I = fem.dirichletbc(u_gamma_fn, fem.locate_dofs_topological(
    OI.V, OI.fdim, OI.facet_tags.find(S_OUTER)))

# Rollers last so they win on the four edges Gamma shares with them.  They
# agree there anyway: u_Gamma comes from Omega_II, which satisfies the same
# rollers.
OI.setup([bc_load, bc_gamma_I] + OI.rollers())

# ===========================================================================
# 5.  Omega_II : pure Neumann on Gamma -- identical to the FE-NO script
# ===========================================================================
print("\n[Omega_II] boundary conditions  (Neumann on Gamma)")
traction = fem.Function(OII.V)           # live Neumann datum
OII.setup(OII.rollers(), iface_tag=S_INNER, traction=traction)


# ===========================================================================
# 6.  the iteration -- the same one FE_NO_static_coupling runs
# ===========================================================================
print(f"\nSchwarz FE-FE (mirrored): TOL={TOL:.0e}  max {NITER} iterations")
print(f"  flux blend:  t <- {THETA:.3g} * sigma_II.n|FE  +  "
      f"{1-THETA:.3g} * sigma_I.n|FE")
print(f"  theta weights the FE (old) side, so larger theta = more damping")
print(f"  u_Gamma^0 = 0, matching the FE-NO script\n")

u_gamma = np.zeros((dI.size, 3))         # in the Omega_I ordering
t_II = None
t_I = None                               # so the final savez cannot NameError
full_prev = None
hist, iterates = [], []
uI_iterates, uII_iterates = [], []      # full FE fields each sweep, kept so
                                        # the per-sweep dataset can be rebuilt
conv = []                               # per-sweep [du_I, du_II] for
                                        # static_convergence_plot.py

for it in range(NITER):
    iterates.append(u_gamma.copy())

    # ---- (a) Omega_I plays the operator: Dirichlet in, traction out ------
    OI.set_disp_at(u_gamma_fn, dI, u_gamma)
    nI = OI.solve()
    t_I = np.einsum("nij,nj->ni", OI.stress_at(dI), e_r)      # the EXACT DtN

    # ---- (b) blend the two sides' estimates of the same flux -------------
    # t_II was read back from Omega_II's SOLUTION at the end of the previous
    # sweep.  On the first sweep there is none, so Omega_I's goes in unblended
    # -- exactly what the FE-NO script does with N(0).
    t = t_I if t_II is None else THETA * t_II + (1.0 - THETA) * t_I

    # ---- (c) Omega_II, whose outward normal on Gamma is -e_r -------------
    OII.set_disp_at(traction, dII[perm], -t)
    nII = OII.solve()
    u_new = OII.disp_at(dII[perm])                            # Omega_I ordering
    t_II = np.einsum("nij,nj->ni", OII.stress_at(dII[perm]), e_r)

    du = np.linalg.norm(u_new - u_gamma)
    u_gamma = u_new

    # ---- (d) convergence : IDENTICAL to the FE-NO script ------------------
    #
    # ALIGNED WITH FE-NO.  The FE-NO script stops on
    #
    #       err = || u_II^k - u_II^{k-1} ||  +  || u_Gamma^k - u_Gamma^{k-1} ||
    #
    # i.e. the Omega_II VOLUME increment plus the INTERFACE increment.  It has
    # no choice: its Omega_I is the operator, which has no mesh and therefore
    # no volume field to difference.
    #
    # The original version of this script used dU_I + dU_II -- both VOLUME
    # increments.  These norms are unnormalised sums over dofs, so their
    # magnitude scales with the dof count: Omega_I carries ~257k points where
    # FE-NO has only the few-thousand-point interface.  Against the same
    # TOL that made this script's test systematically STRICTER, and
    # "FE-FE took N sweeps, FE-NO took M" stopped being a like-for-like
    # statement -- which is the one thing this mirrored script exists to fix.
    #
    # So the stopping test below is now FE-NO's, term for term.  Both scripts
    # difference the SAME Omega_II field (both read Tube_outer.msh, same mesh,
    # same dof order) and the SAME interface increment `du`, defined here
    # exactly as there: || u_new - u_gamma || taken BEFORE u_gamma is updated.
    #
    # dU_I is still computed and still written to `conv` /
    # convergence_mirror.txt / the npz -- it is useful as a diagnostic, it just
    # no longer drives the stop.
    uI_full = OI.uh.x.array.reshape(-1, 3).copy()
    uII_full = OII.uh.x.array.reshape(-1, 3).copy()
    uI_iterates.append(uI_full)         # paired with iterates[it] above
    uII_iterates.append(uII_full)
    if full_prev is None:
        dU_I = dU_II = np.nan
    else:
        dU_I  = float(np.linalg.norm(uI_full  - full_prev[0]))
        dU_II = float(np.linalg.norm(uII_full - full_prev[1]))
    conv.append([dU_I, dU_II])
    err = np.nan if full_prev is None else (dU_II + du)      # <- FE-NO's test
    full_prev = (uI_full, uII_full)

    # physical residual: do the two sides carry the same traction on Gamma?
    nrm = max(np.linalg.norm(t_I), np.linalg.norm(t_II), 1e-30)
    res = np.linalg.norm(t_I - t_II) / nrm
    hist.append([err, res, du])

    es = "   --    " if np.isnan(err) else f"{err:.3e}"
    print(f"  it {it:3d}  ksp {nI:3d}/{nII:3d}  error={es}  "
          f"traction mismatch={res:.3e}")

    if not np.isnan(err) and err < TOL:
        print(f"converged after {it + 1} iterations  ({err:.3e} < {TOL:.0e})")
        break
else:
    print(f"WARNING: not converged in {NITER} iterations")

hist = np.array(hist)
conv = np.array(conv)                    # (n_sweeps, 2) = [du_I, du_II]
print(f"wall time {time.time() - t0:.1f} s")

# ---- the single convergence curve for static_convergence_plot.py ------------
# It loads current/error_list_FE_FE.npy as ONE column, "L2 error vs inner
# iteration".  That column is hist[:,0] = ||Delta u_II|| + ||Delta u_Gamma||,
# the quantity the stop test compares to TOL -- which is why the curve
# terminates at ~TOL.
#
# This is now the SAME quantity FE-NO writes to error_list_FE_NN.npy, computed
# on the same Omega_II mesh, so the two curves may be plotted on one pair of
# axes and their lengths compared directly.  Before the alignment above they
# were different measures and the comparison was meaningless.
#
# The leading sweep has no predecessor, so its NaN is dropped and the curve
# starts at inner iteration 1.  The per-subdomain du_I / du_II are still kept
# separately in `conv` / coupling_history_mirror.npz / convergence_mirror.txt.
error_list_FE_FE = hist[np.isfinite(hist[:, 0]), 0]
np.save(os.path.join(out_dir, "error_list_FE_FE.npy"), error_list_FE_FE)
print(f"wrote error_list_FE_FE.npy ({error_list_FE_FE.size} inner iterations) "
      f"-> feed to static_convergence_plot.py as current/error_list_FE_FE.npy")

# --- the contraction factor of THIS iteration ------------------------------
# This is the rho that belongs in FE_NO_static_coupling's error-amplification
# estimate.  The 0.40 currently hard-coded there came from the ORIGINAL FE-FE
# scheme, which is a different iteration; there is no reason for the two to
# agree.
RHO = np.nan
if hist.shape[0] > 3:
    e = hist[1:, 0]                       # drop the NaN sentinel
    e = e[np.isfinite(e) & (e > 0)]
    if e.size > 3:
        tail = e[-min(5, e.size - 1):] / e[-min(5, e.size - 1) - 1:-1]
        RHO = float(np.mean(tail))
        print(f"\nmeasured contraction factor of this iteration: rho = {RHO:.4f}")
        print(f"  (theta = {THETA:g}.  This is the rho FE-NO's "
              f"error/(1-rho) estimate needs --")
        print(f"   the 0.40 hard-coded there was measured on the ORIGINAL "
              f"FE-FE scheme.)")
        print(f"  an operator interface error e would show up in the coupled "
              f"solution as e/(1-rho) = {1/(1-RHO):.2f} e")


# ===========================================================================
# 7.  validation against the monolithic solution
# ===========================================================================
report = {"theta": THETA, "rho": RHO, "n_iter": hist.shape[0]}
if VALIDATE_AGAINST_MONOLITHIC:
    print("\nsolving the monolithic reference on Tube_entire.msh ...")
    os.chdir(mesh_dir)
    dom, _, ftags = read_msh("Tube_entire.msh", MPI.COMM_WORLD, gdim=3)  # COMPAT
    fdim = dom.topology.dim - 1
    dom.topology.create_connectivity(fdim, dom.topology.dim)
    Vm = fem.functionspace(dom, ("Lagrange", 2, (3,)))

    um_load = fem.Function(Vm)
    um_load.interpolate(load_callable)
    bcs_m = [fem.dirichletbc(um_load, fem.locate_dofs_topological(
        Vm, fdim, ftags.find(E_INNER)))]
    for tag in (E_BOTTOM, E_TOP, E_SYMX, E_SYMY):
        comp = detect_roller(Vm, ftags, tag, fdim)
        bcs_m.append(fem.dirichletbc(
            default_scalar_type(0.0),
            fem.locate_dofs_topological(Vm.sub(comp), fdim, ftags.find(tag)),
            Vm.sub(comp)))
        print(f"  [entire] tag {tag}: locking u_{'xyz'[comp]}")

    u_, v_ = ufl.TrialFunction(Vm), ufl.TestFunction(Vm)
    f_ = fem.Constant(dom, default_scalar_type((0.0, 0.0, 0.0)))
    am = fem.form(ufl.inner(sigma(u_), epsilon(v_)) * ufl.dx)
    Lm = fem.form(ufl.inner(f_, v_) * ufl.dx)
    Am = assemble_matrix(am, bcs=bcs_m); Am.assemble()
    bm = create_rhs(Vm, Lm)                                         # COMPAT
    assemble_vector(bm, Lm)
    apply_lifting(bm, [am], bcs=[bcs_m])
    bm.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(bm, bcs_m)
    uhm = fem.Function(Vm, name="displacement")
    kspm = PETSc.KSP().create(dom.comm); kspm.setOperators(Am)
    kspm.setType(PETSc.KSP.Type.CG); kspm.getPC().setType(PETSc.PC.Type.GAMG)
    kspm.setTolerances(rtol=1e-10)
    kspm.solve(bm, petsc_vec(uhm))                                  # COMPAT
    uhm.x.scatter_forward()

    # strain/stress of the monolithic solution, CG2, same recovery as above
    Vtm = fem.functionspace(dom, ("Lagrange", 2, (3, 3)))
    ipm = ipoints(Vtm)                                              # COMPAT
    sigm = fem.Function(Vtm)
    sigm.interpolate(fem.Expression(sigma(uhm), ipm))

    xm = Vm.tabulate_dof_coordinates()
    dm = fem.locate_dofs_topological(Vm, fdim, ftags.find(E_INTERFACE))
    dist_m, perm_m = cKDTree(xm[dm]).query(xI)
    assert dist_m.max() < 1e-6 * h_iface, "entire mesh not conformal on r=2"

    u_mono = uhm.x.array.reshape(-1, 3)[dm[perm_m]]
    s_mono = sigm.x.array.reshape(-1, 9)[dm[perm_m]].reshape(-1, 3, 3)
    t_mono = np.einsum("nij,nj->ni", s_mono, e_r)

    e_u = np.linalg.norm(u_gamma - u_mono) / np.linalg.norm(u_mono)
    e_inf = np.abs(u_gamma - u_mono).max() / np.abs(u_mono).max()
    e_t = np.linalg.norm(t_I - t_mono) / np.linalg.norm(t_mono)
    report.update(iface_u_rel_L2=e_u, iface_u_rel_Linf=e_inf,
                  iface_t_rel_L2=e_t)

    print(f"\ninterface, coupled vs monolithic:")
    print(f"   displacement  relative L2   {e_u:.3e}")
    print(f"   displacement  relative Linf {e_inf:.3e}")
    print(f"   traction      relative L2   {e_t:.3e}")
    print("   These are the FLOOR for FE-NO on the same iteration: the only")
    print("   difference there is that Omega_I is the operator instead of an")
    print("   exact solve, so whatever FE-NO reports above these numbers is")
    print("   the operator's contribution and nothing else.")

    # =======================================================================
    # 7b.  the same per-component report FE-NO prints, so the two tables can
    #      sit next to each other
    # =======================================================================
    epsm = fem.Function(Vtm)
    epsm.interpolate(fem.Expression(epsilon(uhm), ipm))
    Um = uhm.x.array.reshape(-1, 3)
    Em = epsm.x.array.reshape(-1, 9)[:, VOIGT_IDX]

    def _rel(a, b):
        return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))

    rm = np.hypot(xm[:, 0], xm[:, 1])
    inI = np.flatnonzero((rm > R1 - 1e-9) & (rm < R2 + 1e-9))
    inII = np.flatnonzero(rm > R2 - 1e-9)

    def interp_to(src_pts, src_vals, dst_pts, label=""):
        """Linear interpolation between non-matching meshes, nearest as a
        fallback for the few destination points outside the source hull.

        Tube_entire is NOT the union of Tube_inner and Tube_outer -- only the
        r=1 and r=2 surfaces are conformal, the volumes are meshed
        independently -- so both halves have to come across this way.

        The FE-NO script interpolates only its Omega_II half; its Omega_I half
        is the operator evaluated directly at the monolithic points, because
        the trunk takes arbitrary coordinates.  Here both halves are
        interpolated.  The difference is the interpolation's own error, linear
        on a mesh of spacing ~0.05 acting on a field whose second derivative is
        O(1e-2), so ~1e-4 relative -- two orders below the errors being plotted.
        """
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

    # --- both subdomains' CG2 strain, recovered the same way as the reference
    def _strain_of(sub):
        Vt = fem.functionspace(sub.mesh, ("Lagrange", 2, (3, 3)))
        fn = fem.Function(Vt)
        fn.interpolate(fem.Expression(epsilon(sub.uh), ipoints(Vt)))  # COMPAT
        return fn.x.array.reshape(-1, 9)[:, VOIGT_IDX]

    E_I, E_II_f = _strain_of(OI), _strain_of(OII)

    U_I_on_m = interp_to(OI.dof_x, OI.uh.x.array.reshape(-1, 3), xm[inI],
                         "Omega_I displacement")
    E_I_on_m = interp_to(OI.dof_x, E_I, xm[inI], "Omega_I strain")
    U_II_on_m = interp_to(OII.dof_x, OII.uh.x.array.reshape(-1, 3), xm[inII],
                          "Omega_II displacement")
    E_II_on_m = interp_to(OII.dof_x, E_II_f, xm[inII], "Omega_II strain")

    rows = [("interface  u", _rel(u_gamma, u_mono)),
            ("interface  strain", _rel(
                interp_to(OI.dof_x, E_I, xI, "Omega_I interface strain"),
                Em[dm[perm_m]])),
            (f"Omega_I    u   ({inI.size} pts)", _rel(U_I_on_m, Um[inI])),
            ("Omega_I    strain", _rel(E_I_on_m, Em[inI])),
            (f"Omega_II   u   ({inII.size} pts, interp)",
             _rel(U_II_on_m, Um[inII]))]

    print()
    print(f"  {'quantity':32s} {'relative L2':>12s}")
    for name, v in rows:
        print(f"  {name:32s} {v:12.3e}")
    report["rows"] = rows

    # =======================================================================
    # 7c.  the fields on the MONOLITHIC mesh, then the same 27 PNGs FE-NO
    #      writes, with the same names, so the two runs drop straight into a
    #      side-by-side figure.
    # =======================================================================
    print("\nassembling the 3-D fields on the monolithic mesh ...")
    U_cpl, E_cpl = np.zeros_like(Um), np.zeros_like(Em)
    # Omega_II is written second so it wins on r = 2, where the two grids do
    # coincide and the value is u_Gamma either way.
    U_cpl[inI], E_cpl[inI] = U_I_on_m, E_I_on_m
    U_cpl[inII], E_cpl[inII] = U_II_on_m, E_II_on_m

    u_cpl_fn = fem.Function(Vm, name="u_coupled")
    u_ref_fn = fem.Function(Vm, name="u_monolithic")
    u_err_fn = fem.Function(Vm, name="u_abs_error")
    u_cpl_fn.x.array[:] = U_cpl.reshape(-1)
    u_ref_fn.x.array[:] = Um.reshape(-1)
    u_err_fn.x.array[:] = np.abs(U_cpl - Um).reshape(-1)

    e_cpl_fn = fem.Function(Vtm, name="strain_coupled")
    e_ref_fn = fem.Function(Vtm, name="strain_monolithic")
    e_err_fn = fem.Function(Vtm, name="strain_abs_error")
    _back = [0, 3, 4, 3, 1, 5, 4, 5, 2]            # Voigt -> 3x3, row major
    e_cpl_fn.x.array[:] = E_cpl[:, _back].reshape(-1)
    e_ref_fn.x.array[:] = Em[:, _back].reshape(-1)
    e_err_fn.x.array[:] = np.abs(E_cpl - Em)[:, _back].reshape(-1)

    _cwd = os.getcwd()
    os.chdir(out_dir)
    for fname, fns in (("u_coupled.bp", [u_cpl_fn]),
                       ("u_monolithic.bp", [u_ref_fn]),
                       ("u_error.bp", [u_err_fn]),
                       ("strain_coupled.bp", [e_cpl_fn]),
                       ("strain_monolithic.bp", [e_ref_fn]),
                       ("strain_error.bp", [e_err_fn])):
        with VTXWriter(dom.comm, fname, fns, engine="BP4") as w:
            w.write(0.0)
    print("  wrote 6 .bp files -- open them in ParaView")

    try:
        import pyvista as pv
        from dolfinx import plot
        Vs1 = fem.functionspace(dom, ("Lagrange", 1))
        Vv1 = fem.functionspace(dom, ("Lagrange", 1, (3,)))

        def to_p1_vec(arr):
            f3 = fem.Function(Vm); f3.x.array[:] = arr.reshape(-1)
            g = fem.Function(Vv1); g.interpolate(f3)
            return g.x.array.reshape(-1, 3)

        def to_p1_scalar(vals):
            f1 = fem.Function(Vm)
            a = np.zeros_like(Um); a[:, 0] = vals
            f1.x.array[:] = a.reshape(-1)
            g = fem.Function(Vv1); g.interpolate(f1)
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
            # a SHARED colour range for coupled and reference: without it the
            # two pictures autoscale differently and look alike whatever the
            # error is
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

        print("  wrote 27 PNGs -- same names and same camera as the FE-NO run,"
              " so the two folders line up file for file")
    except Exception as _e:
        print(f"  PyVista skipped: {_e}")
    os.chdir(_cwd)

    np.savez_compressed(os.path.join(out_dir, "field_comparison_mirror.npz"),
                        xyz=xm.astype(np.float32),
                        U_coupled=U_cpl.astype(np.float32),
                        U_monolithic=Um.astype(np.float32),
                        E_coupled=E_cpl.astype(np.float32),
                        E_monolithic=Em.astype(np.float32),
                        inI=inI, inII=inII,
                        names=np.array([n for n, _ in rows], dtype=object),
                        values=np.array([v for _, v in rows]))

    # =======================================================================
    # 10.  PER-SWEEP DATASET  (aligned with the FE-NO dump; static_ux_figure_3d)
    #
    #      ALIGNMENT -- the key point.  This uses the SAME monolithic mesh
    #      (Tube_entire.msh), the SAME r-masks (inI, inII) and the SAME dof
    #      order as the FE-NO script.  DOLFINx numbers the dofs of a given mesh
    #      + element deterministically, so run serially both scripts produce
    #      byte-identical xm, hence identical xm[inII] (-> X/Y/Z) and xm[inI]
    #      (-> X1/Y1/Z1).  The plot can therefore subtract FE-FE from FE-NO
    #      point-for-point with NO interpolation.  (static_ux_figure_3d.py also
    #      asserts this at load time.)
    #
    #      FIGURE domain mapping (2-D example convention, Omega_I = outer):
    #          FIGURE Omega_I  (outer r in [2,4]) = this run's Omega_II -> u  i=j
    #          FIGURE Omega_II (inner r in [1,2]) = this run's Omega_I  -> u2 i=j
    #      Same points as FE-NO; here BOTH halves are exact FE solves, whereas in
    #      FE-NO the inner half (Omega_II) is the neural operator.
    # =======================================================================
    from scipy.spatial import Delaunay
    from scipy.interpolate import LinearNDInterpolator

    COMP, COMP_NAME = 0, "ux"                   # 0=u_x  1=u_y  2=u_z

    # Only these sweeps are written, not every one.  Any index past the actual
    # number of sweeps is dropped, and the TRUE last sweep is always added so the
    # error / final column is the converged field regardless of where it landed.
    SAVE_INDICES = [0, 8, 12, 16, 25]
    nsw = len(iterates)
    save_idx = sorted(set([j for j in SAVE_INDICES if j < nsw] + [nsw - 1]))
    print(f"\n[dataset] {nsw} sweeps ran  ->  last sweep index = {nsw - 1}")
    print(f"[dataset] saving sweeps {save_idx}  "
          f"(requested {SAVE_INDICES}; dropped any >= {nsw})")
    if 25 in SAVE_INDICES and 25 != nsw - 1:
        print(f"[dataset] note: you asked for j=25 as the last step, but the run "
              f"actually ended at j={nsw - 1}.")

    ds_dir  = os.path.join(out_dir, f"dataset_{COMP_NAME}")
    fe_fe_d = os.path.join(ds_dir, "FE-FE")
    gt_d    = os.path.join(ds_dir, "groud_truth")
    for d in (ds_dir, fe_fe_d, gt_d):
        createFolder(d)

    x_fig_I  = xm[inII]                         # FIGURE Omega_I  (outer FE)
    x_fig_II = xm[inI]                          # FIGURE Omega_II (inner FE)
    for arr, stem in ((x_fig_I,  ("X",  "Y",  "Z")),
                      (x_fig_II, ("X1", "Y1", "Z1"))):
        for k, nm in enumerate(stem):
            np.savetxt(os.path.join(ds_dir, nm + ".txt"), arr[:, k])
    np.savetxt(os.path.join(gt_d, "u.txt"), Um)            # u v w  (3 columns)
    np.savetxt(os.path.join(gt_d, "eps.txt"), Em)          # strain (6 Voigt cols)
    for k, nm in enumerate(("X", "Y", "Z")):
        np.savetxt(os.path.join(gt_d, nm + ".txt"), xm[:, k])

    # triangulate each source volume ONCE; the field changes every sweep but the
    # mesh does not, so only the interpolator weights are rebuilt per sweep.
    tri_OI, tri_OII = Delaunay(OI.dof_x), Delaunay(OII.dof_x)

    def _interp(tri, src_pts, vals, dst):
        out = LinearNDInterpolator(tri, vals)(dst)
        bad = ~np.isfinite(out).all(axis=1)
        if bad.any():
            out[bad] = vals[cKDTree(src_pts).query(dst[bad])[1]]
        return out

    # --- P1 DISPLAY MESH for SOLID rendering in the figure ------------------
    # Same linear grid as the PyVista section; saved once, the coupled field of
    # every saved sweep is interpolated to its nodes so the figure draws solid
    # surfaces (thresholded by radius into Omega_I / Omega_II) not a scatter.
    # Best-effort: on failure the txt point dataset is still written.
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

    U_all_I, U_all_II, E_all_I, E_all_II = [], [], [], []
    for j in save_idx:
        # FIGURE Omega_I (outer) = Omega_II FE field this sweep, onto xm[inII]
        OII.uh.x.array[:] = uII_iterates[j].reshape(-1)
        UE_II = np.column_stack([uII_iterates[j], _strain_of(OII)])   # (n, 3+6)
        out_I = _interp(tri_OII, OII.dof_x, UE_II, x_fig_I)
        up_figI, ep_figI = out_I[:, :3], out_I[:, 3:]
        # FIGURE Omega_II (inner) = Omega_I FE field this sweep, onto xm[inI]
        OI.uh.x.array[:] = uI_iterates[j].reshape(-1)
        UE_I = np.column_stack([uI_iterates[j], _strain_of(OI)])      # (n, 3+6)
        out_II = _interp(tri_OI, OI.dof_x, UE_I, x_fig_II)
        up_figII, ep_figII = out_II[:, :3], out_II[:, 3:]

        np.savetxt(os.path.join(fe_fe_d, f"u i = {j} .txt"),  up_figI)    # u v w
        np.savetxt(os.path.join(fe_fe_d, f"u2 i = {j} .txt"), up_figII)   # u v w
        np.savetxt(os.path.join(fe_fe_d, f"eps i = {j} .txt"),  ep_figI)
        np.savetxt(os.path.join(fe_fe_d, f"eps2 i = {j} .txt"), ep_figII)
        U_all_I.append(up_figI);  U_all_II.append(up_figII)
        E_all_I.append(ep_figI);  E_all_II.append(ep_figII)

        # coupled field on the whole monolithic mesh -> P1 nodes, for solid render
        if _P1_OK:
            _U = np.zeros_like(Um)
            _U[inII] = up_figI        # figure Omega_I  (outer, r in [2,4])
            _U[inI]  = up_figII       # figure Omega_II (inner, r in [1,2])
            np.save(os.path.join(fe_fe_d, f"u_p1 i = {j} .npy"),
                    _to_p1(_U).astype(np.float32))
            _E = np.zeros_like(Em)                        # coupled strain, 6 Voigt
            _E[inII] = ep_figI        # figure Omega_I  (outer FE strain)
            _E[inI]  = ep_figII       # figure Omega_II (inner FE strain)
            np.save(os.path.join(fe_fe_d, f"eps_p1 i = {j} .npy"),
                    _to_p1_6(_E).astype(np.float32))

    np.savez_compressed(
        os.path.join(ds_dir, "all_iters_ux.npz"),
        xyz_I=x_fig_I.astype(np.float32),  xyz_II=x_fig_II.astype(np.float32),
        xyz_gt=xm.astype(np.float32),      u_gt=Um.astype(np.float32),
        u_I=np.asarray(U_all_I, np.float32),   u_II=np.asarray(U_all_II, np.float32),
        eps_I=np.asarray(E_all_I, np.float32), eps_II=np.asarray(E_all_II, np.float32),
        sweeps=np.asarray(save_idx),           # which sweep index each slice is
        comp=COMP, comp_name=COMP_NAME, method="FE_FE",
        n_sweeps=nsw, theta=THETA)
    # restore the converged fields the section-8 .bp writes expect
    OI.uh.x.array[:]  = uI_iterates[-1].reshape(-1)
    OII.uh.x.array[:] = uII_iterates[-1].reshape(-1)
    print(f"[dataset] wrote sweeps {save_idx} x 4 txt files -> {fe_fe_d}")
    print(f"[dataset] coords + ground truth -> {ds_dir}")

# ===========================================================================
# 8.  save
# ===========================================================================
os.chdir(out_dir)

# The figure goes first.  Everything after it can fail -- and did, on the first
# run -- and none of it can change what the iteration already produced.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    # same label as the FE-NO figure -- the two curves are now the same measure
    ax[0].semilogy(hist[:, 0], "o-",
                   label=r"$\|\Delta u_{II}\|+\|\Delta u_\Gamma\|$")
    ax[0].semilogy(hist[:, 1], "s-", ms=3, alpha=.7,
                   label="traction mismatch")
    ax[0].axhline(TOL, color="k", ls="--", lw=0.8, label=f"tol {TOL:.0e}")
    ax[0].set_xlabel("Schwarz iteration"); ax[0].set_ylabel("increment")
    ax[0].set_title(rf"FE-FE mirrored, $\theta$={THETA}")
    ax[0].grid(True, which="both", alpha=0.3); ax[0].legend(fontsize=8)

    ax[1].plot([np.linalg.norm(u) for u in iterates], "o-")
    ax[1].set_xlabel("Schwarz iteration"); ax[1].set_ylabel(r"$\|u_\Gamma\|$")
    ax[1].set_title(f"contraction rho = {RHO:.3f}" if np.isfinite(RHO)
                    else "interface norm")
    ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("convergence_mirror.png", dpi=170)
    print("wrote convergence_mirror.png")
except Exception as e:
    print("plot skipped:", e)

# `report` already carries theta / rho / n_iter, so it is the single source for
# them.  Passing them a second time as explicit keywords is what raised
# "got multiple values for keyword argument 'theta'".
_out = dict(history=hist,              # (n, 3) [error, traction mismatch, du]
            convergence=conv,          # (n, 2) [du_I, du_II]  <- for the conv plot
            dU_I=conv[:, 0],           # ||u_I^k  - u_I^{k-1}||   per sweep
            dU_II=conv[:, 1],          # ||u_II^k - u_II^{k-1}||  per sweep
            load_mode=np.array(LOAD_MODE),
            xyz_interface=xI.astype(np.float32),
            u_interface=u_gamma.astype(np.float32),
            t_interface=t_I.astype(np.float32),
            iterates=np.asarray(iterates, np.float32))
_out.update({k: np.float64(v) for k, v in report.items()
             if isinstance(v, (int, float, np.floating))})
np.savez_compressed("coupling_history_mirror.npz", **_out)
print("wrote coupling_history_mirror.npz:", ", ".join(sorted(_out)))

# plain-text convergence table too, in case static_convergence_plot.py loadtxt's
# it: columns = iteration, du_I, du_II  (row 0 is NaN -- no previous sweep yet)
_conv_tbl = np.column_stack([np.arange(conv.shape[0]), conv])
np.savetxt("convergence_mirror.txt", _conv_tbl,
           header="iter    dU_I            dU_II", fmt=["%5d", "%.8e", "%.8e"])
print("wrote convergence_mirror.txt  (iter, dU_I, dU_II)")

with VTXWriter(OI.mesh.comm, "u_omega1_mirror.bp", [OI.uh], engine="BP4") as w:
    w.write(0.0)
with VTXWriter(OII.mesh.comm, "u_omega2_mirror.bp", [OII.uh], engine="BP4") as w:
    w.write(0.0)

print("results in", out_dir)