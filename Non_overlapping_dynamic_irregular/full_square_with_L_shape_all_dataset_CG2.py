import jax
import jax.numpy as jnp
from jax import grad, vmap, random, jit, config
import dolfinx
from dolfinx import fem, default_scalar_type
from dolfinx.fem import functionspace 
from dolfinx.io import XDMFFile, gmshio
from dolfinx.fem import Constant, Function 
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from dolfinx.mesh import create_box, create_unit_square
import ufl 
from ufl import dx
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc
from petsc4py import PETSc
import os 
from tqdm import trange 
import gmsh 
import math
from dolfinx import plot
from scipy.interpolate import Rbf, interp1d, griddata
import logging 
from matplotlib.ticker import ScalarFormatter
import time
from dynamic_utils import plot_mesh, plot_disp, plot_s, plot_relative_error, createFolder
import meshio


start_time = time.time()
#region Save path       
originalDir = "/nfshdd/21040463r/FEM_DeepONet_non_overlapping_coupling/non_overlapping_figures/elasto_dynamic_irregular" #os.getcwd()
print('curent working directory:', originalDir)
os.chdir(os.path.join(originalDir))

foldername = 'full_square_with_L_shape_all_dataset_N100_89_169_CG2'  
createFolder(foldername)
os.chdir(os.path.join(originalDir, './'+ foldername + '/')) 

origin_real = os.path.join(originalDir, './'+ foldername + '/')
os.chdir(origin_real)

msh = meshio.read(
    "L_shape_all.msh",
    file_format="gmsh", 
)
# check the cell types available in the mesh
print("Available cell types:")
print(msh.cells_dict.keys())  # check available cell types
# or print all cell blocks
print("All cell blocks:")
for cell_block in msh.cells:
    print(f"Cell type: {cell_block.type}, Count: {cell_block.data.shape[0]}")
# Extract points and cell information
points = msh.points
cells = msh.cells_dict["triangle"]  
plot_mesh(cells, points, 'Gmsh Mesh')


# region FEM 
mesh, cell_markers, facet_markers  = gmshio.read_from_msh("L_shape_all.msh", MPI.COMM_WORLD) 
V = functionspace(mesh, ("CG", 2, (mesh.geometry.dim, ))) ###without  (mesh1.geometry.dim, ), it is a scalar space not a vector space 
#mesh = create_unit_square(MPI.COMM_WORLD, 100, 100) 
tdim = mesh.topology.dim  #mesh.geometry.dim = 3 mesh.topology.dim = 2
fdim = tdim - 1 # facet dimension
# extract the coordinates of the mesh (every vertices of the mesh)
u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
X, Y = u_geometry[:,0], u_geometry[:,1]
np.savetxt('X.txt', X)
np.savetxt('Y.txt', Y)
#plot_disp(X, Y, X, 'full_domain', 'Full Domain')

tag = 103
# find the facets with tag = 103
facets_103 = facet_markers.indices[facet_markers.values == tag]
# find the dofs on these facets
dofs_103 = fem.locate_dofs_topological(V, fdim, facets_103)
# get the coordinates of all the dofs in V
dof_coords = V.tabulate_dof_coordinates()
# only keep the dof coords on the boundary with tag 103
boundary_dof_coords = dof_coords[dofs_103]
plot_disp(boundary_dof_coords[:,0], boundary_dof_coords[:,1], boundary_dof_coords[:,0], 'BC_inner', 'BC_inner')
np.savetxt('X_BC_in.txt', boundary_dof_coords[:,0])
np.savetxt('Y_BC_in.txt', boundary_dof_coords[:,1])

tag = 101
cells_101 = cell_markers.indices[cell_markers.values == tag]
tdim = mesh.topology.dim
mesh.topology.create_connectivity(tdim, tdim) # ensure connectivity is created (important)
dofs_101 = fem.locate_dofs_topological(V, tdim, cells_101)
dof_coords = V.tabulate_dof_coordinates()
domain_dof_coords = dof_coords[dofs_101]

#plot_disp(domain_dof_coords[:,0], domain_dof_coords[:,1], domain_dof_coords[:,0], 'domain_dof_101', 'Domain DOF 101')

'''
========================================================================
This is just for mesh, instead of the function space V
========================================================================

facet_to_vertex = dolfinx.mesh.entities_to_geometry(mesh, fdim, facets_103)
vertex_indices = np.unique(facet_to_vertex.flatten())
points_103 = mesh.geometry.x[vertex_indices]
#plot_disp(points_103[:,0], points_103[:,1], points_103[:,0], 'boundary_103', 'Boundary 103')'''

# find the inner square points and index 
index_inner_domain = dofs_101

#plot_disp(X[index_inner_domain], Y[index_inner_domain], X[index_inner_domain], 'inner_square', 'Inner Square')
X1 = X[index_inner_domain]
Y1 = Y[index_inner_domain]


def epsilon(u):
    return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

def sigma_(u):
    return lmbda * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)  
# Sub domain for clamp at bottom 
def bt(x):
    return np.isclose(x[1], -0.5)

def left(x):
    return np.isclose(x[0], -1.)
# Sub domain for rotation at top
def top(x):
    return np.isclose(x[1], 1.5) 

def right(x):
    return np.isclose(x[0], 1.)



# top displacement 
def top_disp(t):
    return 0.01 * t


# region Parameters
# steel params 
E  = 1000 #210e9 #Pa
nu = 0.3
mu    =  E / (2.0*(1.0 + nu))
lmbda = E*nu / ((1.0 + nu)*(1.0 - 2.0*nu))

# Mass density
rho = 5 #7800 #kg/m^3

# Newmark method parameters
gamma   =  0.5 #Constant(mesh, 0.5+alpha_f-alpha_m)
beta    =  1 #Constant(mesh, (gamma+0.5)**2/4.)

# Time-stepping parameters
T       = 4.0
Nsteps  = 1e4
dt =  T/Nsteps

# the p-wave for elasto-dynamic problem
c_p =((lmbda + 2*mu)/rho)**0.5

lc = 2 / (60 * 2) # L / (number of elements per wavelength)

print('Obey the Courant-Friedrichs-Lewy (CFL) condition :', dt < lc/c_p)


# External pressure
p0 = 1.
cutoff_Tc = T/5

# Define function space for stresses
Vsig = fem.functionspace(mesh, ("DG", 0, (tdim, tdim)))

# Test and trial functions
du = ufl.TrialFunction(V)
u_ = ufl.TestFunction(V)
# Current (unknown) displacement
u = Function(V, name="Displacement")
# Fields from previous time step (displacement, velocity, acceleration)
u_old = Function(V)
v_old = Function(V)
a_old = Function(V)
p = fem.Function(V)


# Exterior force 
def pressure_expression(t):
    if t <= cutoff_Tc:
        return p0 * t / cutoff_Tc
    else:
        return p0

def update_pressure(t):
    p_array = p.x.petsc_vec.getArray()
    p_array_x = p_array.reshape(-1,2)[:,0]
    p_array_y = p_array.reshape(-1,2)[:,1]
    for i in range(p_array_x.size):
        p_array_x[i] = 0.
    for i in range(p_array_y.size):
        p_array_y[i] = pressure_expression(t)    
    p_array_new = np.hstack([p_array_x.reshape(-1,1),p_array_y.reshape(-1,1)])
    #print(p_array_new.shape)
    p.x.petsc_vec.setArray(p_array_new.flatten())
    

# region Myexpression 
class MyExpression_top_out:
    def __init__(self, x0, y0, value, V_dim):
        self.x0 = x0
        self.y0 = y0
        self.value  = value
        self.V_dim = V_dim        
        self.RBF_1  = Rbf(x0, y0, value)
        
    def eval(self, x):
        #print(x[0], x[1], x[2]) 
        values = np.zeros((self.V_dim, x.shape[1]))
        values[1] = np.where(np.isclose(x[1], 1.5, 1e-3, 1e-3), self.RBF_1(x[0], x[1]), 0)
        return values 

class MyExpression_right_out:
    def __init__(self, x0, y0, value, V_dim):
        self.x0 = x0
        self.y0 = y0
        self.value  = value
        self.V_dim = V_dim        
        self.RBF_1  = Rbf(x0, y0, value)
        
    def eval(self, x):
        #print(x[0], x[1], x[2]) 
        values = np.zeros((self.V_dim, x.shape[1]))
        values[0] = np.where(np.isclose(x[0], 1, 1e-3, 1e-3), self.RBF_1(x[0], x[1]), 0)
        return values


# region Elastic funcs
# Stress tensor
def sigma(r):
    return 2.0*mu*ufl.sym(ufl.grad(r)) + lmbda*ufl.tr(ufl.sym(ufl.grad(r)))*ufl.Identity(len(r))

# Mass form
def m(u, u_):
    return rho*ufl.inner(u, u_)*dx

# Elastic stiffness form
def k(u, u_):
    return ufl.inner(sigma(u), ufl.sym(ufl.grad(u_)))*dx

# Rayleigh damping form
'''def c(u, u_):
    return eta_m*m(u, u_) + eta_k*k(u, u_)'''



# Update formula for acceleration
# a = 2/((u - u0 - v0*dt)/(beta*dt*dt) - (1-beta)*a0)
# region update formula
def update_a(u, u_old, v_old, a_old, ufl=True):
    #print('dt=', type(dt))
    if ufl:
        dt_ = dt
        beta_ = beta

    else:
        dt_ = float(dt)
        beta_ = float(beta)
        # transform vector to array
        u_val = u.array
        u_old_val = u_old.array
        v_old_val = v_old.array
        a_old_val = a_old.array
        
        # update a value 
                # update a value 
        a_new_val = 2*(u_val - u_old_val - dt_ * v_old_val) / (beta_ * dt_**2) - \
                     (1 -  beta_) / beta_* a_old_val
                  
        # back to vector 
        a_old.setArray(a_new_val)
        a_old.assemble()  # assemble vector 
        
    return a_old 

# Update formula for velocity
# v = dt * ((1-gamma)*a0 + gamma*a) + v0
def update_v(a, u_old, v_old, a_old, ufl=True):
    
    if ufl:
        dt_ = dt
        gamma_ = gamma
    else:
        dt_ = float(dt)
        gamma_ = float(gamma)
        
        # transform vector to array
        a_val = a.array
        u_old_val = u_old.array
        v_old_val = v_old.array
        a_old_val = a_old.array
        
        # update a value 
        v_new_val = v_old_val + dt_*((1-gamma_)*a_old_val + gamma_*a_val)
        
        # back to vector 
        v_old.setArray(v_new_val)
        v_old.assemble()  # assemble vector 
        
    return v_old 

def update_fields(u, u_old, v_old, a_old):
    """Update fields at the end of each time step.""" 

    # Get vectors (references)
    u_vec, u0_vec  = u.x.petsc_vec, u_old.x.petsc_vec
    v0_vec, a0_vec = v_old.x.petsc_vec, a_old.x.petsc_vec 

    # use update functions using vector arguments
    a_vec = update_a(u_vec, u0_vec, v0_vec, a0_vec, ufl=False)
    v_vec = update_v(a_vec, u0_vec, v0_vec, a0_vec, ufl=False)

    # Update (u_old <- u)
    v_old.x.petsc_vec[:], a_old.x.petsc_vec[:] = v_vec, a_vec
    u_old.x.petsc_vec[:] = u.x.petsc_vec


def avg(x_old, x_new, alpha):
    return alpha*x_old + (1-alpha)*x_new



# Residual
a_new = update_a(du, u_old, v_old, a_old, ufl=True)
v_new = update_v(a_new, u_old, v_old, a_old, ufl=True)


#region RBF 
# Use double precision to generate data (due to GP sampling)
def RBF(x1, x2, params): #radial basis function 
    output_scale, lengthscales = params
    diffs = np.expand_dims(x1 / lengthscales, 1) - \
            np.expand_dims(x2 / lengthscales, 0)
    r2 = np.sum(diffs**2, axis=2)
    return output_scale**2 * np.exp(-0.5 * r2)

# To generate (x,t) (u, y)
def solve_ADR(key, Nx, Ny, m, P, length_scale):
    """No need explicit resolution 
    """
    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    # Generate subkeys
    subkey, subkey1, subkey2 = random.split(key, 3)
    subkey_x = random.split(subkey1,4)
    subkey_y = random.split(subkey2,4)
    # Generate a GP sample
    N = 512
    gp_params = (0.01, length_scale)
    jitter = 1e-10
    X = jnp.linspace(xmin, xmax, N)[:,None]
    K = RBF(X, X, gp_params)
    L = jnp.linalg.cholesky(K + jitter*np.eye(N))
    
    def gp_sample(key, L):
        gp_sample = jnp.dot(L, random.normal(key, (N,)))
        return gp_sample
    
    gp_sample_x = vmap(gp_sample, (0,None))(subkey_x, L)
    gp_sample_y = vmap(gp_sample, (0,None))(subkey_y, L)
    # Create a callable interpolation function
    f_fn_u = lambda x: vmap(jnp.interp,(None, None, 0))(x, X.flatten(), gp_sample_x)
    f_fn_v = lambda x: vmap(jnp.interp,(None, None, 0))(x, X.flatten(), gp_sample_y)
    # Create grid
    x = jnp.linspace(xmin, xmax, Nx)
    y = jnp.linspace(ymin, ymax, Ny)

    # Input sensor locations and measurements
    yy = jnp.linspace(xmin, xmax, m)
    xx = jnp.linspace(ymin, ymax, m)
    u_l, u_r, u_up, u_bt = f_fn_u(yy)
    v_l, v_r, v_up, v_bt = f_fn_v(xx)
    # Output sensor locations and measurements
    idx = random.randint(subkey, (P,2), 0, max(Nx,Ny))
    h = jnp.concatenate([x[idx[:,0]][:,None], y[idx[:,1]][:,None]], axis = 1)

    return  u_up, u_r


m = 200
P = m
Nx, Ny = m, m
key = random.PRNGKey(int(time.time()))
length_sacle = 0.2
N = 100 # number of samples
keys = random.split(key, N)
config.update("jax_enable_x64", True)
u_up_RBF, u_r_RBF =  vmap(solve_ADR, (0, None, None, None, None, None))(keys, Nx, Ny, m, P, length_sacle)
config.update("jax_enable_x64", False)




# region [M][a] = [F]
# Confusing bug: ufl.rhs and ufl.lhs cannot find the bilinear and linear forms correctly
# especially, when the corresponding fenicsx functions are used in elastic funcs
 
# saved list
V1_list, U1_list, ax_list, ay_list, epsilon_x1_list, \
epsilon_y1_list, epsilon_xy1_list, vx_list, vy_list = [], [], [], [], [], [], [], [], []

V1_lists, U1_lists, ax_lists, ay_lists, epsilon_x1_lists, \
epsilon_y1_lists, epsilon_xy1_lists, vx_lists, vy_lists =  [[] for _ in range(8)],  [[] for _ in range(8)], [[] for _ in range(8)],\
[[] for _ in range(8)], [[] for _ in range(8)], [[] for _ in range(8)], [[] for _ in range(8)], [[] for _ in range(8)], [[] for _ in range(8)]




ts_end1 = 70
#for i in range(Nsteps):
break_list=[]
u_list, v_list, u2_list, v2_list = [], [], [], []
for i in trange(N):
    t=0  # initial time step 
    # Test and trial functions
    du = ufl.TrialFunction(V)
    u_ = ufl.TestFunction(V)
    # Current (unknown) displacement
    u = Function(V, name="Displacement")
    # Fields from previous time step (displacement, velocity, acceleration)
    u_old = Function(V)
    v_old = Function(V)
    a_old = Function(V)
    p = fem.Function(V)
    # region BCs
    # Set up boundary condition at bottom
    bt_b = dolfinx.mesh.locate_entities_boundary(mesh, fdim, bt)
    uD = np.array([0, 0, 0], dtype=default_scalar_type)
    mesh.topology.create_connectivity(fdim, tdim)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, bt_b)
    bc_bt = fem.dirichletbc(uD, boundary_dofs, V)

    bt_l = dolfinx.mesh.locate_entities_boundary(mesh, fdim, left)
    uD = np.array([0, 0, 0], dtype=default_scalar_type)
    mesh.topology.create_connectivity(fdim, tdim)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, bt_l)
    bc_l = fem.dirichletbc(uD, boundary_dofs, V)


    xc = np.linspace(-0.5, 1.5, m)
    yc = np.ones_like(xc) * 1.5
    upper_b = dolfinx.mesh.locate_entities_boundary(mesh, fdim, top)
    uD_c = Function(V)
    uD_c_value = u_up_RBF[i,:]
    #if i == 0:
        #plot_bc(xc, uD_c_value, 'top_bc')
    uD_c_fun = MyExpression_top_out(xc, yc, uD_c_value, mesh.geometry.dim)
    uD_c.interpolate(uD_c_fun.eval)        
    boundary_dofs = fem.locate_dofs_topological(V, fdim, upper_b)
    bc_top = fem.dirichletbc(uD_c, boundary_dofs)

    yr = np.linspace(-1, 1, m) 
    xr = np.ones_like(yr) * 1.
    right_b = dolfinx.mesh.locate_entities_boundary(mesh, fdim, right)
    uD_r = Function(V)
    uD_r_value = u_r_RBF[i,:]
    #if i == 0:
        #plot_bc(xc, uD_r_value, 'right_bc')
    uD_r_fun = MyExpression_right_out(xr, yr, uD_r_value, mesh.geometry.dim)
    uD_r.interpolate(uD_r_fun.eval)        
    boundary_dofs = fem.locate_dofs_topological(V, fdim, right_b)
    bc_r = fem.dirichletbc(uD_r, boundary_dofs)  

    #print('bc_c:', uD_c.x.array)

    bc = [bc_bt, bc_top, bc_l, bc_r]

    for ts in trange(ts_end1 + 100):
        t += dt    
        #update_pressure(t-dt)

        error_list = []
        
        LL = rho*ufl.inner(2*(du) / (beta * dt**2), u_)*dx + ufl.inner(sigma(du), ufl.sym(ufl.grad(u_)))*dx     
        RR = rho*ufl.inner(2*( u_old + dt * v_old) / (beta * dt**2) + (1 -  beta) / beta* a_old , u_)*dx 
        theta = 0.5 # relaxation coefficient
        ## FEM 
        # assemble the matrix and vector
        a_form = fem.form(LL)
        L_form = fem.form(RR)
        # Define solver (Not for reuse)
        A = assemble_matrix(a_form, bcs=bc)
        A.assemble()
        b = create_vector(L_form)
        solver = PETSc.KSP().create(mesh.comm)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.CG)
        solver.getPC().setType(PETSc.PC.Type.JACOBI)

        # Update the right hand side reusing the initial vector
        with b.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b, L_form)
        
        # Apply Dirichlet boundary condition to the vector
        apply_lifting(b, [a_form], [bc])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, bc)
        # Solve linear problem
        solver.solve(b, u.x.petsc_vec)
        u.x.scatter_forward()

        #u_geometry = mesh.geometry.x
        ### atthention mesh1.geometry.x are not accurate as imagine, weired!! 

        u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
        u_values = u.x.array.real
        update_fields(u, u_old, v_old, a_old)
        u_tot = u_values.reshape(-1,3)
        U_, V_, W_= u_tot[:,0], u_tot[:,1], u_tot[:,2]

        # retrieve stress 
        gdim = mesh.geometry.dim
        ### stress is 3*3  (gdim, gdim)
        Function_space_for_epsilon = fem.functionspace(mesh, ("CG", 2, (gdim, gdim)))        
        expr= fem.Expression(epsilon(u), Function_space_for_epsilon.element.interpolation_points())        
        epsilon_values = Function(Function_space_for_epsilon) 
        epsilon_values.interpolate(expr)
        epsilon_tot = epsilon_values.x.array.reshape(-1,9)

        if ts >= ts_end1+19 and ts <= ts_end1 + 29:
            U_1 = U_[index_inner_domain]
            V_1 = V_[index_inner_domain]
            ax = a_old.x.petsc_vec.array.reshape(-1,3)[index_inner_domain, 0]
            ay = a_old.x.petsc_vec.array.reshape(-1,3)[index_inner_domain, 1]
            vx = v_old.x.petsc_vec.array.reshape(-1,3)[index_inner_domain, 0]
            vy = v_old.x.petsc_vec.array.reshape(-1,3)[index_inner_domain, 1]
            epsilon_x1 = epsilon_tot[index_inner_domain,0]
            epsilon_y1 = epsilon_tot[index_inner_domain,4]
            epsilon_xy1 = epsilon_tot[index_inner_domain,1]
            V1_lists[0].append(np.copy(V_1.reshape(-1,1)))
            U1_lists[0].append(np.copy(U_1.reshape(-1,1)))
            ax_lists[0].append(np.copy(ax.reshape(-1,1)))
            ay_lists[0].append(np.copy(ay.reshape(-1,1)))
            epsilon_x1_lists[0].append(np.copy(epsilon_x1.reshape(-1,1)))
            epsilon_y1_lists[0].append(np.copy(epsilon_y1.reshape(-1,1)))
            epsilon_xy1_lists[0].append(np.copy(epsilon_xy1.reshape(-1,1)))
            vx_lists[0].append(np.copy(vx.reshape(-1,1)))
            vy_lists[0].append(np.copy(vy.reshape(-1,1)))
        
        if ts >= ts_end1+29 and ts <= ts_end1 + 39:
            U_1 = U_[index_inner_domain]
            V_1 = V_[index_inner_domain]
            ax = a_old.x.petsc_vec.array.reshape(-1,3)[index_inner_domain, 0]
            ay = a_old.x.petsc_vec.array.reshape(-1,3)[index_inner_domain, 1]
            vx = v_old.x.petsc_vec.array.reshape(-1,3)[index_inner_domain, 0]
            vy = v_old.x.petsc_vec.array.reshape(-1,3)[index_inner_domain, 1]
            epsilon_x1 = epsilon_tot[index_inner_domain,0]
            epsilon_y1 = epsilon_tot[index_inner_domain,4]
            epsilon_xy1 = epsilon_tot[index_inner_domain,1]
            V1_lists[1].append(np.copy(V_1.reshape(-1,1)))
            U1_lists[1].append(np.copy(U_1.reshape(-1,1)))
            ax_lists[1].append(np.copy(ax.reshape(-1,1)))
            ay_lists[1].append(np.copy(ay.reshape(-1,1)))
            epsilon_x1_lists[1].append(np.copy(epsilon_x1.reshape(-1,1)))
            epsilon_y1_lists[1].append(np.copy(epsilon_y1.reshape(-1,1)))
            epsilon_xy1_lists[1].append(np.copy(epsilon_xy1.reshape(-1,1)))
            vx_lists[1].append(np.copy(vx.reshape(-1,1)))
            vy_lists[1].append(np.copy(vy.reshape(-1,1)))
        
        if ts >= ts_end1+39 and ts <= ts_end1 + 49:
            U_1 = U_[index_inner_domain]
            V_1 = V_[index_inner_domain]
            ax = a_old.x.petsc_vec.array.reshape(-1,3)[index_inner_domain, 0]
            ay = a_old.x.petsc_vec.array.reshape(-1,3)[index_inner_domain, 1]
            vx = v_old.x.petsc_vec.array.reshape(-1,3)[index_inner_domain, 0]
            vy = v_old.x.petsc_vec.array.reshape(-1,3)[index_inner_domain, 1]
            epsilon_x1 = epsilon_tot[index_inner_domain,0]
            epsilon_y1 = epsilon_tot[index_inner_domain,4]
            epsilon_xy1 = epsilon_tot[index_inner_domain,1]
            V1_lists[2].append(np.copy(V_1.reshape(-1,1)))
            U1_lists[2].append(np.copy(U_1.reshape(-1,1)))
            ax_lists[2].append(np.copy(ax.reshape(-1,1)))
            ay_lists[2].append(np.copy(ay.reshape(-1,1)))
            epsilon_x1_lists[2].append(np.copy(epsilon_x1.reshape(-1,1)))
            epsilon_y1_lists[2].append(np.copy(epsilon_y1.reshape(-1,1)))
            epsilon_xy1_lists[2].append(np.copy(epsilon_xy1.reshape(-1,1)))
            vx_lists[2].append(np.copy(vx.reshape(-1,1)))
            vy_lists[2].append(np.copy(vy.reshape(-1,1)))

        if ts >= ts_end1+49 and ts <= ts_end1 + 59:
            U_1 = U_[index_inner_domain]
            V_1 = V_[index_inner_domain]
            ax = a_old.x.petsc_vec.array.reshape(-1,3)[index_inner_domain, 0]
            ay = a_old.x.petsc_vec.array.reshape(-1,3)[index_inner_domain, 1]
            vx = v_old.x.petsc_vec.array.reshape(-1,3)[index_inner_domain, 0]
            vy = v_old.x.petsc_vec.array.reshape(-1,3)[index_inner_domain, 1]
            epsilon_x1 = epsilon_tot[index_inner_domain,0]
            epsilon_y1 = epsilon_tot[index_inner_domain,4]
            epsilon_xy1 = epsilon_tot[index_inner_domain,1]
            V1_lists[3].append(np.copy(V_1.reshape(-1,1)))
            U1_lists[3].append(np.copy(U_1.reshape(-1,1)))
            ax_lists[3].append(np.copy(ax.reshape(-1,1)))
            ay_lists[3].append(np.copy(ay.reshape(-1,1)))
            epsilon_x1_lists[3].append(np.copy(epsilon_x1.reshape(-1,1)))
            epsilon_y1_lists[3].append(np.copy(epsilon_y1.reshape(-1,1)))
            epsilon_xy1_lists[3].append(np.copy(epsilon_xy1.reshape(-1,1)))
            vx_lists[3].append(np.copy(vx.reshape(-1,1)))
            vy_lists[3].append(np.copy(vy.reshape(-1,1)))
        
        if ts >= ts_end1+59 and ts <= ts_end1 + 69:
            U_1 = U_[index_inner_domain]
            V_1 = V_[index_inner_domain]
            ax = a_old.x.petsc_vec.array.reshape(-1,3)[index_inner_domain, 0]
            ay = a_old.x.petsc_vec.array.reshape(-1,3)[index_inner_domain, 1]
            vx = v_old.x.petsc_vec.array.reshape(-1,3)[index_inner_domain, 0]
            vy = v_old.x.petsc_vec.array.reshape(-1,3)[index_inner_domain, 1]
            epsilon_x1 = epsilon_tot[index_inner_domain,0]
            epsilon_y1 = epsilon_tot[index_inner_domain,4]
            epsilon_xy1 = epsilon_tot[index_inner_domain,1]
            V1_lists[4].append(np.copy(V_1.reshape(-1,1)))
            U1_lists[4].append(np.copy(U_1.reshape(-1,1)))
            ax_lists[4].append(np.copy(ax.reshape(-1,1)))
            ay_lists[4].append(np.copy(ay.reshape(-1,1)))
            epsilon_x1_lists[4].append(np.copy(epsilon_x1.reshape(-1,1)))
            epsilon_y1_lists[4].append(np.copy(epsilon_y1.reshape(-1,1)))
            epsilon_xy1_lists[4].append(np.copy(epsilon_xy1.reshape(-1,1)))
            vx_lists[4].append(np.copy(vx.reshape(-1,1)))
            vy_lists[4].append(np.copy(vy.reshape(-1,1)))
        
        if ts >= ts_end1+69 and ts <= ts_end1 + 79:
            U_1 = U_[index_inner_domain]
            V_1 = V_[index_inner_domain]
            ax = a_old.x.petsc_vec.array.reshape(-1,3)[index_inner_domain, 0]
            ay = a_old.x.petsc_vec.array.reshape(-1,3)[index_inner_domain, 1]
            vx = v_old.x.petsc_vec.array.reshape(-1,3)[index_inner_domain, 0]
            vy = v_old.x.petsc_vec.array.reshape(-1,3)[index_inner_domain, 1]
            epsilon_x1 = epsilon_tot[index_inner_domain,0]
            epsilon_y1 = epsilon_tot[index_inner_domain,4]
            epsilon_xy1 = epsilon_tot[index_inner_domain,1]
            V1_lists[5].append(np.copy(V_1.reshape(-1,1)))
            U1_lists[5].append(np.copy(U_1.reshape(-1,1)))
            ax_lists[5].append(np.copy(ax.reshape(-1,1)))
            ay_lists[5].append(np.copy(ay.reshape(-1,1)))
            epsilon_x1_lists[5].append(np.copy(epsilon_x1.reshape(-1,1)))
            epsilon_y1_lists[5].append(np.copy(epsilon_y1.reshape(-1,1)))
            epsilon_xy1_lists[5].append(np.copy(epsilon_xy1.reshape(-1,1)))
            vx_lists[5].append(np.copy(vx.reshape(-1,1)))
            vy_lists[5].append(np.copy(vy.reshape(-1,1)))
        
        if ts >= ts_end1+79 and ts <= ts_end1 + 89:
            U_1 = U_[index_inner_domain]
            V_1 = V_[index_inner_domain]
            ax = a_old.x.petsc_vec.array.reshape(-1,3)[index_inner_domain, 0]
            ay = a_old.x.petsc_vec.array.reshape(-1,3)[index_inner_domain, 1]
            vx = v_old.x.petsc_vec.array.reshape(-1,3)[index_inner_domain, 0]
            vy = v_old.x.petsc_vec.array.reshape(-1,3)[index_inner_domain, 1]
            epsilon_x1 = epsilon_tot[index_inner_domain,0]
            epsilon_y1 = epsilon_tot[index_inner_domain,4]
            epsilon_xy1 = epsilon_tot[index_inner_domain,1]
            V1_lists[6].append(np.copy(V_1.reshape(-1,1)))
            U1_lists[6].append(np.copy(U_1.reshape(-1,1)))
            ax_lists[6].append(np.copy(ax.reshape(-1,1)))
            ay_lists[6].append(np.copy(ay.reshape(-1,1)))
            epsilon_x1_lists[6].append(np.copy(epsilon_x1.reshape(-1,1)))
            epsilon_y1_lists[6].append(np.copy(epsilon_y1.reshape(-1,1)))
            epsilon_xy1_lists[6].append(np.copy(epsilon_xy1.reshape(-1,1)))
            vx_lists[6].append(np.copy(vx.reshape(-1,1)))
            vy_lists[6].append(np.copy(vy.reshape(-1,1)))     

        if ts >= ts_end1+89 and ts <= ts_end1 + 99:
            U_1 = U_[index_inner_domain]
            V_1 = V_[index_inner_domain]
            ax = a_old.x.petsc_vec.array.reshape(-1,3)[index_inner_domain, 0]
            ay = a_old.x.petsc_vec.array.reshape(-1,3)[index_inner_domain, 1]
            vx = v_old.x.petsc_vec.array.reshape(-1,3)[index_inner_domain, 0]
            vy = v_old.x.petsc_vec.array.reshape(-1,3)[index_inner_domain, 1]
            epsilon_x1 = epsilon_tot[index_inner_domain,0]
            epsilon_y1 = epsilon_tot[index_inner_domain,4]
            epsilon_xy1 = epsilon_tot[index_inner_domain,1]
            V1_lists[7].append(np.copy(V_1.reshape(-1,1)))
            U1_lists[7].append(np.copy(U_1.reshape(-1,1)))
            ax_lists[7].append(np.copy(ax.reshape(-1,1)))
            ay_lists[7].append(np.copy(ay.reshape(-1,1)))
            epsilon_x1_lists[7].append(np.copy(epsilon_x1.reshape(-1,1)))
            epsilon_y1_lists[7].append(np.copy(epsilon_y1.reshape(-1,1)))
            epsilon_xy1_lists[7].append(np.copy(epsilon_xy1.reshape(-1,1)))
            vx_lists[7].append(np.copy(vx.reshape(-1,1)))
            vy_lists[7].append(np.copy(vy.reshape(-1,1)))   

        if ts == ts_end1 + 19 and i ==1:
            plot_disp(X1, Y1, U1_lists[0][-1], 'displacement U1 ts=' + str(ts), rf'$u_{{\mathrm{{FE}}}}^{{{ts}}}$')
            plot_disp(X1, Y1, V1_lists[0][-1], 'displacement V1 ts=' + str(ts), rf'$v_{{\mathrm{{FE}}}}^{{{ts}}}$')
            plot_disp(X1, Y1, vx_lists[0][-1], 'vx ts=' + str(ts), rf'$v_{{x,\mathrm{{FE}}}}^{{{ts}}}$')
            plot_disp(X1, Y1, vy_lists[0][-1], 'vy ts=' + str(ts), rf'$v_{{y,\mathrm{{FE}}}}^{{{ts}}}$')

        if ts == ts_end1 + 39 and i ==1:
            plot_disp(X1, Y1, U1_lists[2][-1], 'displacement U1 ts=' + str(ts), rf'$u_{{\mathrm{{FE}}}}^{{{ts}}}$')
            plot_disp(X1, Y1, V1_lists[2][-1], 'displacement V1 ts=' + str(ts), rf'$v_{{\mathrm{{FE}}}}^{{{ts}}}$')
            plot_disp(X1, Y1, vx_lists[2][-1], 'vx ts=' + str(ts), rf'$v_{{x,\mathrm{{FE}}}}^{{{ts}}}$')
            plot_disp(X1, Y1, vy_lists[2][-1], 'vy ts=' + str(ts), rf'$v_{{y,\mathrm{{FE}}}}^{{{ts}}}$')

        if ts == ts_end1 + 49 and i ==1:
            plot_disp(X1, Y1, U1_lists[3][-1], 'displacement U1 ts=' + str(ts), rf'$u_{{\mathrm{{FE}}}}^{{{ts}}}$')
            plot_disp(X1, Y1, V1_lists[3][-1], 'displacement V1 ts=' + str(ts), rf'$v_{{\mathrm{{FE}}}}^{{{ts}}}$')
            plot_disp(X1, Y1, vx_lists[3][-1], 'vx ts=' + str(ts), rf'$v_{{x,\mathrm{{FE}}}}^{{{ts}}}$')
            plot_disp(X1, Y1, vy_lists[3][-1], 'vy ts=' + str(ts), rf'$v_{{y,\mathrm{{FE}}}}^{{{ts}}}$')        

        if ts == ts_end1 + 69 and i ==1:
            plot_disp(X1, Y1, U1_lists[4][-1], 'displacement U1 ts=' + str(ts), rf'$u_{{\mathrm{{FE}}}}^{{{ts}}}$')
            plot_disp(X1, Y1, V1_lists[4][-1], 'displacement V1 ts=' + str(ts), rf'$v_{{\mathrm{{FE}}}}^{{{ts}}}$')
            plot_disp(X1, Y1, vx_lists[4][-1], 'vx ts=' + str(ts), rf'$v_{{x,\mathrm{{FE}}}}^{{{ts}}}$')
            plot_disp(X1, Y1, vy_lists[4][-1], 'vy ts=' + str(ts), rf'$v_{{y,\mathrm{{FE}}}}^{{{ts}}}$')        

        if ts == ts_end1 + 79 and i ==1:
            plot_disp(X1, Y1, U1_lists[5][-1], 'displacement U1 ts=' + str(ts), rf'$u_{{\mathrm{{FE}}}}^{{{ts}}}$')
            plot_disp(X1, Y1, V1_lists[5][-1], 'displacement V1 ts=' + str(ts), rf'$v_{{\mathrm{{FE}}}}^{{{ts}}}$')
            plot_disp(X1, Y1, vx_lists[5][-1], 'vx ts=' + str(ts), rf'$v_{{x,\mathrm{{FE}}}}^{{{ts}}}$')
            plot_disp(X1, Y1, vy_lists[5][-1], 'vy ts=' + str(ts), rf'$v_{{y,\mathrm{{FE}}}}^{{{ts}}}$')        
        
        if ts == ts_end1 + 89 and i ==1:
            plot_disp(X1, Y1, U1_lists[6][-1], 'displacement U1 ts=' + str(ts), rf'$u_{{\mathrm{{FE}}}}^{{{ts}}}$')
            plot_disp(X1, Y1, V1_lists[6][-1], 'displacement V1 ts=' + str(ts), rf'$v_{{\mathrm{{FE}}}}^{{{ts}}}$')
            plot_disp(X1, Y1, vx_lists[6][-1], 'vx ts=' + str(ts), rf'$v_{{x,\mathrm{{FE}}}}^{{{ts}}}$')
            plot_disp(X1, Y1, vy_lists[6][-1], 'vy ts=' + str(ts), rf'$v_{{y,\mathrm{{FE}}}}^{{{ts}}}$')        

end_time= time.time()
print('Time cost:', end_time - start_time)

for j in range(8):
    ts0 = 89+j*10
    ts1 = 99+j*10
    ts = str(ts0) + ' - ' + str(ts1)
    #np.savetxt('test ts = ' + str(ts) +'.txt', np.array(test_1_list).squeeze().T)
    np.savetxt('V1 ts = ' + str(ts) +'.txt', np.array(V1_lists[j]).squeeze().T)
    np.savetxt('U1 ts = ' + str(ts) +'.txt', np.array(U1_lists[j]).squeeze().T)
    np.savetxt('ax ts = ' + str(ts) +'.txt', np.array(ax_lists[j]).squeeze().T)
    np.savetxt('ay ts = ' + str(ts) +'.txt', np.array(ay_lists[j]).squeeze().T)
    np.savetxt('epsilon_x1 ts = ' + str(ts) +'.txt', np.array(epsilon_x1_lists[j]).squeeze().T)
    np.savetxt('epsilon_y1 ts = ' + str(ts) +'.txt', np.array(epsilon_y1_lists[j]).squeeze().T)
    np.savetxt('epsilon_xy1 ts = ' + str(ts) +'.txt', np.array(epsilon_xy1_lists[j]).squeeze().T)
    np.savetxt('vx ts = ' + str(ts) +'.txt', np.array(vx_lists[j]).squeeze().T)
    np.savetxt('vy ts = ' + str(ts) +'.txt', np.array(vy_lists[j]).squeeze().T)








        




