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
import pickle
from dynamic_utils import plot_mesh, plot_disp, createFolder, plot_boundary, plot_disp_real
from scipy.spatial import KDTree
import time 
from dolfinx.io import VTKFile
###############Attention##################
# the Dofinx 0.9.0 version is used in this code
# .vector --> .x.petsc_vec
##########################################

#region Save path       
originalDir = os.path.dirname(os.path.abspath(__file__))
print('curent working directory:', originalDir)
os.chdir(os.path.join(originalDir))

foldername = 'FE_NO_elasto_dynamic_L_shape_90_169'  
createFolder(foldername )
os.chdir(os.path.join(originalDir, './'+ foldername + '/')) 

origin_real = os.path.join(originalDir, './'+ foldername + '/')
#### Gmsh generates the geometry 
import meshio
os.chdir(os.path.join(originalDir + '/'+ 'Gmsh_results' + '/'))


msh = meshio.read(
    "L_shape_in.msh",
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
#plot_mesh(cells, points, 'Gmsh Mesh')


# region mesh out 
mesh, cell_markers, facet_markers  = gmshio.read_from_msh("L_shape_in.msh", MPI.COMM_WORLD) 
os.chdir(origin_real)
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
X_BC, Y_BC = boundary_dof_coords[:,0], boundary_dof_coords[:,1]
np.savetxt('X_BC.txt', X_BC)
np.savetxt('Y_BC.txt', Y_BC)

os.chdir(os.path.join(originalDir, './' + 'L_shape_ground_truth' + '/'))
X_BC_in, Y_BC_in = np.loadtxt('X_BC_in.txt'), np.loadtxt('Y_BC_in.txt')


# resort the boundary dof coords to match the order in the ground truth BC_in
index = np.array([np.where((np.isclose(x0, X_BC_in)) & (np.isclose(y_0, Y_BC_in)))[0] for x0, y_0 in zip(X_BC, Y_BC)]).flatten()
os.chdir(origin_real)

np.savetxt('X_BC_in_r.txt', X_BC_in[index])
np.savetxt('Y_BC_in_r.txt', Y_BC_in[index])
np.savetxt('index_BC.txt', index)

X_BC_in_r, Y_BC_in_r = X_BC_in[index], Y_BC_in[index]

print('index length:', len(index))
print(X_BC_in_r[0:10], X_BC[0:10])




def epsilon(u):
    return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

def sigma_(u):
    return lmbda * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)  

# region locators
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


# region Parameters
# steel params 
E  = 1000 
nu = 0.3
mu    =  E / (2.0*(1.0 + nu))
lmbda = E*nu / ((1.0 + nu)*(1.0 - 2.0*nu))

# Mass density
rho = 5 
# Newmark method parameters
gamma   =  0.5
beta    =  1 

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


#region Exterior loadings  
# top displacement 
def top_disp(t):
    return 0.01 * t




# region Measure dss
# Create mesh function over the cell facets
domain = mesh 
# locate the boundary facets
d_facets = dolfinx.mesh.locate_entities_boundary(domain, fdim, top)
# Mark the facets
d_facets_mark = np.zeros_like(d_facets) + 3
# Create mesh tag (assign a number to the specific facets)
ft = dolfinx.mesh.meshtags(mesh, fdim, np.array(d_facets).astype(np.int32), 
                   np.array(d_facets_mark).astype(np.int32))
mesh.topology.create_connectivity(fdim, tdim)
# Define measure for boundary condition integral
dss = ufl.Measure('ds', domain=mesh, subdomain_data=ft)
#To verify the inner surface is well-difined 
facet_values = ft.values
facet_indices = ft.indices
# Define a color map for the tags
colors = {3: 'red'}  # Assign colors for each tag (1, 2, 3)
### Attention fdim = 1, tdim = 2 ####



mask_103 = facet_markers.values == 103
# Define a color map for the tags
colors = {103: 'red'}  
facet_values = facet_markers.values[mask_103]
facet_indices = facet_markers.indices[mask_103]
#plot_boundary(mesh, facet_values, facet_indices, colors, 'inner_boundary', fdim)

ds = ufl.Measure("ds", domain=mesh, subdomain_data = facet_markers) # tag =103 we only need inner boundnary ds


Traction = fem.Function(V)
# region Expression 
'''
-------------------------------------------------------
expression for the interpolation of the boundary values
-------------------------------------------------------     
'''
class MyExpression:
    def __init__(self, x0, y0, value, V_dim):
        self.x0 = x0
        self.y0 = y0
        self.value  = value
        self.V_dim = V_dim        
        self.Tx_fun  = Rbf(x0, y0, value[0])
        self.Ty_fun  = Rbf(x0, y0, value[1])
        
    def eval(self, x):

        para1 = E/((1 + nu)*(1 - 2*nu))

        values = np.zeros((self.V_dim, x.shape[1]))

        condition = np.any([np.isclose(x_, x[0], atol = 1e-5) & np.isclose(y_, x[1], atol = 1e-5) 
                      for x_, y_ in zip(self.x0, self.y0)], axis = 0) # should be array of shape
        
        values[0] = np.where(condition, self.Tx_fun(x[0], x[1]), 0)
        values[1] = np.where(condition, self.Ty_fun(x[0], x[1]), 0)
                             
        return values 

class MyExpression_bc:
    def __init__(self, x0, y0, value, V_dim):
        self.x0 = x0
        self.y0 = y0
        self.value  = value
        self.V_dim = V_dim        
        self.Tx_fun  = Rbf(x0, y0, value[0])
        self.Ty_fun  = Rbf(x0, y0, value[1])
        
    def eval(self, x):

        para1 = E/((1 + nu)*(1 - 2*nu))

        values = np.zeros((self.V_dim, x.shape[1]))

        condition = np.any([np.isclose(x_, x[0], atol = 1e-5) & np.isclose(y_, x[1], atol = 1e-5) 
                      for x_, y_ in zip(self.x0, self.y0)], axis = 0) # should be array of shape
        
        values[0] = np.where(condition, self.Tx_fun(x[0], x[1]), 0)
        values[1] = np.where(condition, self.Ty_fun(x[0], x[1]), 0)
                             
        return values 

# region Elastic funcs
# Stress tensor
def sigma(r):
    return 2.0*mu*ufl.sym(ufl.grad(r)) + lmbda*ufl.tr(ufl.sym(ufl.grad(r)))*ufl.Identity(len(r))

# Mass form
def mass(u, u_):
    return rho*ufl.inner(u, u_)*dx

# Elastic stiffness form
def k(u, u_):
    return ufl.inner(sigma(u), ufl.sym(ufl.grad(u_)))*dx

# Rayleigh damping form
'''def c(u, u_):
    return eta_m*m(u, u_) + eta_k*k(u, u_)'''

# Work of external forces
def Wext(u_):
    return ufl.dot(u_, p)*dss(3)

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


# update formula for NN 
def update_fields_NN(u, u_old, v_old, a_old):
    """
    Update fields at the end of each time step.
    a = 2/((u - u0 - v0*dt)/(beta*dt*dt) - (1-beta)*a0)
    v = dt * ((1-gamma)*a0 + gamma*a) + v0
    """ 
    a = 2*(u - u_old - dt * v_old) / (beta * dt**2) - \
                    (1 -  beta) / beta* a_old
    
    v = v_old + dt*((1-gamma)*a_old + gamma*a)

    # Update (u_old <- u)
    v_old[:], a_old[:] = v, a
    u_old[:] = u


def avg(x_old, x_new, alpha):
    return alpha*x_old + (1-alpha)*x_new


# region BCs
# Set up boundary condition at bottom
bt_b = dolfinx.mesh.locate_entities_boundary(mesh, fdim, bt)
uD = np.array([0, 0, 0], dtype=default_scalar_type)
mesh.topology.create_connectivity(fdim, tdim)
boundary_dofs = fem.locate_dofs_topological(V, fdim, bt_b)
bc_bt = fem.dirichletbc(uD, boundary_dofs, V)

bt_l = dolfinx.mesh.locate_entities_boundary(mesh, fdim, left)
uD_l = np.array([0, 0, 0], dtype=default_scalar_type)
mesh.topology.create_connectivity(fdim, tdim)
boundary_dofs = fem.locate_dofs_topological(V, fdim, bt_l)
bc_l = fem.dirichletbc(uD_l, boundary_dofs, V)


# Set up boundary condition at top 
top_b = dolfinx.mesh.locate_entities_boundary(mesh, fdim, top)
uD_top = np.array([0, 0.01, 0], dtype=default_scalar_type)
mesh.topology.create_connectivity(fdim, tdim)
boundary_dofs = fem.locate_dofs_topological(V, fdim, top_b)
bc_top = fem.dirichletbc(uD_top, boundary_dofs, V)

top_r = dolfinx.mesh.locate_entities_boundary(mesh, fdim, right)
uD_r = np.array([0.01, 0, 0], dtype=default_scalar_type)
mesh.topology.create_connectivity(fdim, tdim)
boundary_dofs = fem.locate_dofs_topological(V, fdim, top_r)
bc_r = fem.dirichletbc(uD_r, boundary_dofs, V)



bc = [bc_bt, bc_l, bc_top, bc_r]


# Residual
a_new = update_a(du, u_old, v_old, a_old, ufl=True)
v_new = update_v(a_new, u_old, v_old, a_old, ufl=True)

# region [M][a] = [F]
# Confusing bug: ufl.rhs and ufl.lhs cannot find the bilinear and linear forms correctly
# especially, when the corresponding fenicsx functions are used in elastic funcs
LL = rho*ufl.inner(2*(du) / (beta * dt**2), u_)*dx + ufl.inner(sigma(du), ufl.sym(ufl.grad(u_)))*dx     
RR = rho*ufl.inner(2*( u_old + dt * v_old) / (beta * dt**2) + (1 -  beta) / beta* a_old , u_)*dx  
 

# region Iteration part 
### later these parameters will be replaced by DeepONet 
os.chdir(os.path.join(originalDir + '/'+ 'Gmsh_results' + '/'))
mesh2, cell_markers2, facet_markers2  = gmshio.read_from_msh("L_shape_in_real.msh", MPI.COMM_WORLD) 
os.chdir(origin_real)
V2 = functionspace(mesh2, ("CG", 2, (mesh2.geometry.dim, )))

tag_1 = 104
# find the facets with tag = 104
facets_104 = facet_markers2.indices[facet_markers2.values == tag_1]
# find the dofs on these facets
dofs_104 = fem.locate_dofs_topological(V2, fdim, facets_104)
# get the coordinates of all the dofs in V2
dof_coords = V2.tabulate_dof_coordinates()
# only keep the dof coords on the boundary with tag 104
boundary_dof_coords = dof_coords[dofs_104]
plot_disp(boundary_dof_coords[:,0], boundary_dof_coords[:,1], boundary_dof_coords[:,0], 'BC1', 'BC1')
X_BC1, Y_BC1 = boundary_dof_coords[:,0], boundary_dof_coords[:,1]
u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V2)
X1, Y1 = u_geometry[:,0], u_geometry[:,1]

np.savetxt('X1.txt', X1)
np.savetxt('Y1.txt', Y1)

np.savetxt('X_BC1.txt', X_BC1)
np.savetxt('Y_BC1.txt', Y_BC1)



du2 = ufl.TrialFunction(V2)
u_2 = ufl.TestFunction(V2)
# Current (unknown) displacement
u2 = Function(V2, name="Displacement")
# Fields from previous time step (displacement, velocity, acceleration)
u_old2 = Function(V2)
v_old2 = Function(V2)
a_old2 = Function(V2)

# region DeepONet 
from prepare_DeepONet_Elasto_dynamic_ts_89_169_non_overlapping_irregular import PI_DeepONet


m = X_BC.shape[0]
d = 2 # dimension of the input data
ela_model = dict()
ela_model['E'] = 1000e-8 #1000 
ela_model['nu'] = 0.3 
ela_model['rho'] = 5e-8 #5

branch_layers_1 =  [2*m, 100, 100, 100, 100, 800]
trunk_layers =  [d, 100, 100, 100, 100, 800]
model = PI_DeepONet(branch_layers_1, trunk_layers, **ela_model)

os.chdir(os.path.join(originalDir, './' + 'prepare_DeepONet_disk_dynamic_square_disk_ts_89_169_200w_test5' + '/'))
X1_real = np.loadtxt('X1_NO.txt')
Y1_real = np.loadtxt('Y1_NO.txt')
x_c_real = np.loadtxt('x_c_NO.txt')
y_c_real = np.loadtxt('y_c_NO.txt')
print('c_NO verification:', np.all(np.abs(x_c_real - X_BC) < 1e-8), np.all(np.abs(y_c_real - Y_BC) < 1e-8)) # these two sets of coordinates should be the same
print(os.getcwd())
with open('DeepONet_ED_89_169.pkl', 'rb') as f:
    params0 = pickle.load(f)

os.chdir(origin_real)

t=0  # initial time step 
ts_end = 90 # final time step for FE_FE coupling
params = params0 # initialize the parameters for DeepONet 
iter_NN =  0  # count the iteration of NN

break_list=[]


'''
===========================================================
The handshake between subdomains is information exchange
at the overlapping boundaries
-----------------------------------------------------------
The index and coordinates of the boundary points extraction
Every domain has inner and outer edges in overlapping region 
===========================================================
'''

#coor_r_out = np.array([np.where(np.isclose(x0, X)& (np.isclose(y_0, Y)))[0] for x0, y_0 in zip(X_BC, Y_BC)]).flatten() # outer circle coordinates in out domain mesh
coor_r1_out = np.array([np.where(np.isclose(x0, X1) & (np.isclose(y0, Y1)))[0] for x0, y0 in zip(X_BC1, Y_BC1)]).flatten() # boundary coordinates in inner domain mesh

# create target point array
target_points = np.column_stack((X1_real, Y1_real))
# create source point array
source_points = np.column_stack((X1, Y1))
# build KDTree for fast nearest neighbor search
tree = KDTree(source_points)
distances, index_XY = tree.query(target_points)



print(len(index_XY), np.max(np.abs(X1 - X1_real)), np.max(np.abs(X1[index_XY]- X1_real)), np.max(np.abs(Y1[index_XY]-Y1_real)))

#print(len(index_xy_c), np.max(np.abs(x_c_NO - x_c_NO)), np.max(np.abs(x_c_NO[index_xy_c]- x_c_NO)), np.max(np.abs(y_c_NO[index_xy_c]-y_c_NO)))
os.chdir(origin_real)

# The boundary of irregular shape 
n_bc = ufl.FacetNormal(mesh)


# region extract BC
os.chdir(os.path.join(originalDir + '/'+ 'Gmsh_results' + '/'))
_ , _, facet_markers_bc  = gmshio.read_from_msh("L_shape_in_all_bc1.msh", MPI.COMM_WORLD) 

tags_list = range(104,116)
bc_list = []
coor_out_list = []
length_list = []
facets_list = []
for tag in tags_list:
    facets_bc = facet_markers_bc.indices[facet_markers_bc.values == tag]
    dofs_bc = fem.locate_dofs_topological(V, fdim, facets_bc)
    dof_coords = V.tabulate_dof_coordinates()
    # only keep the dof coords on the boundary with tag 103
    boundary_dof_coords = dof_coords[dofs_bc]
    bc_x, bc_y = boundary_dof_coords[:,0], boundary_dof_coords[:,1]
    bc_list.append(np.hstack([bc_x.reshape(-1, 1), bc_y.reshape(-1, 1)]))
    coor_out_val = np.array([np.where(np.isclose(x0, X)& (np.isclose(y_0, Y)))[0] for x0, y_0 in zip(bc_x, bc_y)]).flatten()
    coor_out_list.append(coor_out_val)
    length_list.append(bc_x.shape[0])
    facets_list.append(facets_bc)
    plot_disp(bc_x, bc_y, bc_x*10, 'BC_tag_'+ str(tag), 'BC_tag_'+ str(tag))

coor_out_test = np.hstack(coor_out_list)

# Get the unique values and the index of each value in the unique-value array
unique_vals, first_indices, counts = np.unique(
    coor_out_test, 
    return_index=True, 
    return_counts=True
)

# Mark which values are duplicates
duplicate_indices = first_indices[counts > 1] 
# The boundary of irregular shape 
nx_c_104 = (bc_list[0][:,0] + 0.4)/(0.1)
ny_c_104 = (bc_list[0][:,1] - 0.9)/(0.1)

nx_c_105 = (bc_list[1][:,0] + 0.4)/(0.1)
ny_c_105 = (bc_list[1][:,1] - 0.1)/(0.1)

nx_c_108 = (bc_list[4][:,0] + 0.1)/(0.1)
ny_c_108 = (bc_list[4][:,1] - 0.9)/(0.1)

nx_c_109 = -(bc_list[5][:,0] - 0.1)/(0.1)
ny_c_109 = -(bc_list[5][:,1] - 0.6)/(0.1)

nx_c_110 = (bc_list[6][:,0] - 0.4)/(0.1)
ny_c_110 = (bc_list[6][:,1] - 0.4)/(0.1)

nx_c_111 = (bc_list[7][:,0] - 0.4)/(0.1)
ny_c_111 = (bc_list[7][:,1] - 0.1)/(0.1)

nx_c_106 = np.ones_like(bc_list[2][:,0]) * (-1)
ny_c_106 = np.ones_like(bc_list[2][:,0]) * 0

nx_c_107 = np.ones_like(bc_list[3][:,0]) * 0
ny_c_107 = np.ones_like(bc_list[3][:,0]) * 1

nx_c_112 = np.ones_like(bc_list[8][:,0]) * 1
ny_c_112 = np.ones_like(bc_list[8][:,0]) * 0

nx_c_113 = np.ones_like(bc_list[9][:,0]) * 0
ny_c_113 = np.ones_like(bc_list[9][:,0]) * 1

nx_c_114 = np.ones_like(bc_list[10][:,0]) * 1
ny_c_114 = np.ones_like(bc_list[10][:,0]) * 0

nx_c_115 = np.ones_like(bc_list[11][:,0]) * 0
ny_c_115 = np.ones_like(bc_list[11][:,0]) * (-1)

nx_c = np.hstack([nx_c_104, nx_c_105, nx_c_106, nx_c_107, \
                  nx_c_108, nx_c_109, nx_c_110, nx_c_111, \
                  nx_c_112, nx_c_113, nx_c_114, nx_c_115]).reshape(-1, 1)
ny_c = np.hstack([ny_c_104, ny_c_105, ny_c_106, ny_c_107, \
                  ny_c_108, ny_c_109, ny_c_110, ny_c_111, \
                  ny_c_112, ny_c_113, ny_c_114, ny_c_115]).reshape(-1, 1)

n_total = np.hstack([np.delete(nx_c, duplicate_indices, axis = 0), np.delete(ny_c, duplicate_indices, axis = 0)])
coor_r_out_real = np.delete(coor_out_test, duplicate_indices, axis = 0)

np.savetxt('n_total.txt', n_total)
np.savetxt('coor_r_out_real.txt', coor_r_out_real)


n_total = np.loadtxt('n_total.txt')
nx_c = n_total[:,0]
ny_c = n_total[:,1]
coor_r_out = np.loadtxt('coor_r_out_real.txt').astype(np.int32)
os.chdir(origin_real)

target_points = np.column_stack((X[coor_r_out], Y[coor_r_out]))
# create source point array
source_points = np.column_stack((X_BC1, Y_BC1))
# build KDTree for fast nearest neighbor search
tree = KDTree(source_points)
distances, index_xy_c = tree.query(target_points)

print('three BC=',X_BC1[index_xy_c][0:10], X[coor_r_out][0:10], x_c_real[0:10])
X_BC, Y_BC = X[coor_r_out], Y[coor_r_out]
index_out2in = np.array([np.where((np.isclose(x0, X[coor_r_out])) & (np.isclose(y0, Y[coor_r_out])))[0] 
                                for x0, y0 in zip(X_BC1, Y_BC1)]).flatten()
print(X_BC[index_out2in][0:10], X_BC1[0:10])


target_points = np.column_stack((x_c_real, y_c_real))
# create source point array
source_points = np.column_stack((X_BC1, Y_BC1))
# build KDTree for fast nearest neighbor search
tree = KDTree(source_points)
distances1, index_xy_c_NO = tree.query(target_points)
print('three BC1=',X_BC1[index_xy_c_NO][0:10], x_c_real[0:10])


for ts in trange(170):
    if ts == 92:
        start_time = time.time()
    t += dt    
    #update_pressure(t-dt)

    error_list = []
    u_list, v_list, u2_list, v2_list = [], [], [], []
    niter = 1000
    theta = 0.5 # relaxation coefficient

    #### Coupling  
    for iter in range(0, niter):
        #region First FE solver 
        if ts == 0 and iter == 0:
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

            u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
            u_values = u.x.array.real
            
        
        if  (ts == 0 and iter > 0) or (ts > 0 and iter >= 0): 

            # Assign the stored values back to u_old, v_old, and a_old (- delta_t)
            u_old.x.petsc_vec.setArray(u_old_0)
            v_old.x.petsc_vec.setArray(v_old_0)
            a_old.x.petsc_vec.setArray(a_old_0)

            # region Handshake 
            LL3 = rho*ufl.inner(2*(du) / (beta * dt**2), u_)*dx + \
                      ufl.inner(sigma(du), ufl.sym(ufl.grad(u_)))*dx     
            
            RR3 = rho*ufl.inner(2*( u_old + dt * v_old) / (beta * dt**2) + \
                                (1 -  beta) / beta * a_old , u_)*dx  +\
                                 ufl.dot(Traction, u_) * ds(103) # here tag = 103, defined in Gmsh 
            

            '''traction_term = fem.form(ufl.dot(Traction, u_) * ds(103))
            integral_value = fem.assemble_scalar(traction_term)
            print(f"Boundary integral value for the given u_: {integral_value}")'''

            a_form3 = fem.form(LL3)
            L_form3 = fem.form(RR3)
            A3 = assemble_matrix(a_form3, bcs=bc)
            A3.assemble()
            b3 = create_vector(L_form3)

            solver = PETSc.KSP().create(mesh.comm)
            solver.setOperators(A3)
            solver.setType(PETSc.KSP.Type.CG)
            solver.getPC().setType(PETSc.PC.Type.JACOBI)

            # Update the right hand side reusing the initial vector
            with b3.localForm() as loc_b3:
                loc_b3.set(0)
            assemble_vector(b3, L_form3)
            
            # Apply Dirichlet boundary condition to the vector
            apply_lifting(b3, [a_form3], [bc])
            b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b3, bc)
            # Solve linear problem
            solver.solve(b3, u.x.petsc_vec)
            u.x.scatter_forward()

            u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
            u_values = u.x.array.real
        
        # Store the current values of u_old, v_old, a_old for the next iteration
        u_old_0 = np.copy(u_old.x.petsc_vec.array)
        v_old_0 = np.copy(v_old.x.petsc_vec.array)
        a_old_0 = np.copy(a_old.x.petsc_vec.array)

        # update the u, u_old, v_old, a_old(+ delta_t) 
        # Attention: we use stored values u_old_0, v_old_0, a_old_0 in one inner iteration
        update_fields(u, u_old, v_old, a_old)
        
        u_tot = u_values.reshape(-1,3)
        U_, V_, W_= u_tot[:,0], u_tot[:,1], u_tot[:,2]

        # Extract displacment at outer circle 
        u_c = U_[coor_r_out].reshape(1,-1)
        v_c = V_[coor_r_out].reshape(1,-1)
    
    
        # retrieve stress 
        gdim = mesh.geometry.dim
        ### stress is 3*3  (gdim, gdim)
        Function_space_for_sigma = fem.functionspace(mesh, ("CG", 2, (gdim, gdim)))        
        expr= fem.Expression(sigma(u), Function_space_for_sigma.element.interpolation_points())        
        sigma_values = Function(Function_space_for_sigma) 
        sigma_values.interpolate(expr)
        sigma_tot = sigma_values.x.array.reshape(-1,9)



        sigma_xx_c0 = sigma_tot[coor_r_out, 0].reshape(1,-1)
        sigma_yy_c0 = sigma_tot[coor_r_out, 4].reshape(1,-1)
        sigma_xy_c0 = sigma_tot[coor_r_out, 1].reshape(1,-1)

        Tx_c0 = -(sigma_xx_c0 * nx_c + sigma_xy_c0 * ny_c)
        Ty_c0 = -(sigma_xy_c0 * nx_c + sigma_yy_c0 * ny_c)


        # region handshake
        # interpolation the displacement from the outer region to the inner region

        u2c_test = u_c[0, index_out2in].reshape(1,-1)
        v2c_test = v_c[0, index_out2in].reshape(1,-1)
        fdim2 = mesh2.topology.dim -1 

        uD_c = Function(V2)
        uD_c_value = np.vstack([u2c_test,v2c_test])
        #print(x_up.shape, u_up.shape, uD_up_value.shape)
       
        uD_c_fun = MyExpression_bc(X_BC1, Y_BC1, uD_c_value, mesh2.geometry.dim)
        uD_c.interpolate(uD_c_fun.eval)        
        boundary_dofs = fem.locate_dofs_topological(V2, fdim2, facets_104)
        bc_c = fem.dirichletbc(uD_c, boundary_dofs)            
        bc2  = [bc_c]


        if (ts == 0 and iter >=1) or (ts >= 1 and ts < ts_end):
            # Assign the stored values back to u_old, v_old, and a_old (- delta_t)
            '''
            =========================================================
            Atthenion: before convergence, this part is used to 
            assign the stored values back to u_old, v_old, and a_old
            instead of stepping forward 
            =========================================================
            '''
            u_old2.x.petsc_vec.setArray(u_old_1)
            v_old2.x.petsc_vec.setArray(v_old_1)
            a_old2.x.petsc_vec.setArray(a_old_1)

        if ts >= ts_end:
            # Assign the stored values back to u_old, v_old, and a_old (- delta_t)
            # ts_end is the FE-FE coupling end time step
            # in FE-NO, u,v and a are no longer petsc vectors
            u_old2 = np.copy(u_old_1)
            v_old2 = np.copy(v_old_1)
            a_old2 = np.copy(a_old_1)


        if ts < ts_end:   
            #region Second FE solver      
            LL2 = rho*ufl.inner(2*(du2) / (beta * dt**2), u_2)*dx + \
                ufl.inner(sigma(du2), ufl.sym(ufl.grad(u_2)))*dx    
                
            RR2 = rho*ufl.inner(2*( u_old2 + dt * v_old2) / (beta * dt**2) + \
                                (1 -  beta) / beta* a_old2 , u_2)*dx 
                
            a_form2 = fem.form(LL2)
            L_form2 = fem.form(RR2)

            A2 = assemble_matrix(a_form2, bcs=bc2)
            A2.assemble()
            b2 = create_vector(L_form2)
            solver = PETSc.KSP().create(mesh2.comm)
            solver.setOperators(A2)
            solver.setType(PETSc.KSP.Type.CG)
            solver.getPC().setType(PETSc.PC.Type.JACOBI)
            
            with b2.localForm() as loc_b2:
                loc_b2.set(0)
            assemble_vector(b2, L_form2)

            # Apply Dirichlet boundary condition to the vector
            apply_lifting(b2, [a_form2], [bc2])
            b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b2, bc2)
            # Solve linear problem
            solver.solve(b2, u2.x.petsc_vec)
            u2.x.scatter_forward()
            
            # Store the current values of u_old, v_old, a_old
            u_old_1 = np.copy(u_old2.x.petsc_vec.array)
            v_old_1 = np.copy(v_old2.x.petsc_vec.array)
            a_old_1 = np.copy(a_old2.x.petsc_vec.array)
            
            # Update old fields with new quantities
            # Attention: we use stored values u_old_1, v_old_1, a_old_1 in one inner iteration
            update_fields(u2, u_old2, v_old2, a_old2)
                
            topology, u_cell_types, u_geometry = plot.vtk_mesh(V2)
            u_values = u2.x.array.real
            u_tot = u_values.reshape(-1,3)
            U_1, V_1= u_tot[:,0], u_tot[:,1]

            # retrieve stress
            gdim = mesh2.geometry.dim
            ### stress is 3*3  (gdim, gdim)
            Function_space_for_sigma = fem.functionspace(mesh2, ("CG", 2, (gdim, gdim))) 
            expr= fem.Expression(sigma(u2), Function_space_for_sigma.element.interpolation_points())        
            sigma_values = Function(Function_space_for_sigma) 
            sigma_values.interpolate(expr)
            sigma_tot1 = sigma_values.x.array.reshape(-1,9)

            #stress at inner boundary 
            sigma_xx_c = sigma_tot1[coor_r1_out,0]
            sigma_yy_c = sigma_tot1[coor_r1_out,4]
            sigma_xy_c = sigma_tot1[coor_r1_out,1]

            Tx_c_new = -(sigma_xx_c[index_xy_c].reshape(1,-1) * nx_c + sigma_xy_c[index_xy_c].reshape(1,-1) * ny_c)
            Ty_c_new = -(sigma_xy_c[index_xy_c].reshape(1,-1) * nx_c + sigma_yy_c[index_xy_c].reshape(1,-1) * ny_c)

            # relaxation formula to get new traction values
            Tx_c_tot = theta * Tx_c0 + (1-theta)*Tx_c_new
            Ty_c_tot = theta * Ty_c0 + (1-theta)*Ty_c_new

            T_c_tot = np.vstack([Tx_c_tot.reshape(1,-1), Ty_c_tot.reshape(1,-1)])  

            # Create a FEniCSX expression for the new traction values
            # and interpolate it to the function space V to set bounary conditions for outer domain
            T_c_fun = MyExpression(X[coor_r_out], Y[coor_r_out], T_c_tot, mesh.geometry.dim)
            Traction.interpolate(T_c_fun.eval)

    
        if ts >=ts_end: 
            # region Inner NN 
            '''
            =============================================================
            Each parmas serve for 10 time steps (can be further extended)
            =============================================================
            '''
            iter_NN += 1
            y_test = np.hstack([X1.reshape(-1,1), Y1.reshape(-1,1)])

            # get the displacement values on the outer domain outer edges
            u_c_out, v_c_out = U_[coor_r_out], V_[coor_r_out]
            x_c_out, y_c_out = X[coor_r_out], Y[coor_r_out]
            # Rbf interpolation to get the displacement values on the outer domain outer edges (avoid mismatch)
            u_c_real = Rbf(x_c_out, y_c_out, u_c_out)(X1[coor_r1_out], Y1[coor_r1_out])
            v_c_real = Rbf(x_c_out, y_c_out, v_c_out)(X1[coor_r1_out], Y1[coor_r1_out])


            
            u_old_, v_old_, vx_old, vy_old = u_old2.reshape(-1,3)[:, 0], u_old2.reshape(-1,3)[:, 1],\
                                                v_old2.reshape(-1,3)[:, 0], v_old2.reshape(-1,3)[:, 1]
            
            # input of Branch2 (boundary conditions)
            u_bc = u_c_real[index_xy_c_NO].reshape(1,-1)
            v_bc = v_c_real[index_xy_c_NO].reshape(1,-1)

            # satisify the order in PointNet traning 
            u_old_ = u_old_[index_XY].reshape(1,-1)
            v_old_ = v_old_[index_XY].reshape(1,-1)
            vx_old = vx_old[index_XY].reshape(1,-1)
            vy_old = vy_old[index_XY].reshape(1,-1)

            # the scale factor 100 is used to match the scale of the input data    
            u_test = np.hstack([X1_real.reshape(1,-1), Y1_real.reshape(1,-1), u_old_, v_old_, vx_old/100, 
                                vy_old/100, u_bc, v_bc]) 
            
            v_test = np.hstack([X1_real.reshape(1,-1), Y1_real.reshape(1,-1), u_old_, v_old_, vx_old/100, 
                                vy_old/100, u_bc, v_bc]) 
            
            s_u_pred, s_v_pred, s_e_xx_pred_tot, s_e_yy_pred_tot, s_e_xy_pred_tot = model.predict_s(params, u_test, v_test, y_test)
            
            y_test_c = np.hstack([x_c_out.reshape(-1,1), y_c_out.reshape(-1,1)])
            _, _, s_e_xx_pred, s_e_yy_pred, s_e_xy_pred = model.predict_s(params, u_test, v_test, y_test_c)

            para1 = E/((1 + nu)*(1 - 2*nu))
            Tx_c_new = -(para1 * ((1 - nu) * s_e_xx_pred + nu * s_e_yy_pred) * nx_c\
                                    + para1 * ((1 - 2 * nu) * s_e_xy_pred)* ny_c)
            
            Ty_c_new = -(para1 * ((1 - 2 * nu) * s_e_xy_pred) * nx_c\
                                    + para1 * ((1 - nu) * s_e_yy_pred + nu * s_e_xx_pred)* ny_c)
            

            # Store the current values of u_old, v_old, a_old
            u_old_1 = np.copy(u_old2)
            v_old_1 = np.copy(v_old2)
            a_old_1 = np.copy(a_old2)
            
            u2_NN = np.hstack((s_u_pred.reshape(-1,1), s_v_pred.reshape(-1,1), np.zeros(s_u_pred.shape).reshape(-1,1))).flatten()
            # Update old fields with new quantities obtained from NN 
            # Attention: we use stored values u_old_1, v_old_1, a_old_1 before convergence 
            
            update_fields_NN(u2_NN, u_old2, v_old2, a_old2)

            # relaxation formula to get new traction values
            Tx_c_tot = theta * Tx_c0 + (1-theta)*Tx_c_new
            Ty_c_tot = theta * Ty_c0 + (1-theta)*Ty_c_new

            T_c_tot = np.vstack([Tx_c_tot.reshape(1,-1), Ty_c_tot.reshape(1,-1)])   

            # Create a FEniCSX expression for the new traction values
            # and interpolate it to the function space V to set bounary conditions for outer domain
            T_c_fun = MyExpression(X[coor_r_out], Y[coor_r_out], T_c_tot, mesh.geometry.dim)
            Traction.interpolate(T_c_fun.eval)


        U_copy = np.copy(U_)
        V_copy = np.copy(V_)
        U_1_copy = np.copy(U_1)
        V_1_copy = np.copy(V_1)

        u_list.append(U_copy)
        v_list.append(V_copy)
        u2_list.append(U_1_copy)
        v2_list.append(V_1_copy)

        break_list.append(sigma_tot[:,0])
        if len(u_list) > 1:
            # L2 error for U and V in outer and inner domains 
            uv_L2 = np.linalg.norm(np.sqrt((u_list[-1] - u_list[-2])**2 + (v_list[-1] - v_list[-2])**2)) 
            uv_L2_2 = np.linalg.norm(np.sqrt((u2_list[-1] - u2_list[-2])**2 + (v2_list[-1] - v2_list[-2])**2))
            print('\n' ,'error', uv_L2 + uv_L2_2)
            error_list.append(uv_L2 + uv_L2_2)

            if  error_list[-1] < 1e-5:

                if ts >= ts_end - 2 and ts < ts_end:
                    plot_disp(X, Y, u.x.array.reshape(-1,3)[:,0],'U_x_FE t=' + str(ts) + ' iter = ' + str(iter), 'n = ' + str(ts))
                    plot_disp(X, Y, u.x.array.reshape(-1,3)[:,1],'U_y_FE t=' + str(ts) + ' iter = ' + str(iter), 'n = ' + str(ts))
                    plot_disp(X1, Y1, U_1, 'U_x_FE-FE t=' + str(ts) + ' iter = ' + str(iter), 'n = ' + str(ts))
                    plot_disp(X1, Y1, V_1, 'U_y_FE-FE t=' + str(ts) + ' iter = ' + str(iter), 'n = ' + str(ts))

                if ts == ts_end:
                    plot_disp(X, Y, u.x.array.reshape(-1,3)[:,0],'U_x_FE t=' + str(ts) + ' iter = ' + str(iter), 'n = ' + str(ts))
                    plot_disp(X, Y, u.x.array.reshape(-1,3)[:,1],'U_y_FE t=' + str(ts) + ' iter = ' + str(iter), 'n = ' + str(ts))
                    plot_disp_real(X1, Y1, s_u_pred,'U_x_NN t=' + str(ts) + ' iter = ' + str(iter), 'n = ' + str(ts))
                    plot_disp_real(X1, Y1, s_v_pred,'U_y_NN t=' + str(ts) + ' iter = ' + str(iter), 'n = ' + str(ts))
                    plot_disp(X1, Y1, s_e_xx_pred_tot,'strain_x_NN t=' + str(ts) + ' iter = ' + str(iter), rf'$\epsilon_{{xx,\mathrm{{FE-NO}},\Omega_{{II}}}}^{{{ts},{iter}}}$')
                    plot_disp(X1, Y1, s_e_yy_pred_tot,'strain_y_NN t=' + str(ts) + ' iter = ' + str(iter), rf'$\epsilon_{{yy,\mathrm{{FE-NO}},\Omega_{{II}}}}^{{{ts},{iter}}}$')
                    
                    # Save the results to files
                    np.savetxt('U ts = ' + str(ts) +'.txt', U_)
                    np.savetxt('V ts = ' + str(ts) +'.txt', V_)
                    np.savetxt('V1 ts = ' + str(ts) +'.txt', s_v_pred)  
                    np.savetxt('U1 ts = ' + str(ts) +'.txt', s_u_pred)
                    np.savetxt('s_e_xx_pred ts = ' + str(ts) +'.txt', s_e_xx_pred_tot)
                    np.savetxt('s_e_yy_pred ts = ' + str(ts) +'.txt', s_e_yy_pred_tot)
                    np.savetxt('s_e_xy_pred ts = ' + str(ts) +'.txt', s_e_xy_pred_tot)
                    np.savetxt(f'error_list_FE_NO_elasto ts=' + str(ts) + ' iter=' + str(iter) + '.txt', error_list)
                if ts >= ts_end:
                    # ---- Outer-domain FE solution: u is already a Function on V, write it directly ----
                    with VTKFile(mesh.comm, f"U_outer_ts_{ts}.pvd", "w") as vtk:
                        vtk.write_function(u, t)

                    # ---- Inner-domain NN displacement: build a vector Function on V2, then write it ----
                    # s_u_pred / s_v_pred are ordered the same as (X1, Y1),
                    # which matches the block order of array.reshape(-1,3) for V2 (CG2 vector space)
                    u2_nn = Function(V2, name="Displacement_NN")
                    u2_nn.x.array[:] = np.hstack((
                        s_u_pred.reshape(-1, 1),
                        s_v_pred.reshape(-1, 1),
                        np.zeros_like(s_u_pred).reshape(-1, 1),
                    )).flatten()
                    with VTKFile(mesh2.comm, f"U_inner_NN_ts_{ts}.pvd", "w") as vtk:
                        vtk.write_function(u2_nn, t)

                    # ---- Inner-domain NN strain: scalar fields, placed in a scalar CG2 space ----
                    Vs = functionspace(mesh2, ("CG", 2))
                    e_xx = Function(Vs, name="e_xx"); e_xx.x.array[:] = s_e_xx_pred_tot
                    e_yy = Function(Vs, name="e_yy"); e_yy.x.array[:] = s_e_yy_pred_tot
                    e_xy = Function(Vs, name="e_xy"); e_xy.x.array[:] = s_e_xy_pred_tot
                    with VTKFile(mesh2.comm, f"strain_NN_ts_{ts}.pvd", "w") as vtk:
                        vtk.write_function([e_xx, e_yy, e_xy], t)
                        
                if ts == 92:
                    end_time = time.time()
                    print(f'FE-NO coupling time for one time: {end_time - start_time} seconds')     

                # Attention: only meet the tolerance, we can break the inner iteration
                # and the stored values are updated to the new values (+ delta_t)
                u_old_0 = np.copy(u_old.x.petsc_vec.array)
                v_old_0 = np.copy(v_old.x.petsc_vec.array)
                a_old_0 = np.copy(a_old.x.petsc_vec.array)

                if ts < ts_end:
                    # use in fenicsx
                    u_old_1 = np.copy(u_old2.x.petsc_vec.array)
                    v_old_1 = np.copy(v_old2.x.petsc_vec.array)
                    a_old_1 = np.copy(a_old2.x.petsc_vec.array)                    
                       
                else:
                    # use in NN
                    u_old_1 = np.copy(u_old2)
                    v_old_1 = np.copy(v_old2)
                    a_old_1 = np.copy(a_old2)

                break



