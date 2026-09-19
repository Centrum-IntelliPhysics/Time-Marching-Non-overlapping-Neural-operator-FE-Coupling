'''
===============================================================================================
FE_full_static.py is to solve the static problem of the full intact square.
-----------------------------------------------------------------------------------------------
The results of displacement u and v serve as the ground truth for the DeepONet training and 
FE-NO coupling
'''

import os
import jax.numpy as jnp
from jax import grad, vmap, random, jit, config
from mpi4py import MPI
from dolfinx.io import XDMFFile, gmshio
import gmsh
from dolfinx.fem import functionspace, Function 
from dolfinx import mesh
from dolfinx import fem
import numpy as np
import ufl
from dolfinx import default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import pyvista
import matplotlib.pyplot as plt
from dolfinx import plot
from dolfinx import io
from pathlib import Path
import pickle
from scipy.interpolate import Rbf, interp1d, griddata
import math 
from matplotlib.ticker import ScalarFormatter
import dolfinx 
import meshio
from utils import createFolder, plot_disp, plot_bc
import time 
from tqdm import trange

#region Save path       
originalDir = os.path.dirname(os.path.abspath(__file__)) 
print('curent working directory:', originalDir)
os.chdir(os.path.join(originalDir))

foldername = 'static_data_ground_truth_uv_001_RBF_N_1000_CG2'  
createFolder(foldername )
os.chdir(os.path.join(originalDir, './'+ foldername + '/')) 

origin_real = os.path.join(originalDir, './'+ foldername + '/')
#### Gmsh generates the geometry 
lc = 0.04

# region gmsh
# Define circle parameters
center = (0.0, 0.5, 0.0)  # Center of the circle
radius = 0.3             # Radius of the circle
radius1 = 0.3 - 0.05    # Radius of the inner circle
num_points = 100      # Number of points on the circumference #so important !!!0 
num_points1 = 100  


# region Hole 
# initialize GMSH
gmsh.initialize()
gmsh.model.add("model")
# Define inner square
# add the square points
# add the square points
gmsh.model.occ.addPoint(-1, -0.5, 0, lc, 1)
gmsh.model.occ.addPoint(1, -0.5, 0, lc, 2)
gmsh.model.occ.addPoint(1, 1.5, 0, lc, 3)
gmsh.model.occ.addPoint(-1, 1.5, 0, lc, 4)

# add the square lines
line1 = gmsh.model.occ.addLine(1, 2)
line2 = gmsh.model.occ.addLine(2, 3)
line3 = gmsh.model.occ.addLine(3, 4)
line4 = gmsh.model.occ.addLine(4, 1)

# create a loop from the lines
rec_loop = gmsh.model.occ.addCurveLoop([line1, line2, line3, line4])

# create a surface from the loop
background = gmsh.model.occ.addPlaneSurface([rec_loop])
#background = gmsh.model.occ.addRectangle(-1, -0.5, 0, 2, 2, 100)
gmsh.model.occ.synchronize()

points1 = []
for i in range(num_points):
    angle1 = 2 * math.pi * i / num_points
    x1 = center[0] + radius * math.cos(angle1)
    y1 = center[1] + radius * math.sin(angle1)
    points1.append(gmsh.model.occ.addPoint(x1, y1, center[2], lc))

# Create lines between consecutive points
lines1 = []
for i in range(num_points):
    p1 = points1[i]
    p2 = points1[(i + 1) % num_points]
    lines1.append(gmsh.model.occ.addLine(p1, p2))

# Create a closed loop
circle_loop1 = gmsh.model.occ.addCurveLoop(lines1)

# Create a surface from the loop
surface_hole1 = gmsh.model.occ.addPlaneSurface([circle_loop1])


gmsh.model.occ.synchronize()

    # Fragment the surfaces
out_dim_tags, out_dim_tags_map = gmsh.model.occ.fragment(
    [(2,background),(2, surface_hole1)],  # Target entities
    []  # Tool entities
)

# Synchronize after boolean operation
gmsh.model.occ.synchronize()

# Generate 2D mesh
gmsh.model.mesh.generate(2)

# Get all surfaces after fragmentation
surfaces = gmsh.model.getEntities(2) #2 --> 2d surface

# Create physical groups for the two regions
# Now we use the actual tags from the fragment operation
outer_region = gmsh.model.addPhysicalGroup(2, [surfaces[0][1]], tag=1)
inner_region = gmsh.model.addPhysicalGroup(2, [surfaces[1][1]], tag=2)

# Create physical groups for the two regions
# Now we use the actual tags from the fragment operation
gmsh.model.setPhysicalName(2, outer_region, "Outer_Square")
gmsh.model.setPhysicalName(2, inner_region, "Inner_Square")

gmsh.model.addPhysicalGroup(1, [line1, line2, line3, line4] , name="LargerSquareEdges")# facet tag = 3
gmsh.model.addPhysicalGroup(1, lines1 , name="InnerSquareEdges")# facet tag = 3

# Write the mesh to a file (optional)
gmsh.write("full_square.msh")
# Finalize GMSH
gmsh.finalize()


# region Hole 
# initialize GMSH
gmsh.initialize()
gmsh.model.add("model")
# Define inner square
# add the square points
points1 = []
for i in range(num_points):
    angle1 = 2 * math.pi * i / num_points
    x1 = center[0] + radius * math.cos(angle1)
    y1 = center[1] + radius * math.sin(angle1)
    points1.append(gmsh.model.occ.addPoint(x1, y1, center[2], lc))

# Create lines between consecutive points
lines1 = []
for i in range(num_points):
    p1 = points1[i]
    p2 = points1[(i + 1) % num_points]
    lines1.append(gmsh.model.occ.addLine(p1, p2))

# Create a closed loop
circle_loop1 = gmsh.model.occ.addCurveLoop(lines1)

# Create a surface from the loop
surface_hole1 = gmsh.model.occ.addPlaneSurface([circle_loop1])


gmsh.model.occ.synchronize()

# Synchronize after boolean operation
gmsh.model.occ.synchronize()

# Generate 2D mesh
gmsh.model.mesh.generate(2)

# Get all surfaces after fragmentation
surfaces = gmsh.model.getEntities(2)

# Create physical groups for the two regions
# Now we use the actual tags from the fragment operation
outer_region = gmsh.model.addPhysicalGroup(2, [surfaces[0][1]], tag=1)

# Create physical groups for the two regions
# Now we use the actual tags from the fragment operation
gmsh.model.setPhysicalName(2, outer_region, "Outer_Square")
gmsh.model.addPhysicalGroup(1, lines1 , name="InnerSquareEdges")
gmsh.write("inner_hole.msh")

# Finalize GMSH
gmsh.finalize()



'''# visualize the mesh
msh = meshio.read("full_square.msh")
points = msh.points
cells = msh.cells_dict["triangle"]  
plt.figure(figsize=(8, 8))
for cell in cells:
    polygon = points[cell]
    polygon = np.vstack([polygon, polygon[0]])
    plt.plot(polygon[:, 0], polygon[:, 1], 'k-', linewidth=0.5) 
plt.gca().set_aspect('equal')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gmsh Mesh Visualization')
plt.savefig('Full square' + ".jpg", dpi=700)
plt.show()'''


# import msh file 
mesh1, cell_markers, facet_markers  = gmshio.read_from_msh("full_square.msh", MPI.COMM_WORLD) 
V = functionspace(mesh1, ("CG", 2, (mesh1.geometry.dim, ))) ###without  (mesh1.geometry.dim, ), it is a scalar space not a vector space 
u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
X, Y = u_geometry[:,0], u_geometry[:,1]

mesh2, _, _  = gmshio.read_from_msh("inner_hole.msh", MPI.COMM_WORLD) 
V2 = functionspace(mesh2, ("CG", 2, (mesh2.geometry.dim, ))) ###without  (mesh1.geometry.dim, ), it is a scalar space not a vector space 
_, _, u_geometry2 = plot.vtk_mesh(V2)
X1, Y1 = u_geometry2[:,0], u_geometry2[:,1]
np.savetxt('X.txt', X)
np.savetxt('Y.txt', Y)
np.savetxt('X1.txt', X1)
np.savetxt('Y1.txt', Y1)

# save the outer circle
def on_cricle(x):
    xc = x[0][np.where(np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius**2, 1e-4, 1e-4))]
    yc = x[1][np.where(np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius**2, 1e-4, 1e-4))]
    return xc, yc 
x_c_out, y_c_out = on_cricle(np.vstack([X.reshape(1,-1), Y.reshape(1,-1)]))

index_y_positive = np.where(y_c_out >= center[1])[0]
x_c_out_pos = x_c_out[index_y_positive]
y_c_out_pos = y_c_out[index_y_positive]

index_resort = np.argsort(x_c_out_pos)[::-1]  # Resort the indices in descending order
x_c_out_pos = x_c_out_pos[index_resort]
y_c_out_pos = y_c_out_pos[index_resort]

index_y_negative = np.where(y_c_out < center[1])[0]
x_c_out_neg = x_c_out[index_y_negative]
y_c_out_neg = y_c_out[index_y_negative]

index_resort_neg = np.argsort(x_c_out_neg) # Resort the indices in ascending order
x_c_out_neg = x_c_out_neg[index_resort_neg]
y_c_out_neg = y_c_out_neg[index_resort_neg]

x_c_out = np.concatenate((x_c_out_pos, x_c_out_neg))
y_c_out = np.concatenate((y_c_out_pos, y_c_out_neg))

np.savetxt('x_c1_out.txt', x_c_out)
np.savetxt('y_c1_out.txt', y_c_out) 


index_c = []
for i in range(x_c_out.shape[0]):
    index_1 = np.where(np.isclose(X, x_c_out[i])&np.isclose(Y, y_c_out[i]))[0]
    index_c.append(index_1)

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
    subkey_x = random.split(subkey1,2)
    subkey_y = random.split(subkey2,2)
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
    u_up, _ = f_fn_u(yy)
    v_up, _ = f_fn_v(xx)
    # Output sensor locations and measurements
    idx = random.randint(subkey, (P,2), 0, max(Nx,Ny))
    h = jnp.concatenate([x[idx[:,0]][:,None], y[idx[:,1]][:,None]], axis = 1)

    return  u_up, v_up


m = 200
P = m
Nx, Ny = m, m
key = random.PRNGKey(int(time.time()))
length_sacle = 0.2
N = 1000 # number of samples
keys = random.split(key, N)
config.update("jax_enable_x64", True)
u_up_RBF, v_up_RBF =  vmap(solve_ADR, (0, None, None, None, None, None))(keys, Nx, Ny, m, P, length_sacle)
config.update("jax_enable_x64", False)



### linear materials parameters 
E = 0.210e-2 
nu = 0.3 

def epsilon(u):
    return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)


def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

def depsilon_xy_dx(u):
    return ufl.Dx(epsilon(u)[0, 1], 0)  # Second derivative of epsilon_xy with respect to x and y

mu = E/(2 * (1 + nu))
lambda_ = E*nu/((1 + nu)*(1 - 2*nu))

# region Expression func
class MyExpression_top_out:
    def __init__(self, x0, y0, value0, value1, V_dim):
        self.x0 = x0
        self.y0 = y0
        self.value0  = value0
        self.value1 = value1
        self.V_dim = V_dim 
        self.RBF_0 = Rbf(x0, y0, value0)       
        self.RBF_1  = Rbf(x0, y0, value1)
        
    def eval(self, x):
        #print(x[0], x[1], x[2]) 
        values = np.zeros((self.V_dim, x.shape[1]))
        values[0] = np.where(np.isclose(x[1], 1.5, 1e-4, 1e-4), self.RBF_0(x[0], x[1]), 0)
        values[1] = np.where(np.isclose(x[1], 1.5, 1e-4, 1e-4), self.RBF_1(x[0], x[1]), 0)
        return values 
    
def top(x):
    return np.isclose(x[1], 1.5)

# Find the boundaries and set the boundary conditions
tdim = mesh1.topology.dim
fdim = tdim - 1
domain = mesh1

top_facets = mesh.locate_entities_boundary(domain, fdim, top)
u_up = fem.Function(V)
# Locate DOFs on the boundary
boundary_dofs = fem.locate_dofs_topological(V, fdim, top_facets)
# Create Dirichlet BC
bc_u_top = fem.dirichletbc(u_up, boundary_dofs)

x_up = np.linspace(-1, 1, m)
y_up = np.ones_like(x_up) * 1.5

upper_b = dolfinx.mesh.locate_entities_boundary(mesh1, fdim, top)
uD_up = Function(V)

u_list, v_list, e_xx_list, e_xy_list, e_yy_list = [], [], [], [], []

start_time = time.time()
for i in trange(N):
    uD_up_value = u_up_RBF[i,:]
    vD_up_value = v_up_RBF[i,:]

    uD_up_fun = MyExpression_top_out(x_up, y_up, uD_up_value, vD_up_value, mesh1.geometry.dim)
    uD_up.interpolate(uD_up_fun.eval)        
    boundary_dofs = fem.locate_dofs_topological(V, fdim, upper_b)
    bc_top = fem.dirichletbc(uD_up, boundary_dofs)

    bottom_point = dolfinx.mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], -0.5))
    uD = np.array([0, 0, 0], dtype=default_scalar_type)
    domain.topology.create_connectivity(fdim, tdim)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, bottom_point)
    bc_bt = fem.dirichletbc(uD, boundary_dofs, V)

    bcs = [bc_top, bc_bt]


    # Define the variational problem
    T = fem.Constant(domain, default_scalar_type((0, 0, 0)))
    ds = ufl.Measure("ds", domain=domain)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(domain, default_scalar_type((0, 0, 0)))
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds


    problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}) #PETSc（Portable, Extensible Toolkit for Scientific Computation）

    # Solve the problem
    uh = problem.solve()
    u_values = uh.x.array.real
    u_tot = u_values.reshape(-1,3)
    u, v, w= u_tot[:,0], u_tot[:,1], u_tot[:,2]

    # get the strain tensor
    Function_space_for_epsilon = fem.functionspace(mesh1, ("CG", 2, (3, 3)))        
    expr= fem.Expression(epsilon(uh), Function_space_for_epsilon.element.interpolation_points())        
    epsilon_values = Function(Function_space_for_epsilon) 
    epsilon_values.interpolate(expr)
    epsilon_tot1 = epsilon_values.x.array.reshape(-1,9)

    '''if i == 0:
        plot_disp(X, Y, u, 'U', rf'$u_{{\mathrm{{FE}}}}$')
        plot_disp(X, Y, v, 'V', rf'$v_{{\mathrm{{FE}}}}$')
        plot_bc(np.linspace(0,1,X[index_c].shape[0]), u[index_c], 'u index')
        plot_bc(np.linspace(0,1,X[index_c].shape[0]), v[index_c], 'v index')
        plot_bc(np.linspace(0,1,X[index_c].shape[0]), epsilon_tot1[index_c,0], 'strain_xx index')
        plot_bc(np.linspace(0,1,X[index_c].shape[0]), epsilon_tot1[index_c,1], 'strain_xy index')
        plot_bc(np.linspace(0,1,X[index_c].shape[0]), epsilon_tot1[index_c,4], 'strain_yy index')
        plot_bc(np.linspace(0,1,X[index_c].shape[0]), u_up_RBF[i,:], 'top u RBF index')
        plot_bc(np.linspace(0,1,X[index_c].shape[0]), v_up_RBF[i,:], 'top v RBF index')'''
    
    u_list.append(u[index_c].reshape(-1,1))
    v_list.append(v[index_c].reshape(-1,1))
    e_xx_list.append(epsilon_tot1[index_c,0].reshape(-1,1))
    e_xy_list.append(epsilon_tot1[index_c,1].reshape(-1,1))
    e_yy_list.append(epsilon_tot1[index_c,4].reshape(-1,1))

u_c_value = np.hstack(u_list)
v_c_value = np.hstack(v_list)
epsilon_xx_c_value = np.hstack(e_xx_list)
epsilon_xy_c_value = np.hstack(e_xy_list)
epsilon_yy_c_value = np.hstack(e_yy_list)

# save the data
np.savetxt('u_c_value.txt', u_c_value)
np.savetxt('v_c_value.txt', v_c_value)
np.savetxt('epsilon_xx_c_value.txt', epsilon_xx_c_value)
np.savetxt('epsilon_xy_c_value.txt', epsilon_xy_c_value)
np.savetxt('epsilon_yy_c_value.txt', epsilon_yy_c_value)

end_time = time.time()
print('Total time for solving the problem:', end_time - start_time)




time1 = time.time()
s_xx_func = Rbf(X, Y, epsilon_tot1[:,0])
time2 = time.time()
print('RBF time for s_xx_func:', time2 - time1)
plot_disp(X1, Y1, s_xx_func(X1, Y1), 's_xx_1_domain', rf'$\epsilon_{{xx,\mathrm{{FE-FE}},\Omega_{{II}}}}$')










'''plot_disp(X, Y, epsilon_tot1[:,0], 'strain_xx',rf'$\epsilon_{{xx,\mathrm{{FE-FE}},\Omega_{{II}}}}$')
plot_disp(X, Y, epsilon_tot1[:,1], 'strain_xy', rf'$\epsilon_{{xy,\mathrm{{FE-FE}},\Omega_{{II}}}}$')
plot_disp(X, Y, epsilon_tot1[:,4], 'strain_yy', rf'$\epsilon_{{yy,\mathrm{{FE-FE}},\Omega_{{II}}}}$')

plot_disp(X,Y,u, 'displacement u', rf'$u_{{\mathrm{{FE}}}}$')
plot_disp(X,Y,v,'displacement v', rf'$v_{{\mathrm{{FE}}}}$')'''

'''# save the data for interpolation to calculate the error 
np.savetxt('u.txt', u)
np.savetxt('v.txt', v)

u_func = Rbf(X, Y, u)
V_func = Rbf(X, Y, v)

plot_disp(X1, Y1, u_func(X1, Y1), 'u_1', rf'$u_{{\mathrm{{FE, \Omega_{{II}}}}}}$')
plot_disp(X1, Y1, V_func(X1, Y1), 'v_1', rf'$v_{{\mathrm{{FE, \Omega_{{II}}}}}}$')

np.savetxt('u1.txt', u_func(X1, Y1))
np.savetxt('v1.txt', V_func(X1, Y1))'''

'''function_space_for_depsilon = fem.functionspace(mesh1, ("CG", 2, (1, 1)))
expr_depsilon_xy_dx = fem.Expression(depsilon_xy_dx(uh), function_space_for_depsilon.element.interpolation_points())
depsilon_xy_dx_values = Function(function_space_for_depsilon)
depsilon_xy_dx_values.interpolate(expr_depsilon_xy_dx)
depsilon_xy_dx_values = depsilon_xy_dx_values.x.array.reshape(-1, 1)

plot_disp(X, Y, depsilon_xy_dx_values, 'd2epsilon_xy_dx', rf'$\frac{{\partial \epsilon_{{xy,\mathrm{{FE-FE}},\Omega_{{II}}}}}}{{ \partial x}}$')

depsilon_xy_dx_func = Rbf(X, Y, depsilon_xy_dx_values)

plot_disp(X1, Y1, depsilon_xy_dx_func(X1, Y1), 'd2epsilon_xy_dxdy', rf'$\frac{{\partial \epsilon_{{xy,\mathrm{{FE-FE}},\Omega_{{II}}}}}}{{ \partial x}}$')
np.savetxt('depsilon_xy_dx.txt', depsilon_xy_dx_func(X1, Y1))'''

'''time1 = time.time()
s_xx_func = Rbf(X, Y, epsilon_tot1[:,0])
time2 = time.time()
print('RBF time for s_xx_func:', time2 - time1)
#s_xy_func = Rbf(X, Y, epsilon_tot1[:,1])
#s_yy_func = Rbf(X, Y, epsilon_tot1[:,4])

plot_disp(X1, Y1, s_xx_func(X1, Y1), 's_xx_1_domain', rf'$\epsilon_{{xx,\mathrm{{FE-FE}},\Omega_{{II}}}}$')
x_list = []
y_list = []
for i in range(num_points * 2):
    angle = 2 * math.pi * i / (num_points * 2)
    x = center[0] + radius * math.cos(angle)
    y = center[1] + radius * math.sin(angle)
    x_list.append(x)
    y_list.append(y)  

x_c = np.array(x_list).flatten()
y_c = np.array(y_list).flatten()
s_xx_func_c = s_xx_func(x_c, y_c)
plot_bc(np.linspace(0,1,x_c.shape[0]), s_xx_func_c , 's_xx_1_index')

np.savetxt('strain_xx_c_interpolate.txt', s_xx_func_c)'''


'''plot_disp(X1, Y1, s_xy_func(X1, Y1), 's_xy_1', rf'$\epsilon_{{xy,\mathrm{{FE-FE}},\Omega_{{II}}}}$')
plot_disp(X1, Y1, s_yy_func(X1, Y1), 's_yy_1', rf'$\epsilon_{{yy,\mathrm{{FE-FE}},\Omega_{{II}}}}$')

np.savetxt('strain_xx1.txt', s_xx_func(X1, Y1))
np.savetxt('strain_xy1.txt', s_xy_func(X1, Y1))
np.savetxt('strain_yy1.txt', s_yy_func(X1, Y1))

print('averate strain:', np.mean(s_xx_func(X1, Y1)), np.mean(s_xy_func(X1, Y1)), np.mean(s_yy_func(X1, Y1)))'''










        




