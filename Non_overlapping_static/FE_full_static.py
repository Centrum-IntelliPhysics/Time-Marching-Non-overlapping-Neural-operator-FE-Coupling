'''
===============================================================================================
FE_full_static.py is to solve the static problem of the full intact square.
-----------------------------------------------------------------------------------------------
The results of displacement u and v serve as the ground truth for the DeepONet training and 
FE-NO coupling
'''

import os
import jax
import jax.numpy as jnp
from jax import grad, vmap
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
from utils import createFolder, plot_disp
import time
#region Save path       
originalDir =os.path.dirname(os.path.abspath(__file__))
print('curent working directory:', originalDir)
os.chdir(os.path.join(originalDir))

foldername = 'static_data_ground_truth_uv_001'  
createFolder(foldername )
os.chdir(os.path.join(originalDir, './'+ foldername + '/')) 

origin_real = os.path.join(originalDir, './'+ foldername + '/')
#### Gmsh generates the geometry 
lc = 0.04

# region gmsh
# Define circle parameters
center = (0.0, 0.5, 0.0)  # Center of the circle
radius = 0.3             # Radius of the circle
num_points = 100      
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

# Generate 2D mesh
gmsh.model.mesh.generate(2)

# Create physical groups for the two regions
# Now we use the actual tags from the fragment operation
outer_region = gmsh.model.addPhysicalGroup(2, [surface_hole1], tag=1)

#gmsh.model.addPhysicalGroup(2, [surface_hole], name=" Hole")


# Create physical groups for the two regions
# Now we use the actual tags from the fragment operation
gmsh.model.setPhysicalName(2, outer_region, "Outer_Square")
gmsh.model.addPhysicalGroup(1, lines1 , name="InnerSquareEdges")# facet tag = 3
gmsh.write("inner_hole.msh")

# Finalize GMSH
gmsh.finalize()




# visualize the mesh
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
plt.show()


# import msh file 
mesh1, cell_markers, facet_markers  = gmshio.read_from_msh("full_square.msh", MPI.COMM_WORLD) 
V = functionspace(mesh1, ("CG", 2, (mesh1.geometry.dim, ))) ###without  (mesh1.geometry.dim, ), it is a scalar space not a vector space 
u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
X, Y = u_geometry[:,0], u_geometry[:,1]

mesh2, cell_markers, facet_markers  = gmshio.read_from_msh("inner_hole.msh", MPI.COMM_WORLD) 
V2 = functionspace(mesh2, ("CG", 2, (mesh2.geometry.dim, ))) 
u_topology1, u_cell_types1, u_geometry1 = plot.vtk_mesh(V2)
X1, Y1 = u_geometry1[:,0], u_geometry1[:,1]
print('X1.shape', X1.shape, 'Y1.shape', Y1.shape)
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


# Find the boundaries and set the boundary conditions
tdim = mesh1.topology.dim
fdim = tdim - 1
domain = mesh1

upper_point = dolfinx.mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 1.5))
uD = np.array([0.01, 0.01, 0], dtype=default_scalar_type)
domain.topology.create_connectivity(fdim, tdim)
boundary_dofs = fem.locate_dofs_topological(V, fdim, upper_point)
bc_upper = fem.dirichletbc(uD, boundary_dofs, V)

bottom_point = dolfinx.mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], -0.5))
uD = np.array([0, 0, 0], dtype=default_scalar_type)
domain.topology.create_connectivity(fdim, tdim)
boundary_dofs = fem.locate_dofs_topological(V, fdim, bottom_point)
bc_bt = fem.dirichletbc(uD, boundary_dofs, V)

bcs = [bc_upper, bc_bt]


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

np.savetxt('strain_xx.txt', epsilon_tot1[:,0])
np.savetxt('strain_xy.txt', epsilon_tot1[:,1])
np.savetxt('strain_yy.txt', epsilon_tot1[:,4])

plot_disp(X, Y, epsilon_tot1[:,0], 'strain_xx',rf'$\epsilon_{{xx,\mathrm{{FE-FE}},\Omega_{{II}}}}$')
plot_disp(X, Y, epsilon_tot1[:,1], 'strain_xy', rf'$\epsilon_{{xy,\mathrm{{FE-FE}},\Omega_{{II}}}}$')
plot_disp(X, Y, epsilon_tot1[:,4], 'strain_yy', rf'$\epsilon_{{yy,\mathrm{{FE-FE}},\Omega_{{II}}}}$')

plot_disp(X,Y,u, 'displacement u', rf'$u_{{x, \mathrm{{FE}}}}$')
plot_disp(X,Y,v,'displacement v', rf'$u_{{y, \mathrm{{FE}}}}$')

# save the data for interpolation to calculate the error 
np.savetxt('u.txt', u)
np.savetxt('v.txt', v)

u_func = Rbf(X, Y, u)
V_func = Rbf(X, Y, v)

plot_disp(X1, Y1, u_func(X1, Y1), 'u_1', rf'$u_{{\mathrm{{FE, \Omega_{{II}}}}}}$')
plot_disp(X1, Y1, V_func(X1, Y1), 'v_1', rf'$v_{{\mathrm{{FE, \Omega_{{II}}}}}}$')

np.savetxt('u1.txt', u_func(X1, Y1))
np.savetxt('v1.txt', V_func(X1, Y1))


index_inner_domain = np.where((X - center[0])**2 + (Y - center[1])**2 <= radius**2 + 1e-5)[0]

e_xx_inner_domain = epsilon_tot1[index_inner_domain, 0]
e_xy_inner_domain = epsilon_tot1[index_inner_domain, 1]
e_yy_inner_domain = epsilon_tot1[index_inner_domain, 4]

plot_disp(X[index_inner_domain], Y[index_inner_domain], e_xx_inner_domain, 'e_xx_inner_domain', rf'$\epsilon_{{xx,\mathrm{{FE}},\Omega_{{II}}}}$')
plot_disp(X[index_inner_domain], Y[index_inner_domain], e_yy_inner_domain, 'e_yy_inner_domain', rf'$\epsilon_{{yy,\mathrm{{FE}},\Omega_{{II}}}}$')
plot_disp(X[index_inner_domain], Y[index_inner_domain], e_xy_inner_domain, 'e_xy_inner_domain', rf'$\epsilon_{{xy,\mathrm{{FE}},\Omega_{{II}}}}$')

e_xx_id_func = Rbf(X[index_inner_domain], Y[index_inner_domain], e_xx_inner_domain)
e_xy_id_func = Rbf(X[index_inner_domain], Y[index_inner_domain], e_xy_inner_domain)
e_yy_id_func = Rbf(X[index_inner_domain], Y[index_inner_domain], e_yy_inner_domain)

np.savetxt('strain_xx1.txt', e_xx_id_func(X1, Y1))
np.savetxt('strain_xy1.txt', e_xy_id_func(X1, Y1))
np.savetxt('strain_yy1.txt', e_yy_id_func(X1, Y1))

plot_disp(X1, Y1, e_xx_id_func(X1, Y1), 'e_xx_1', rf'$\epsilon_{{xx,\mathrm{{FE}},\Omega_{{II}}}}$')
plot_disp(X1, Y1, e_xy_id_func(X1, Y1), 'e_xy_1', rf'$\epsilon_{{xy,\mathrm{{FE}},\Omega_{{II}}}}$')
plot_disp(X1, Y1, e_yy_id_func(X1, Y1), 'e_yy_1', rf'$\epsilon_{{yy,\mathrm{{FE}},\Omega_{{II}}}}$')










        




