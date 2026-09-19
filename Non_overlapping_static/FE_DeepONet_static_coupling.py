'''
======================================================================
The FE-DeepONet or FE-NO coupling framework for linear elastic materials 
under static loading conditions.
----------------------------------------------------------------------
The non-overlapping boundary is used. 
The domain decomposition and Schwartz alternating method is used by
exchanging the displacement at the non-overlapping boundary.
======================================================================
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
import time
from utils import plot_disp, plot_relative_error, createFolder


#region Save path       
originalDir = os.path.dirname(os.path.abspath(__file__))
print('current working directory is ' + originalDir)
os.chdir(os.path.join(originalDir))

foldername = 'FE_DeepONet_static_non_overlapping_coupling_results_real'  
createFolder(foldername )
os.chdir(os.path.join(originalDir, './'+ foldername + '/')) 

originalDir_real = os.path.join(originalDir, './'+ foldername + '/')
#### Gmsh generates the geometry 
lc = 0.04

# region gmsh
#===================================================
# Using Gmsh to generate two regions: 
# 1. Outer region with a square and a hole (FE region)
# 2. Inner region with a disk (DeepONet region)
# overlapping region exists between the two regions
#---------------------------------------------------
# Inner region mesh is not used for FE, it provides
# the coordinates for the DeepONet model
#===================================================

# Define circle parameters
center = (0.0, 0.5, 0.0)  # Center of the circle
radius = 0.3             # Radius of the circle
num_points = 100      # Number of points on the circumference #so important !!!0 
num_points1 = 100  

gmsh.initialize()
gmsh.model.add("model")
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

# cut the hole 
# Define the points for the circle
points = []
for i in range(num_points):
    angle = 2 * math.pi * i / num_points
    x = center[0] + radius * math.cos(angle)
    y = center[1] + radius * math.sin(angle)
    points.append(gmsh.model.occ.addPoint(x, y, center[2], lc))

# Create lines between consecutive points
lines = []
for i in range(num_points):
    p1 = points[i]
    p2 = points[(i + 1) % num_points]
    lines.append(gmsh.model.occ.addLine(p1, p2))

# Create a closed loop
circle_loop = gmsh.model.occ.addCurveLoop(lines)

# Create a surface from the loop
surface_hole = gmsh.model.occ.addPlaneSurface([circle_loop])

# Cut the larger square with the hole
gmsh.model.occ.cut([(2, background)], [(2, surface_hole)])
# Synchronize the model to GMSH
gmsh.model.occ.synchronize()


# Create physical groups for the two regions
# Now we use the actual tags from the fragment operation
outer_region = gmsh.model.addPhysicalGroup(2, [background], tag=1)
# Generate 2D mesh
gmsh.model.mesh.generate(2)

gmsh.model.addPhysicalGroup(1, [line1, line2, line3, line4], name="LargerSquareEdges") # facet tag = 7
gmsh.model.addPhysicalGroup(1, lines, name="smallHoleEdges")# facet tag = 9



# Write the mesh to a file (optional)
gmsh.write("outer_region.msh")

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



import meshio

# read .msh file
'''msh = meshio.read("outer_region.msh")
# extract points and cell information
points = msh.points
cells = msh.cells_dict["triangle"]  
plt.figure(figsize=(8, 8))

#plot every triangle 
for cell in cells:
    polygon = points[cell]
    polygon = np.vstack([polygon, polygon[0]])
    plt.plot(polygon[:, 0], polygon[:, 1], 'k-', linewidth=0.5)  # 用黑色线条绘制

#set axis 
plt.gca().set_aspect('equal')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gmsh Mesh Visualization2')
plt.savefig('outer_region' + ".jpg", dpi=700)
plt.show()

# read .msh file 
msh = meshio.read("inner_hole.msh")
# extract points and cell information
points = msh.points
cells = msh.cells_dict["triangle"]  
# plot the mesh 
plt.figure(figsize=(8, 8))

# plot every triangle
for cell in cells:
    polygon = points[cell]
    polygon = np.vstack([polygon, polygon[0]])
    plt.plot(polygon[:, 0], polygon[:, 1], 'k-',linewidth=0.5)  # 用黑色线条绘制

plt.gca().set_aspect('equal')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gmsh Mesh Visualization')
plt.savefig('inner_hole' + ".jpg", dpi=700)
plt.show()'''

#region FEM 
import dolfinx 

def on_cricle(x):
    xc = x[0][np.where(np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius**2, 1e-4, 1e-4))]
    yc = x[1][np.where(np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius**2, 1e-4, 1e-4))]
    return xc, yc 

# import gmsh file
mesh1, cell_markers, facet_markers  = gmshio.read_from_msh("outer_region.msh", MPI.COMM_WORLD) 
num_cells_global = mesh1.topology.index_map(mesh1.topology.dim).size_global
print(f"global number of elements: {num_cells_global}")
V = functionspace(mesh1, ("CG", 2, (mesh1.geometry.dim, ))) ###without  (mesh1.geometry.dim, ), it is a scalar space not a vector space 

u_topology1, u_cell_types1, u_geometry1 = plot.vtk_mesh(V)
X, Y = u_geometry1[:,0], u_geometry1[:,1]
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

coor_r_out = np.array([np.where((X == x_val) & (Y == y_val))[0] 
                    for x_val, y_val in zip(x_c_out, y_c_out)])[:,0]




print('mesh number', mesh1.topology.index_map(2).size_local)
print('DOF number', V.dofmap.index_map.size_local)

### linear materials parameters 
E = 1000 # Young's modulus 
nu = 0.3    # Poisson's ratio

# strain function in ufl(a domain specific language for declaration of finite element discretizations of variational forms)
def epsilon(u):
    return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

# stress function in ufl
def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

mu = E/(2 * (1 + nu))
lambda_ = E*nu/((1 + nu)*(1 - 2*nu))

# Find the boudnary and define the boundary conditions for outer region 
tdim = mesh1.topology.dim
fdim = tdim - 1
domain = mesh1
upper_point = dolfinx.mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 1.5))
uD = np.array([0.01, 0.01, 0], dtype=default_scalar_type)
domain.topology.create_connectivity(fdim, tdim)
boundary_dofs = fem.locate_dofs_topological(V, fdim, upper_point)
bc_upper = fem.dirichletbc(uD, boundary_dofs, V)

upper_point = dolfinx.mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], -0.5))
uD = np.array([0, 0, 0], dtype=default_scalar_type)
domain.topology.create_connectivity(fdim, tdim)
boundary_dofs = fem.locate_dofs_topological(V, fdim, upper_point)
bc_bt = fem.dirichletbc(uD, boundary_dofs, V)

bcs = [bc_upper, bc_bt]


# region Mypression
#==================================================
# MyExpression is used to interpolate the displacement at the boundary
#==================================================

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
        
        values[0] = np.where(np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius**2, 1e-4, 1e-4), self.Tx_fun(x[0], x[1]), 0)
        values[1] = np.where(np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius**2, 1e-4, 1e-4), self.Ty_fun(x[0], x[1]), 0)
                             
        return values 


# define the variational problem in FEM
T = fem.Constant(domain, default_scalar_type((0, 0, 0)))
Traction = fem.Function(V)
ds = ufl.Measure("ds", domain=domain)
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(domain, default_scalar_type((0, 0, 0)))
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds


# inner hole boundary
mesh2, cell_markers, facet_markers  = gmshio.read_from_msh("inner_hole.msh", MPI.COMM_WORLD) 
num_cells_global = mesh2.topology.index_map(mesh2.topology.dim).size_global
print(f"global number of elements: {num_cells_global}")
V2 = functionspace(mesh2, ("CG", 2, (mesh2.geometry.dim, ))) 
u_topology1, u_cell_types1, u_geometry1 = plot.vtk_mesh(V2)
X1, Y1 = u_geometry1[:,0], u_geometry1[:,1]
#print('X1.shao  ', X1.shape, 'Y1.shao', Y1.shape)
x_c1_out, y_c1_out = on_cricle(np.vstack([X1.reshape(1,-1), Y1.reshape(1,-1)]))
index_y_positive = np.where(y_c1_out >= center[1])[0]
x_c1_out_pos = x_c1_out[index_y_positive]
y_c1_out_pos = y_c1_out[index_y_positive]

index_resort = np.argsort(x_c1_out_pos)[::-1]  # Resort the indices in descending order
x_c1_out_pos = x_c1_out_pos[index_resort]
y_c1_out_pos = y_c1_out_pos[index_resort]

index_y_negative = np.where(y_c1_out < center[1])[0]
x_c1_out_neg = x_c1_out[index_y_negative]
y_c1_out_neg = y_c1_out[index_y_negative]

index_resort_neg = np.argsort(x_c1_out_neg) # Resort the indices in ascending order
x_c1_out_neg = x_c1_out_neg[index_resort_neg]
y_c1_out_neg = y_c1_out_neg[index_resort_neg]

x_c1_out = np.concatenate((x_c1_out_pos, x_c1_out_neg))
y_c1_out = np.concatenate((y_c1_out_pos, y_c1_out_neg))

coor_r1_out = np.array([np.where((X1 == x_val) & (Y1 == y_val))[0] 
                    for x_val, y_val in zip(x_c1_out, y_c1_out)])[:,0]
np.savetxt('X1.txt', X1)
np.savetxt('Y1.txt', Y1)
np.savetxt('xc1_out.txt', x_c1_out)
np.savetxt('yc1_out.txt', y_c1_out)

# region Measure ds 
#==================================================
# Define the measure of the traction boundary
#==================================================
def circle_boundary(x):

    return np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius**2, 1e-4, 1e-4)

boundary_facets_c = mesh.locate_entities_boundary(
    domain, domain.topology.dim-1, circle_boundary)

marked_facets_c = np.hstack([boundary_facets_c])
marked_values_c = np.hstack([np.full_like(boundary_facets_c, 100)])

boundary_tags_c = mesh.meshtags(domain, domain.topology.dim-1, 
                             marked_facets_c, marked_values_c)

ds = ufl.Measure("ds", domain=mesh1, subdomain_data=boundary_tags_c) 


# region DeepONet 
#=========================================================
# Pretrained DeepONet model is used to predict the displacement
# for the inner region receiving the boundary displacement from FEM
#=========================================================


from prepare_DeepONet_static_uv_strain_bcs_test5 import PI_DeepONet
m = num_points1 * 2 
d = 2 # dimension of the input data
ela_model = dict()
ela_model['E'] = E #1000 
ela_model['nu'] = nu

branch_layers =  [2*m, 100, 100, 100, 100, 800]
trunk_layers =  [d, 100, 100, 100, 100, 800]
model = PI_DeepONet(branch_layers, trunk_layers, **ela_model)

os.chdir(os.path.join(originalDir, './' + 'prepare_DeepONet_static_linear_elastic_test5' + '/'))
print(os.getcwd())

# load the pretrained model parameters
with open('DeepONet_static.pkl', 'rb') as f:
    params = pickle.load(f)

os.chdir(originalDir_real)


start_time = time.time()

u_list, v_list, u2_list, v2_list =[],[],[],[]
break_list = []
error_list = []
#region Coupling 
niter = 1000
theta = 0.5 # relaxation coefficient
#### Coupling  
for iter in range(0, niter):
    ## FEM 
    if iter == 0:
        problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uh = problem.solve()### first iteration, assume T at interface = 0 
        print('success initiation')
    
    else:     
        # interpolate to handshake 
        Traction.interpolate(T_fun.eval) 

        f = fem.Constant(mesh1, default_scalar_type((0, 0, 0)))
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        a = ufl.inner(sigma(u), epsilon(v))*ufl.dx
        L = ufl.dot(f, v)*ufl.dx  + ufl.dot(Traction, v) * ds(100) # 100 is the tag for the interface boundary condition

        problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uh = problem.solve()
                
        print('success loop =' + f'{iter}')
        
    u_values = uh.x.array.real
    u_tot = u_values.reshape(-1,3)
    u, v, w= u_tot[:,0], u_tot[:,1], u_tot[:,2]
    
    # Extract displacment at outer circle 
    u_c = u[coor_r_out].reshape(1,-1)
    v_c = v[coor_r_out].reshape(1,-1)
    
    gdim = mesh1.geometry.dim
    ### strain is 3*3  (gdim, gdim)
    Function_space_for_strain = fem.functionspace(mesh1, ("CG", 2, (gdim, gdim)))
    expr= fem.Expression(epsilon(uh), Function_space_for_strain.element.interpolation_points())
    strain_values = Function(Function_space_for_strain)
    strain_values.interpolate(expr)
    strain_tot = strain_values.x.array.reshape(-1,9)

    ### stress is 3*3  (gdim, gdim)
    Function_space_for_sigma = fem.functionspace(mesh1, ("CG", 2, (gdim, gdim)))        
    expr= fem.Expression(sigma(uh), Function_space_for_sigma.element.interpolation_points())        
    sigma_values = Function(Function_space_for_sigma) 
    
    preA = sigma_values.x.array
    sigma_values.interpolate(expr)
    sigma_tot = sigma_values.x.array.reshape(-1,9)
    ###normal vector 
    x_c, y_c = X[coor_r_out], Y[coor_r_out]
    # attention the '-' sign, because the normal vector is pointing outward the circle
    n1 = -x_c/((x_c**2 + (y_c-0.5)**2)**0.5)
    n2 = -(y_c-0.5)/((x_c**2 + (y_c-0.5)**2)**0.5)


    # region Handshake
    # interpolation the displacement from the outer region to the inner region
    uc_fun = Rbf(x_c_out, y_c_out, u_c)
    vc_fun = Rbf(x_c_out, y_c_out, v_c)
    u2c = uc_fun(x_c1_out, y_c1_out).reshape(1,-1)
    v2c = vc_fun(x_c1_out, y_c1_out).reshape(1,-1)

    u_test = np.hstack([u2c, v2c]).reshape(1,-1) 
    v_test = np.hstack([u2c, v2c]).reshape(1,-1)
    hc_test = np.hstack([X[coor_r_out].reshape(-1,1), Y[coor_r_out].reshape(-1,1)])
    # predict the displacement at the inner boundary
    #=================================================
    # the u_test and v_test are the displacement at
    # the outer boundary as boundary condition
    # the hc_test is the coordinate of the inner boundary
    #=================================================
    s_uc_pred_in, s_vc_pred_in, s_e_xx_pred, s_e_yy_pred, s_e_xy_pred = model.predict_s(params, u_test, v_test, hc_test)

    #displacement at inner boundary 
    u_c2 = s_uc_pred_in.reshape(1,-1)
    v_c2 = s_vc_pred_in.reshape(1,-1)
    s_e_xx_pred = s_e_xx_pred.reshape(1,-1)
    s_e_yy_pred = s_e_yy_pred.reshape(1,-1)
    s_e_xy_pred = s_e_xy_pred.reshape(1,-1)
    gdim = mesh2.geometry.dim

    #==================================================
    # Assemble the traction at the non-overlapping boundary
    #==================================================

    fdim2 = mesh2.topology.dim -1 
    para1 = E/((1 + nu)*(1 - 2*nu))
    T_x1 = (para1 * ((1 - nu) * s_e_xx_pred + nu * s_e_yy_pred) * n1\
                             + para1 * ((1 - 2 * nu) * s_e_xy_pred)* n2)
    T_x2 = (para1 * ((1 - 2 * nu) * s_e_xy_pred) * n1\
                            + para1 * ((1 - nu) * s_e_yy_pred + nu * s_e_xx_pred)* n2)
    
    T_c_new = np.vstack([T_x1.reshape(1,-1), T_x2.reshape(1,-1)])
    # the traction from the outer region
    # T = sigma . n
    T_c0 =  np.vstack([(sigma_tot[coor_r_out, 0]*n1 + sigma_tot[coor_r_out, 1]*n2).reshape(1,-1), 
                        (sigma_tot[coor_r_out, 1]*n1 + sigma_tot[coor_r_out, 4]*n2).reshape(1,-1)])

    T_c = theta * T_c0 + (1-theta)*T_c_new

    # use MyExpression_inner_hole to interpolate the displacement at the inner boundary (from inner hole to outer region)
    T_fun = MyExpression(X[coor_r_out], Y[coor_r_out], T_c, mesh1.geometry.dim)
    h_test = np.hstack([X1.reshape(-1,1), Y1.reshape(-1,1)])
    s_u_pred_tot, s_v_pred_tot, s_e_xx_pred_tot, s_e_yy_pred_tot, s_e_xy_pred_tot = model.predict_s(params, u_test, v_test, h_test)

    
    if iter ==0 :
        np.savetxt('X.txt', X)
        np.savetxt('Y.txt', Y)
        np.savetxt('X1.txt', X1)
        np.savetxt('Y1.txt', Y1)      

    if iter>= 0: #== 0 or iter == 5 or iter == 11:
        plot_disp(X1,Y1,s_u_pred_tot,'u inner i = ' + str(iter),  rf'$u_{{x, \mathrm{{FE-NO}},\Omega_{{II}}}}^{{{iter}}}$')
        plot_disp(X1,Y1,s_v_pred_tot,'v inner i = ' + str(iter),  rf'$u_{{y, \mathrm{{FE-NO}},\Omega_{{II}}}}^{{{iter}}}$')
        plot_disp(X1, Y1, s_e_xx_pred_tot, 'e_xx inner i = ' + str(iter), rf'$\epsilon_{{xx,\mathrm{{FE-NO}},\Omega_{{II}}}}^{{{iter}}}$')
        plot_disp(X1, Y1, s_e_yy_pred_tot, 'e_yy inner i = ' + str(iter), rf'$\epsilon_{{yy,\mathrm{{FE-NO}},\Omega_{{II}}}}^{{{iter}}}$')
        plot_disp(X1, Y1, s_e_xy_pred_tot, 'e_xy inner i = ' + str(iter), rf'$\epsilon_{{xy,\mathrm{{FE-NO}},\Omega_{{II}}}}^{{{iter}}}$')
        plot_disp(X, Y, strain_tot[:,0].reshape(-1,1), 'e_xx outer i = ' + str(iter), rf'$\epsilon_{{xx,\mathrm{{FE}},\Omega_{{I}}}}^{{{iter}}}$')
        plot_disp(X, Y, strain_tot[:,4].reshape(-1,1), 'e_yy outer i = ' + str(iter), rf'$\epsilon_{{yy,\mathrm{{FE}},\Omega_{{I}}}}^{{{iter}}}$')
        plot_disp(X, Y, strain_tot[:,1].reshape(-1,1), 'e_xy outer i = ' + str(iter), rf'$\epsilon_{{xy,\mathrm{{FE}},\Omega_{{I}}}}^{{{iter}}}$')
    
        plot_disp(X,Y,u, 'u outer i = ' + str(iter), rf'$u_{{x, \mathrm{{FE-NO, \Omega_I}}}}^{{{iter}}}$')
        plot_disp(X,Y,v, 'v outer i = ' + str(iter), rf'$u_{{y, \mathrm{{FE-NO, \Omega_I}}}}^{{{iter}}}$')

        np.savetxt('u2 i = '+ str(iter) + ' .txt', s_u_pred_tot)
        np.savetxt('v2 i = '+ str(iter) + ' .txt', s_v_pred_tot)
        np.savetxt('e_xx i = '+ str(iter) + ' .txt', s_e_xx_pred_tot)
        np.savetxt('e_yy i = '+ str(iter) + ' .txt', s_e_yy_pred_tot)
        np.savetxt('e_xy i = '+ str(iter) + ' .txt', s_e_xy_pred_tot)
        np.savetxt('e_xx_outer i = '+ str(iter) + ' .txt', strain_tot[:,0])
        np.savetxt('e_yy_outer i = '+ str(iter) + ' .txt', strain_tot[:,4])
        np.savetxt('e_xy_outer i = '+ str(iter) + ' .txt', strain_tot[:,1])
        np.savetxt('u i = '+ str(iter) + ' .txt', u)
        np.savetxt('v i = '+ str(iter) + ' .txt', v)

    u_list.append(u)
    v_list.append(v)
    u2_list.append(s_u_pred_tot)
    v2_list.append(s_v_pred_tot)

    if len(u_list) > 1:
        uv_L2 = np.linalg.norm(np.sqrt((u_list[-1] - u_list[-2])**2 + (v_list[-1] - v_list[-2])**2)) 
        uv_L2_2 = np.linalg.norm(np.sqrt((u2_list[-1] - u2_list[-2])**2 + (v2_list[-1] - v2_list[-2])**2))
        print('\n' ,'error', uv_L2 + uv_L2_2)
        error_list.append(uv_L2 + uv_L2_2)
        if error_list[-1] < 1e-3:
            # L2 error is smaller than 1e-3, break the loop
            break

np.save('error_list_FE_NN', error_list)    
end_time = time.time()  
print('time', end_time - start_time)










        




