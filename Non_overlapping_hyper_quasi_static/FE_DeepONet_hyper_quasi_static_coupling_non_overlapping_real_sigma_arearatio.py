'''
======================================================================
The FE-DeepONet or FE-NO coupling framework for hyper elastic materials 
under quasi-static loading conditions.
----------------------------------------------------------------------
The non-overlapping boundary is used. 
The domain decomposition and Schwartz alternating method is used by
exchanging the displacement at the overlapping boundary.

Two tractions (displacements) are applied on the top and right edege
======================================================================
'''
from dolfinx import log, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import os 
import jax
import jax.numpy as jnp
from jax import grad, vmap
import dolfinx
from dolfinx import fem, default_scalar_type, la
from dolfinx.fem import functionspace 
from dolfinx.io import XDMFFile, gmshio
from dolfinx.fem import Constant, Function 
from mpi4py import MPI
import numpy as np
from dolfinx.mesh import create_box, create_unit_square
import ufl 
from ufl import dx
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc, create_matrix
from petsc4py import PETSc
import os 
from tqdm import trange 
import gmsh 
import math
from dolfinx import plot
from scipy.interpolate import Rbf, interp1d, griddata
import logging 
import meshio
import pyvista
import pickle
import time
import dolfinx 
from Hyper_utils import createFolder, plot_disp, plot_relative_error, plot_boundary



#region Save path       
originalDir ='/nfshdd/21040463r/FEM_DeepONet_non_overlapping_coupling/non_overlapping_hyper_clean'
print('curent working directory:', originalDir)
os.chdir(os.path.join(originalDir))

foldername = 'FE_DeepONet_hyper_elasticity_quasi_static_coupling_results_1e_3_DeepONet_arearatio'  
createFolder(foldername )
os.chdir(os.path.join(originalDir, './'+ foldername + '/')) 

originalDir_real = os.path.join(originalDir, './'+ foldername + '/') 

# region gmsh
#### Gmsh generates the geometry 
lc = 0.04

# Define circle parameters
center = (0.0, 0.5, 0.0)  # Center of the circle
radius = 0.3             # Radius of the circle
num_points = 80     # Number of points on the circumference 



#========================================================
# Use FEM to solve the outer region for hyper-elasticity
#========================================================

#region FEM 
# load the gmsh file
os.chdir(os.path.join(originalDir, './Hyper_elastic_Gmsh_tri/'))
mesh1, cell_markers, facet_markers  = gmshio.read_from_msh("outer_region.msh", MPI.COMM_WORLD) 
V = functionspace(mesh1, ("CG", 2, (mesh1.geometry.dim, ))) ###without  (mesh1.geometry.dim, ), it is a scalar space not a vector space 
os.chdir(originalDir_real)
### materials parameters 
E = 1000
nu = 0.3 

# strain function in ufl(a domain specific language for declaration of finite element discretizations of variational forms)
def epsilon(u):
    """格林-拉格朗日应变 - 适用于大变形""" # for large deformation
    I = ufl.Identity(gdim)
    F = I + ufl.grad(u)  # 变形梯度 
    C = F.T * F          # 右柯西-格林张量
    E = 0.5 * (C - I)    # 格林-拉格朗日应变
    return E

def sigma(u):
    I = ufl.Identity(gdim)
    F = I + ufl.grad(u)
    J = ufl.det(F)
    C = F.T * F
    Cinv = ufl.inv(C)
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    S = mu * (I - Cinv) + lmbda * ufl.ln(J) * Cinv
    return (1 / J) * F * S * F.T

'''def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)'''

mu = E/(2 * (1 + nu))
lambda_ = E*nu/((1 + nu)*(1 - 2*nu))


# find the boundaries and set the boundary conditions
tdim = mesh1.topology.dim
fdim = tdim - 1
domain = mesh1
# region Location BC
def bt(x):
    return np.isclose(x[1], -0.5)

def left(x):
    return np.isclose(x[0], -1.)

def top(x):
    return np.isclose(x[1], 1.5) 

def right(x):
    return np.isclose(x[0], 1.)

# the fixed boundaries for bottom edge and left edge 
bottom_point = dolfinx.mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], -0.5))
uD = np.array([0, 0, 0], dtype=default_scalar_type)
domain.topology.create_connectivity(fdim, tdim)
boundary_dofs = fem.locate_dofs_topological(V, fdim, bottom_point)
bc_bt = fem.dirichletbc(uD, boundary_dofs, V)

left_points = dolfinx.mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], -1.))
uD = np.array([0, 0, 0], dtype=default_scalar_type)
domain.topology.create_connectivity(fdim, tdim)
boundary_dofs = fem.locate_dofs_topological(V, fdim, left_points)
bc_l = fem.dirichletbc(uD, boundary_dofs, V)


# the functions to find the outer cricle and inner circle for overlapping boundaries
# input: [2, n] --> [2, m] m points on the circular boundaries 
def on_cricle(x):
    xc = x[0][np.where(np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius**2, 1e-3, 1e-3))]
    yc = x[1][np.where(np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius**2, 1e-3, 1e-3))]
    return xc, yc 

# region Mypression
#=====================================================================
# MyExpression is used to interpolate the displacement at the boundary
#=====================================================================
'''class MyExpression:
    def __init__(self, x0, y0, value, V_dim):
        self.x0 = x0
        self.y0 = y0
        self.value  = value
        self.V_dim = V_dim        
        self.Tx_fun  = Rbf(x0, y0, value[0])
        self.Ty_fun  = Rbf(x0, y0, value[1])
        
    def eval(self, x):

        values = np.zeros((self.V_dim, x.shape[1]))
        
        values[0] = np.where(np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius**2, 1e-4, 1e-4), self.Tx_fun(x[0], x[1]), 0)
        values[1] = np.where(np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius**2, 1e-4, 1e-4), self.Ty_fun(x[0], x[1]), 0)
        # here we should use 1e-4 to find the points on the circle 
                             
        #print('max value', np.max(values[0]), np.max(values[1]))
        values0 = values[0]
        np.savetxt('values0.txt', values0[np.where(np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius**2, 1e-4, 1e-4))])
        return values '''

class MyExpression:
    def __init__(self, x0, y0, value, V_dim):
        self.x0 = x0
        self.y0 = y0
        self.value  = value
        self.V_dim = V_dim        
        self.RBF_0  = Rbf(x0, y0, value[0])
        self.RBF_1  = Rbf(x0, y0, value[1])
        
    def eval(self, x):
        
        values = np.zeros((self.V_dim, x.shape[1]))
        values[0] = np.where(np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius**2, 1e-4, 1e-4), 
                            self.RBF_0(x[0], x[1]), 0)
        values[1] = np.where(np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius**2, 1e-4, 1e-4),
                            self.RBF_1(x[0], x[1]), 0)
        '''values0 = values[0]
        np.savetxt('values0.txt', values0[np.where(np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius**2, 1e-3, 1e-3))])
        print(np.where(np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius**2, 1e-3, 1e-3))[0].shape)'''
        return values


# region measure ds
def circle_boundary(x):

    return np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius**2, 1e-3, 1e-3)

boundary_facets_c = dolfinx.mesh.locate_entities_boundary(
    domain, domain.topology.dim-1, circle_boundary)

marked_facets_c = np.hstack([boundary_facets_c])
marked_values_c = np.hstack([np.full_like(boundary_facets_c, 105)])

boundary_tags_c = dolfinx.mesh.meshtags(domain, domain.topology.dim-1, 
                             marked_facets_c, marked_values_c)

ds = ufl.Measure("ds", domain=mesh1, subdomain_data=boundary_tags_c) 
# doublecheck the measure ds 

facet_values = boundary_tags_c.values
facet_indices = boundary_tags_c.indices
colors = {105: 'red'}  
#plot_boundary(mesh1, facet_values, facet_indices, colors, 'top_boundary', fdim)
print('The number of the c edge dofs:', len(facet_values))


# region Hyper Eqs 
# define the variational problem in FEM     
B = fem.Constant(domain, default_scalar_type((0, 0, 0)))
#T = fem.Constant(domain, default_scalar_type((0, 0, 0))) 
Traction = fem.Function(V)
v = ufl.TestFunction(V)
uh = fem.Function(V)
# Spatial dimension
d = len(uh)
# Identity tensor
I = ufl.variable(ufl.Identity(d))
# Deformation gradient
F_grad = ufl.variable(I + ufl.grad(uh))
# Right Cauchy-Green tensor
C = ufl.variable(F_grad.T * F_grad)
# Invariants of deformation tensors
Ic = ufl.variable(ufl.tr(C))
J = ufl.variable(ufl.det(F_grad))
# Elasticity parameters
mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
# Stored strain energy density (compressible neo-Hookean model)
psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J))**2
# Stress (first Piola–Kirchhoff stress tensor)
P = ufl.diff(psi, F_grad)
# Define form F (we want to find u such that F(u) = 0)
F = ufl.inner(ufl.grad(v), P) * dx  - ufl.inner(v, B) * dx 


# out hole boundary
u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
X, Y = u_geometry[:,0], u_geometry[:,1]
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


# inner hole boundary
os.chdir(os.path.join(originalDir, './Hyper_elastic_Gmsh_tri/'))
mesh2, cell_markers, facet_markers  = gmshio.read_from_msh("inner_hole.msh", MPI.COMM_WORLD) 
V2 = functionspace(mesh2, ("CG", 2, (mesh2.geometry.dim, ))) 
os.chdir(originalDir_real)
u_topology1, u_cell_types1, u_geometry1 = plot.vtk_mesh(V2)
X1, Y1 = u_geometry1[:,0], u_geometry1[:,1]
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

os.chdir(os.path.join(originalDir, './FE_full_square_hyper_all_dataset_N_200_one_tractions_square_CG2_dense_u_v_top_resort_real_sigma_0'))
import numpy as npr
x_c_real = npr.loadtxt('x_c.txt')
y_c_real = npr.loadtxt('y_c.txt')

from scipy.spatial import KDTree
# 创建目标点数组
target_points = npr.column_stack((x_c_real, y_c_real))
# 创建源点数组  
source_points = npr.column_stack((x_c1_out, y_c1_out))

# 构建KDTree进行快速最近邻搜索
tree = KDTree(source_points)
distances, index_xy_c = tree.query(target_points)

print('x_c difference', np.max(np.abs(x_c_real - x_c1_out)), np.max(np.abs(x_c_real - x_c1_out[index_xy_c])))

os.chdir(originalDir_real)

#print('difference of circular coordinates', x_c1_out, x_c_out, np.max(np.abs(x_c_out - x_c1_out)))
np.savetxt('X1.txt', X1)
np.savetxt('Y1.txt', Y1)
np.savetxt('xc1_out.txt', x_c1_out)
np.savetxt('yc1_out.txt', y_c1_out)


# region DeepONet 
#=========================================================
# Pretrained DeepONet model is used to predict the displacement
# for the inner region receiving the boundary displacement from FEM
#=========================================================
from prepare_DeepONet_hyper_elastic import PI_DeepONet

m = x_c_out.shape[0]  # number of sensor points on the circular boundary 
print('number of sensor points on the circular boundary:', m)
d = 2 # dimension of the input data
ela_model = dict()
#=====================================================
#We predict sigma directly, so E is important to be
#consistent with the pretraining values 
# In linear-elastic,we use epsilon to predict sigma,
# E just need to be real value. 
#=====================================================
ela_model['E'] = E * 1e-6  #1000 in DeepONet we should scale the E  
ela_model['nu'] = nu

branch_layers =  [2*m, 100, 100, 100, 100, 800]
trunk_layers =  [d, 100, 100, 100, 100, 800]
model = PI_DeepONet(branch_layers, trunk_layers, **ela_model)
# 1211 is for disk case and it is correct
os.chdir(os.path.join(originalDir, './' + 'prepare_DeepONet_hyper_elastic_100w_uv_bcs_strain_one_traction_N_800_batch_100_uv_top_resort_real_sigma' + '/'))
print(os.getcwd())
with open('DeepONet_DR.pkl', 'rb') as f:
    params = pickle.load(f)

X11 = np.linspace(center[0]-radius-0.05, center[0]+radius+0.05, m)
Y11 = np.linspace(center[1]-radius-0.05, center[1]+radius+0.05, m)
X1_, Y1_ = np.meshgrid(X11, Y11)

os.chdir(originalDir_real)



time_0 = time.time()
theta = 0.5 #0.9  # relaxation parameter for traction
#region Coupling 
niter =1000
ts_tot = 5 # total quasi-static step
for ts in range(0, ts_tot):

    top_b = dolfinx.mesh.locate_entities_boundary(mesh1, fdim, top)
    uD_top = np.array([0.05*(ts+1), 0.05*(ts+1), 0], dtype=default_scalar_type)
    mesh1.topology.create_connectivity(fdim, tdim)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, top_b)
    bc_top = fem.dirichletbc(uD_top, boundary_dofs, V)

    bcs  = [bc_top, bc_bt] 

    error_list = []
    break_list = []
    u_list, v_list, u2_list, v2_list = [], [], [], []
    #### Coupling  
    for iter in range(0, niter):
        ## FEM 
        if iter == 0 and ts == 0:
            # first hyper elastic solver  
            problem = NonlinearProblem(F, uh, bcs)
            solver = NewtonSolver(mesh1.comm, problem) 
            # Set Newton solver options
            solver.atol = 1e-8
            solver.rtol = 1e-8
            solver.convergence_criterion = "incremental"
            num_its, converged = solver.solve(uh)
            assert (converged)
            uh.x.scatter_forward()
            print(f"Time step {ts}, inner_iter{iter}, Number of iterations {num_its}, disp {uD_top}")
        
        else:     
            
            # Define form F (we want to find u such that F(u) = 0)
            F = ufl.inner(ufl.grad(v), P) * dx  - ufl.inner(v, B) * dx - area_ratio * ufl.inner(v, Traction) * ds(105)  
            problem = NonlinearProblem(F, uh, bcs)
            solver = NewtonSolver(mesh1.comm, problem)
            # Set Newton solver options
            solver.atol = 1e-8
            solver.rtol = 1e-8
            solver.convergence_criterion = "incremental"
            num_its, converged = solver.solve(uh)
            assert (converged)
            uh.x.scatter_forward()
            print(f"Time step {ts}, inner_iter{iter}, Number of iterations {num_its}, disp {uD_top}")
            print('success initiation')
        
        
        if ts >= ts_tot - 1 and iter == 0:
            start1 = time.time()
            
       
        u_values = uh.x.array.real
        u_tot = u_values.reshape(-1,3)
        U_, V_, W_= u_tot[:,0], u_tot[:,1], u_tot[:,2]
        # the displacement obtained from outer region (FEM) at the outer circle of overlapping boundary 
        u_c = U_[coor_r_out].reshape(1,-1)
        v_c = V_[coor_r_out].reshape(1,-1)
        
        gdim = mesh1.geometry.dim
        ### stress is 3*3  (gdim, gdim)
        Function_space_for_sigma = fem.functionspace(mesh1, ("CG", 2, (gdim, gdim)))        
        expr= fem.Expression(sigma(uh), Function_space_for_sigma.element.interpolation_points())        
        sigma_values = Function(Function_space_for_sigma) 
        sigma_values.interpolate(expr)
        sigma_tot = sigma_values.x.array.reshape(-1,9)

        '''v2 = ufl.TestFunction(V2)
        uh2 = fem.Function(V2)

        # Spatial dimension
        d2 = len(uh2)

        # Identity tensor
        I2 = ufl.variable(ufl.Identity(d2))

        # Deformation gradient
        F_grad2 = ufl.variable(I2 + ufl.grad(uh2))

        # Right Cauchy-Green tensor
        C2 = ufl.variable(F_grad2.T * F_grad2)

        # Invariants of deformation tensors
        Ic2 = ufl.variable(ufl.tr(C2))
        J2 = ufl.variable(ufl.det(F_grad2))

        # Elasticity parameters
        mu2 = E / (2 * (1 + nu))
        lmbda2 =  E * nu / ((1 + nu) * (1 - 2 * nu))
        # Stored strain energy density (compressible neo-Hookean model)
        psi2 = (mu2 / 2) * (Ic2 - 3) - mu2 * ufl.ln(J2) + (lmbda2 / 2) * (ufl.ln(J2))**2
        # Stress
        # Hyper-elasticity
        P2 = ufl.diff(psi2, F_grad2)

        # Define form F (we want to find u such that F(u) = 0)
        F2 = ufl.inner(ufl.grad(v2), P2) * dx


        fdim2 = mesh2.topology.dim -1 
        #print(fdim2)
        c_line = dolfinx.mesh.locate_entities_boundary(mesh2, fdim2, lambda x: 
                            np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius**2, 1e-3, 1e-3))
        
        uD_c = Function(V2)
        uD_c_value = np.vstack([u_c,v_c])
        #print(x_up.shape, u_up.shape, uD_up_value.shape)
        uD_c_fun = MyExpression(X[coor_r_out], Y[coor_r_out], uD_c_value, mesh2.geometry.dim)
        uD_c.interpolate(uD_c_fun.eval)        
        boundary_dofs = fem.locate_dofs_topological(V2, fdim2, c_line)
        bc_c = fem.dirichletbc(uD_c, boundary_dofs)            
        bc2  = [bc_c]
        
        problem2 = NonlinearProblem(F2, uh2, bc2)
        solver2 = NewtonSolver(mesh2.comm, problem2)
        # Set Newton solver options
        solver2.atol = 1e-8
        solver2.rtol = 1e-8
        solver2.convergence_criterion = "incremental"
        num_its, converged = solver2.solve(uh2)
        assert (converged)
        uh2.x.scatter_forward()
        print(f"Time step {ts}, inner_iter{iter}, Number of iterations {num_its}, disp {uD_top}")

        u_values2 = uh2.x.array.real
        u_tot2 = u_values2.reshape(-1,3)
        U2_, V2_= u_tot2[:,0], u_tot2[:,1]

        #displacement at inner boundary 
        u_c2 = U2_[coor_r1_out].reshape(1,-1)
        v_c2 = V2_[coor_r1_out].reshape(1,-1)
        #### test square fem
        
        gdim = mesh2.geometry.dim
        ### stress is 3*3  (gdim, gdim)
        Function_space_for_sigma1 = fem.functionspace(mesh2, ("CG", 2, (gdim, gdim)))        
        expr1= fem.Expression(sigma(uh2), Function_space_for_sigma1.element.interpolation_points())        
        sigma_values1 = Function(Function_space_for_sigma1) 
        sigma_values1.interpolate(expr1)
        sigma_tot1 = sigma_values.x.array.reshape(-1,9)

        s_e_xx_pred = sigma_tot1[coor_r1_out, 0]
        s_e_yy_pred = sigma_tot1[coor_r1_out, 4]
        s_e_xy_pred = sigma_tot1[coor_r1_out, 1]

        n1 = x_c_out/((x_c_out**2 + (y_c_out-0.5)**2)**0.5)
        n2 = (y_c_out-0.5)/((x_c_out**2 + (y_c_out-0.5)**2)**0.5)

        #==================================================
        # find the traction at the circular boundary
        #==================================================


        T_x1 = -(sigma_tot1[coor_r1_out, 0]*n1 + sigma_tot1[coor_r1_out, 1]*n2)
        T_x2 = -(sigma_tot1[coor_r1_out, 1]*n1 + sigma_tot1[coor_r1_out, 4]*n2)
        T_x1_new = Rbf(x_c1_out, y_c1_out, T_x1)(x_c_out, y_c_out)
        T_x2_new = Rbf(x_c1_out, y_c1_out, T_x2)(x_c_out, y_c_out)
        T_c_new = np.vstack([T_x1_new.reshape(1,-1), T_x2_new.reshape(1,-1)])

        T_c0 =  np.vstack([-(sigma_tot[coor_r_out, 0]*n1 + sigma_tot[coor_r_out, 1]*n2).reshape(1,-1), 
                -(sigma_tot[coor_r_out, 1]*n1 + sigma_tot[coor_r_out, 4]*n2).reshape(1,-1)])            

        # relaxation formula for traction
        T_c = theta * T_c0 + (1-theta)*T_c_new
        T_fun = MyExpression(X[coor_r_out], Y[coor_r_out], T_c, mesh1.geometry.dim)
        Traction.interpolate(T_fun.eval)'''
        #test_T = Traction.x.array[Traction.x.array != 0]
        #print(test_T.shape, test_T)



        if iter >= 0:
            # region handshake

            # Get the boundary dispalcement by using Rbf interpolation 
            uc_fun = Rbf(x_c_out, y_c_out, u_c)
            vc_fun = Rbf(x_c_out, y_c_out, v_c)

            u2c = uc_fun(x_c1_out, y_c1_out)
            v2c = vc_fun(x_c1_out, y_c1_out)

            u2c = u2c[index_xy_c].reshape(1,-1)
            v2c = v2c[index_xy_c].reshape(1,-1)

            # predict the displacement at the inner boundary
            #=================================================
            # the u_test and v_test are the displacement at
            # the outer boundary as boundary condition
            # the hc_test is the coordinate of the inner boundary
            #=================================================
            u_test = np.hstack([u2c, v2c]).reshape(1,-1) 
            v_test = np.hstack([u2c, v2c]).reshape(1,-1)
            hc_test = np.hstack([X1[coor_r1_out].reshape(-1,1), Y1[coor_r1_out].reshape(-1,1)])

            s_uc_pred, s_vc_pred, _, _, _ = \
                                    model.predict_s(params, u_test, v_test, hc_test)
            
            hc_test_out = np.hstack([x_c_out.reshape(-1,1), y_c_out.reshape(-1,1)])

            _, _, sigma_xx_pred, sigma_yy_pred, sigma_xy_pred = \
                                    model.predict_s(params, u_test, v_test, hc_test_out)            

            u_c2 = s_uc_pred.reshape(1,-1)
            v_c2 = s_vc_pred.reshape(1,-1)
            sigma_xx_c = sigma_xx_pred.reshape(1,-1)
            sigma_yy_c = sigma_yy_pred.reshape(1,-1)
            sigma_xy_c = sigma_xy_pred.reshape(1,-1)

            n1 = x_c_out/((x_c_out**2 + (y_c_out-0.5)**2)**0.5)
            n2 = (y_c_out-0.5)/((x_c_out**2 + (y_c_out-0.5)**2)**0.5)

            n1_1 = x_c1_out/((x_c1_out**2 + (y_c1_out-0.5)**2)**0.5)
            n2_1 = (y_c1_out-0.5)/((x_c1_out**2 + (y_c1_out-0.5)**2)**0.5)

            N_ref = ufl.FacetNormal(mesh1) # reference vector 
            # 当前构型法向量（非单位）
            #n_current = J * ufl.dot(ufl.inv(F_grad).T, N_ref)
            n_current  = ufl.dot(ufl.cofac(F_grad), N_ref)

            # 面积变化比例
            area_ratio = ufl.sqrt(ufl.dot(n_current, n_current))

            #==================================================
            # find the traction at the circular boundary
            #==================================================
            
            h_test = np.hstack([X1.reshape(-1,1), Y1.reshape(-1,1)])
            s_u_pred_tot, s_v_pred_tot, sigma_xx_tot, sigma_yy_tot, sigma_xy_tot = model.predict_s(params, u_test, v_test, h_test)
            F11_grad_c1, F12_grad_c1, F21_grad_c1, F22_grad_c1 = model.predict_F(params, u_test, v_test, hc_test)
            # For 2 * 2 matrix, cofac(F) = [[F22, -F21],[-F12, F11]]

            n1_current_x = F22_grad_c1 * n1_1 + (-F21_grad_c1)* n2_1
            n1_current_y = (-F12_grad_c1) * n1_1 + F11_grad_c1 * n2_1

            n1_current_x_unit = n1_current_x/np.sqrt(n1_current_x**2 + n1_current_y**2)
            n1_current_y_unit = n1_current_y/np.sqrt(n1_current_x**2 + n1_current_y**2)

            U2_, V2_ = s_u_pred_tot.reshape(-1,1), s_v_pred_tot.reshape(-1,1) 
            W2_ = np.zeros_like(U2_)
            '''uh2 = fem.Function(V2)
            U_pred = np.hstack([U2_.reshape(-1,1), V2_.reshape(-1,1), W2_.reshape(-1,1)]).flatten()
            uh2.x.array.real = U_pred  
            print(uh2.x.array.real.shape)

            #plot_disp(X1, Y1, U2_, 'U2', 'U2')
            #plot_disp(X1, Y1, uh2.x.array.real.reshape(-1,3)[:,0], 'U2_1', 'U2_1')

            Function_space_for_sigma1 = fem.functionspace(mesh2, ("CG", 2, (gdim, gdim)))        
            expr1= fem.Expression(sigma(uh2), Function_space_for_sigma1.element.interpolation_points())        
            sigma_values1 = Function(Function_space_for_sigma1) 
            sigma_values1.interpolate(expr1)
            sigma_tot1 = sigma_values1.x.array.reshape(-1,9)

            sigma_xx_c = sigma_tot1[coor_r1_out, 0]
            sigma_yy_c = sigma_tot1[coor_r1_out, 4]
            sigma_xy_c = sigma_tot1[coor_r1_out, 1]

            print('difference of stress', np.max(np.abs(sigma_xx_tot.flatten()[coor_r1_out] - sigma_xx_c0)), 
                                    np.max(np.abs(sigma_yy_tot.flatten()[coor_r1_out] - sigma_yy_c0)),
                                    np.max(np.abs(sigma_xy_tot.flatten()[coor_r1_out] - sigma_xy_c0)))
            
            print('difference of stress',sigma_xx_tot.flatten()[coor_r1_out], sigma_xx_c0)'''
            '''if iter >= 0 and ts == 0 : 
                npr.savetxt('u_test iter = ' + str(iter) + '.txt', u_test)
                npr.savetxt('sigma_xx iter = ' + str(iter) + '.txt', sigma_tot1[:, 0])
                plot_disp(X1, Y1, sigma_tot1[:, 0],'sigma_xx iter = ' + str(iter),'sigma_xx')'''

            T_x1 = -(sigma_xx_c * n1_current_x_unit + sigma_xy_c * n1_current_y_unit)
            T_x2 = -(sigma_xy_c * n1_current_x_unit + sigma_yy_c * n1_current_y_unit)
            T_x1_new = Rbf(x_c1_out, y_c1_out, T_x1)(x_c_out, y_c_out)
            T_x2_new = Rbf(x_c1_out, y_c1_out, T_x2)(x_c_out, y_c_out)
            #print(T_x1, T_x2)
            T_c_new = np.vstack([T_x1_new.reshape(1,-1), T_x2_new.reshape(1,-1)])

            T_c0 =  np.vstack([-(sigma_tot[coor_r_out, 0]*n1_current_x_unit + sigma_tot[coor_r_out, 1]*n1_current_y_unit).reshape(1,-1), 
                               -(sigma_tot[coor_r_out, 1]*n1_current_x_unit + sigma_tot[coor_r_out, 4]*n1_current_y_unit).reshape(1,-1)])            

            # relaxation formula for traction
            T_c = theta * T_c0 + (1-theta)*T_c_new
            T_fun = MyExpression(X[coor_r_out], Y[coor_r_out], T_c, mesh1.geometry.dim)
            Traction.interpolate(T_fun.eval)




        u_list.append(U_)
        v_list.append(V_)
        u2_list.append(U2_)
        v2_list.append(V2_)
        break_list.append(sigma_tot[:,0])
        if len(u_list) > 1:
            uv_L2 = np.linalg.norm(np.sqrt((u_list[-1] - u_list[-2])**2 + (v_list[-1] - v_list[-2])**2)) 
            uv_L2_2 = np.linalg.norm(np.sqrt((u2_list[-1] - u2_list[-2])**2 + (v2_list[-1] - v2_list[-2])**2))
            print('\n' ,'error', uv_L2 + uv_L2_2)
            error_list.append(uv_L2 + uv_L2_2)
            if  error_list[-1] < 1e-3:
        
                plot_disp(X,Y,U_,'displacement u ts=' +str(ts) +' iter=' + str(iter),rf'$u_{{\mathrm{{FE-NO}}}}^{{{ts}}}$')
                plot_disp(X,Y,V_,'displacement v ts=' +str(ts) +' iter=' + str(iter),rf'$v_{{\mathrm{{FE-NO}}}}^{{{ts}}}$')
                

                np.savetxt('u ts=' +str(ts) +' .txt', U_)
                np.savetxt('v ts=' +str(ts) +' .txt', V_)
                np.savetxt('u2 ts=' +str(ts) +' .txt', U2_)
                np.savetxt('v2 ts=' +str(ts) +' .txt', V2_)
                np.savetxt('sigma_xx2 ts=' + str(ts) +'.txt', sigma_xx_tot)
                np.savetxt('sigma_yy2 ts=' + str(ts) +'.txt', sigma_yy_tot)
                np.savetxt('sigma_xy2 ts=' + str(ts) +'.txt', sigma_xy_tot)

                plot_disp(X1,Y1,U2_,'displacement u inner ts=' +str(ts) +' iter=' + str(iter), rf'$u_{{\mathrm{{FE-NO}}}}^{{{ts}}}$')
                plot_disp(X1,Y1,V2_,'displacement v inner ts=' +str(ts) +' iter=' + str(iter), rf'$v_{{\mathrm{{FE-NO}}}}^{{{ts}}}$')
                plot_disp(X1,Y1,sigma_xx_tot,'sigma xx inner ts=' +str(ts) +' iter=' + str(iter), rf'$\sigma_{{xx, \mathrm{{FE-NO}}}}^{{{ts}}}$')
                plot_disp(X1,Y1,sigma_yy_tot,'sigma yy inner ts=' +str(ts) +' iter=' + str(iter), rf'$\sigma_{{yy, \mathrm{{FE-NO}}}}^{{{ts}}}$')
                plot_disp(X1,Y1,sigma_xy_tot,'sigma xy inner ts=' +str(ts) +' iter=' + str(iter), rf'$\sigma_{{xy, \mathrm{{FE-NO}}}}^{{{ts}}}$')


                U2_tot = np.hstack([U2_.reshape(-1,1), V2_.reshape(-1,1),np.zeros((V2_.shape[0],1))])
                np.savetxt(f'error_list_FE_NN_hyper ts=' + str(ts) + ' iter=' + str(iter) + '.txt', error_list)
                #s_uc_pred, s_vc_pred, _, _, _ = model.predict_s(params, u_test, v_test, hc_test)
                '''plot_relative_error(X1[coor_r1_out], Y1[coor_r1_out], np.abs(s_uc_pred.reshape(-1,1) - u2c.reshape(-1,1)),
                                    'error u_bc ts=' + str(ts) + ' iter=' + str(iter),
                                    rf'$|u_{{\mathrm{{FE}},\Omega_{{II}}}}^{{{ts}}} - u_{{\mathrm{{NO}},\Omega_{{II}}}}^{{{ts}}}|$')'''

                # L2 error is smaller than 1e-3, break the loop
                break

np.savetxt('X.txt', X)
np.savetxt('Y.txt', Y)
np.savetxt('X1.txt', X1)
np.savetxt('Y1.txt', Y1)

time_1 = time.time()
print('time tot', time_1 - time_0)
print('time 4', time_1 - start1)
np.savetxt('consuming time',np.array([time_1 - time_0, time_1 - start1]))

# region plot error
#=========================================================
# Plot the error between the predicted displacement and the ground truth
# Ground truth data is generated by FEM in full square
#=========================================================

'''try:
    i = 4
    os.chdir(os.path.join(originalDir, './' + 'FE_DeepONet_hyper_elasticity_quasi_static_coupling_results' + '/'))
    X_NN, Y_NN, X1_NN, Y1_NN = np.loadtxt('X.txt'), np.loadtxt('Y.txt'), np.loadtxt('X1.txt'), np.loadtxt('Y1.txt')
    # for interpolation FE_NN coupling
    u_FE_NN_4, v_FE_NN_4 = np.loadtxt('u ts=' + str(i) + ' .txt'), np.loadtxt('v ts=' + str(i) + ' .txt')
    u2_FE_NN_4, v2_FE_NN_4 = np.loadtxt('u2 ts=' + str(i) + ' .txt'), np.loadtxt('v2 ts=' + str(i) + ' .txt')

    os.chdir(os.path.join(originalDir, './' + 'Hyper_elasticity_quasi_static_ground_truth' + '/'))
    X_full, Y_full= np.loadtxt('X.txt'), np.loadtxt('Y.txt')
    # for interpolation FE_NN coupling
    u_full_4, v_full_4 = np.loadtxt('u ts=' + str(i) + ' .txt'), np.loadtxt('v ts=' + str(i) + ' .txt')
    start_0 = time.time()
    u_full_4_outer = Rbf(X_full, Y_full, u_full_4)(X_NN, Y_NN)
    v_full_4_outer = Rbf(X_full, Y_full, v_full_4)(X_NN, Y_NN)
    u_full_4_inner = Rbf(X_full, Y_full, u_full_4)(X1_NN, Y1_NN)
    v_full_4_inner = Rbf(X_full, Y_full, v_full_4)(X1_NN, Y1_NN)
    end_0 = time.time()
    print('time for interpolation:', end_0 - start_0)

    os.chdir(originalDir_real)
    np.savetxt('u_full_4_outer ts=' + str(i) + '.txt', u_full_4_outer)
    np.savetxt('v_full_4_outer ts=' + str(i) + '.txt', v_full_4_outer)
    np.savetxt('u_full_4_inner ts=' + str(i) + '.txt', u_full_4_inner)
    np.savetxt('v_full_4_inner ts=' + str(i) + '.txt', v_full_4_inner)
    os.chdir(originalDir_real)
    u_full_4_outer = np.loadtxt('u_full_4_outer ts=' + str(i) + '.txt')
    v_full_4_outer = np.loadtxt('v_full_4_outer ts=' + str(i) + '.txt')
    u_full_4_inner = np.loadtxt('u_full_4_inner ts=' + str(i) + '.txt')
    v_full_4_inner = np.loadtxt('v_full_4_inner ts=' + str(i) + '.txt')

    # plot the relative error between the FE_NN coupling and FE_FE coupling
    u_error_coupling_4 = np.abs(u_FE_NN_4 - u_full_4_outer)
    v_error_coupling_4 = np.abs(v_FE_NN_4 - v_full_4_outer)
    u2_error_coupling_4 = np.abs(u2_FE_NN_4 - u_full_4_inner)
    v2_error_coupling_4 = np.abs(v2_FE_NN_4 - v_full_4_inner)

    plot_relative_error(X_NN, Y_NN, u_error_coupling_4, 'u_FE_NO_error ts=' + str(i), rf'$|u_{{\mathrm{{FE}}}}^{{{i}}} - u_{{\mathrm{{FE-NO}},\Omega_{{I}}}}^{{{i}}}|$')
    plot_relative_error(X_NN, Y_NN, v_error_coupling_4, 'v_FE_NO_error ts=' + str(i), rf'$|v_{{\mathrm{{FE}}}}^{{{i}}} - v_{{\mathrm{{FE-NO}},\Omega_{{I}}}}^{{{i}}}|$')
    plot_relative_error(X1_NN, Y1_NN, u2_error_coupling_4, 'u2_FE_NO_error ts=' + str(i), rf'$|u_{{\mathrm{{FE}}}}^{{{i}}} - u_{{\mathrm{{FE-NO}},\Omega_{{II}}}}^{{{i}}}|$')
    plot_relative_error(X1_NN, Y1_NN, v2_error_coupling_4, 'v2_FE_NO_error ts=' + str(i), rf'$|v_{{\mathrm{{FE}}}}^{{{i}}} - v_{{\mathrm{{FE-NO}},\Omega_{{II}}}}^{{{i}}}|$')

except Exception as e:
    print(f"Failed to open the directory with ground truth data generated by FEM")'''




