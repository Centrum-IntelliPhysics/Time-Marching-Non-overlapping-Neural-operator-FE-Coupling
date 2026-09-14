from dolfinx import log, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
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
from ufl import dx
import time
from Hyper_utils import createFolder, plot_disp, plot_relative_error, plot_deformation_uy

#region Save path       
originalDir = os.path.dirname(os.path.abspath(__file__))
print('curent working directory:', originalDir)
os.chdir(os.path.join(originalDir))

foldername = 'FE_full_square_hyper_one_traction_ground_truth'  
createFolder(foldername )
os.chdir(os.path.join(originalDir, './'+ foldername + '/')) 

origin_real = os.path.join(originalDir, './'+ foldername + '/')
#### Gmsh generates the geometry 
lc = 0.04

# region gmsh
# Define circle parameters
center = (0.0, 0.5, 0.0)  # Center of the circle
radius = 0.3             # Radius of the circle
num_points = 100     # Number of points on the circumference 





#region FEM 
#### FEM  
import dolfinx 
# 导入 Gmsh 生成的 .msh 文件 
os.chdir(os.path.join(originalDir, './Hyper_elastic_Gmsh_tri/'))
mesh1, cell_markers, facet_markers  = gmshio.read_from_msh("full_square.msh", MPI.COMM_WORLD) 
V = functionspace(mesh1, ("CG", 2, (mesh1.geometry.dim, ))) ###without  (mesh1.geometry.dim, ), it is a scalar space not a vector space 

os.chdir(origin_real)

### linear materials parameters 
E = 1000 #10GPa
nu = 0.3 

mu = E/(2 * (1 + nu))
lambda_ = E*nu/((1 + nu)*(1 - 2*nu))
# 找到边界面
tdim = mesh1.topology.dim
fdim = tdim - 1
domain = mesh1

# region Location BC
def bt(x):
    return np.isclose(x[1], -0.5)

def left(x):
    return np.isclose(x[0], -1.)
# Sub domain for rotation at top
def top(x):
    return np.isclose(x[1], 1.5) 

def right(x):
    return np.isclose(x[0], 1.)


circle = dolfinx.mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius**2, 1e-3, 1e-3))
#circle_1 = dolfinx.mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True ))
#circle_rec = dolfinx.mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0],-1.5) | np.isclose(x[0],2.5) | np.isclose(x[1], -0.5) | np.isclose(x[1], 1.5) )

def disk_out(x):
    xc = x[0][np.where((x[0]-center[0])**2 + (x[1]-center[1])**2 >= radius**2)]
    yc = x[1][np.where((x[0]-center[0])**2 + (x[1]-center[1])**2 >= radius**2)]
    return xc, yc 

def disk1(x):
    xc = x[0][np.where((x[0]-center[0])**2 + (x[1]-center[1])**2 <= radius**2 + 1e-4)]
    yc = x[1][np.where((x[0]-center[0])**2 + (x[1]-center[1])**2 <= radius**2 + 1e-4)]
    return xc, yc 


# region Mypression
class MyExpression:
    def __init__(self, x0, y0, value, V_dim):
        self.x0 = x0
        self.y0 = y0
        self.value  = value
        self.V_dim = V_dim        
        self.RBF_0  = Rbf(x0, y0, value[0])
        self.RBF_1  = Rbf(x0, y0, value[1])
        
    def eval(self, x):
        # Added some spatial variation here. Expression is sin(x)
        #print(x.shape)
        #print(x[0], x[1], x[2]) 
        
        values = np.zeros((self.V_dim, x.shape[1]))

        #values[0] = self.RBF_0(x[0], x[1])
        #values[1] = self.RBF_1(x[0], x[1])
        values[0] = np.where(np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius**2, 1e-4, 1e-4), 
                            self.RBF_0(x[0], x[1]), 0)
        values[1] = np.where(np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius**2, 1e-4, 1e-4),
                            self.RBF_1(x[0], x[1]), 0)
        
        #AAA1 = x[0][np.where(np.isclose((x[0]-center[0])**2 +
        #                          (x[1]-center[1])**2, radius**2, 1e-3, 1e-3))]
        
        #np.savetxt('AA', AAA1)
        return values 



    
B = fem.Constant(domain, default_scalar_type((0, 0, 0)))
T = fem.Constant(domain, default_scalar_type((0, 0, 0))) ## later T.value[2] refer to this value 

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
# Stress
# Hyper-elasticity
P = ufl.diff(psi, F_grad)

# Define form F (we want to find u such that F(u) = 0)
F = ufl.inner(ufl.grad(v), P) * dx
### ds will be redefined later

gdim = mesh1.geometry.dim

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

### define the inner boundary for handshake # the new defined ds 
boundaries = [(1, lambda x: np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius**2, 1e-3, 1e-3))]

# out hole boundary
u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
X, Y = u_geometry[:,0], u_geometry[:,1]

x_out, y_out = disk_out(np.vstack([X.reshape(1,-1), Y.reshape(1,-1)]))
x_disk1, y_disk1 = disk1(np.vstack([X.reshape(1,-1), Y.reshape(1,-1)]))

x_close = np.isclose(X[:, None], x_disk1)
y_close = np.isclose(Y[:, None], y_disk1)
both_close = x_close & y_close

# find the indices in X, Y that are close to any point in x_disk1, y_disk1
has_match = np.any(both_close, axis=1)
index_disk1 = np.where(has_match)[0]

#plot_disp(x_disk1, y_disk1, x_disk1, 'disk1', 'disk1')

bt_points = dolfinx.mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], -0.5))
uD = np.array([0, 0, 0], dtype=default_scalar_type)
domain.topology.create_connectivity(fdim, tdim)
boundary_dofs = fem.locate_dofs_topological(V, fdim, bt_points)
bc_bt = fem.dirichletbc(uD, boundary_dofs, V)

left_points = dolfinx.mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], -1.))
uD = np.array([0, 0, 0], dtype=default_scalar_type)
domain.topology.create_connectivity(fdim, tdim)
boundary_dofs = fem.locate_dofs_topological(V, fdim, left_points)
bc_l = fem.dirichletbc(uD, boundary_dofs, V)

# resort the boundary points in a circle
index_c = np.where(np.isclose((X - center[0])**2 + (Y - center[1])**2, radius**2, 1e-3, 1e-3))[0]
x_c = X[index_c]
y_c = Y[index_c]

x_c_pos = x_c[x_c >= 0]
x_c_neg = x_c[x_c < 0]

index_c_pos = np.argsort(x_c_pos) # form small to large
index_c_neg = np.argsort(x_c_neg) 

x_c_pos = x_c_pos[index_c_pos]
x_c_neg = x_c_neg[index_c_neg]

y_c_pos = y_c[x_c >= 0][index_c_pos]
y_c_neg = y_c[x_c < 0][index_c_neg]

x_c = np.hstack([x_c_neg, x_c_pos])
y_c = np.hstack([y_c_neg, y_c_pos])

index_c_resort = np.array([np.where(np.isclose(X, x_val) & np.isclose(Y, y_val) )[0][0] for x_val, y_val in zip(x_c, y_c)])

#plot_disp(x_c, y_c, x_c, 'circle_bc', 'circle_bc')
niter =1000
theta = 0.5 # relaxation coefficient
ts_tot = 5
consume_time = []
#### Coupling  
for ts in range(0, ts_tot):
    time_start = time.time()
    top_b = dolfinx.mesh.locate_entities_boundary(mesh1, fdim, top)
    uD_top = np.array([0.05*(ts+1), 0.05*(ts+1), 0], dtype=default_scalar_type)
    mesh1.topology.create_connectivity(fdim, tdim)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, top_b)
    bc_top = fem.dirichletbc(uD_top, boundary_dofs, V)
    
    right_b = dolfinx.mesh.locate_entities_boundary(mesh1, fdim, right)
    uD_right = np.array([0.05*(ts+1), 0 , 0], dtype=default_scalar_type)
    mesh1.topology.create_connectivity(fdim, tdim)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, right_b)
    bc_r = fem.dirichletbc(uD_right, boundary_dofs, V)

    error_list = []
    break_list = []
    u_list, v_list, u2_list, v2_list = [], [], [], []
    #### Coupling  
    bcs  = [bc_top, bc_bt]  #bc_l, bc_r,
    problem = NonlinearProblem(F, uh, bcs)
    solver = NewtonSolver(mesh1.comm, problem)
    # Set Newton solver options
    solver.atol = 1e-8
    solver.rtol = 1e-8
    solver.max_it = 50
    solver.convergence_criterion = "incremental"
    num_its, converged = solver.solve(uh)
    assert (converged)
    uh.x.scatter_forward()
    print(f"Time step {ts}, Number of iterations {num_its}, disp {uD_top}")

    u_values = uh.x.array.real
    u_tot = u_values.reshape(-1,3)
    U_, V_, W_= u_tot[:,0], u_tot[:,1], u_tot[:,2]

    time_end_ts = time.time()
    print('time for ts', ts, 'is', time_end_ts - time_start)
    consume_time.append(time_end_ts - time_start)

    ### strain is 3*3  (gdim, gdim)
    Function_space_for_strain = fem.functionspace(mesh1, ("CG", 2, (gdim, gdim)))        
    expr= fem.Expression(epsilon(uh), Function_space_for_strain.element.interpolation_points())        
    strain_values = Function(Function_space_for_strain) 
    strain_values.interpolate(expr)
    strain_tot = strain_values.x.array.reshape(-1,9)    

    Function_space_for_stress = fem.functionspace(mesh1, ("CG", 2, (gdim, gdim)))        
    expr= fem.Expression(sigma(uh), Function_space_for_stress.element.interpolation_points())        
    stress_values = Function(Function_space_for_stress) 
    stress_values.interpolate(expr)
    sigma_tot = stress_values.x.array.reshape(-1,9)  

    plot_disp(X,Y,U_,'displacement u ts=' +str(ts), rf'$u_{{x,\mathrm{{FE}}}}^{{{ts}}}$')
    plot_disp(X,Y,V_,'displacement v ts=' +str(ts), rf'$u_{{y,\mathrm{{FE}}}}^{{{ts}}}$')
    plot_disp(x_disk1, y_disk1, strain_tot[index_disk1,0], 'e_xx_in ts=' +str(ts), rf'$\epsilon_{{xx, \mathrm{{FE}}, \Omega_{{II}}}}^{{{ts}}}$')
    plot_disp(x_disk1, y_disk1, strain_tot[index_disk1,4], 'e_yy_in ts=' +str(ts), rf'$\epsilon_{{yy, \mathrm{{FE}}, \Omega_{{II}}}}^{{{ts}}}$')
    plot_disp(x_disk1, y_disk1, strain_tot[index_disk1,1], 'e_xy_in ts=' +str(ts), rf'$\epsilon_{{xy, \mathrm{{FE}}, \Omega_{{II}}}}^{{{ts}}}$')

    plot_disp(x_disk1, y_disk1, sigma_tot[index_disk1,0], 's_xx_in ts=' +str(ts),  rf'$\sigma_{{xx, \mathrm{{FE}}, \Omega_{{II}}}}^{{{ts}}}$')
    plot_disp(x_disk1, y_disk1, sigma_tot[index_disk1,4], 's_yy_in ts=' +str(ts), rf'$\sigma_{{yy, \mathrm{{FE}}, \Omega_{{II}}}}^{{{ts}}}$')
    plot_disp(x_disk1, y_disk1, sigma_tot[index_disk1,1], 's_xy_in ts=' +str(ts), rf'$\sigma_{{xy, \mathrm{{FE}}, \Omega_{{II}}}}^{{{ts}}}$')

    plot_deformation_uy(uh, V, 'deformation uy outer ts=' +str(ts))

    np.savetxt('u ts=' +str(ts) +' .txt', U_)
    np.savetxt('v ts=' +str(ts) +' .txt', V_)
    np.savetxt('U_in ts=' +str(ts) +' .txt', u_tot[index_disk1,0])
    np.savetxt('V_in ts=' +str(ts) +' .txt', u_tot[index_disk1,1])
    np.savetxt('e_xx_in ts=' +str(ts) +' .txt', strain_tot[index_disk1,0])
    np.savetxt('e_yy_in ts=' +str(ts) +' .txt', strain_tot[index_disk1,4])
    np.savetxt('e_xy_in ts=' +str(ts) +' .txt', strain_tot[index_disk1,1])
    np.savetxt('s_xx_in ts=' +str(ts) +' .txt', sigma_tot[index_disk1,0])
    np.savetxt('s_yy_in ts=' +str(ts) +' .txt', sigma_tot[index_disk1,4])
    np.savetxt('s_xy_in ts=' +str(ts) +' .txt', sigma_tot[index_disk1,1])
    np.savetxt('s_xx ts=' +str(ts) +' .txt', sigma_tot[:, 0])
    np.savetxt('s_yy ts=' +str(ts) +' .txt', sigma_tot[:, 4])
    np.savetxt('s_xy ts=' +str(ts) +' .txt', sigma_tot[:, 1])
    
np.savetxt('X.txt', X)
np.savetxt('Y.txt', Y)
np.savetxt('X1.txt', x_disk1)
np.savetxt('Y1.txt', y_disk1)
np.savetxt('x_c.txt', x_c)
np.savetxt('y_c.txt', y_c)
np.savetxt('consuming time',np.array(consume_time))
        










        




