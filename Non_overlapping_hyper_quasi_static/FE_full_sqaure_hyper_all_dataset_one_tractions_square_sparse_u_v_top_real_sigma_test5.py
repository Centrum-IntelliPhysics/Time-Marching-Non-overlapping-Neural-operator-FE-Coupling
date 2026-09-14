from dolfinx import log, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import os
import jax
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
from ufl import dx
import time 
from tqdm import trange 
from utils import plot_bc
#### utils 

def createFolder(folder_name):
    try:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    except OSError:
        print ('Error: Creating folder. ' +  folder_name)



'''def epsilon(u):
    return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)


def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)'''


# region Plot funcs
def plot_disp(X, Y, u, foldername, title):
    '''
    Plot the displacement field
    '''
    plt.figure(figsize=(10, 8.5))
    
    # Set a compact layout between the main plot and the colorbar
    ax = plt.gca()  # Get the current axis
    scatter = ax.scatter(X, Y, c=u, cmap='seismic')

    # Add a colorbar
    cbar = plt.colorbar(scatter, orientation='vertical', pad=0.02, location='right')  # Reduce the padding
    cbar.formatter = ScalarFormatter()  # Set the default formatter
    cbar.formatter.set_scientific(True)  # Enable scientific notation
    cbar.formatter.set_powerlimits((-2, 2))  # Show scientific notation for numbers smaller than 0.01
    cbar.ax.get_yaxis().offsetText.set_fontsize(20)  # Set scientific notation size  # fontsize of formatter 12
    #cbar.ax.get_yaxis().offsetText.set_x(1.1)  # Move scientific notation to the right 
    cbar.ax.tick_params(labelsize=20)  # Adjust the font size of the colorbar labels
    # Set font properties
    ax.set_xlabel('x', fontsize=24)
    ax.set_ylabel('y', fontsize=24)
    ax.set_title(title, fontsize=36, fontweight='bold', y = 1.05)  # Increase the distance between the title and the plot

    # Adjust tick size and font size
    ax.tick_params(axis='both', which='major', labelsize=20, length=10, width=2)  # Major ticks
    ax.tick_params(axis='both', which='minor', labelsize=18, length=5, width=1)  # Minor ticks

    # Adjust the subplot layout
    plt.tight_layout(rect=[0, 0, 0.95, 1])  # The rect parameter controls the overall layout (reduce right margin)

    # Save and display
    plt.savefig(foldername + ".jpg", dpi=700, bbox_inches='tight')  # bbox_inches='tight' reduces the extra margins
    plt.show()


def plot_deformation_uy(u, V2, foldername):
    # Start virtual framebuffer if off-screen rendering is needed
    pyvista.start_xvfb()
    # Create plotter and pyvista grid
    p = pyvista.Plotter(off_screen=True)
    topology, cell_types, geometry = plot.vtk_mesh(V2)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    # Attach vector values to grid and warp grid by vector
    grid["u"] = u.x.array.reshape((geometry.shape[0], 3))
    grid["uy"] = u.x.array.reshape((geometry.shape[0], 3))[:, 1]
    # Add the grid to the plotter
    #actor_0 = p.add_mesh(grid, style="wireframe", color="k")
    
    warped = grid.warp_by_vector("u", factor=1)
    #actor_1 = p.add_mesh(warped, show_edges=True)
    actor_1 = p.add_mesh(warped, scalars= 'uy', cmap='seismic', show_edges= False) ## warped for deformation
    p.remove_scalar_bar()
    ## scalars for add values on the mesh
    p.add_scalar_bar(
    vertical=True,
    position_x=0.8,
    position_y=0.15,
    label_font_size=20,
    title_font_size=22,      # Font size for the title
    height=0.7,              # Height of the colorbar (adjust based on your needs)
    width=0.07             # Width of the colorbar (adjust based on your needs))  # Adjust the font size of the labels  # Move colorbar to the right side
    )
    p.show_axes()
    #p.add_mesh(grid, show_edges=True)

    # Adjust the view to align with the XY plane
    p.view_xy()

    # Show the plot (necessary for off-screen rendering)
    p.show()

    # Save the screenshot
    p.screenshot(foldername + ".png")

def plot_deformation_u(u, V2, foldername):
    # Start virtual framebuffer if off-screen rendering is needed
    pyvista.start_xvfb()
    # Create plotter and pyvista grid
    p = pyvista.Plotter(off_screen=True)
    topology, cell_types, geometry = plot.vtk_mesh(V2)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    # Attach vector values to grid and warp grid by vector
    grid["u"] = u.x.array.reshape((geometry.shape[0], 3))
    # Add the grid to the plotter
    #actor_0 = p.add_mesh(grid, style="wireframe", color="k")
    warped = grid.warp_by_vector("u", factor=1)
    #actor_1 = p.add_mesh(warped, show_edges=True)
    actor_1 = p.add_mesh(warped, scalars= "u", cmap="seismic", show_edges= False) ## warped for deformation
    ## scalars for add values on the mesh

    p.show_axes()
    #p.add_mesh(grid, show_edges=True)

    # Adjust the view to align with the XY plane
    p.view_xy()

    # Show the plot (necessary for off-screen rendering)
    p.show()

    # Save the screenshot
    p.screenshot(foldername + ".png")

#region Save path       
originalDir ='/nfshdd/21040463r/FEM_DeepONet_non_overlapping_coupling/non_overlapping_hyper_clean'
print('curent working directory:', originalDir)
os.chdir(os.path.join(originalDir))

foldername = 'FE_full_square_hyper_all_dataset_N_200_one_tractions_square_CG2_dense_u_v_top_resort_real_sigma_1_test5'  
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

# Set mesh size
#gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.2)

# set quadrilateral elements
#gmsh.model.mesh.setRecombine(2, out_dim_tags[0][1])  # choose quadrilateral elements
#gmsh.model.mesh.setRecombine(2, out_dim_tags[1][1])  # choose quadrilateral elements



# Get all surfaces after fragmentation
surfaces = gmsh.model.getEntities(2) #2 --> 2d surface

# Create physical groups for the two regions
# Now we use the actual tags from the fragment operation
outer_region = gmsh.model.addPhysicalGroup(2, [surfaces[0][1]], tag=1)
inner_region = gmsh.model.addPhysicalGroup(2, [surfaces[1][1]], tag=2)

#gmsh.model.addPhysicalGroup(2, [surface_hole], name=" Hole")


# Create physical groups for the two regions
# Now we use the actual tags from the fragment operation
gmsh.model.setPhysicalName(2, outer_region, "Outer_Square")
gmsh.model.setPhysicalName(2, inner_region, "Inner_Square")
gmsh.model.addPhysicalGroup(1, [line1, line2, line3, line4] , name="LargerSquareEdges")# facet tag = 3
gmsh.model.addPhysicalGroup(1, lines1 , name="InnerSquareEdges")# facet tag = 3

# set the algorithm as transfinite 
#gmsh.option.setNumber("Mesh.RecombineAll", 1) # set quadral mesh 
#gmsh.option.setNumber("Mesh.Algorithm", 8)  # 8 = Transfinite
# Generate 2D mesh
gmsh.model.mesh.generate(2)

# Write the mesh to a file (optional)
gmsh.write("hyper_non_over.msh")

# Finalize GMSH
gmsh.finalize()

import meshio

# 读取Gmsh生成的网格文件 (.msh)
msh = meshio.read("hyper_non_over.msh")

# 提取点和单元信息
points = msh.points
cells = msh.cells_dict["triangle"]  
# 使用三角形单元

# 创建图形
plt.figure(figsize=(8, 8))

# 绘制每个三角形
for cell in cells:
    polygon = points[cell]
    # 将多边形闭合
    polygon = np.vstack([polygon, polygon[0]])
    plt.plot(polygon[:, 0], polygon[:, 1], 'k-', linewidth=0.5)  # 用黑色线条绘制

# 设置坐标轴
plt.gca().set_aspect('equal')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gmsh Mesh Visualization2')
plt.savefig('Gmsh Mesh Visualization2' + ".jpg", dpi=700)
plt.show()


#region FEM 
#### FEM  
import dolfinx 
# 导入 Gmsh 生成的 .msh 文件 
mesh1, cell_markers, facet_markers  = gmshio.read_from_msh("hyper_non_over.msh", MPI.COMM_WORLD) 
V = functionspace(mesh1, ("CG", 2, (mesh1.geometry.dim, ))) ###without  (mesh1.geometry.dim, ), it is a scalar space not a vector space 


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


class MyExpression_top_out:
    def __init__(self, x0, y0, value_0, value_1, V_dim):
        self.x0 = x0
        self.y0 = y0
        self.value_0  = value_0
        self.value_1  = value_1
        self.V_dim = V_dim        
        self.RBF_0  = Rbf(x0, y0, value_0)
        self.RBF_1  = Rbf(x0, y0, value_1)
        
    def eval(self, x):
        #print(x[0], x[1], x[2]) 
        values = np.zeros((self.V_dim, x.shape[1]))
        values[0] = np.where(np.isclose(x[1], 1.5, 1e-4, 1e-4), self.RBF_0(x[0], x[1]), 0)
        values[1] = np.where(np.isclose(x[1], 1.5, 1e-4, 1e-4), self.RBF_1(x[0], x[1]), 0)
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
        values[0] = np.where(np.isclose(x[0], 1., 1e-4, 1e-4), self.RBF_1(x[0], x[1]), 0)
        return values 




#region RBF 
# Use double precision to generate data (due to GP sampling)
def RBF(x1, x2, params): #radial basis function 
    output_scale, lengthscales = params
    diffs = np.expand_dims(x1 / lengthscales, 1) - \
            np.expand_dims(x2 / lengthscales, 0)
    r2 = np.sum(diffs**2, axis=2)
    return output_scale**2 * np.exp(-0.5 * r2)

# To generate (x,t) (u, y)
def Generate_RBF(key, Nx, Ny, m, P, length_scale):
    """No need explicit resolution 
    """
    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    # Generate subkeys
    subkey, subkey1, subkey2, subkey3 = random.split(key, 4)
    subkey1 = random.split(subkey1, 1)
    subkey2 = random.split(subkey2, 1)
    subkey3 = random.split(subkey3, 1)
    # Generate a GP sample
    N = 512

    gp_params1 = (0.1, length_scale)
    gp_params2 = (0.1, length_scale)
    jitter = 1e-10
    X = jnp.linspace(xmin, xmax, N)[:,None]
    K1 = RBF(X, X, gp_params1)
    L1 = jnp.linalg.cholesky(K1 + jitter*np.eye(N))
    
    K2 = RBF(X, X, gp_params2)
    L2 = jnp.linalg.cholesky(K2 + jitter*np.eye(N))


    def gp_sample(key, L):
        gp_sample = jnp.dot(L, random.normal(key, (N,)))
        return gp_sample
    
    gp_sample_1 = vmap(gp_sample, (0,None))(subkey1, L1)
    gp_sample_2 = vmap(gp_sample, (0,None))(subkey2, L2)

    # Create a callable interpolation function
    f_fn1 = lambda x: vmap(jnp.interp,(None, None, 0))(x, X.flatten(), gp_sample_1)
    f_fn2 = lambda x: vmap(jnp.interp,(None, None, 0))(x, X.flatten(), gp_sample_2)

    # Input sensor locations and measurements
    yy = jnp.linspace(xmin, xmax, m)
    u_up, v_up = f_fn1(yy), f_fn2(yy)


    return  u_up, v_up


m = 200
P = m
Nx, Ny = m, m
key = random.PRNGKey(int(time.time()))
length_scale = 1
N = 5 # number of samples
keys = random.split(key, N)
config.update("jax_enable_x64", True)
u_up, v_up =  vmap(Generate_RBF, (0, None, None, None, None, None))(keys, Nx, Ny, m, P, length_scale)
config.update("jax_enable_x64", False)



#region FEM     
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

bt_points = dolfinx.mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], -0.5))
uD = np.array([0, 0, 0], dtype=default_scalar_type)
domain.topology.create_connectivity(fdim, tdim)
boundary_dofs = fem.locate_dofs_topological(V, fdim, bt_points)
bc_bt = fem.dirichletbc(uD, boundary_dofs, V)

left_points = dolfinx.mesh.locate_entities_boundary(domain, fdim, left)
uD = np.array([0, 0, 0], dtype=default_scalar_type)
domain.topology.create_connectivity(fdim, tdim)
boundary_dofs = fem.locate_dofs_topological(V, fdim, left_points)
bc_left = fem.dirichletbc(uD, boundary_dofs, V)


np.savetxt('X.txt', X)
np.savetxt('Y.txt', Y)

# Define the inner boundary for handshake
# resort the boundary points in a circle
index_c = np.where(np.isclose((X - center[0])**2 + (Y - center[1])**2, radius**2, 1e-3, 1e-3))[0]
x_c = X[index_c]
y_c = Y[index_c]

x_c_pos0 = x_c[(x_c >= 0) & (y_c >=0.5)]
x_c_pos1 = x_c[(x_c >= 0) & (y_c <0.5)]
x_c_neg0 = x_c[(x_c < 0) & (y_c >=0.5)]
x_c_neg1 = x_c[(x_c < 0) & (y_c <0.5)]


index_c_pos0 = np.argsort(x_c_pos0) # form small to large
index_c_pos1 = np.argsort(-x_c_pos1) # form large to small
index_c_neg0 = np.argsort(x_c_neg0) 
index_c_neg1 = np.argsort(-x_c_neg1) 


x_c_pos0 = x_c_pos0[index_c_pos0]
x_c_pos1 = x_c_pos1[index_c_pos1]
x_c_neg0 = x_c_neg0[index_c_neg0]
x_c_neg1 = x_c_neg1[index_c_neg1]

y_c_pos0 = y_c[(x_c >= 0) & (y_c >=0.5)][index_c_pos0]
y_c_pos1 = y_c[(x_c >= 0) & (y_c <0.5)][index_c_pos1]
y_c_neg0 = y_c[(x_c < 0) & (y_c >=0.5)][index_c_neg0]
y_c_neg1 = y_c[(x_c < 0) & (y_c <0.5)][index_c_neg1]

x_c = np.hstack([x_c_pos0, x_c_pos1, x_c_neg1, x_c_neg0])
y_c = np.hstack([y_c_pos0, y_c_pos1, y_c_neg1, y_c_neg0])

index_c_resort = np.array([np.where(np.isclose(X, x_val) & np.isclose(Y, y_val) )[0][0] for x_val, y_val in zip(x_c, y_c)])
np.savetxt('x_c.txt', x_c)
np.savetxt('y_c.txt', y_c)

index_inner_square = np.where(
(X-center[0])**2 + (Y-center[1])**2 < (radius)**2 + 1e-7
)[0]

#print('index_inner_sqaure:', index_inner_square.shape)
plot_disp(X[index_inner_square], Y[index_inner_square], X[index_inner_square], 'inner_square', 'Inner Square')
X1 = X[index_inner_square]
Y1 = Y[index_inner_square]
np.savetxt('X1.txt', X1)
np.savetxt('Y1.txt', Y1)



ts_tot = 10 # fake step for loading (avoid crash due to large deformation)
uc_list, vc_list, e_xx_c_list, e_yy_c_list, e_xy_c_list = [], [], [], [],[]
s_xx_c_list, s_yy_c_list, s_xy_c_list = [], [], []
U1_list, V1_list= [], []
error_index_list = []   


upper_b = dolfinx.mesh.locate_entities_boundary(mesh1, fdim, top)
right_b = dolfinx.mesh.locate_entities_boundary(mesh1, fdim, right)
vD_up = Function(V)
uD_r = Function(V)

x_up = np.linspace(-1., 1., m)
y_up = np.ones_like(x_up)*1.5

x_r = np.ones_like(x_up)*1.
y_r = np.linspace(-0.5, 1.5, m)
start_time_once = time.time()

for i in trange(N):
    uh.x.array[:] = np.zeros(uh.x.array.shape)  # reset to zero for each sample
    try:
        for ts in range(ts_tot):

            vD_up_value = v_up[i,:]*(ts+1)/ts_tot  # the displacement at top boundary
            uD_up_value = u_up[i,:]*(ts+1)/ts_tot  # the displacement at top boundary

            vD_up_fun = MyExpression_top_out(x_up, y_up, uD_up_value, vD_up_value, mesh1.geometry.dim)
            vD_up.interpolate(vD_up_fun.eval)        
            boundary_dofs = fem.locate_dofs_topological(V, fdim, upper_b)
            bc_top = fem.dirichletbc(vD_up, boundary_dofs)

            '''uD_r_value = u_r[i,:]*(ts+1)/ts_tot  # the displacement at right boundary
            uD_r_fun = MyExpression_right_out(x_r, y_r, uD_r_value, mesh1.geometry.dim)
            uD_r.interpolate(uD_r_fun.eval)        
            boundary_dofs = fem.locate_dofs_topological(V, fdim, right_b)
            bc_right = fem.dirichletbc(uD_r, boundary_dofs)'''

            print('max vD_up', np.max(vD_up_value), 'min vD_up', np.min(vD_up_value), 'max uD_up', np.max(uD_up_value), 'min uD_up', np.min(uD_up_value))
            #vD_up_value = vD_up.x.array

            bcs = [bc_bt, bc_top ] #,bc_left, bc_right
            problem = NonlinearProblem(F, uh, bcs)
            solver = NewtonSolver(mesh1.comm, problem)
            # Set Newton solver options
            solver.atol = 1e-8
            solver.rtol = 1e-8
            solver.max_it = 100
            solver.convergence_criterion = "incremental"
            num_its, converged = solver.solve(uh)
            assert (converged)
            uh.x.scatter_forward()
            print(f"Time step {ts}, Number of iterations {num_its}")

            '''if ts == 0:
                plot_disp(X,Y,U_,'displacement u ts=' +str(ts), rf'$u_{{\mathrm{{FE}}}}^{{{ts}}}$')
                plot_disp(X[index], Y[index], uc, 'displacement uc ts=' +str(ts), rf'$u_{{\mathrm{{FE}}}}^{{{ts}}}$')
                plot_disp(x_c, y_c, vc, 'displacement vc ts=' +str(ts), rf'$v_{{\mathrm{{FE}}}}^{{{ts}}}$')'''

            '''if i == 0:
                plot_disp(X,Y,U_,'displacement u ts=' +str(ts), rf'$u_{{\mathrm{{FE}}}}^{{{ts}}}$')
                plot_disp(X,Y,V_,'displacement v ts=' +str(ts), rf'$v_{{\mathrm{{FE}}}}^{{{ts}}}$')
                plot_deformation_uy(uh, V, 'deformation uy outer ts=' +str(ts))'''
        
        u_values = uh.x.array.real
        u_tot = u_values.reshape(-1,3)
        U_, V_, W_= u_tot[:,0], u_tot[:,1], u_tot[:,2]

        gdim = mesh1.geometry.dim
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

        uc = U_[index_c_resort]  
        vc = V_[index_c_resort]
        U_1 = U_[index_inner_square]
        V_1 = V_[index_inner_square]
        e_xx_c = strain_tot[index_c_resort, 0]  # epsilon_xx
        e_yy_c = strain_tot[index_c_resort, 4]  # epsilon_yy
        e_xy_c = strain_tot[index_c_resort, 1]  # epsilon_xy

        s_xx_c = sigma_tot[index_c_resort,0]
        s_yy_c = sigma_tot[index_c_resort,4]
        s_xy_c = sigma_tot[index_c_resort,1]


        uc_list.append(np.copy(uc.reshape(-1,1)))
        vc_list.append(np.copy(vc.reshape(-1,1)))
        e_xx_c_list.append(np.copy(e_xx_c.reshape(-1,1)))
        e_yy_c_list.append(np.copy(e_yy_c.reshape(-1,1)))
        e_xy_c_list.append(np.copy(e_xy_c.reshape(-1,1)))
        s_xx_c_list.append(np.copy(s_xx_c.reshape(-1,1)))
        s_yy_c_list.append(np.copy(s_yy_c.reshape(-1,1)))
        s_xy_c_list.append(np.copy(s_xy_c.reshape(-1,1)))  
        U1_list.append(np.copy(U_1.reshape(-1,1)))
        V1_list.append(np.copy(V_1.reshape(-1,1)))     

        if len(uc_list) == 1:
            plot_disp(X, Y, U_,'displacement u i=' +str(i), rf'$u_{{\mathrm{{FE}}}}^{{{i}}}$')
            plot_disp(X, Y, V_,'displacement v i=' +str(i), rf'$v_{{\mathrm{{FE}}}}^{{{i}}}$')
            plot_disp(x_disk1, y_disk1,  strain_tot[index_disk1,0], 'strain_xx', rf'$\epsilon_{{xx}}$')
            plot_bc(np.linspace(0,1,m), e_xx_c, 'strain_xx_bc')
            plot_bc(np.linspace(0,1,m), s_xy_c, 'stress_xy_bc')
            end_time_once = time.time()
            print('time cost for one sample (s):', end_time_once - start_time_once)

    except:
        print('error happened at sample:', i)
        error_index_list.append(i)
        continue
    
        
np.savetxt('u_c.txt', np.array(uc_list).squeeze().T)
np.savetxt('v_c.txt', np.array(vc_list).squeeze().T)
np.savetxt('e_xx_c.txt', np.array(e_xx_c_list).squeeze().T)
np.savetxt('e_yy_c.txt', np.array(e_yy_c_list).squeeze().T)
np.savetxt('e_xy_c.txt', np.array(e_xy_c_list).squeeze().T)
np.savetxt('s_xx_c.txt', np.array(s_xx_c_list).squeeze().T)
np.savetxt('s_yy_c.txt', np.array(s_yy_c_list).squeeze().T)
np.savetxt('s_xy_c.txt', np.array(s_xy_c_list).squeeze().T)
np.savetxt('U1.txt', np.array(U1_list).squeeze().T)
np.savetxt('V1.txt', np.array(V1_list).squeeze().T)

np.savetxt('error_index_list.txt', np.array(error_index_list))









        




