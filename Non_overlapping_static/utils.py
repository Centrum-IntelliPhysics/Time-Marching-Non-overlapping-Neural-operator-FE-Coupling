
import os
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
import numpy as np
import pyvista
from dolfinx import plot


# region utils 
#utils 
def plot_disp(X, Y, u, foldername, title):
    '''
    Plot the displacement field
    '''
    plt.figure(figsize=(10, 8.5))
    
    # Set a compact layout between the main plot and the colorbar
    ax = plt.gca()  # Get the current axis
    scatter = ax.scatter(X, Y, c=u, cmap='seismic')
    #scatter = ax.scatter(X, Y, c=u, cmap='viridis')  # Use 'viridis' colormap for better visibility

    # Add a colorbar
    cbar = plt.colorbar(scatter, orientation='vertical', pad=0.02, location='right')  # Reduce the padding
    cbar.formatter = ScalarFormatter()  # Set the default formatter
    cbar.formatter.set_scientific(True)  # Enable scientific notation
    cbar.formatter.set_powerlimits((-2, 2))  # Show scientific notation for numbers smaller than 0.01
    cbar.ax.get_yaxis().offsetText.set_fontsize(32)  # Set scientific notation size  # fontsize of formatter 12
    #cbar.ax.get_yaxis().offsetText.set_x(1.1)  # Move scientific notation to the right 
    cbar.ax.tick_params(labelsize=32)  # Adjust the font size of the colorbar labels
    # Set font properties
    ax.set_xlabel('x', fontsize=32, fontweight='bold')
    ax.set_ylabel('Y', fontsize=32, fontweight='bold')
    ax.set_title(title, fontsize=40, fontweight='bold', y = 1.05)  # Increase the distance between the title and the plot

    # Adjust tick size and font size
    ax.tick_params(axis='both', which='major', labelsize=32, length=10, width=2)  # Major ticks
    ax.tick_params(axis='both', which='minor', labelsize=32, length=5, width=1)  # Minor ticks

    # Adjust the subplot layout
    plt.tight_layout(rect=[0, 0, 0.95, 1])  # The rect parameter controls the overall layout (reduce right margin)

    # Save and display
    plt.savefig(foldername + ".jpg", dpi=700, bbox_inches='tight')  # bbox_inches='tight' reduces the extra margins
    plt.show()
    plt.close()

def plot_relative_error(X, Y, u, foldername, title):
    '''
    Plot the relative error field
    '''
    plt.figure(figsize=(10, 8.5))
    
    # Set a compact layout between the main plot and the colorbar
    ax = plt.gca()  # Get the current axis
    scatter = ax.scatter(X, Y, c=u, cmap='viridis')

    # Add a colorbar
    cbar = plt.colorbar(scatter, orientation='vertical', pad=0.02, location='right')  # Reduce the padding
    cbar.formatter = ScalarFormatter()  # Set the default formatter
    cbar.formatter.set_scientific(True)  # Enable scientific notation
    cbar.formatter.set_powerlimits((-2, 2))  # Show scientific notation for numbers smaller than 0.01
    
    
    
    # Adjust scientific notation offset text (e.g., "1e-3")
    cbar.ax.get_yaxis().offsetText.set_fontsize(32)  # Set scientific notation size
    cbar.ax.get_yaxis().offsetText.set_position((1.1, 0.5))  # Move offset text further right
    
    cbar.ax.tick_params(labelsize=32)  # Adjust the font size of the colorbar labels
    cbar.ax.yaxis.offsetText.set_position((3.0, 0.5))  # Move offset text further right

    # Set font properties
    ax.set_xlabel('X', fontsize=32, fontweight='bold')
    ax.set_ylabel('Y', fontsize=32, fontweight='bold')
    ax.set_title(title, fontsize=40, fontweight='bold')
    
    # Adjust tick size and font size
    ax.tick_params(axis='both', which='major', labelsize=32, length=10, width=2)  # Major ticks
    ax.tick_params(axis='both', which='minor', labelsize=32, length=5, width=1)  # Minor ticks

    # Adjust the subplot layout
    plt.tight_layout(rect=[0, 0, 0.95, 1])  # The rect parameter controls the overall layout (reduce right margin)

    # Save and display
    plt.savefig(foldername + ".jpg", dpi=700, bbox_inches='tight')  # bbox_inches='tight' reduces the extra margins
    plt.show()
    plt.close()


def plot_loss(loss_bcs_log, loss_res_log, loss_bcs_trains_log):

    fig, ax = plt.subplots(figsize=(10, 8.5))
    ax.plot(np.arange(len(loss_bcs_log)) * 1e3 ,loss_bcs_log, lw=2, label='bcs')
    ax.plot(np.arange(len(loss_bcs_log)) * 1e3 ,loss_res_log, lw=2, label='res')
    ax.plot(np.arange(len(loss_bcs_log)) * 1e3 ,loss_bcs_trains_log, lw=2, label='bcs_train')

    plt.xlabel('Iteration', fontsize=32)
    plt.ylabel('Loss', fontsize=32)
    plt.yscale('log')
    plt.legend(
        loc='upper right', bbox_to_anchor=(0.95, 1), frameon=False, fontsize=36
    )
    # Adjust tick size and font size
    ax.tick_params(axis='both', which='major', labelsize=32, length=10, width=2)  # Major ticks
    ax.tick_params(axis='both', which='minor', labelsize=32, length=5, width=1)  # Minor ticks

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0)) 
    ax.xaxis.set_major_formatter(formatter)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.xaxis.get_offset_text().set_fontsize(32)

    plt.tight_layout()
    plt.savefig('loss' + ".jpg", dpi=700)
    plt.show()
    plt.close()
    
def plot_bc(bc, dis, filename):
    plt.figure(figsize = (6,5))
    plt.plot(bc,dis, lw=2, label=filename)
    plt.xlabel('x')
    plt.ylabel('displament')
    #plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename + ".jpg", dpi=700)
    plt.show()
    plt.close()    
    


def createFolder(folder_name):
    try:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    except OSError:
        print ('Error: Creating folder. ' +  folder_name)


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

def plot_loss4(loss_bcs_log, loss_res_log, loss_bcs_trains_log, loss_test_log):

    fig, ax = plt.subplots(figsize=(10, 8.5))
    ax.plot(np.arange(len(loss_bcs_log)) * 1e3 ,loss_bcs_log, lw=2, label='bcs')
    ax.plot(np.arange(len(loss_bcs_log)) * 1e3 ,loss_res_log, lw=2, label='res')
    ax.plot(np.arange(len(loss_bcs_log)) * 1e3 ,loss_bcs_trains_log, lw=2, label='bcs_train')
    ax.plot(np.arange(len(loss_bcs_log)) * 1e3 ,loss_test_log, lw=2, label='test')

    plt.xlabel('Iteration', fontsize=32)
    plt.ylabel('Loss', fontsize=32)
    plt.yscale('log')
    plt.legend(
        loc='upper right', bbox_to_anchor=(0.95, 1), frameon=False, fontsize=36
    )
    # Adjust tick size and font size
    ax.tick_params(axis='both', which='major', labelsize=32, length=10, width=2)  # Major ticks
    ax.tick_params(axis='both', which='minor', labelsize=32, length=5, width=1)  # Minor ticks

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0)) 
    ax.xaxis.set_major_formatter(formatter)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.xaxis.get_offset_text().set_fontsize(32)

    plt.tight_layout()
    plt.savefig('loss' + ".jpg", dpi=700)
    plt.show()
    plt.close()