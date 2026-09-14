import os 
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

def createFolder(folder_name):
    try:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    except OSError:
        print ('Error: Creating folder. ' +  folder_name)


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
    Plot the displacement field
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
    cbar.ax.get_yaxis().offsetText.set_fontsize(32)  # Set scientific notation size  # fontsize of formatter 12
    #cbar.ax.get_yaxis().offsetText.set_x(1.1)  # Move scientific notation to the right 
    cbar.ax.tick_params(labelsize=32)  # Adjust the font size of the colorbar labels
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


def plot_loss(loss_bcs_log, loss_res_log, loss_comp_log):

    fig, ax = plt.subplots(figsize=(10, 8.5))
    ax.plot(np.arange(len(loss_bcs_log)) * 1e3 ,loss_bcs_log, lw=2, label='bcs')
    ax.plot(np.arange(len(loss_bcs_log)) * 1e3 ,loss_res_log, lw=2, label='res')
    ax.plot(np.arange(len(loss_bcs_log)) * 1e3 ,loss_comp_log, lw=2, label='bcs_strain')

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

def plot_boundary(mesh, facet_values, facet_indices,colors, foldername, fdim = 1):
    '''
    plot the boundary of the mesh
    params: 
            mesh: the mesh object
            facet_values: the values of the facets
            facet_indices: the indices of the facets
            colors: the color map for the tags
            foldername: the foldername to save the plot
    
    '''

    plt.figure(figsize=(8, 8))
    # Loop through the unique tag values and plot the facets with the same color
    for tag in np.unique(facet_values):
        facet_indices_with_tag = facet_indices[facet_values == tag]
        color = colors.get(tag, 'black')  # Use 'black' if the tag is not in the color map
        for facet_idx in facet_indices_with_tag:
            # retrieve the dofs of the facet  (cause Lagrangian 1 order element CG 1)
            # DOF is the vertex of the mesh
            facet_dofs = mesh.topology.connectivity(fdim, 0).links(facet_idx)
            # retrieve the coordinates of the facet
            facet_coords = mesh.geometry.x[facet_dofs]
            plt.plot(facet_coords[:, 0], facet_coords[:, 1], color=color, label=f"Tag {tag}")
            plt.savefig(foldername)

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