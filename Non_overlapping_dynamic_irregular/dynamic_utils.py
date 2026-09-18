import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import os
# region utils 
def createFolder(folder_name):
    try:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    except OSError:
        print ('Error: Creating folder. ' +  folder_name)

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
    ax.set_xlabel('X', fontsize=32, fontweight='bold')
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

# The vmax and vmin are used to set the colorbar range
def plot_disp_real(X, Y, u, foldername, title):
    '''
    Plot the displacement field
    for bounded value 
    '''
    plt.figure(figsize=(10, 8.5))
    
    # Set a compact layout between the main plot and the colorbar
    ax = plt.gca()  # Get the current axis
    scatter = ax.scatter(X, Y, c=u, cmap='seismic',vmin=0, vmax=0.01)

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

def plot_loss(loss_bcs_log, loss_res_log, loss_bcs_strain_log):

    fig, ax = plt.subplots(figsize=(10, 8.5))
    ax.plot(np.arange(len(loss_bcs_log)) * 1e3 ,loss_bcs_log, lw=2, label='bcs')
    ax.plot(np.arange(len(loss_bcs_log)) * 1e3 ,loss_res_log, lw=2, label='res')
    ax.plot(np.arange(len(loss_bcs_log)) * 1e3 ,loss_bcs_strain_log, lw=2, label='bcs_strain')

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

def plot_2loss(loss_bcs_log, loss_res_log):

    fig, ax = plt.subplots(figsize=(10, 8.5))
    ax.plot(np.arange(len(loss_bcs_log)) * 1e3 ,loss_bcs_log, lw=2, label='bcs')
    ax.plot(np.arange(len(loss_bcs_log)) * 1e3 ,loss_res_log, lw=2, label='res')

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



def plot_error_list(error_list, foldername):
    """
    Plots the error list as a line figure with a logarithmic scale on the y-axis.

    Parameters:
    error_list (list or array-like): List of error values to plot.
    """
    plt.figure(figsize=(10, 8))
    plt.plot(error_list, marker='o', linestyle='-', color='b', label='Error')
    plt.yscale('log')
    plt.title(foldername)
    plt.xlabel('Iteration')
    plt.ylabel('Error (log scale)')
    plt.legend()
    plt.grid(True)
    plt.savefig(foldername + ".jpg", dpi=700)
    plt.show()


def plot_mesh(cells, points, foldername):
    '''
    plot the mesh from gmsh
    '''
    plt.figure(figsize=(8, 8))
    for cell in cells:
        polygon = points[cell]
        polygon_closed = np.vstack([polygon, polygon[0]])  
        plt.plot(polygon_closed[:, 0], polygon_closed[:, 1], 'k-', linewidth=0.5)

    plt.gca().set_aspect('equal')
    plt.xlabel('X', fontsize=32, fontweight='bold')
    plt.ylabel('Y', fontsize=32, fontweight='bold')
    plt.title(foldername, fontsize=40, fontweight='bold')
    plt.savefig( foldername + ".jpg", dpi=700)
    plt.show()

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
    
    # Remove duplicate labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

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

def plot_s(XX, TT, S_pred, filename):
    plt.figure(figsize=(6,5))
    ax = plt.subplot(1,1,1)
    #plt.pcolor(XX, TT, S_pred, cmap='seismic')
    sc = ax.scatter(XX, TT, c=S_pred, s=5, cmap='seismic')
    plt.colorbar(sc)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title(filename)
    plt.tight_layout()
    plt.savefig(filename + ".jpg", dpi=700)
    plt.show()
    plt.close()


def plot_error_list_max(ts,error_list1, error_list2, label1, label2, foldername):
    """
    Plots two error lists as line figures with a logarithmic scale on the y-axis.

    Parameters:
    error_list1 (list or array-like): First list of error values to plot (blue line).
    error_list2 (list or array-like): Second list of error values to plot (red line).
    foldername (str): Name of the file for saving the plot.
    """
    fig, ax = plt.subplots(figsize=(16, 11))
    
    # Plot the first error list
    ax.plot(
        ts, error_list1, marker='o', markersize=10, linestyle='-', linewidth=3, color='b', label=label1
    )
    
    # Plot the second error list
    ax.plot(
        ts, error_list2, marker='s', markersize=10, linestyle='--', linewidth=3, color='r', label=label2
    )

    # Adjust tick size and font size
    ax.tick_params(axis='both', which='major', labelsize=32, length=10, width=2)  # Major ticks
    ax.tick_params(axis='both', which='minor', labelsize=32, length=5, width=1)  # Minor ticks

    #plt.yscale('log')
    #plt.title(foldername)
    plt.xlabel('time step '+ rf'n', fontsize=40)
    plt.xlim([ts[0], ts[-1] + 1])
    plt.ylabel('absolute error' , fontsize=40)
    plt.ylim([0, max(max(error_list1), max(error_list2))*1.28])
    # Add legends without border and place them in the upper right corner
    plt.legend(
        loc='upper left', bbox_to_anchor=(0, 1), frameon=False, fontsize=36
    )
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0)) 
    ax.yaxis.set_major_formatter(formatter)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(32)

    # Adjust the subplot parameters to reduce right-side whitespace
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.12)
    #plt.grid(True)
    plt.savefig(foldername + ".jpg", dpi=700)
    plt.show()    
    plt.close()

def plot_error_list_max_U(ts,error_list1, error_list2, label1, label2, foldername):
    """
    Plots two error lists as line figures with a logarithmic scale on the y-axis.

    Parameters:
    error_list1 (list or array-like): First list of error values to plot (blue line).
    error_list2 (list or array-like): Second list of error values to plot (red line).
    foldername (str): Name of the file for saving the plot.
    """
    fig, ax = plt.subplots(figsize=(16, 11))
    
    # Plot the first error list
    ax.plot(
        ts, error_list1, marker='o', markersize=10, linestyle='-', linewidth=3, color='b', label=label1
    )
    
    # Plot the second error list
    ax.plot(
        ts, error_list2, marker='s', markersize=10, linestyle='--', linewidth=3, color='r', label=label2
    )

    # Adjust tick size and font size
    ax.tick_params(axis='both', which='major', labelsize=32, length=10, width=2)  # Major ticks
    ax.tick_params(axis='both', which='minor', labelsize=32, length=5, width=1)  # Minor ticks

    #plt.yscale('log')
    #plt.title(foldername)
    plt.xlabel('time step '+ rf'n', fontsize=40)
    plt.xlim([ts[0], ts[-1] + 1])
    plt.ylabel('absolute error' , fontsize=40)
    # Add legends without border and place them in the upper right corner
    plt.legend(
        loc='upper left', bbox_to_anchor=(0, 1), frameon=False, fontsize=36
    )
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0)) 

    ax.yaxis.set_major_formatter(formatter)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(32)

    # Adjust the subplot parameters to reduce right-side whitespace
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.12)
    #plt.grid(True)
    plt.savefig(foldername + ".jpg", dpi=700)
    plt.show()    
    plt.close()



def plot_error_list_max_strain(ts,error_list1, error_list2, label1, label2, foldername):
    """
    Plots two error lists as line figures with a logarithmic scale on the y-axis.

    Parameters:
    error_list1 (list or array-like): First list of error values to plot (blue line).
    error_list2 (list or array-like): Second list of error values to plot (red line).
    foldername (str): Name of the file for saving the plot.
    """
    fig, ax = plt.subplots(figsize=(16, 11))
    
    # Plot the first error list
    ax.plot(
        ts, error_list1, marker='o', markersize=10, linestyle='-', linewidth=3, color='b', label=label1
    )
    
    # Plot the second error list
    ax.plot(
        ts, error_list2, marker='s', markersize=10, linestyle='--', linewidth=3, color='r', label=label2
    )

    # Adjust tick size and font size
    ax.tick_params(axis='both', which='major', labelsize=32, length=10, width=2)  # Major ticks
    ax.tick_params(axis='both', which='minor', labelsize=32, length=5, width=1)  # Minor ticks

    #plt.yscale('log')
    #plt.title(foldername)
    plt.xlabel('time step '+ rf'n', fontsize=40)
    plt.xlim([ts[0], ts[-1] + 1])
    plt.ylabel('absolute error' , fontsize=40)
    # Add legends without border and place them in the upper right corner
    plt.legend(
        loc='upper right', bbox_to_anchor=(1, 1), frameon=False, fontsize=36
    )
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0)) 
    ax.yaxis.set_major_formatter(formatter)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(32)

    # Adjust the subplot parameters to reduce right-side whitespace
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.12)
    #plt.grid(True)
    plt.savefig(foldername + ".jpg", dpi=700)
    plt.show()    
    plt.close()


def plot_error_list(error_list1, error_list2, foldername):
    """
    Plots two error lists as line figures with a logarithmic scale on the y-axis.

    Parameters:
    error_list1 (list or array-like): First list of error values to plot (blue line).
    error_list2 (list or array-like): Second list of error values to plot (red line).
    foldername (str): Name of the file for saving the plot.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot the first error list
    ax.plot(
        np.linspace(1, error_list1.shape[0], error_list1.shape[0]), error_list1, marker='s', markersize=20, linestyle='-', linewidth=3, color='b', label='FE-NO'
    )
    
    # Plot the second error list
    ax.plot(
        np.linspace(1, error_list2.shape[0], error_list2.shape[0]), error_list2, marker='o', markersize=12, linestyle='--', linewidth=3, color='r', label='FE-FE'
    )

    # Adjust tick size and font size
    ax.tick_params(axis='both', which='major', labelsize=20, length=10, width=2)  # Major ticks
    ax.tick_params(axis='both', which='minor', labelsize=18, length=5, width=1)  # Minor ticks

    plt.yscale('log')
    #plt.title(foldername)
    plt.xlabel(r'inner iteration #', fontsize=24)
    plt.ylabel(r'$L_2$ error', fontsize=24)
    plt.xlim([0.9, len(error_list2)+1])
    # Add legends without border and place them in the upper right corner
    plt.legend(
        loc='upper right', bbox_to_anchor=(0.95, 1), frameon=False, fontsize=20
    )
    plt.grid(True)
    plt.savefig(foldername + ".jpg", dpi=700)
    plt.show()  