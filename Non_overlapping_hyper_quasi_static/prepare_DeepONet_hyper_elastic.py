
# -*- coding: utf-8 -*-
"""PI DeepONet for 2D elasticity problem
   New structure for DeepONet
"""

# Commented out IPython magic to ensure Python compatibility.
import os
import jax
import jax.numpy as np
from jax import random, grad, vmap, jit, hessian, lax
from jax.example_libraries import optimizers
from jax.nn import relu
from jax import config
#from jax.ops import index_update, index
from jax.flatten_util import ravel_pytree

import itertools
from functools import partial
#from torch.utils import data
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import math  
from scipy.interpolate import griddata
import pickle
import scipy
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import Rbf, interp1d, griddata
# %matplotlib inline

# region neural net
# Define the neural net
def MLP(layers, activation=relu):
  ''' Vanilla MLP'''
  def init(rng_key):
      def init_layer(key, d_in, d_out):
          k1, k2 = random.split(key)
          glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
          W = glorot_stddev * random.normal(k1, (d_in, d_out))
          b = np.zeros(d_out)
          return W, b
      key, *keys = random.split(rng_key, len(layers))
      params = list(map(init_layer, keys, layers[:-1], layers[1:])) # d_in d_out initiate
      return params
  def apply(params, inputs):
      for W, b in params[:-1]:
          outputs = np.dot(inputs, W) + b
          inputs = activation(outputs)
      W, b = params[-1]
      outputs = np.dot(inputs, W) + b
      return outputs
  return init, apply

# region model 
# Define the model
class PI_DeepONet:
    def __init__(self, branch_layers, trunk_layers, **ela_model):
        # Network initialization and evaluation functions
        self.branch_init, self.branch_apply = MLP(branch_layers, activation=np.tanh)
        self.trunk_init, self.trunk_apply = MLP(trunk_layers, activation=np.tanh)
        
        #elastic_model 
        self.E = ela_model['E']
        self.nu = ela_model['nu']
        
        # Initialize
        branch_params = self.branch_init(rng_key = random.PRNGKey(1234))
        trunk_params = self.trunk_init(rng_key = random.PRNGKey(4321))
        branch_params_v = self.branch_init(rng_key = random.PRNGKey(12341))
        branch_params_ax = self.branch_init(rng_key = random.PRNGKey(12342))
        branch_params_ay = self.branch_init(rng_key = random.PRNGKey(12343))
        #trunk_params_v = self.trunk_init(rng_key = random.PRNGKey(43211))
        
        params_u = (branch_params, trunk_params)
        params_v = (branch_params_v, trunk_params)

        params = (params_u, params_v)
        
        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(optimizers.exponential_decay(1e-4,
                                                                      decay_steps=50000,
                                                                      decay_rate=0.9))
        self.opt_state = self.opt_init(params)

        # Used to restore the trained model parameters
        _, self.unravel_params = ravel_pytree(params)

        self.itercount = itertools.count()

        # Loggers
        self.loss_log = []
        self.loss_bcs_log = []
        self.loss_res_log = []
        self.loss_bcs_strain_log = [] 

    # region architecture 
    # Define DeepONet architecture
    @partial(jax.jit, static_argnums=(0,))
    def operator_net_u(self, params, u, x, y):
        branch_params, trunk_params = params
        h = np.hstack([x.reshape(-1,1), y.reshape(-1,1)])
        B = self.branch_apply(branch_params, u)  # B(u(xm)) 
        T = self.trunk_apply(trunk_params, h)
        print('B.shape=', B.shape, 'T.shape=', T.shape)
        # Compute the final output
        # Input shapes:
        # branch: [batch_size, 4m]
        # trunk: [p, 2]
        # output: [batch_size, p]
        outputs = np.einsum('ij,kj->ik', B, T)
        print(outputs.shape)
        return  outputs
    
    @partial(jax.jit, static_argnums=(0,))
    def operator_net_v(self, params, v, x, y):
        branch_params, trunk_params = params
        h = np.hstack([x.reshape(-1,1), y.reshape(-1,1)])
        B = self.branch_apply(branch_params, v)  # B(u(xm)) 
        T = self.trunk_apply(trunk_params, h)
        # Compute the final output
        # Input shapes:
        # branch: [batch_size, 4m]
        # trunk: [p, 2]
        # output: [batch_size, p]
        outputs = np.einsum('ij,kj->ik', B, T)
        return  outputs

       
    # region Def_grad 
    # Deformation gradient F = I + grad(u) 
    # F11 = 1 + u_x
    @partial(jax.jit, static_argnums=(0,))
    def F11(self, params_u, u, x, y):
        s_u_x= jax.jvp(lambda x: self.operator_net_u(params_u, u, x, y), (x,), (np.ones_like(x),))[1]
        
        return 1 + s_u_x

    # F12 = u_y
    @partial(jax.jit, static_argnums=(0,))
    def F12(self, params_u, u, x, y):
        s_u_y= jax.jvp(lambda y: self.operator_net_u(params_u, u, x, y), (y,), (np.ones_like(y),))[1]
        
        return s_u_y
    
    # F21 = v_x
    @partial(jax.jit, static_argnums=(0,))
    def F21(self, params_v, v, x, y):
        s_v_x= jax.jvp(lambda x: self.operator_net_v(params_v, v, x, y), (x,), (np.ones_like(x),))[1]
        
        return s_v_x
    
    # F22 = 1 + v_y
    @partial(jax.jit, static_argnums=(0,))
    def F22(self, params_v, v, x, y):
        s_v_y= jax.jvp(lambda y: self.operator_net_v(params_v, v, x, y), (y,), (np.ones_like(y),))[1]
        
        return 1 + s_v_y
    
    # the strain energy density function of hyper-elastic material
    @partial(jax.jit, static_argnums=(0,))
    def Psi(self, F11, F12, F21, F22):
        
        J_def = F11 * F22 - F12 * F21
        # assure the positive definiteness of J
        J = J_def
        #J = np.maximum(J_def, 1e-8) 
        kapa = self.E/3*(1-2*self.nu) # bulk modulus
        mu = self.E/(2*(1 + self.nu)) # shear modulus
        lmbda = self.nu*self.E/((1 + self.nu)*(1 - 2*self.nu)) # Lame's first parameter
        I1 = F11**2 + F12**2+ F21**2 + F22**2
        #epsilon = 1e-8 # smooth the log function
        Psi = 1/2*mu*(I1 - 3) + lmbda/2*np.log(J)**2 - mu*np.log(J)
        #Psi = mu / 2 * (I1 - 3 - 2 * np.log(J)) + lmbda / 2 * (J - 1) ** 2
        #Psi = mu/2 * (I1 - 3)  + lmbda/2 * (J - 1)**2
        return Psi

    # First derivative of Psi
    @partial(jax.jit, static_argnums=(0,))
    def Psi_deri(self, params_u, params_v, u, v, x, y):
        F11 = self.F11(params_u, u, x, y)
        F12 = self.F12(params_u, u, x, y)
        F21 = self.F21(params_v, v, x, y)
        F22 = self.F22(params_v, v, x, y)
        Psi = self.Psi(F11, F12, F21, F22)
        Psi_F11 = jax.jvp(lambda F11 : self.Psi(F11, F12, F21, F22), (F11,), (np.ones_like(F11),))[1]
        Psi_F12 = jax.jvp(lambda F12 : self.Psi(F11, F12, F21, F22), (F12,), (np.ones_like(F12),))[1]
        Psi_F21 = jax.jvp(lambda F21 : self.Psi(F11, F12, F21, F22), (F21,), (np.ones_like(F21),))[1]
        Psi_F22 = jax.jvp(lambda F22 : self.Psi(F11, F12, F21, F22), (F22,), (np.ones_like(F22),))[1]
        return Psi_F11, Psi_F12, Psi_F21, Psi_F22


    # region residual net 
    # Define ODE/PDE residual
    def residual_net(self, params, u, v, x, y):
        #s = self.operator_net(params, u, x, y)
        #s_t = grad(self.operator_net, argnums=3)(params, u, x, t)
        params_u, params_v = params
        
        F11 = self.F11(params_u, u, x, y)
        F12 = self.F12(params_u, u, x, y)
        F21 = self.F21(params_v, v, x, y)
        F22 = self.F22(params_v, v, x, y)

        Psi_dF11dx = jax.jvp(lambda x: self.Psi_deri(params_u, params_v, u, v, x, y)[0], (x,), (np.ones_like(x),))[1]
        Psi_dF12dx = jax.jvp(lambda x: self.Psi_deri(params_u, params_v, u, v, x, y)[1], (x,), (np.ones_like(x),))[1]
        Psi_dF21dy = jax.jvp(lambda y: self.Psi_deri(params_u, params_v, u, v, x, y)[2], (y,), (np.ones_like(y),))[1]
        Psi_dF22dy = jax.jvp(lambda y: self.Psi_deri(params_u, params_v, u, v, x, y)[3], (y,), (np.ones_like(y),))[1]

        ###Newton's second law in plane strain 
        # res0 = div(sigma) - f = 0 in configuration coordinates
        res0 = Psi_dF11dx + Psi_dF12dx
        res1 = Psi_dF21dy + Psi_dF22dy

        return res0, res1


    # region strain 
    # (infinitesimal)
    '''def strain(self, params, u, v, x, y):
        params_u, params_v = params
        #u, v = u.reshape(-1,1), v.reshape(-1,1)
        
        #print(s_u.shape)
        s_u_y = jax.jvp(lambda y: self.operator_net_u(params_u, u, x, y), (y,), (np.ones_like(y),))[1]
        
        s_u_x= jax.jvp(lambda x: self.operator_net_u(params_u, u, x, y), (x,), (np.ones_like(x),))[1]

        s_v_y= jax.jvp(lambda y: self.operator_net_v(params_v, v, x, y), (y,), (np.ones_like(y),))[1]
                       
        s_v_x=  jax.jvp(lambda x: self.operator_net_v(params_v, v, x, y), (x,), (np.ones_like(x),))[1]

        e_xx = s_u_x
        e_yy = s_v_y
        e_xy = (s_u_y + s_v_x)/2

        return e_xx, e_yy, e_xy'''
    # large deformation
    '''def strain(self, params, u, v, x, y):
        params_u, params_v = params

        F11 = self.F11(params_u, u, x, y)
        F12 = self.F12(params_u, u, x, y)
        F21 = self.F21(params_v, v, x, y)
        F22 = self.F22(params_v, v, x, y)

        # right Cauchy-Green deformation tensor C = F^T * F
        C11 = F11**2 + F21**2
        C22 = F12**2 + F22**2
        C12 = F11 * F12 + F21 * F22  
        #C21 = C12  

        I = np.eye(2)

        # Green-Lagrange strain tensor E = 0.5 * (C - I)
        E11 = 0.5 * (C11 - 1)
        E22 = 0.5 * (C22 - 1)
        E12 = 0.5 * (C12 - 0)

        return E11, E22, E12'''

    def stress(self, params, u, v, x, y):
        params_u, params_v = params

        F11 = self.F11(params_u, u, x, y)
        F12 = self.F12(params_u, u, x, y)
        F21 = self.F21(params_v, v, x, y)
        F22 = self.F22(params_v, v, x, y)

        J_def = F11 * F22 - F12 * F21

        C11 = F11**2 + F21**2
        C22 = F12**2 + F22**2
        C12 = F11 * F12 + F21 * F22
        C21 = C12

        det_C = C11 * C22 - C12 * C21
        
        Cinv11 = C22 / det_C
        Cinv22 = C11 / det_C
        Cinv12 = -C12 / det_C
        Cinv21 = Cinv12

        

        mu = self.E / (2 * (1 + self.nu)) * 1e6 # E 1000 Pa 
        lmbda = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu)) * 1e6

        S11 = mu * (1 - Cinv11) + lmbda * np.log(J_def) * Cinv11
        S22 = mu * (1 - Cinv22) + lmbda * np.log(J_def) * Cinv22
        S12 = mu * (0 - Cinv12) + lmbda * np.log(J_def) * Cinv12
        S21 = S12

        sigma_11 = (1 / J_def) * (F11 * (S11 * F11 + S12 * F12) + F12 * (S21 * F11 + S22 * F12))
        sigma_22 = (1 / J_def) * (F21 * (S11 * F21 + S12 * F22) + F22 * (S21 * F21 + S22 * F22))
        sigma_12 = (1 / J_def) * (F11 * (S11 * F21 + S12 * F22) + F12 * (S21 * F21 + S22 * F22))

        return sigma_11, sigma_22, sigma_12

    # region loss 
    # Define boundary loss
    def loss_bcs(self, params, batch):
        inputs, outputs = batch
        u, v, h = inputs
        params_u, params_v = params
        # Compute forward pass
        s_u_pred = self.operator_net_u(params_u, u, h[:, 0], h[:, 1])
        s_v_pred = self.operator_net_v(params_v, v, h[:, 0], h[:, 1])

        loss =  np.mean((outputs[0] - s_u_pred)**2) + np.mean((outputs[1] - s_v_pred)**2)
        return loss

    def loss_bcs_strain(self, params, batch):
        inputs, outputs = batch
        u, v, h = inputs
        # Compute forward pass
        e_xx_pred, e_yy_pred, e_xy_pred = self.stress(params, u, v, h[:, 0], h[:, 1])
        # Compute loss
        loss = np.mean((outputs[2] - e_xx_pred)**2) + np.mean((outputs[3] - e_yy_pred)**2) + np.mean((outputs[4] - e_xy_pred)**2)
        return 1e-6 * loss

    # Define residual loss
    def loss_res(self, params, batch):
        # Fetch data
        # inputs: (u1, y), shape = (Nxm, m), (Nxm,1)
        # outputs: u2, shape = (Nxm, 1)
        ## u2 is s_res, in this case s_res = u(x)
        
        inputs, outputs = batch
        u, v, h = inputs
        # Compute forward pass
        res0, res1 = self.residual_net(params, u, v, h[:,0], h[:,1])
        #print(pred.shape, outputs.shape)
        # Compute loss
        loss = np.mean((outputs[0] - res0)**2) + np.mean((outputs[1] - res1)**2)
       
        return 1e3 * loss



    # Define total loss
    def loss(self, params, bcs_batch, res_batch):
        loss_bcs = self.loss_bcs(params, bcs_batch)
        loss_res = self.loss_res(params, res_batch)
        loss_bcs_strain = self.loss_bcs_strain(params, bcs_batch)

        loss = loss_bcs + loss_res + loss_bcs_strain
        return loss

    # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, bcs_batch, res_batch):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, bcs_batch, res_batch)
        return self.opt_update(i, g, opt_state)

    # region training 
    # Optimize parameters in a loop
    def train(self, bcs_dataset, res_dataset, nIter = 10000):
        # Define data iterators
        bcs_data = iter(bcs_dataset)
        res_data = iter(res_dataset)

        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            # Fetch data
            bcs_batch= next(bcs_data)
            res_batch = next(res_data)

            self.opt_state = self.step(next(self.itercount), self.opt_state, bcs_batch, res_batch)

            if it % 1000 == 0:
                params = self.get_params(self.opt_state)

                # Compute losses
                loss_value = self.loss(params, bcs_batch, res_batch)
                loss_bcs_value = self.loss_bcs(params, bcs_batch)
                loss_res_value = self.loss_res(params, res_batch)
                loss_bcs_strain_value = self.loss_bcs_strain(params, bcs_batch)

                # Store losses
                self.loss_log.append(loss_value)
                self.loss_bcs_log.append(loss_bcs_value)
                self.loss_res_log.append(loss_res_value)
                self.loss_bcs_strain_log.append(loss_bcs_strain_value)

                # Print losses
                pbar.set_postfix({'bcs' : loss_bcs_value,
                                  'res': loss_res_value,
                                  'bcs_strain': loss_bcs_strain_value})

    # Evaluates predictions at test points
    @partial(jit, static_argnums=(0,))
    def predict_s(self, params, U_star, V_star, Y_star):
        params_u, params_v = params
        s_u_pred = self.operator_net_u(params_u, U_star, Y_star[:,0], Y_star[:,1])
        s_v_pred = self.operator_net_v(params_v, V_star, Y_star[:,0], Y_star[:,1])
        s_e_xx_pred, s_e_yy_pred, s_e_xy_pred = self.stress(params, U_star, V_star, Y_star[:,0], Y_star[:,1])

        return s_u_pred, s_v_pred, s_e_xx_pred, s_e_yy_pred, s_e_xy_pred

    @partial(jit, static_argnums=(0,))
    def predict_F(self, params, U_star, V_star, Y_star):
        params_u, params_v = params

        F11 = self.F11(params_u, U_star, Y_star[:,0], Y_star[:,1])
        F12 = self.F12(params_u, U_star, Y_star[:,0], Y_star[:,1])
        F21 = self.F21(params_v, V_star, Y_star[:,0], Y_star[:,1])
        F22 = self.F22(params_v, V_star, Y_star[:,0], Y_star[:,1])

        return F11, F12, F21, F22

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
    


def createFolder(folder_name):
    try:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    except OSError:
        print ('Error: Creating folder. ' +  folder_name)

# region Data generator
class DataGenerator():
    def __init__(self, u, v, h, s_u, s_v, s_e_xx, s_e_yy, s_e_xy, batch_size, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.u = u # input sample
        self.v = v
        self.h = h # location
        self.s_u = s_u # labeled data evulated at y (solution measurements, BC/IC conditions, etc.)
        self.s_v = s_v
        self.s_e_xx = s_e_xx
        self.s_e_yy = s_e_yy
        self.s_e_xy = s_e_xy
        self.N = u.shape[0]
        self.batch_size = batch_size  
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        #print('key', subkey)
        inputs, outputs = self.__data_generation(subkey)
        return inputs, outputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        rand_num = random.randint(key, (1,), 0, self.N + 1)[0]
        s_u = self.s_u[idx,:]
        s_v = self.s_v[idx,:]
        s_e_xx = self.s_e_xx[idx,:]
        s_e_yy = self.s_e_yy[idx,:]
        s_e_xy = self.s_e_xy[idx,:]
        h = self.h[rand_num,:,:]
        u = self.u[idx,:]
        v = self.v[idx,:]
        # Construct batch
        inputs = (u, v, h)
        outputs = (s_u, s_v, s_e_xx, s_e_yy, s_e_xy)
        return inputs, outputs

#region RBF 
### symetric RBF
def RBF(x1, x2, params): #radial basis function 
    output_scale, lengthscales = params
    diffs = np.expand_dims(x1 / lengthscales , 1) - \
            np.expand_dims(x2 / lengthscales , 0)
    r2 = np.abs(np.sum(diffs, axis=2))
    ub = np.ones(r2.shape)*np.max(x1 / lengthscales)
    d = ub -r2
    C  = np.minimum(r2, d)

    return output_scale**2 * np.exp(-0.5 * C**2)* (0.5 * C)


# To generate (x,t) (u, y)
def solve_ADR(key, Nx, Nt, P, length_scale):
    """No need explicit resolution 
    """
    # Generate subkeys
    xmin, xmax = 0, 10
    key, subkey0, subkey1= random.split(key, 3)
    subkeys0 = random.split(subkey0, 6)
    subkeys1 = random.split(subkey1, 6)
    subkeys2 = random.split(key, 6)
    subkeys3 = random.split(key, 6)
    # Generate a GP sample
    N = 512
    length_scale_u = length_scale[0]
    length_scale_v = length_scale[1]
    gp_params_u1 = (0.0008, length_scale_u)
    gp_params_u2 = (0.0008, length_scale_u)
    gp_params_v1 = (0.005, length_scale_v)
    gp_params_v2 = (0.005, length_scale_v)


    jitter = 1e-20
    X = np.linspace(xmin, xmax, N)[:,None]

    K_u1 = RBF(X, X, gp_params_u1)
    ### symetric (why np.linalg.cholesky() cannot work)
    import numpy as npr
    D, V = npr.linalg.eigh(K_u1 + jitter*np.eye(N))
    D = np.maximum(D, 0)
    L_u1 = V @ np.diag(np.sqrt(D))

    K_u2 = RBF(X, X, gp_params_u2)
    D_u2, V_u2 = npr.linalg.eigh(K_u2 + jitter*np.eye(N))
    D_u2 = np.maximum(D_u2, 0)
    L_u2 = V_u2 @ np.diag(np.sqrt(D_u2))


    K_v1 = RBF(X, X, gp_params_v1)
    D_v1, V_v1 = npr.linalg.eigh(K_v1 + jitter*np.eye(N))
    D_v1 = np.maximum(D_v1, 0)
    L_v1 = V_v1 @ np.diag(np.sqrt(D_v1))

    K_v2 = RBF(X, X, gp_params_v2)
    D_v2, V_v2 = npr.linalg.eigh(K_v2 + jitter*np.eye(N))
    D_v2 = np.maximum(D_v2, 0)
    L_v2 = V_v2 @ np.diag(np.sqrt(D_v2))
    
    def gp_sample(key, L):
        gp_sample = np.dot(L, random.normal(key, (N,)))
        return gp_sample
    
    gp_sample_u1 = vmap(gp_sample, (0,None))(subkeys0, L_u1)
    gp_sample_u2 = vmap(gp_sample, (0,None))(subkeys1, L_u2)

    gp_sample_v1 = vmap(gp_sample, (0,None))(subkeys1, L_v1)
    gp_sample_v2 = vmap(gp_sample, (0,None))(subkeys1, L_v2)
    # Create a callable interpolation function
    f_fn_u1 = lambda x: vmap(np.interp,(None, None, 0))(x, X.flatten(), gp_sample_u1)
    f_fn_u2 = lambda x: vmap(np.interp,(None, None, 0))(x, X.flatten(), gp_sample_u2)

    f_fn_v1 = lambda x: vmap(np.interp,(None, None, 0))(x, X.flatten(), gp_sample_v1)
    f_fn_v2 = lambda x: vmap(np.interp,(None, None, 0))(x, X.flatten(), gp_sample_v2)
    x_c = np.linspace(xmin, xmax, m)
    u_c1, u_c_p, u_c_n, _, _, _ = f_fn_u1(x_c)
    u_c2, _, _, _, _, _= f_fn_u2(x_c)
    v_c1, v_c_p, v_c_n, _, _, _ = f_fn_v1(x_c)
    v_c2, _, _, _, _, _ = f_fn_v2(x_c)

    # positive and negative
    '''u_c_p, v_c_p = np.maximum(u_c_p, 0), np.maximum(v_c_p, 0)
    u_c_n, v_c_n = np.minimum(u_c_n, 0), np.minimum(v_c_n, 0)'''

    # Input sensor locations and measurements

    return u_c1, v_c1 #, u_c_p, v_c_p, u_c_n, v_c_n 

# region training data 
def generate_one_training_data(key, P, Q, N):
    keys = random.split(key, N)
    keys = random.split(key, N)
    # Generate N samples of boundary conditions
    # shape of u_c and v_c is (N, m)
    #e_xx_c, e_yy_c = vmap(RBF_data_generation, (0, None, None))(keys, m, length_scale)
    os.chdir(os.path.join(originalDir, './'+ 'FE_full_square_hyper_all_dataset_N_200_one_tractions_square_CG2_dense_u_v_top_resort_real_sigma_0' + '/'))
    # Load the data
    import numpy as npr
    u_c0 = npr.loadtxt('u_c.txt').T
    v_c0 = npr.loadtxt('v_c.txt').T
    e_xx_c0 = npr.loadtxt('s_xx_c.txt').T
    e_yy_c0 = npr.loadtxt('s_yy_c.txt').T
    e_xy_c0 = npr.loadtxt('s_xy_c.txt').T
    x_c = npr.loadtxt('x_c.txt')
    y_c = npr.loadtxt('y_c.txt')
    os.chdir(os.path.join(originalDir, './'+ 'FE_full_square_hyper_all_dataset_N_200_one_tractions_square_CG2_dense_u_v_top_resort_real_sigma_1' + '/'))
    u_c1 = npr.loadtxt('u_c.txt').T
    v_c1 = npr.loadtxt('v_c.txt').T
    e_xx_c1 = npr.loadtxt('s_xx_c.txt').T
    e_yy_c1 = npr.loadtxt('s_yy_c.txt').T
    e_xy_c1 = npr.loadtxt('s_xy_c.txt').T


    u_c = np.vstack([u_c0, u_c1])
    v_c = np.vstack([v_c0, v_c1])
    e_xx_c = np.vstack([e_xx_c0, e_xx_c1])
    e_yy_c = np.vstack([e_yy_c0, e_yy_c1])
    e_xy_c = np.vstack([e_xy_c0, e_xy_c1])

    os.chdir(origin_real)


    # traning data for boundary conditions
    h_train_l = np.hstack([x_c.reshape(-1,1), y_c.reshape(-1,1)])
    h_train =np.tile(h_train_l, (N,1,1)).reshape(N, m, 2)    

    # plot the generated boundary conditions (test RBF's performance)
    plot_bc(np.linspace(0,1,m),e_xx_c.reshape(-1,m)[0,:],'e_xx_l_0')
    plot_bc(np.linspace(0,1,m),e_xx_c.reshape(-1,m)[5,:],'e_xx_l_5')
    plot_bc(np.linspace(0,1,m),e_yy_c.reshape(-1,m)[0,:],'e_yy_l_0')
    plot_bc(np.linspace(0,1,m),e_yy_c.reshape(-1,m)[5,:],'e_yy_l_5')   
    plot_bc(np.linspace(0,1,m),e_xy_c.reshape(-1,m)[0,:],'e_xy_l_0')
    plot_bc(np.linspace(0,1,m),e_xy_c.reshape(-1,m)[5,:],'e_xy_l_5')
    plot_disp(x_c, y_c, e_xx_c.reshape(-1,m)[0,:], 'e_xx_l_0', 'e_xx_l_0')

    s_e_xx_train = e_xx_c.reshape(-1,1)
    s_e_yy_train = e_yy_c.reshape(-1,1)
    s_e_xy_train = e_xy_c.reshape(-1,1)
    
    u_train = np.hstack([u_c.reshape(-1,m), v_c.reshape(-1,m)])
    v_train = np.hstack([u_c.reshape(-1,m), v_c.reshape(-1,m)])
    s_u_train = u_c.reshape(-1,1)
    s_v_train = v_c.reshape(-1,1)
    # Sample collocation points
    def generate_collocation_points(key, Q):
        subkeys = random.split(key, 2)
        r = random.uniform(subkeys[0], (Q,1), minval=0, maxval=radius)
        theta = random.uniform(subkeys[1], (Q,1), minval=0, maxval=2*np.pi)   
        xc = center[0] + r * np.cos(theta)
        yc = center[1] + r * np.sin(theta)
        
        return np.hstack([xc, yc])
    subkeys_ = random.split(key, N)
    #print('u.shape=', u.shape)
    
    # Training data for the PDE residual
    u_r_train = np.hstack([u_c, v_c])
    v_r_train = np.hstack([u_c, v_c])
    h_r_train = vmap(generate_collocation_points, (0,None))(subkeys_, Q)
    print('h_r_train.shape=', h_r_train[0,:5,:5])
    s_r_train = np.zeros((N,Q,1))  # here we can see the N(s) = u(x), we have m = Nx 
   
    #print('u_r_train.shape=', u_r_train.shape)
    return u_train, v_train,h_train, s_u_train, s_v_train, s_e_xx_train, s_e_yy_train, \
            s_e_xy_train, u_r_train, v_r_train, h_r_train, s_r_train


# Geneate training data corresponding to N input sample
def generate_training_data(key, N, P, Q):
    config.update("jax_enable_x64", True)
    u_train, v_train,h_train, s_u_train, s_v_train, s_e_xx_train, s_e_yy_train,\
                s_e_xy_train, u_r_train, v_r_train, h_r_train, s_r_train = generate_one_training_data(key, P, Q ,N)
    #print(u_r_train.shape)  ## vmap return N * u_r_train different  (N, P, m)
    print('u_train',u_train.shape)
    u_train = np.float32(u_train.reshape(N ,-1))  #turn to be (N,m)
    v_train = np.float32(v_train.reshape(N ,-1)) 
    h_train = np.float32(h_train.reshape(N , P,-1))
    s_u_train = np.float32(s_u_train.reshape(N,-1))
    s_v_train = np.float32(s_v_train.reshape(N,-1))
    s_e_xx_train = np.float32(s_e_xx_train.reshape(N,-1))
    s_e_yy_train = np.float32(s_e_yy_train.reshape(N,-1))
    s_e_xy_train = np.float32(s_e_xy_train.reshape(N,-1))
    u_r_train = np.float32(u_r_train.reshape(N ,-1))
    v_r_train = np.float32(v_r_train.reshape(N ,-1))
    h_r_train = np.float32(h_r_train.reshape(N , Q ,-1))
    print('h_r_train.shape1=', h_r_train[0,:5,:5])
    s_r_train = np.float32(s_r_train.reshape(N ,-1))

    config.update("jax_enable_x64", False)
    return     u_train, v_train,h_train, s_u_train, s_v_train, s_e_xx_train, s_e_yy_train, \
                s_e_xy_train, u_r_train, v_r_train, h_r_train, s_r_train






# region main 
if __name__ == "__main__":
    originalDir ='/nfshdd/21040463r/FEM_DeepONet_non_overlapping_coupling/non_overlapping_hyper_clean'
    os.chdir(os.path.join(originalDir))

    foldername = 'prepare_DeepONet_hyper_elastic_200w_uv_bcs_strain_one_traction_N_800_batch_100_uv_top_resort_real_sigma'  
    createFolder(foldername)
    os.chdir(os.path.join(originalDir, './'+ foldername + '/'))
    origin_real  = os.path.join(originalDir, './'+ foldername + '/')
    # GRF length scale
    length_scale = [2, 2] #0.2 for symetric RBF big length_scale
    # Resolution of the solution
    Nx = 200
    Ny = 200
    d = 2
    os.chdir(os.path.join(originalDir, './'+ 'FE_full_square_hyper_all_dataset_N_200_one_tractions_square_CG2_dense_u_v_top_resort_real_sigma_0' + '/'))
    # Load the data
    import numpy as npr
    u_c = npr.loadtxt('u_c.txt').T
    os.chdir(origin_real)

    N = u_c.shape[0] * 2#1000 # number of input samples
    print('N=', N)
    m = Nx   # number of input sensors
    P_train = m # number of output sensors, 100 for each side
    Q_train = 1600 #400  # number of collocation points for each input sample
    center = (0.0, 0.5, 0.0)  # Center of the circle
    radius = 0.3             # Radius of the circle    

    ### define the elastic model 
    ela_model = dict()
    ela_model['E'] = 0.1e-2 #1000 
    ela_model['nu'] = 0.3 

    #os.chdir('/nfsv4/21040463r/PINN/DeepONet_DR_no_ADR_to_ul_ur_vl_vr_test_rerun_0731_uxy_elastic')
    
    key = random.PRNGKey(0)
    u_bcs_train, v_bcs_train, h_bcs_train, s_u_train, s_v_train, s_e_xx_train, s_e_yy_train, s_e_xy_train, u_res_train, v_res_train, h_res_train, s_res_train\
            = generate_training_data(key, N, P_train, Q_train)
   
    # Initialize model
    branch_layers = [2*m, 100, 100, 100, 100, 800]
    trunk_layers =  [d, 100, 100, 100, 100, 800]
    model = PI_DeepONet(branch_layers, trunk_layers, **ela_model)
    
    # Create data set
    batch_size =  100 #100 
    bcs_dataset = DataGenerator(u_bcs_train, v_bcs_train, h_bcs_train, s_u_train, s_v_train, s_e_xx_train, s_e_yy_train, s_e_xy_train, batch_size )
    
    res_dataset = DataGenerator(u_res_train, v_res_train, h_res_train, s_res_train, s_res_train, s_res_train, s_res_train, s_res_train, batch_size)
    
    
    # Train
    model.train(bcs_dataset, res_dataset, nIter=2000000)
    
    # Test data
    #N_test = 100 # number of input samples
    P_test = m   # number of sensors
    #key_test = random.PRNGKey(1234567)
    #keys_test = random.split(key_test, N_test) ##for compute error with ADR
    
    # region prediction 
    # Predict

    with open('DeepONet_DR.pkl', 'rb') as f:
        params = pickle.load(f)

    '''params = model.get_params(model.opt_state)
    with open('DeepONet_DR.pkl', 'wb') as f:
        pickle.dump(params, f)


    #Plot for loss function
    plot_loss(model.loss_bcs_log, model.loss_res_log, model.loss_bcs_strain_log)
    import numpy as npr # jnp donesn't have loadtxt
    npr.savetxt('loss_bcs_log.txt', model.loss_bcs_log)
    npr.savetxt('loss_res_log.txt', model.loss_res_log)
    npr.savetxt('loss_bcs_strain_log.txt', model.loss_bcs_strain_log)'''

    ts = 4 
    import numpy as npr
    os.chdir(originalDir + '/'+ 'FE_full_square_hyper_one_traction_ground_truth_square_CG2_u_v_top_real_sigma' + '/')
    X1 = np.array(npr.loadtxt('X1.txt')).reshape(1,-1)
    Y1 = np.array(npr.loadtxt('Y1.txt')).reshape(1,-1)
    e_xx_real = np.array(npr.loadtxt('s_xx_in ts=' + str(ts) + ' .txt')).reshape(1,-1)
    e_xy_real = np.array(npr.loadtxt('s_xy_in ts=' + str(ts) + ' .txt')).reshape(1,-1)
    e_yy_real = np.array(npr.loadtxt('s_yy_in ts=' + str(ts) + ' .txt')).reshape(1,-1)
    u_real = np.array(npr.loadtxt('U_in ts=' + str(ts) + ' .txt')).reshape(1,-1)
    v_real = np.array(npr.loadtxt('V_in ts=' + str(ts) + ' .txt')).reshape(1,-1)

    os.chdir(origin_real)  


    os.chdir(os.path.join(originalDir, './'+ 'FE_full_square_hyper_all_dataset_N_200_one_tractions_square_CG2_dense_u_v_top_resort_real_sigma_0' + '/'))
    # Load the data
    x_c = npr.loadtxt('x_c.txt')
    y_c = npr.loadtxt('y_c.txt')
    os.chdir(origin_real)


    e_xx_real_fun = Rbf(X1, Y1, e_xx_real)
    e_xy_real_fun = Rbf(X1, Y1, e_xy_real)
    e_yy_real_fun = Rbf(X1, Y1, e_yy_real)
    u_real_fun = Rbf(X1, Y1, u_real)
    v_real_fun = Rbf(X1, Y1, v_real)

    # interpolate the data to the ground truth
    e_xx_real_c = e_xx_real_fun(x_c, y_c).reshape(1,-1)
    e_xy_real_c = e_xy_real_fun(x_c, y_c).reshape(1,-1)
    e_yy_real_c = e_yy_real_fun(x_c, y_c).reshape(1,-1)
    u_real_c = u_real_fun(x_c, y_c).reshape(1,-1)
    v_real_c = v_real_fun(x_c, y_c).reshape(1,-1)


    npr.savetxt('e_xx_real_c.txt', e_xx_real_c)
    npr.savetxt('e_xy_real_c.txt', e_xy_real_c)
    npr.savetxt('e_yy_real_c.txt', e_yy_real_c)
    npr.savetxt('u_real_c.txt', u_real_c)
    npr.savetxt('v_real_c.txt', v_real_c)

    e_xx_real_c = npr.loadtxt('e_xx_real_c.txt').reshape(1,-1)
    e_xy_real_c = npr.loadtxt('e_xy_real_c.txt').reshape(1,-1)
    e_yy_real_c = npr.loadtxt('e_yy_real_c.txt').reshape(1,-1)
    u_real_c = npr.loadtxt('u_real_c.txt').reshape(1,-1)
    v_real_c = npr.loadtxt('v_real_c.txt').reshape(1,-1)

    u_test = np.hstack([u_real_c, v_real_c]).reshape(1,-1)
    v_test = np.hstack([u_real_c, v_real_c]).reshape(1,-1)


    y_test = np.hstack([x_c.reshape(-1,1), y_c.reshape(-1,1)])
    s_u_pred, s_v_pred, s_e_xx_pred, s_e_yy_pred, s_e_xy_pred = model.predict_s(params, u_test, v_test, y_test)
    plot_bc(np.linspace(0,1,int(s_e_xx_pred[0,:].shape[0])),s_e_xx_pred[0,:],'test e_xx_bc')
    plot_bc(np.linspace(0,1,int(e_xx_real_c.shape[1])),e_xx_real_c.flatten(),'test e_xx_bc_real')
    plot_bc(np.linspace(0,1,int(e_xx_real_c.shape[1])),s_e_xx_pred[0,:] - e_xx_real_c.flatten(),'test e_xx_error')

    plot_bc(np.linspace(0,1,int(s_e_yy_pred[0,:].shape[0])),s_e_yy_pred[0,:],'test e_yy_bc')
    plot_bc(np.linspace(0,1,int(e_yy_real_c.shape[1])), e_yy_real_c.flatten(),'test e_yy_bc_real')
    plot_bc(np.linspace(0,1,int(e_yy_real_c.shape[1])),s_e_yy_pred[0,:] - e_yy_real_c.flatten(),'test e_yy_error')

    plot_bc(np.linspace(0,1,int(s_e_xy_pred[0,:].shape[0])),s_e_xy_pred[0,:],'test e_xy_bc')
    plot_bc(np.linspace(0,1,int(e_xy_real_c.shape[1])), e_xy_real_c.flatten(),'test e_xy_bc_real')
    plot_bc(np.linspace(0,1,int(e_xy_real_c.shape[1])),s_e_xy_pred[0,:] - e_xy_real_c.flatten(),'test e_xy_error')

    plot_bc(np.linspace(0,1,int(s_u_pred[0,:].shape[0])),s_u_pred[0,:],'test u_bc')
    plot_bc(np.linspace(0,1,int(u_real_c.shape[1])), u_real_c.flatten(),'test u_bc_real')
    plot_bc(np.linspace(0,1,int(u_real_c.shape[1])),s_u_pred[0,:] - u_real_c.flatten(),'test u_error')

    plot_bc(np.linspace(0,1,int(s_v_pred[0,:].shape[0])),s_v_pred[0,:],'test v_bc')
    plot_bc(np.linspace(0,1,int(v_real_c.shape[1])), v_real_c.flatten(),'test v_bc_real')
    plot_bc(np.linspace(0,1,int(v_real_c.shape[1])),s_v_pred[0,:] - v_real_c.flatten(),'test v_error')


    y_test = np.hstack([X1.reshape(-1,1), Y1.reshape(-1,1)])
    s_u_pred, s_v_pred, s_e_xx_pred, s_e_yy_pred, s_e_xy_pred = model.predict_s(params, u_test, v_test, y_test)

    plot_disp(X1,Y1, s_u_pred, 's_u_test1_0',  rf'$u_{{\mathrm{{NO}},\Omega_{{II}}}}$')
    plot_disp(X1,Y1, s_v_pred, 's_v_test1_0',  rf'$v_{{\mathrm{{NO}},\Omega_{{II}}}}$')   
    plot_disp(X1,Y1, u_real, 's_u_real_0', rf'$u_{{\mathrm{{FE}},\Omega_{{II}}}}$')
    plot_disp(X1,Y1, v_real, 's_v_real_0', rf'$v_{{\mathrm{{FE}},\Omega_{{II}}}}$') 
    plot_relative_error(X1,Y1, np.abs(s_u_pred - u_real), 's_u_error_01', rf'$|u_{{\mathrm{{FE}},\Omega_{{II}}}} - u_{{\mathrm{{NO}},\Omega_{{II}}}}|$')
    plot_relative_error(X1,Y1, np.abs(s_v_pred - v_real), 's_v_error_01', rf'$|v_{{\mathrm{{FE}},\Omega_{{II}}}} - v_{{\mathrm{{NO}},\Omega_{{II}}}}|$')

    plot_disp(X1,Y1, s_e_xx_pred, 's_e_xx_test1_0', rf'$\epsilon_{{xx, \mathrm{{NO}},\Omega_{{II}}}}$')
    plot_disp(X1,Y1, s_e_xy_pred, 's_e_xy_test1_0', rf'$\epsilon_{{xy, \mathrm{{NO}},\Omega_{{II}}}}$')
    plot_disp(X1,Y1, s_e_yy_pred, 's_e_yy_test1_0', rf'$\epsilon_{{yy, \mathrm{{NO}},\Omega_{{II}}}}$')
    plot_disp(X1,Y1, e_xx_real, 's_e_xx_real_0', rf'$\epsilon_{{xx,\mathrm{{FE}},\Omega_{{II}}}}$')
    plot_disp(X1,Y1, e_xy_real, 's_e_xy_real_0', rf'$\epsilon_{{xy,\mathrm{{FE}},\Omega_{{II}}}}$')
    plot_disp(X1,Y1, e_yy_real, 's_e_yy_real_0', rf'$\epsilon_{{yy,\mathrm{{FE}},\Omega_{{II}}}}$')


    plot_relative_error(X1,Y1, np.abs(s_e_xx_pred - e_xx_real), 's_e_xx_error_01', rf'$|\epsilon_{{xx,\mathrm{{FE}},\Omega_{{II}}}} - \epsilon_{{xx,\mathrm{{NO}},\Omega_{{II}}}}|$')
    plot_relative_error(X1,Y1, np.abs(s_e_xy_pred - e_xy_real), 's_e_xy_error_01', rf'$|\epsilon_{{xy,\mathrm{{FE}},\Omega_{{II}}}} - \epsilon_{{xy,\mathrm{{NO}},\Omega_{{II}}}}|$')
    plot_relative_error(X1,Y1, np.abs(s_e_yy_pred - e_yy_real), 's_e_yy_error_01', rf'$|\epsilon_{{yy,\mathrm{{FE}},\Omega_{{II}}}} - \epsilon_{{yy,\mathrm{{NO}},\Omega_{{II}}}}|$')














