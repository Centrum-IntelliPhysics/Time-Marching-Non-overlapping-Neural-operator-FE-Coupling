"""
=======================================================================
Part of the FEM-DeepONet coupling work
-----------------------------------------------------------------------
DeepONet Model for elasto-dynamic for time steps 89 to 149
2D plane strain problem
Square + disk domain (non-overlapping boundary)
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
from scipy.interpolate import griddata, Rbf
import pickle
import scipy
import jax.nn as jnn
from jax.lax import conv_general_dilated as conv_lax
from interpax import Interpolator2D
import flax.linen as fnn
from dynamic_utils import plot_disp, plot_relative_error, plot_loss4, createFolder

import torch
import torch.nn as nn
import torch.nn.functional as F
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



#region PointNet
# Initialize the Glorot (Xavier) normal distribution for weight initialization
initializer = jax.nn.initializers.glorot_normal()
rng_key = random.PRNGKey(0)
key1, key2, key3, key4 = random.split(rng_key, 4)

def conv1d(x, w, b, pool_window=2, pool_strides=2, use_pooling = False):
    """1D Convolution operation with VALID padding"""
    # x shape: (L, C_in) - 1D input with length L and C_in channels
    # w shape: (K, C_in, C_out) - Kernel with length K, C_in input channels, C_out output channels
    
    conv_out = conv_lax(
        lhs=x[None, ..., None],  # Add batch and dummy height dimensions: (1, L, 1, C_in)
        rhs=w[..., None, :],     # Reshape kernel: (K, 1, C_in, C_out)
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    
    # Remove batch and dummy height dimensions, add bias
    conv_out = conv_out[0, :, 0, :] + b
    
    # 1D average pooling
    if use_pooling:
        pooled_out = fnn.avg_pool(
            conv_out[None, :, None, :],  # Add batch and dummy height dimensions: (1, L, 1, C_out)
            window_shape=(1, pool_window),
            strides=(1, pool_strides),
            padding='VALID'
        )
        
        # Remove batch and dummy height dimensions
        return pooled_out[0, :, 0, :]
    else:
        return conv_out  # Return without pooling


def init_SimplePointNet_params(p, key=random.PRNGKey(0)):
    """
    Initialize Simplified PointNet_2D parameters for the branch network as a list of tuples.
    Each tuple contains (weights, biases) for a layer.

    Returns:
    list: List of tuples, each containing weights and biases for a layer
    """
    key1, key2, key3, key4, key5, key6 = random.split(key, 6)
    input_channels = 6 # (x, y, u, v, vx, vy) coordinates + displacements + velocities
    out_put_channels1 = 12#32
    out_put_channels2 = 24#64
    out_put_channels3 = 48#256
    # Conv1: (3,3,4,6) - (3 * 3) kernal input has 4 channels, 6 output filters ### error maybe 
    # 4 channels for u v vx vy in the previous step 
    conv1_w = random.normal(key1, (1, input_channels, out_put_channels1)) * 0.1 #kernal size is 1 * 1 for conv1d
    conv1_b = random.normal(key2, (out_put_channels1,)) * 0.1                               

    conv2_w = random.normal(key2, (1, out_put_channels1, out_put_channels2)) * 0.1
    conv2_b = random.normal(key3, (out_put_channels2,)) * 0.1

    conv3_w = random.normal(key3, (1, out_put_channels2, out_put_channels3)) * 0.1
    conv3_b = random.normal(key4, (out_put_channels3,)) * 0.1

    # Calculate the output size after the convolutions and pooling
    '''conv_output_length = N_num_ptx # number of input points
    flat_size = conv_output_length * out_put_channels3 # no pooling and 1D conv'''

    dense1_w = random.normal(key4, (out_put_channels3, 32)) * np.sqrt(2.0 / out_put_channels3)
    dense1_b = np.zeros(32)

    dense2_w = random.normal(key5, (32, 24)) * np.sqrt(2.0 / 128)
    dense2_b = np.zeros(24)

    dense3_w = random.normal(key6, (24, p)) * np.sqrt(2.0 / 64)
    dense3_b = np.zeros(p)

    return [
        (conv1_w, conv1_b),
        (conv2_w, conv2_b),
        (conv3_w, conv3_b),
        (dense1_w, dense1_b),
        (dense2_w, dense2_b),
        (dense3_w, dense3_b)
    ]



def BranchNet_dil_PointNet(params, x):
    """
    CNN-based branch network for the DeepONet.

    Args:
    params (list): List of tuples containing weights and biases
    x (array): Input tensor of shape (batch_size, N_num_ptx, 6)

    Returns:
    array: Output tensor of shape (batch_size, 2*p)
    """
    print('x shape in PointNet branch net', x.shape)
    X1_in, Y1_in, U1, V1, vx, vy = x[:, :m_s], x[:, m_s:2*m_s], x[:, 2*m_s:3*m_s], x[:, 3*m_s:4*m_s], x[:, 4*m_s:5*m_s], x[:, 5*m_s:6*m_s]
    #print('X1_in shape', X1_in.shape)
    x = np.concatenate([X1_in[:, :, None], Y1_in[:, :, None], U1[:, :, None], 
                                     V1[:, :, None], vx[:, :, None], vy[:, :, None]], axis = 2) # (Batch, N, 6)
    print('x shape after concat in PointNet branch net', x.shape)

    def single_forward(params, x):
        # Unpack conv and dense layer parameters
        (conv1_w, conv1_b), (conv2_w, conv2_b), (conv3_w, conv3_b), \
        (dense1_w, dense1_b), (dense2_w, dense2_b), (dense3_w, dense3_b) = params

        '''# Reshape input to (m_i, m_i, 4) - adding channel dimension
        #x = np.transpose(x.reshape(4, nx1, nx1), (1,2,0))'''

        # x shape 1batch: (X, Y, u, v, vx, vy)  (N, 6)

        # Convolution layers with SiLU activation
        x = np.tanh(conv1d(x, conv1_w, conv1_b))
        x = np.tanh(conv1d(x, conv2_w, conv2_b))
        x = np.tanh(conv1d(x, conv3_w, conv3_b))

        print('x shape before max pool', x.shape)
        x = np.max(x, axis=0, keepdims=False)  # (N, C) --> (C,) max pooling over points
        print('x shape after max pool', x.shape)

        # Flatten
        #x = x.reshape(-1) # add the boundary sensor points
        # Dense layers
        x = jnn.silu(np.dot(x, dense1_w) + dense1_b)
        x = jnn.silu(np.dot(x, dense2_w) + dense2_b)
        outputs = np.dot(x, dense3_w) + dense3_b

        return outputs

    return vmap(partial(single_forward, params))(x)




# region model 
# Define the model
class PI_DeepONet:
    def __init__(self, branch_layers_1, trunk_layers, *test_data, **ela_model):
        # Network initialization and evaluation functions
        self.branch_init_1, self.branch_apply_1 = MLP(branch_layers_1, activation=np.tanh)
        self.trunk_init, self.trunk_apply = MLP(trunk_layers, activation=np.tanh)
        self.branch_apply = BranchNet_dil_PointNet
        #elastic_model 
        self.E = ela_model['E']
        self.nu = ela_model['nu']
        self.rho = ela_model['rho']
        
        # Initialize
        branch_params = init_SimplePointNet_params(trunk_layers[-1],  key = random.PRNGKey(1234))
        branch_params_v = init_SimplePointNet_params(trunk_layers[-1], key = random.PRNGKey(12341))

        branch_params_1 = self.branch_init_1(rng_key = random.PRNGKey(123411))
        branch_params_v_1 = self.branch_init_1(rng_key = random.PRNGKey(1234111))

        trunk_params_u = self.trunk_init(rng_key = random.PRNGKey(4321))
        trunk_params_v = self.trunk_init(rng_key = random.PRNGKey(43211))
        
        #trunk_params_v = self.trunk_init(rng_key = random.PRNGKey(43211))
        
        params_u = (branch_params, branch_params_1, trunk_params_u)
        params_v = (branch_params_v, branch_params_v_1, trunk_params_v)


        params = (params_u, params_v)
        
        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(optimizers.exponential_decay(1e-3,
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
        self.loss_test_log = []

        self.test_data = test_data


    # region architecture 
    # Define DeepONet architecture
    @partial(jax.jit, static_argnums=(0,))
    def operator_net_u(self, params, v, x, y):
        branch_params_v, branch_params_v_1, trunk_params_v = params
        h = np.hstack([x.reshape(-1,1), y.reshape(-1,1)])
        B = self.branch_apply(branch_params_v, v[:,:-2*m])  
        B_1 = self.branch_apply_1(branch_params_v_1, v[:,-2*m:])
        T = self.trunk_apply(trunk_params_v, h)
        # Compute the final output
        # Input shapes:
        # branch: [batch_size, 4m1]
        # branch_1: [batch_size, 2m]
        # trunk: [p, 2]
        # output: [batch_size, p]
        B_tot = B * B_1 # element-wise multiplication keep the same shape
        outputs = np.einsum('ij,kj->ik', B_tot, T)
        return  outputs
    
    @partial(jax.jit, static_argnums=(0,))
    def operator_net_v(self, params, v, x, y):
        branch_params_v, branch_params_v_1, trunk_params_v = params
        h = np.hstack([x.reshape(-1,1), y.reshape(-1,1)])
        B = self.branch_apply(branch_params_v, v[:,:-2*m])  
        B_1 = self.branch_apply_1(branch_params_v_1, v[:,-2*m:])
        T = self.trunk_apply(trunk_params_v, h)
        # Compute the final output
        # Input shapes:
        # branch: [batch_size, 4m1]
        # branch_1: [batch_size, 2m]
        # trunk: [p, 2]
        # output: [batch_size, p]
        B_tot = B * B_1 # element-wise multiplication keep the same shape
        outputs = np.einsum('ij,kj->ik', B_tot, T)
        return  outputs

       
    # region residual net 
    # Define ODE/PDE residual
    def residual_net(self, params, u, v, x, y):
        #s = self.operator_net(params, u, x, y)
        #s_t = grad(self.operator_net, argnums=3)(params, u, x, t)
        params_u, params_v = params

        s_u_yy= jax.jvp(lambda y : jax.jvp(lambda y: self.operator_net_u(params_u, u, x, y), (y,), (np.ones_like(y),))[1]
                        , (y,), (np.ones_like(y),))[1]
        s_u_xx= jax.jvp(lambda x : jax.jvp(lambda x: self.operator_net_u(params_u, u, x, y), (x,), (np.ones_like(x),))[1]    
                        , (x,), (np.ones_like(x),))[1]
        s_u_xy= jax.jvp(lambda x : jax.jvp(lambda y: self.operator_net_u(params_u, u, x, y), (y,), (np.ones_like(y),))[1]
                        , (x,), (np.ones_like(x),))[1]
        s_v_yy= jax.jvp(lambda y : jax.jvp(lambda y: self.operator_net_v(params_v, v, x, y), (y,), (np.ones_like(y),))[1]
                        , (y,), (np.ones_like(y),))[1]
        s_v_xx= jax.jvp(lambda x : jax.jvp(lambda x: self.operator_net_v(params_v, v, x, y), (x,), (np.ones_like(x),))[1]
                        , (x,), (np.ones_like(x),))[1] 
        s_v_xy= jax.jvp(lambda x : jax.jvp(lambda y: self.operator_net_v(params_v, v, x, y), (y,), (np.ones_like(y),))[1]
                        , (x,), (np.ones_like(x),))[1]
        
        
        u_old, v_old, vx_old, vy_old = u[:, 2*m_s:3*m_s], u[:, 3*m_s:4*m_s], u[:, 4*m_s:5*m_s]*100, u[:, 5*m_s:6*m_s]*100 # :ms for X1_in

        s_ax = -2/(dt*beta)**2*(u_old + vx_old*dt - self.operator_net_u(params_u, u, x, y)) #+ (1-beta)/(beta) * ax_old
        s_ay = -2/(dt*beta)**2*(v_old + vy_old*dt - self.operator_net_v(params_v, v, x, y)) #+ (1-beta)/(beta) * ay_old

        para1 = self.E/((1 + self.nu)*(1-2*self.nu))
        ###Newton's second law in plane strain 
        res0 = para1*((1-self.nu) * s_u_xx + self.nu * s_v_xy) + para1*(1-2*self.nu)/2*(s_u_yy + s_v_xy) - self.rho * s_ax
        res1 = para1*((1-self.nu) * s_v_yy + self.nu * s_u_xy) + para1*(1-2*self.nu)/2*(s_v_xx + s_u_xy) - self.rho * s_ay

        return res0, res1
    
    
    # region stress
    def stress(self, params, u, v, Y_star):
        params_u, params_v = params
        x, y = Y_star[:,0], Y_star[:,1]
        
        s_u_y = jax.jvp(lambda y: self.operator_net_u(params_u, u, x, y), (y,), (np.ones_like(y),))[1]
        
        s_u_x= jax.jvp(lambda x: self.operator_net_u(params_u, u, x, y), (x,), (np.ones_like(x),))[1]

        s_v_y= jax.jvp(lambda y: self.operator_net_v(params_v, v, x, y), (y,), (np.ones_like(y),))[1]
                       
        s_v_x=  jax.jvp(lambda x: self.operator_net_v(params_v, v, x, y), (x,), (np.ones_like(x),))[1]

        para1 = self.E/((1 + self.nu)*(1-2*self.nu))*1e8
        sigma_x = para1*((1-self.nu) * s_u_x + self.nu * s_v_y) 
        sigma_y = para1*((1-self.nu) * s_v_y + self.nu * s_u_x) 
        sigma_xy = para1*(1-2*self.nu) * (s_u_y + s_v_x)/2  

        return sigma_x, sigma_y, sigma_xy

    # region strain
    def strain(self, params, u, v, x, y):
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

        return e_xx, e_yy, e_xy        
    
    # region loss 
    # Define boundary loss
    def loss_bcs(self, params, batch):
        inputs, outputs = batch
        u, v, h = inputs
        params_u, params_v, = params
        # Compute forward pass
        s_u_pred = self.operator_net_u(params_u, u, h[:, 0], h[:, 1])
        s_v_pred = self.operator_net_v(params_v, v, h[:, 0], h[:, 1])
        ## outputs always be 0 
        loss =  np.mean((outputs[0]- s_u_pred)**2) + np.mean((outputs[1] - s_v_pred)**2)
        return loss

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
        # Compute loss
        loss = np.mean((outputs[0] - res0)**2) + np.mean((outputs[1] - res1)**2)
       
        return loss

    # Define boundary loss for strain
    def loss_bcs_strain(self, params, batch):
        inputs, outputs = batch
        u, v, h = inputs
        # Compute forward pass
        e_xx_pred, e_yy_pred, e_xy_pred = self.strain(params, u, v, h[:, 0], h[:, 1])
        # Compute loss
        loss = np.mean((outputs[2] - e_xx_pred)**2) + np.mean((outputs[3] - e_yy_pred)**2) + np.mean((outputs[4] - e_xy_pred)**2)
        return 1e-3 * loss

    # Define test loss
    @partial(jit, static_argnums=(0,))
    def loss_test(self, params):
        params_u, params_v = params
        U1_test, V1_test, UV_pred = self.test_data
        y_test = np.hstack([X1_real.reshape(-1,1), Y1_real.reshape(-1,1)])
        print('y_test shape in test loss', y_test.shape)
        s_u_pred = self.operator_net_u(params_u, U1_test, y_test[:,0], y_test[:,1])
        s_v_pred = self.operator_net_v(params_v, V1_test, y_test[:,0], y_test[:,1])
        s_uv_pred = np.hstack([s_u_pred, s_v_pred])
        print('UV_pred shape in test loss', UV_pred.shape, 's_u_pred shape', s_u_pred.shape, 's_uv_pred shape', s_uv_pred.shape)
        loss = np.mean((UV_pred- s_uv_pred)**2)
        return loss


    # Define total loss
    def loss(self, params, bcs_batch, res_batch):
        loss_bcs = self.loss_bcs(params, bcs_batch)
        loss_res = self.loss_res(params, res_batch)
        loss_bcs_strain =self.loss_bcs_strain(params, bcs_batch)
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
                loss_bcs_value_strain = self.loss_bcs_strain(params, bcs_batch)
                loss_test_value = self.loss_test(params)

                # Store losses
                self.loss_log.append(loss_value)
                self.loss_bcs_log.append(loss_bcs_value)
                self.loss_res_log.append(loss_res_value)
                self.loss_bcs_strain_log.append(loss_bcs_value_strain)
                self.loss_test_log.append(loss_test_value)


                # Print losses
                pbar.set_postfix({#'Loss': loss_value,
                                  'bcs' : loss_bcs_value,
                                  'res': loss_res_value,
                                  'strain': loss_bcs_value_strain,
                                  'test': loss_test_value
                                  })

    # Evaluates predictions at test points
    @partial(jit, static_argnums=(0,))
    def predict_s(self, params, U_star, V_star, Y_star):
        params_u, params_v = params
        s_u_pred = self.operator_net_u(params_u, U_star, Y_star[:,0], Y_star[:,1])
        s_v_pred = self.operator_net_v(params_v, V_star, Y_star[:,0], Y_star[:,1])

        s_e_xx_pred, s_e_yy_pred, s_e_xy_pred = self.strain(params, U_star, V_star, Y_star[:,0], Y_star[:,1])
        return s_u_pred, s_v_pred, s_e_xx_pred, s_e_yy_pred, s_e_xy_pred

    @partial(jit, static_argnums=(0,))
    def predict_res(self, params, U_star, V_star, Y_star):
        r_pred = vmap(self.residual_net, (None, 0, 0, 0))(params, U_star, V_star, Y_star[:,0], Y_star[:,1])
        return r_pred  
    

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
        inputs, outputs = self.__data_generation(subkey)
        return inputs, outputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        rand_num = random.randint(key, (1,), 0, self.N + 1)[0]
        s_u = self.s_u[idx,:]
        s_v = self.s_v[idx,:]
        s_e_xx_train = self.s_e_xx[idx,:]
        s_e_yy_train = self.s_e_yy[idx,:]
        s_e_xy_train = self.s_e_xy[idx,:]

        h = self.h[rand_num,:,:]
        u = self.u[idx,:]
        v = self.v[idx,:]
        # Construct batch
        inputs = (u, v, h)
        outputs = (s_u, s_v, s_e_xx_train, s_e_yy_train, s_e_xy_train)
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

    return output_scale**2 * np.exp(-0.5 * C**2)

    

# To generate (x,t) (u, y)
def solve_ADR(key, Nx, Nt, P, length_scale):
    """No need explicit resolution 
    """
    # Generate subkeys
    xmin, xmax = 0, 10
    key, subkey0, subkey1= random.split(key, 3)
    subkeys0 = random.split(subkey0, 6)
    subkeys1 = random.split(subkey1, 6)
    # Generate a GP sample
    N = 512
    length_scale_u = length_scale[0]
    length_scale_a = length_scale[1]
    gp_params = (0.01, length_scale_u)
    gp_params_a = (0.03, length_scale_a)

    jitter = 1e-10
    X = np.linspace(xmin, xmax, N)[:,None]
    K = RBF(X, X, gp_params)
    ### symetric (why np.linalg.cholesky() cannot work)
    import numpy as npr
    D, V = npr.linalg.eigh(K + jitter*np.eye(N))
    D = np.maximum(D, 0)
    L = V @ np.diag(np.sqrt(D))

    K_a = RBF(X, X, gp_params_a)
    ### symetric (why np.linalg.cholesky() cannot work)
    D_a, V_a = npr.linalg.eigh(K_a + jitter*np.eye(N))
    D_a = np.maximum(D_a, 0)
    L_a = V_a @ np.diag(np.sqrt(D_a))
    
    def gp_sample(key, L):
        gp_sample = np.dot(L, random.normal(key, (N,)))
        return gp_sample
    
    gp_sample_u = vmap(gp_sample, (0,None))(subkeys0, L)
    gp_sample_a = vmap(gp_sample, (0,None))(subkeys1, L_a)
    # Create a callable interpolation function
    f_fn_uv = lambda x: vmap(np.interp,(None, None, 0))(x, X.flatten(), gp_sample_u)
    f_fn_a = lambda x: vmap(np.interp,(None, None, 0))(x, X.flatten(), gp_sample_a)
    x_c = np.linspace(xmin, xmax, m)
    u_c, v_c, u_c_p, v_c_p, u_c_n, v_c_n = f_fn_uv(x_c)
    # positive and negative
    # exchange the positive and negative profile (since ax_c_n ay_c_n have larger zero internal)
    ax_c, ay_c, ax_c_p, ay_c_p, ax_c_n, ay_c_n = f_fn_a(x_c)
    
    u_c_p, v_c_p = np.maximum(ax_c_p, 0)/3, np.maximum(ay_c_p, 0)/3
    u_c_n, v_c_n = np.minimum(ax_c_n, 0)/3, np.minimum(ay_c_n, 0)/3
    ax_c_p, ay_c_p = np.maximum(u_c_p, 0)*3, np.maximum(v_c_p, 0)*3
    ax_c_n, ay_c_n = np.minimum(u_c_n, 0)*3, np.minimum(v_c_n, 0)*3


    # Create grid
    num_points = m           # m is the sensor number  
    x_list = []
    y_list = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        x_list.append(x)
        y_list.append(y)  

    x = np.array(x_list).reshape(-1,1)
    y = np.array(y_list).reshape(-1,1)
    # Input sensor locations and measurements

    return u_c, v_c, u_c_p, v_c_p, u_c_n, v_c_n


# region training data 
def generate_one_training_data(key, P, Q, N):
    keys = random.split(key, int(N/2))
    # load correct dataset 
    os.chdir(os.path.join(originalDir, './' + 'dataload_from_full_square_disk_dataset_89_169_epsilon_CG2_dense' + '/'))
    U1 = npr.loadtxt('U1.txt') 
    V1 = npr.loadtxt('V1.txt') 
    vx = npr.loadtxt('vx.txt')
    vy = npr.loadtxt('vy.txt')
    e_xx_c = npr.loadtxt('e_xx_c.txt')
    e_yy_c = npr.loadtxt('e_yy_c.txt')
    e_xy_c = npr.loadtxt('e_xy_c.txt')
    u_c = npr.loadtxt('u_c.txt')
    v_c = npr.loadtxt('v_c.txt')
    X1 = npr.loadtxt('X1.txt') 
    Y1 = npr.loadtxt('Y1.txt')
    x_c = npr.loadtxt('x_c.txt')
    y_c = npr.loadtxt('y_c.txt')

    #U1_d = npr.loadtxt('U1_d.txt')  
    #print('U1 shape', U1.shape, U1_d.shape)
    row_delete = list(range(80, U1.shape[0], 81))

    U1 = npr.delete(U1, row_delete, axis=0)
    V1 = npr.delete(V1, row_delete, axis=0)
    vx = npr.delete(vx, row_delete, axis=0)
    vy = npr.delete(vy, row_delete, axis=0)

    row_delete_bc = list(range(0, u_c.shape[0], 81))
    u_c = npr.delete(u_c, row_delete_bc, axis=0)
    v_c = npr.delete(v_c, row_delete_bc, axis=0)
    e_xx_c = npr.delete(e_xx_c, row_delete_bc, axis=0)
    e_yy_c = npr.delete(e_yy_c, row_delete_bc, axis=0)
    e_xy_c = npr.delete(e_xy_c, row_delete_bc, axis=0)
    
    os.chdir(originalDir_real)
    X1_in = np.tile(X1, (N,1))
    Y1_in = np.tile(Y1, (N,1))

    # Sample points from the boundary and the inital conditions

    h_train_tot = np.hstack([x_c.reshape(-1,1), y_c.reshape(-1,1)])  # (m, 2)
    h_train =np.tile(h_train_tot, (N,1,1)).reshape(N, m, 2)    

    # Training data for BC and IC

    #print('shape', X1_in.shape, Y1_in.shape, U1.shape, V1.shape, vx.shape, vy.shape, u_c.shape, v_c.shape)
    u_train = np.hstack([X1_in, Y1_in, U1, V1, vx, vy, u_c, v_c])

    v_train = np.hstack([X1_in, Y1_in, U1, V1, vx, vy, u_c, v_c])
    
    s_u_train = u_c
    s_v_train = v_c
    s_e_xx_train = e_xx_c
    s_e_yy_train = e_yy_c
    s_e_xy_train = e_xy_c
    
    # Training data for the PDE residual
    u_r_train = np.hstack([X1_in, Y1_in, U1, V1, vx, vy, u_c, v_c])
    v_r_train = np.hstack([X1_in, Y1_in, U1, V1, vx, vy, u_c, v_c])
    h_r_train = np.tile(np.hstack([X1.flatten().reshape(-1,1), Y1.flatten().reshape(-1,1)]), (N,1,1))
    s_r_train = np.zeros((N, X1.shape[0], 1))  # here we can see the N(s) = u(x), we have m = Nx 

    return u_train, v_train, h_train, s_u_train, s_v_train, s_e_xx_train, s_e_yy_train, s_e_xy_train, \
             u_r_train, v_r_train, h_r_train, s_r_train



# Geneate training data corresponding to N input sample
def generate_training_data(key, N, P, Q):
    config.update("jax_enable_x64", True)
    u_train, v_train, h_train, s_u_train, s_v_train, s_e_xx_train, s_e_yy_train, s_e_xy_train,\
            u_r_train, v_r_train, h_r_train, s_r_train = generate_one_training_data(key, P, Q ,N)

    u_train = np.float32(u_train.reshape(N ,-1))  #turn to be (N, 4*m1 + 2*4*m)
    v_train = np.float32(v_train.reshape(N ,-1)) 
    h_train = np.float32(h_train.reshape(N ,-1, 2))
    s_u_train = np.float32(s_u_train.reshape(N,-1))
    s_v_train = np.float32(s_v_train.reshape(N,-1))
    s_e_xx_train = np.float32(s_e_xx_train.reshape(N,-1))
    s_e_yy_train = np.float32(s_e_yy_train.reshape(N,-1))
    s_e_xy_train = np.float32(s_e_xy_train.reshape(N,-1))
    u_r_train = np.float32(u_r_train.reshape(N ,-1))
    v_r_train = np.float32(v_r_train.reshape(N ,-1))
    h_r_train = np.float32(h_r_train.reshape(N , -1 ,2))
    s_r_train = np.float32(s_r_train.reshape(N ,-1))

    config.update("jax_enable_x64", False)
    return     u_train, v_train, h_train, s_u_train, s_v_train, s_e_xx_train, s_e_yy_train, s_e_xy_train,\
                u_r_train, v_r_train, h_r_train, s_r_train
    


# region save path 
originalDir = '/nfshdd/21040463r/FEM_DeepONet_non_overlapping_coupling/non_overlapping_figures/elasto_dynamic_irregular'
os.chdir(os.path.join(originalDir))
foldername = 'prepare_DeepONet_disk_dynamic_square_disk_ts_89_169_200w_test5'
createFolder(foldername)
os.chdir(os.path.join(originalDir, './' + foldername + '/'))
originalDir_real = os.path.join(originalDir, './' + foldername + '/')

os.chdir(os.path.join(originalDir, './' + 'dataload_from_full_square_disk_dataset_89_169_epsilon_CG2_dense' + '/'))
# GRF length scale
length_scale = [1, 0.4] #0.2 for symetric RBF big length_scale
import numpy as npr 
# Resolution of the solution
center = (0.0, 0.5, 0.0)  # Center of the circle
radius = 0.3             # Radius of the circle
X1 = npr.loadtxt('X1.txt')
Y1 = npr.loadtxt('Y1.txt')
x_c = npr.loadtxt('x_c.txt')
y_c = npr.loadtxt('y_c.txt')
# resort the sequence of the index (easy for the following application in coupling)
N_num_ptx = X1.shape[0]

#nx1 = index_up.shape[0]
#nx2 = index_up.shape[0]

m_s = N_num_ptx # number of sensors in the square
d = 2
N = 8000 # number of input samples
m = x_c.shape[0] # number of input sensors
P_train = m # number of output sensors, 100 for each side
Q_train = 400 #400  # number of collocation points for each input sample

print('ms, m', m_s, m)
#nemark method
beta = 1
# Time-stepping parameters
T       = 4.0
Nsteps  = 1e4
dt =  T/Nsteps

# test data generation 
#real test   
os.chdir(os.path.join(originalDir, './' + 'L_shape_ground_truth' + '/'))
import numpy as npr # jnp donesn't have loadtxt
trans = 0.3
X1_real = npr.loadtxt('X1.txt') 
Y1_real = npr.loadtxt('Y1.txt') 
print('X1_real shape', X1_real.shape)

U1 = npr.loadtxt('U1 ts = 131.txt')
V1 = npr.loadtxt('V1 ts = 131.txt')
U1_new = npr.loadtxt('U1 ts = 132.txt')
V1_new = npr.loadtxt('V1 ts = 132.txt')
vx = npr.loadtxt('vx ts = 131.txt')/100
vy = npr.loadtxt('vy ts = 131.txt')/100

e_xx_new = npr.loadtxt('epsilon_x1 ts = 132.txt')
e_yy_new = npr.loadtxt('epsilon_y1 ts = 132.txt')
e_xy_new = npr.loadtxt('epsilon_xy1 ts = 132.txt')

index_c = np.concatenate([
np.where((Y1_real == y_val) & (X1_real == x_val))[0]
for x_val, y_val in zip(x_c, y_c)
])  

u_c = U1_new[index_c]
v_c = V1_new[index_c]
print('u_r shape', u_c.shape)

os.chdir(os.path.join(originalDir, './' + 'dataload_from_full_square_disk_dataset_89_169_epsilon_CG2_dense_test5' + '/'))
U1_test5 = npr.loadtxt('U1.txt') 
V1_test5 = npr.loadtxt('V1.txt') 
vx_test5 = npr.loadtxt('vx.txt')
vy_test5 = npr.loadtxt('vy.txt')
e_xx_c_test5 = npr.loadtxt('e_xx_c.txt')
e_yy_c_test5 = npr.loadtxt('e_yy_c.txt')
e_xy_c_test5 = npr.loadtxt('e_xy_c.txt')
u_c_test5 = npr.loadtxt('u_c.txt')
v_c_test5 = npr.loadtxt('v_c.txt')
X1_test5 = npr.loadtxt('X1.txt') 
Y1_test5 = npr.loadtxt('Y1.txt')
x_c_test5 = npr.loadtxt('x_c.txt')
y_c_test5 = npr.loadtxt('y_c.txt')

U1_0_old, U1_0_new, V1_0_old, V1_0_new = U1_test5[0,:], U1_test5[1,:], V1_test5[0,:], V1_test5[1,:]
U1_1_old, U1_1_new, V1_1_old, V1_1_new = U1_test5[12,:], U1_test5[13,:], V1_test5[12,:], V1_test5[13,:]
U1_2_old, U1_2_new, V1_2_old, V1_2_new = U1_test5[24,:], U1_test5[25,:], V1_test5[24,:], V1_test5[25,:]
U1_3_old, U1_3_new, V1_3_old, V1_3_new = U1_test5[36,:], U1_test5[37,:], V1_test5[36,:], V1_test5[37,:]
U1_4_old, U1_4_new, V1_4_old, V1_4_new = U1_test5[48,:], U1_test5[49,:], V1_test5[48,:], V1_test5[49,:]

vx_0_old, vy_0_old, vx_1_old, vy_1_old, vx_2_old, vy_2_old, vx_3_old, vy_3_old, vx_4_old, vy_4_old = \
    vx_test5[0,:], vy_test5[0,:], vx_test5[12,:], vy_test5[12,:], vx_test5[24,:], vy_test5[24,:], \
    vx_test5[36,:], vy_test5[36,:], vx_test5[48,:], vy_test5[48,:]

u_c_0_old, v_c_0_old, u_c_1_old, v_c_1_old, u_c_2_old, v_c_2_old, u_c_3_old, v_c_3_old, u_c_4_old, v_c_4_old = \
    u_c_test5[0,:], v_c_test5[0,:], u_c_test5[12,:], v_c_test5[12,:], u_c_test5[24,:], v_c_test5[24,:], \
    u_c_test5[36,:], v_c_test5[36,:], u_c_test5[48,:], v_c_test5[48,:]

U1_0_test = np.hstack([U1_0_old.reshape(1, -1), V1_0_old.reshape(1, -1), vx_0_old.reshape(1, -1), vy_0_old.reshape(1, -1), u_c_0_old.reshape(1, -1), v_c_0_old.reshape(1, -1)])
U1_1_test = np.hstack([U1_1_old.reshape(1, -1), V1_1_old.reshape(1, -1), vx_1_old.reshape(1, -1), vy_1_old.reshape(1, -1), u_c_1_old.reshape(1, -1), v_c_1_old.reshape(1, -1)])
U1_2_test = np.hstack([U1_2_old.reshape(1, -1), V1_2_old.reshape(1, -1), vx_2_old.reshape(1, -1), vy_2_old.reshape(1, -1), u_c_2_old.reshape(1, -1), v_c_2_old.reshape(1, -1)])
U1_3_test = np.hstack([U1_3_old.reshape(1, -1), V1_3_old.reshape(1, -1), vx_3_old.reshape(1, -1), vy_3_old.reshape(1, -1), u_c_3_old.reshape(1, -1), v_c_3_old.reshape(1, -1)])
U1_4_test = np.hstack([U1_4_old.reshape(1, -1), V1_4_old.reshape(1, -1), vx_4_old.reshape(1, -1), vy_4_old.reshape(1, -1), u_c_4_old.reshape(1, -1), v_c_4_old.reshape(1, -1)])
Y_test = np.tile(np.hstack([X1.real.reshape(1,-1), Y1.real.reshape(1,-1)]), (5, 1))

U1_0_pred = np.hstack([U1_0_new.reshape(1, -1), V1_0_new.reshape(1, -1)])
U1_1_pred = np.hstack([U1_1_new.reshape(1, -1), V1_1_new.reshape(1, -1)])
U1_2_pred = np.hstack([U1_2_new.reshape(1, -1), V1_2_new.reshape(1, -1)])
U1_3_pred = np.hstack([U1_3_new.reshape(1, -1), V1_3_new.reshape(1, -1)])
U1_4_pred = np.hstack([U1_4_new.reshape(1, -1), V1_4_new.reshape(1, -1)])

U1_pred5 = np.vstack([U1_0_pred, U1_1_pred, U1_2_pred, U1_3_pred, U1_4_pred])
U1_test5 = np.hstack([Y_test, np.vstack([U1_0_test, U1_1_test, U1_2_test, U1_3_test, U1_4_test])])
V1_test5 = U1_test5.copy()
test_data = U1_test5, V1_test5, U1_pred5
os.chdir(originalDir_real)

npr.savetxt('X1_NO.txt', X1_real)
npr.savetxt('Y1_NO.txt', Y1_real)
npr.savetxt('x_c_NO.txt', x_c)
npr.savetxt('y_c_NO.txt', y_c)
# region main 
# let the out disk data be zero

if __name__ == "__main__":
    
    # the coefficient means a lot? 
    ### define the elastic model 
    ela_model = dict()
    ela_model['E'] = 1000e-8 #1000 
    ela_model['nu'] = 0.3 
    ela_model['rho'] = 5e-8 #5
    #os.chdir('/nfsv4/21040463r/PINN/DeepONet_DR_no_ADR_to_ul_ur_vl_vr_test_rerun_0731_uxy_elastic')
    
    key = random.PRNGKey(0)
    u_bcs_train, v_bcs_train, h_bcs_train, s_u_train, s_v_train, s_e_xx_train, s_e_yy_train, s_e_xy_train,\
        u_res_train, v_res_train, h_res_train, s_res_train= generate_training_data(key, N, P_train, Q_train)
   
    # Initialize model
    branch_layers_1 =  [2*m, 100, 100, 100, 100, 800]
    trunk_layers =  [d, 100, 100, 100, 100, 800]
    model = PI_DeepONet(branch_layers_1, trunk_layers, *test_data, **ela_model)
    
    # Create data set
    batch_size =  10 #10000
    bcs_dataset = DataGenerator(u_bcs_train, v_bcs_train, h_bcs_train, s_u_train, 
                               s_v_train, s_e_xx_train, s_e_yy_train, s_e_xy_train, batch_size)
    
    res_dataset = DataGenerator(u_res_train, v_res_train, h_res_train, s_res_train, 
                                s_res_train, s_res_train, s_res_train, s_res_train, batch_size)
    
    # batch_size big is more accurate. 
    
    #bcs_dataset = inputs, outputs   
    #inputs = (u, y) #outputs = s   
    
    # Train
    model.train(bcs_dataset, res_dataset, nIter=2000000)
    
    # Test data
    #N_test = 100 # number of input samples
    P_test = m   # number of sensors
    
    # region prediction 
    # Predict
    '''with open('DeepONet_ED_89_169.pkl', 'rb') as f:
        params = pickle.load(f)'''
        
    params = model.get_params(model.opt_state)
    with open('DeepONet_ED_89_169.pkl', 'wb') as f:
        pickle.dump(params, f)


    #Plot for loss function
    plot_loss4(model.loss_bcs_log, model.loss_res_log, model.loss_bcs_strain_log, model.loss_test_log)
    npr.savetxt('loss_bcs_log.txt', npr.array(model.loss_bcs_log))
    npr.savetxt('loss_res_log.txt', npr.array(model.loss_res_log))
    npr.savetxt('loss_bcs_strain_log.txt', npr.array(model.loss_bcs_strain_log))
    npr.savetxt('loss_test_log.txt', npr.array(model.loss_test_log))
    

    '''# interpolate data 
    U1_d = Rbf(X1_real, Y1_real, U1)(X1_, Y1_)
    V1_d = Rbf(X1_real, Y1_real, V1)(X1_, Y1_)
    vx_d = Rbf(X1_real, Y1_real, vx)(X1_, Y1_)
    vy_d = Rbf(X1_real, Y1_real, vy)(X1_, Y1_)

    u_l = Rbf(X1_real, Y1_real, U1_new)(X1[index_left], Y1[index_left])
    u_r = Rbf(X1_real, Y1_real, U1_new)(X1[index_right], Y1[index_right])
    u_up = Rbf(X1_real, Y1_real, U1_new)(X1[index_up], Y1[index_up])
    u_down = Rbf(X1_real, Y1_real, U1_new)(X1[index_down], Y1[index_down])
    v_l = Rbf(X1_real, Y1_real, V1_new)(X1[index_left], Y1[index_left])
    v_r = Rbf(X1_real, Y1_real, V1_new)(X1[index_right], Y1[index_right])
    v_up = Rbf(X1_real, Y1_real, V1_new)(X1[index_up], Y1[index_up])
    v_down = Rbf(X1_real, Y1_real, V1_new)(X1[index_down], Y1[index_down])'''


    u_test =np.hstack([X1.real.reshape(1,-1), Y1.real.reshape(1,-1), U1.reshape(1,-1), V1.reshape(1,-1), 
                       vx.reshape(1,-1), vy.reshape(1,-1), u_c.reshape(1,-1), v_c.reshape(1,-1)]) 
    
    v_test =np.hstack([X1.real.reshape(1,-1), Y1.real.reshape(1,-1), U1.reshape(1,-1), V1.reshape(1,-1), 
                       vx.reshape(1,-1), vy.reshape(1,-1), u_c.reshape(1,-1), v_c.reshape(1,-1)])
      
    #y_test = np.hstack([XX.flatten()[:,None], YY.flatten()[:,None]])
    y_test = np.hstack([X1_real.reshape(-1,1), Y1_real.reshape(-1,1)])
    s_u_pred, s_v_pred, s_e_xx_pred, s_e_yy_pred, s_e_xy_pred = model.predict_s(params, u_test, v_test, y_test)

    # Plot
    plot_disp(X1_real, Y1_real, s_u_pred, 'u_x_pred', rf'$u_{{x, \mathrm{{NO}},\Omega_{{II}}}}^{{{132}}}$')
    plot_disp(X1_real, Y1_real, s_v_pred,'u_y_pred',rf'$u_{{y, \mathrm{{NO}},\Omega_{{II}}}}^{{{132}}}$')
    plot_disp(X1_real, Y1_real,  U1_new.flatten(), 'u_x_truth', rf'$u_{{x, \mathrm{{FE}},\Omega_{{II}}}}^{{{132}}}$')
    plot_disp(X1_real, Y1_real,  V1_new.flatten(), 'u_y_truth', rf'$u_{{y, \mathrm{{FE}},\Omega_{{II}}}}^{{{132}}}$')
    plot_relative_error(X1_real, Y1_real, np.abs(s_u_pred.flatten() - U1_new.flatten()), 'u_x_error',rf'$|u_{{x, \mathrm{{FE}},\Omega_{{II}}}}^{{{132}}} - u_{{x, \mathrm{{NO}},\Omega_{{II}}}}^{{{132}}}|$')
    plot_relative_error(X1_real, Y1_real, np.abs(s_v_pred.flatten() - V1_new.flatten()), 'u_y_error',rf'$|u_{{y, \mathrm{{FE}},\Omega_{{II}}}}^{{{132}}} - u_{{y, \mathrm{{NO}},\Omega_{{II}}}}^{{{132}}}|$')

    plot_disp(X1_real, Y1_real, s_e_xx_pred, 'e_xx_pred', rf'$\epsilon_{{xx, \mathrm{{NO}},\Omega_{{II}}}}^{{{132}}}$')
    plot_disp(X1_real, Y1_real, s_e_yy_pred, 'e_yy_pred', rf'$\epsilon_{{yy, \mathrm{{NO}},\Omega_{{II}}}}^{{{132}}}$')
    plot_disp(X1_real, Y1_real, s_e_xy_pred, 'e_xy_pred', rf'$\epsilon_{{xy, \mathrm{{NO}},\Omega_{{II}}}}^{{{132}}}$')
    plot_disp(X1_real, Y1_real, e_xx_new.flatten(), 'e_xx_truth', rf'$\epsilon_{{xx, \mathrm{{FE}},\Omega_{{II}}}}^{{{132}}}$')
    plot_disp(X1_real, Y1_real, e_yy_new.flatten(), 'e_yy_truth', rf'$\epsilon_{{yy, \mathrm{{FE}},\Omega_{{II}}}}^{{{132}}}$')
    plot_disp(X1_real, Y1_real, e_xy_new.flatten(), 'e_xy_truth', rf'$\epsilon_{{xy, \mathrm{{FE}},\Omega_{{II}}}}^{{{132}}}$')

    plot_relative_error(X1_real, Y1_real, np.abs(s_e_xx_pred.flatten() - e_xx_new.flatten()), 'e_xx_error',rf'$|\epsilon_{{xx, \mathrm{{FE}},\Omega_{{II}}}}^{{{132}}} - \epsilon_{{xx, \mathrm{{NO}},\Omega_{{II}}}}^{{{132}}}|$')
    plot_relative_error(X1_real, Y1_real, np.abs(s_e_yy_pred.flatten() - e_yy_new.flatten()), 'e_yy_error',rf'$|\epsilon_{{yy, \mathrm{{FE}},\Omega_{{II}}}}^{{{132}}} - \epsilon_{{yy, \mathrm{{NO}},\Omega_{{II}}}}^{{{132}}}|$')
    plot_relative_error(X1_real, Y1_real, np.abs(s_e_xy_pred.flatten() - e_xy_new.flatten()), 'e_xy_error',rf'$|\epsilon_{{xy, \mathrm{{FE}},\Omega_{{II}}}}^{{{132}}} - \epsilon_{{xy, \mathrm{{NO}},\Omega_{{II}}}}^{{{132}}}|$')














