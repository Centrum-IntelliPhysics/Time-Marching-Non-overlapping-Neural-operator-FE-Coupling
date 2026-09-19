import numpy as np
import os 
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from joblib import Parallel, delayed
import time 
from dynamic_utils import plot_s, createFolder


originalDir =os.path.dirname(os.path.abspath(__file__))
print('curent working directory:', originalDir)
os.chdir(os.path.join(originalDir))

foldername = 'dataload_from_full_square_disk_dataset_89_169_epsilon_CG2_dense'
createFolder(foldername)
os.chdir(os.path.join(originalDir, './' + foldername + '/'))
originalDir_real = os.path.join(originalDir, './' + foldername + '/')


U1_list, V1_list, vx_list, vy_list, U1c_new_list, V1c_new_list = [], [], [], [], [], []
e_xx_list, e_yy_list, e_xy_list = [], [], []
ax_list, ay_list = [], []   

# load the data from the full square dataset
for i in range(1):
    os.chdir(os.path.join(originalDir, './' + f'full_square_with_L_shape_all_dataset_N100_89_169_CG2_all_ts_together' + '/'))
    print(os.getcwd())
    U1 = np.loadtxt('U1 ts = 89 - 169.txt')
    print(U1.shape)
    V1 = np.loadtxt('V1 ts = 89 - 169.txt')
    vx = np.loadtxt('vx ts = 89 - 169.txt')/100
    vy = np.loadtxt('vy ts = 89 - 169.txt')/100
    ax = np.loadtxt('ax ts = 89 - 169.txt')/10000
    ay = np.loadtxt('ay ts = 89 - 169.txt')/10000
    e_xx = np.loadtxt('epsilon_x1 ts = 89 - 169.txt')
    e_yy = np.loadtxt('epsilon_y1 ts = 89 - 169.txt')
    e_xy = np.loadtxt('epsilon_xy1 ts = 89 - 169.txt')

    U1_list.append(U1)
    V1_list.append(V1)
    vx_list.append(vx)
    vy_list.append(vy)
    ax_list.append(ax)
    ay_list.append(ay)
    e_xx_list.append(e_xx)
    e_yy_list.append(e_yy)
    e_xy_list.append(e_xy)



X1 = np.loadtxt('X1.txt')
Y1 = np.loadtxt('Y1.txt')
X_BC_in = np.loadtxt('X_BC_in.txt').flatten()
Y_BC_in = np.loadtxt('Y_BC_in.txt').flatten()
X_BC, Y_BC = np.loadtxt('X_BC.txt').flatten(), np.loadtxt('Y_BC.txt').flatten()

# use X_BC and Y_BC to prepare the coupling with FEM


index_c = np.array([np.where((np.isclose(x0, X1)) & (np.isclose(y_0, Y1)))[0] for x0, y_0 in zip(X_BC, Y_BC)]).flatten()

print('index_c shape=', index_c.shape)

os.chdir(originalDir_real)
np.savetxt('X1.txt', X1)
np.savetxt('Y1.txt', Y1)

plot_s(X1, Y1, U1[:,0], 'U1_0_1')
plot_s(X1, Y1, U1[:,11], 'U1_11_1')
plot_s(X1, Y1, U1[:,21], 'U1_21_1')

plot_s(X1, Y1, e_xy[:,0], 'e_xy_0_1')
plot_s(X1, Y1, e_xy[:,11], 'e_xy_11_1')
plot_s(X1, Y1, e_xy[:,21], 'e_xy_21_1') 

# transpose the data (points, ts*N) --> (ts*N, points)
U1 = np.hstack(U1_list).T
V1 = np.hstack(V1_list).T
vx = np.hstack(vx_list).T
vy = np.hstack(vy_list).T
ax = np.hstack(ax_list).T
ay = np.hstack(ay_list).T
e_xx = np.hstack(e_xx_list).T
e_yy = np.hstack(e_yy_list).T
e_xy = np.hstack(e_xy_list).T

# save the raw data 
np.savetxt('U1.txt', U1)
np.savetxt('V1.txt', V1)
np.savetxt('vx.txt', vx)
np.savetxt('vy.txt', vy)
np.savetxt('ax.txt', ax)
np.savetxt('ay.txt', ay)
np.savetxt('e_xx.txt', e_xx)
np.savetxt('e_yy.txt', e_yy)
np.savetxt('e_xy.txt', e_xy)

print('U1.T shape=', U1.shape)

N = U1.shape[0]

x_c = X1[index_c]
y_c = Y1[index_c]

np.savetxt('x_c.txt', x_c)
np.savetxt('y_c.txt', y_c)

u_c, v_c = U1[:, index_c], V1[:, index_c]
e_xx_c, e_yy_c, e_xy_c = e_xx[:, index_c], e_yy[:, index_c], e_xy[:, index_c]
print('u_c', u_c.shape)
np.savetxt('u_c.txt', u_c)
np.savetxt('v_c.txt', v_c)
np.savetxt('e_xx_c.txt', e_xx_c)
np.savetxt('e_yy_c.txt', e_yy_c)
np.savetxt('e_xy_c.txt', e_xy_c)








