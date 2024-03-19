import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from scipy.signal import savgol_filter

from src.quaternion_ds.src.util.process_tools import pre_process
from src.quaternion_ds.src.quat_ds import quat_ds as quat_ds_class

from src.quaternion_ds.src.util.plot_tools import plot_gmm_prob, overlay_train_test_4d

# from quaternion_ds.src.util.plot_tools import plot_pose
# from quaternion_ds.src.util.plot_tools import overlay_train_test_4d, plot_gmm_on_traj, overlay_train_test_4d_iros

from src.damm_lpvds.src.damm_lpvds import damm_lpvds as damm_lpvds_class
from src.damm_lpvds.src.util.load_tools import processDataStructure
# from damm_lpvds.src.util.plot_tools import plot_reference_trajectories_DS, plot_train_test_4d_demo_pos


# read data from process_bag... filter out stationary point for position...

from src.util.load_tools import load_data


# Initialize quat_ds

p_in, q_in, index_list  = load_data("demo")

q_in, q_out, q_init, q_att, index_list  = pre_process(q_in, index_list, opt= "slerp")

quat_ds = quat_ds_class(q_in, q_out, q_init, q_att, index_list, K_init=4)


# Initialize damm_lpvds

L = len(index_list)
p_arr = np.empty((L, 1), dtype=object)

for l in range(L):

    p_in_l  = p_in[index_list[l], :].T
    p_out_l = savgol_filter(p_in_l, window_length=81, polyorder=2, deriv=1, delta=0.01, axis=1, mode="nearest")

    p_arr[l, 0] = np.vstack((p_in_l, p_out_l))

Data, Data_sh, att, x0_all, dt, _, traj_length = processDataStructure(p_arr)

damm_lpvds = damm_lpvds_class(Data, Data_sh, att, x0_all, dt, traj_length)



# Begin both damm_lpvds and quaternion_ds

quat_ds.begin()
damm_lpvds.begin()



# Start simulation with dt = ... ?

dt = 0.01

t_max = 10E3
p_i = np.mean(x0_all, axis=1, keepdims=True).reshape(-1, 1)
q_i = q_init

p_list = [p_i[:,0]]
q_test = [q_i]
w_test = []

i = 0
tol = 10E-3

while np.linalg.norm(p_list[-1]-att[:, 0]) >= tol:
# while i <= 300:

    if i > int(t_max):
        print("exceed the max iteration")

#     # if i >= 85 and i < 105:
#     #     # noise = - np.array([4E-3, 4E-3, 4E-3]).reshape(-1, 1)
#     #     noise =  np.array([0, +8E-3, 0]).reshape(-1, 1)
#     #     p_i += noise


#     # noise = np.random.normal(0, 10E-4, size=(3,1))
#     # p_i += noise

    p_i = damm_lpvds.step(p_i, dt) 
    q_i, w_i = quat_ds.step(q_i, dt/2)
    
    q_test.append(q_i)

    w_test.append(w_i[:, 0])

    p_list.append(p_i[:,0])

    i+=1





# Plot results

plot_gmm_prob(np.array(w_test), title="GMM Posterior Probability of Reproduced Data")

p_arr = np.array(p_list)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')
line = ax.plot(p_arr[:, 0], p_arr[:, 1], p_arr[:, 2], 'o')



plt.show()










