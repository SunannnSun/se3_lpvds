import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from scipy.signal import savgol_filter

from quaternion_ds.src.util.load_tools import load_clfd_dataset
from quaternion_ds.src.util.process_tools import pre_process
from quaternion_ds.src.quat_ds import quat_ds as quat_ds_class

from quaternion_ds.src.util.plot_tools import plot_quat
from quaternion_ds.src.util.plot_tools import plot_gmm_prob, plot_gmm_prob_overlay

from quaternion_ds.src.util.plot_tools import plot_pose
from quaternion_ds.src.util.plot_tools import overlay_train_test_4d, plot_gmm_on_traj, overlay_train_test_4d_iros

from damm_lpvds.src.damm_lpvds import damm_lpvds as damm_lpvds_class
from damm_lpvds.src.util.load_tools import processDataStructure, load_data
from damm_lpvds.src.util.plot_tools import plot_reference_trajectories_DS, plot_train_test_4d_demo_pos


# We run se(3) on collected data only ... hence reading it 


"""
Using load_clfd_dataset from quaternion_ds to load p_in and q_in from dataset; process the 
quaternion data using pre_process; convert position into proper scale
"""

# p_in, q_in, index_list  = load_data(3)


p_in, q_in, index_list                  = load_clfd_dataset(task_id=2, num_traj=9, sub_sample=2)
q_in, q_out, q_init, q_att, index_list  = pre_process(q_in, index_list, opt= "slerp")

quat_ds = quat_ds_class(q_in, q_out, q_init, q_att, index_list, K_init=4)



"""
Run savgol filter to append velocity; covert into 
correct format; run pre-processing and initiating the damm_lpvds class
"""



p_out = savgol_filter(p_in, window_length=81, polyorder=2, deriv=1, delta=0.01, axis=0, mode="nearest")

p = np.hstack((p_in, p_out))

L = len(index_list)
p_arr = np.zeros((L, 1), dtype=object)
for l in range(L):
    p_arr[l, 0] = p[index_list[l], :].T


Data, Data_sh, att, x0_all, dt, _, traj_length = processDataStructure(p_arr)

damm_lpvds = damm_lpvds_class(Data, Data_sh, att, x0_all, dt, traj_length)




plot_reference_trajectories_DS(Data, att, 100, 20)


"""
Begin both damm_lpvds and quaternion_ds
"""

quat_ds.begin()
damm_lpvds.begin()


# plot_gmm_on_traj(p_in, q_in, quat_ds.gmm) 

"""
Start simulation
"""

dt = 0.01
# dt =0.001

t_max = 10E3
p_i = np.mean(x0_all, axis=1, keepdims=True).reshape(-1, 1)
q_i = q_init

p_list = [p_i[:,0]]
q_list = [q_i]
w_test = []

i = 0
tol = 10E-3

while np.linalg.norm(p_list[-1]-att[:, 0]) >= tol:
# while i <= 115:

    if i > int(t_max):
        print("exceed the max iteration")

    # if i >= 85 and i < 105:
    #     # noise = - np.array([4E-3, 4E-3, 4E-3]).reshape(-1, 1)
    #     noise =  np.array([0, +8E-3, 0]).reshape(-1, 1)
    #     p_i += noise


    # noise = np.random.normal(0, 10E-4, size=(3,1))
    # p_i += noise

    p_i = damm_lpvds.step(p_i, dt) 
    # q_i, w_i = quat_ds.step(p_i.T, q_i, dt)
    q_i, w_i = quat_ds.step(q_i, dt)

    q_list.append(q_i)

    w_test.append(w_i[:, 0])

    p_list.append(p_i[:,0])

    i+=1


# np.save("w.npy", np.array(w_test))


"""
Plot the simulation Results
"""
plot_gmm_prob(np.array(w_test), title="GMM Posterior Probability of Reproduced Data")



plot_quat(q_list)

# p_arr = np.array(p_list)

# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(projection='3d')
# line = ax.plot(p_arr[:, 0], p_arr[:, 1], p_arr[:, 2], 'o')




"""
Plot the overlaying results of pose between demo and reproduction
"""

# plot_pose(p_in, p_arr, q_out=q_out, label=quat_ds.gmm.assignment_arr)

# plot_train_test_4d_demo(q_in, index_list, q_list)
# plot_train_test_4d(q_in, index_list, q_list)


# plot_train_test_4d_demo_pos(p_in, att, p_arr)


plt.show()










