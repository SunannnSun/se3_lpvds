import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from scipy.signal import savgol_filter

from quaternion_ds.util.load_tools import load_clfd_dataset
from quaternion_ds.util.process_tools import pre_process
from quaternion_ds.quat_ds_coupled import quat_ds as quat_ds_class
from quaternion_ds.util.plot_tools import plot_quat
from quaternion_ds.util.plot_tools import plot_gmm_prob

from damm_lpvds.main import damm_lpvds as damm_lpvds_class
from damm_lpvds.util.load_tools import processDataStructure
from damm_lpvds.util.plot_tools import plot_reference_trajectories_DS



"""
Using load_clfd_dataset from quaternion_ds to load p_in and q_in from dataset; process the 
quaternion data using pre_process; convert position into proper scale
"""


p_in, q_in, index_list                  = load_clfd_dataset(task_id=0, num_traj=2, sub_sample=1)
q_in, q_out, q_init, q_att, index_list  = pre_process(q_in, index_list, opt= "slerp")

p_in /= 100


quat_ds = quat_ds_class(p_in, q_in, q_out, q_att, index_list, K_init=4)


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

output_path  = 'output.json'
damm_lpvds = damm_lpvds_class(Data, Data_sh, att, x0_all, dt, traj_length, output_path)

plot_reference_trajectories_DS(Data, att, 100, 20)


"""
Begin both damm_lpvds and quaternion_ds
"""

quat_ds.begin()

damm_lpvds.begin()



"""
Start simulation
"""

dt = 0.05
t_max = 10E4
p_i = p_in[0, :].reshape(-1, 1)
q_i = q_init

p_list = [p_i[:,0]]
q_list = [q_i]
w_test = []

i = 0
tol = 10E-3
while np.linalg.norm((q_list[-1] * quat_ds.q_att.inv()).as_rotvec()) >= tol:
    if i > int(t_max):
        print("exceed the max iteration")

    p_i = damm_lpvds.step(p_i, dt) 
    q_i, w_i = quat_ds.step(p_i.T, q_i, dt)
    
    q_list.append(q_i)

    w_test.append(w_i[:, 0])

    p_list.append(p_i[:,0])




"""
Plot the simulation Results
"""

plot_gmm_prob(np.array(w_test), title="GMM Posterior Probability of Reproduced Data")

plot_quat(q_list)

p_arr = np.array(p_list)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')
line = ax.plot(p_arr[:, 0], p_arr[:, 1], p_arr[:, 2], 'o')


plt.show()










