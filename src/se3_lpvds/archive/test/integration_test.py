import numpy as np
import matplotlib.pyplot as plt
from quat_ds import quat_ds as quat_ds_class
from util import plot_tools, traj_generator, quat_tools
from scipy.spatial.transform import Rotation as R

"""
Test to ensure proper integration and propoer transformation between world and body frame
"""


# dt is 0.1; q_train in global frame; w_train in local frame
q_init, q_att, q_train, w_train = traj_generator.generate_traj(K=1, N=80)

dt = 0.1
N = len(q_train)


# Reproduce the q_train from q_init using local angular velcoity
q_train_local = [q_init] * N

for i in range(N-1):
    q_k   = q_train_local[i]
    w_k   = w_train[i]
    d_q_k = w_k.as_rotvec() * dt 
    q_kp1 = R.from_rotvec(d_q_k) * q_k
    q_train_local[i+1] = q_kp1



# Reproduce the q_train from q_init using global angular velcoity
q_train_global = [q_init] * N

for i in range(N-1):
    q_k   = q_train_local[i]
    w_k   = q_k * w_train[i] * q_k.inv()
    d_q_k = w_k.as_rotvec() * dt 
    q_kp1 = q_k * R.from_rotvec(d_q_k) 
    q_train_global[i+1] = q_kp1



# Verify the reproduced q_train from both frames are the same as the origin
q_train_arr         = quat_tools.list_to_arr(q_train)
q_train_local_arr   = quat_tools.list_to_arr(q_train_local)
q_train_global_arr  = quat_tools.list_to_arr(q_train_global)


plot_tools.plot_quat(q_train)
plot_tools.plot_quat(q_train_local)
plot_tools.plot_quat(q_train_global)

plt.show()
