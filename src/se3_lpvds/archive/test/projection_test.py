import numpy as np
import matplotlib.pyplot as plt
from quat_ds import quat_ds as quat_ds_class
from util import plot_tools, traj_generator, quat_tools
from scipy.spatial.transform import Rotation as R


"""
Goal: 
    During optimization, riem_log is used to project w_train wrt points of tangency. There are multiple candidates
    of such points corresponding to how w_train is expressed in either body frame or world frame. 


Result: 
    Body frame:
        - Problematic because it does not converge to zero or converge in the very beginning then diverge after projection
        - Such behaviour is kind of hard to achieve via a linear system plus no stability guarantee
        - Mainly due to the coupling between the current orientation and relative angular velcoity

    - World frame:    
        - Good tendency to converge and remain independent of the current orientation after projection wrt 
          either identity or attractor

Remark:
    - Unusually large number in riem_log and prallel_transport; needs to be investigated
"""


# Produce a trajectory and extract w_train in local frame
rng_seed =  np.random.RandomState(seed=2)
q_init, q_att, q_train, w_train_local, dt= traj_generator.generate_traj(K=2, N=60, q_init=R.random())

w_train_local_arr  = quat_tools.list_to_arr(w_train_local)
# plot_tools.plot_4d_coord(w_train_local_arr, title='w_train_local in quaternion')

N = len(q_train)
dt= 0.1



# Transform w_train wrt global frame
w_train_global = [q_train[i]*w_train_local[i]*q_train[i].inv() for i in range(N-1)]
w_train_global_arr  = quat_tools.list_to_arr(w_train_global)
# plot_tools.plot_4d_coord(w_train_global_arr, title='w_train_global in quaternion')



# Quick verification by integrating w_train_global
q_train_global = [q_init] * N
for i in range(N-1):
    q_k   = q_train_global[i]
    w_k   = w_train_global[i]
    d_q_k = w_k.as_rotvec() * dt 
    q_kp1 = q_k * R.from_rotvec(d_q_k) 
    q_train_global[i+1] = q_kp1

q_train_arr         = quat_tools.list_to_arr(q_train)
q_train_global_arr  = quat_tools.list_to_arr(q_train_global)

plot_tools.plot_quat(q_train)
# plot_tools.plot_quat(q_train_global)



# Project w_train_local wrt their corresponding q_train
w_train_local_wrt_body  = quat_tools.riem_log(q_train[:-1], w_train_local)
plot_tools.plot_4d_coord(w_train_local_wrt_body, title="w_train_local_wrt_body")


# Parallel transport the projected w_train_local from body to attractor
# w_train_local_wrt_att  =  quat_tools.parallel_transport(q_train[:-1], q_att, w_train_local_wrt_body)
# plot_tools.plot_4d_coord(w_train_local_wrt_att, title='w_train_local_wrt_att')


# Project w_train_global wrt identity
w_train_global_wrt_id  = quat_tools.riem_log(R.identity(), w_train_global)
plot_tools.plot_4d_coord(w_train_global_wrt_id, title='w_train_global_wrt_id')


# Parallel transport the projected w_train_global from identity to attractor
w_train_global_wrt_att  = quat_tools.parallel_transport(R.identity().inv(), q_att, w_train_global_wrt_id)
# plot_tools.plot_4d_coord(w_train_global_wrt_att, title='w_train_global_wrt_att')


plt.show()
