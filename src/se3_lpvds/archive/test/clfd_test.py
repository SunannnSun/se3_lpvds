import os
from tkinter import W
import numpy as np
from util import plot_tools
from scipy.spatial.transform import Rotation as R
from util import plot_tools, optimize_tools, quat_tools
import matplotlib.pyplot as plt
from util.gmm import gmm as gmm_class
from util.quat_tools import *
from scipy.signal import savgol_filter
from load_pos_ori import load_clfd_dataset


"""
Remark: Loading clfd result Done; extracting and smoothing velocity Done;


STILL not solving the double cover of quaternion tho
"""

q_init, q_att, q_train_local, w_train_local, dt = load_clfd_dataset(task_id=2, num_traj=1, sub_sample=10)

N = len(q_train_local)

"""
Goal: verify if w_train is right; THIS IS "forward simulation"
Result: YES
"""
q_train_global = [q_init] * N

for i in range(N-1):
    q_k   = q_train_local[i]
    w_k   = q_k * w_train_local[i] * q_k.inv()
    d_q_k = w_k.as_rotvec() * dt 
    q_kp1 = q_k * R.from_rotvec(d_q_k) 
    q_train_global[i+1] = q_kp1

plot_tools.plot_quat(q_train_local, title="Ground Truth")
plot_tools.plot_quat(q_train_global, title="Simulation from smoothed velocity")


"""
Goal: Plot angular velocity in global frame and check smoothness
Result: very noisy if no filter applied when in global frame!
"""

w_train_global  = [q_train_local[i]*w_train_local[i]*q_train_local[i].inv() for i in range(len(w_train_local))]
# plot_tools.plot_quat(w_train_local, title = 'w_train_local')
# plot_tools.plot_quat(w_train_global, title ='w_train_global')


"""
Goal: Implement filter intrinsically within the function to load data so that filter applied to each trajectory separately

Result: DONE! If tuning on filter is needed, ajust within the loading function
"""


"""
Goal: Lastly, apply riem_log and see how things look out
"""

w_train_global_wrt_id  = quat_tools.riem_log(R.identity().inv(), w_train_global)
# plot_tools.plot_4d_coord(w_train_global_wrt_id, title='w_train_global_wrt_id')


w_train_global_wrt_att  = quat_tools.parallel_transport(R.identity().inv(), q_att, w_train_global_wrt_id)
# plot_tools.plot_4d_coord(w_train_global_wrt_att, title='w_train_global_wrt_att')


"""
Animate the q_train, and simulate using w_train forward; should produce the same results
"""

# plot_tools.animate_rotated_axes(q_train_local)
# plot_tools.animate_rotated_axes(q_train_global)



plt.show()
