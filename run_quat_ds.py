import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Import Packages
from src.quaternion_ds.src.quat_ds import quat_ds as quat_ds_class
from src.quaternion_ds.src.util import plot_tools, traj_generator, quat_tools, load_tools, process_tools


"""####### LOAD AND PROCESS DATA ########"""
p_in, q_in, index_list                  = load_tools.load_clfd_dataset(task_id=0, num_traj=9, sub_sample=1)
q_in, q_out, q_init, q_att, index_list  = process_tools.pre_process(q_in, index_list, opt= "slerp")


"""############ PERFORM QUAT-DS ############"""
quat_ds = quat_ds_class(q_in, q_out, q_init, q_att, index_list, K_init=5)
quat_ds.begin()

# q_init = R.from_quat(-q_init.as_quat())

q_test, w_test = quat_ds.sim(q_init, dt=0.01)


"""############ PLOT RESULTS #############"""

plot_tools.plot_gamma_over_time(w_test, title=r"$\gamma(\cdot)$ over Time for Reproduced Data")
plot_tools.overlay_train_test_4d(q_in, index_list, q_test)

plt.show()
