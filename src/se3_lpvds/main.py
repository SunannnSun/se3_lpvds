import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


from src.se3_lpvds import se3_lpvds as se3_lpvds_class
from src.util import plot_tools, traj_generator, quat_tools, load_tools, process_tools


"""
Index list should go ... , which is mainly for plotting, but we can just plot the axis-angle on a unit sphere insted...

p_in: M by N array, M: number of points, N: dimension
q_in: list of M (scipy) Rotation objects ... avoid ambiguity about notation - scalar first versus scalar last

usually no velocity is provided, other than the time stamp, hence computing velocity is needed
"""


p_raw, q_raw, t_raw = load_tools.load_clfd_dataset(task_id=1, num_traj=9, sub_sample=1)

p_in, q_in, t_in    = process_tools.pre_process(p_raw, q_raw, t_raw, opt= "savgol")

p_out, q_out        = process_tools.compute_output(p_in, q_in, t_in)

p_init, q_init, p_att, q_att = process_tools.extract_state(p_in, q_in)

p_in, q_in, p_out, q_out     = process_tools.rollout_list(p_in, q_in, p_out, q_out)



se3_lpvds = se3_lpvds_class(p_in, q_in, p_out, q_out, p_att, q_att, K_init=4)
se3_lpvds.begin()



# """############ PERFORM QUAT-DS ############"""
# quat_ds = quat_ds_class(q_in, q_out, q_init, q_att, index_list, K_init=4)
# quat_ds.begin()

# # q_init = R.from_quat(-q_init.as_quat())

# q_test, w_test = quat_ds.sim(q_init, dt=0.01)


# """############ PLOT RESULTS #############"""

# plot_tools.plot_gamma_over_time(w_test, title=r"$\gamma(\cdot)$ over Time for Reproduced Data")
# plot_tools.overlay_train_test_4d(q_in, index_list, q_test)

# plt.show()
