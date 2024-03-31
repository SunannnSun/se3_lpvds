import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


from src.se3_lpvds import se3_lpvds as se3_lpvds_class
from src.util import plot_tools, traj_generator, quat_tools, load_tools, process_tools



p_raw, q_raw, t_raw = load_tools.load_clfd_dataset(task_id=1, num_traj=9, sub_sample=2)

p_in, q_in, t_in    = process_tools.pre_process(p_raw, q_raw, t_raw, opt= "savgol")

p_out, q_out        = process_tools.compute_output(p_in, q_in, t_in)

p_init, q_init, p_att, q_att = process_tools.extract_state(p_in, q_in)

p_in, q_in, p_out, q_out     = process_tools.rollout_list(p_in, q_in, p_out, q_out)



se3_lpvds = se3_lpvds_class(p_in, q_in, p_out, q_out, p_att, q_att, K_init=4)

se3_lpvds.begin()

p_test, q_test, w_test = se3_lpvds.sim(p_init[0], q_init[0], dt=0.01)




# """############ PLOT RESULTS #############"""

plot_tools.plot_gamma_over_time(w_test, title=r"$\gamma(\cdot)$ over Time for Reproduced Data")
# plot_tools.overlay_train_test_4d(q_in, index_list, q_test)

plot_tools.plot_quat(q_test)


p_arr = np.vstack(p_test)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')
ax.plot(p_arr[:, 0], p_arr[:, 1], p_arr[:, 2], 'o')


plt.show()
