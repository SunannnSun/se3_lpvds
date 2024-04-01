import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from src.se3_lpvds.src.se3_class import se3_class
from src.se3_lpvds.src.util import plot_tools, load_tools, process_tools




# Load data (Optional)
p_raw, q_raw, t_raw = load_tools.load_clfd_dataset(task_id=1, num_traj=9, sub_sample=1)


# Process data (Optional)
p_in, q_in, t_in             = process_tools.pre_process(p_raw, q_raw, t_raw, opt= "savgol")
p_out, q_out                 = process_tools.compute_output(p_in, q_in, t_in)
p_init, q_init, p_att, q_att = process_tools.extract_state(p_in, q_in)
p_in, q_in, p_out, q_out     = process_tools.rollout_list(p_in, q_in, p_out, q_out)


# Run se3_lpvds
se3_obj = se3_class(p_in, q_in, p_out, q_out, p_att, q_att, K_init=4)
se3_obj.begin()

p_test, q_test, gamma_test = se3_obj.sim(p_init[0], q_init[0], dt=0.01)


# Plot results (Not done yet)
plot_tools.plot_gamma_over_time(gamma_test)
plot_tools.plot_quat(q_test)

p_arr = np.vstack(p_test)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')
ax.plot(p_arr[:, 0], p_arr[:, 1], p_arr[:, 2], 'o')

plt.show()
