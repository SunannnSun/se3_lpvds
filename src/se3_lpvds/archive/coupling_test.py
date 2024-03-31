import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


from quat_ds import quat_ds as quat_ds_class
from util import plot_tools, traj_generator, quat_tools, load_tools, process_tools



"""####### LOAD AND PROCESS DATA ########"""
q_in, p_in, index_list                  = load_tools.load_clfd_dataset(task_id=2, num_traj=3, sub_sample=1)
q_in, q_out, q_init, q_att, index_list  = process_tools.pre_process(q_in, index_list, opt= "slerp")


from damm_lpvds.util import load_tools, plot_tools
from scipy.signal import savgol_filter
# q_new_att = savgol_filter(q_in_att, window_length=80, polyorder=2, axis=0, mode="nearest")

from damm_lpvds.damm.main   import damm   as damm_class
from damm_lpvds.ds_opt.main import ds_opt as dsopt_class
import os


input_data = np.zeros((len(p_in), 1), dtype=object)
for l in range(len(p_in)):
    pos = p_in[l]
    vel = savgol_filter(pos, window_length=80, deriv= 1, polyorder=2, axis=1, mode="nearest")

    input_data[l, 0] = np.vstack((pos, vel))

Data, Data_sh, att, x0_all, dt, _, traj_length = load_tools.processDataStructure(input_data)
plot_tools.plot_reference_trajectories_DS(Data, att, 100, 20)

dim = Data.shape[0]

damm_config ={
    "mu_0":           np.zeros((dim, )), 
    "sigma_0":        1 * np.eye(dim),
    "nu_0":           dim,
    "kappa_0":        1,
    "sigma_dir_0":    1,
    "min_threshold":  50
}

damm = damm_class(damm_config)         
damm.begin(Data)
damm.evaluate()
damm.plot()

plt.show()


# ds optimization 
ds_opt_config = {
    "Data": Data,
    "Data_sh": Data_sh,
    "att": np.array(att),
    "x0_all": x0_all,
    "dt": dt,
    "traj_length":traj_length
}

output_path  = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output.json')

ds_opt = dsopt_class(ds_opt_config, output_path)
ds_opt.begin()
ds_opt.evaluate()
ds_opt.plot()








# """############ PERFORM QUAT-DS ############"""

# quat_ds = quat_ds_class(q_in, q_out, q_att, index_list, K_init=4)
# quat_ds.begin()
# q_test, w_test = quat_ds.sim(q_init)


# # """############ PLOT RESULTS #############"""

# plot_tools.plot_quat(q_test, title='q_test')
# plot_tools.plot_gmm_prob(w_test, title="GMM Posterior Probability of Reproduced Data")

# plt.show()