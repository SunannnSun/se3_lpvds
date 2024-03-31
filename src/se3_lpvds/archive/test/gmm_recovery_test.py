import numpy as np
import matplotlib.pyplot as plt
from quat_ds import quat_ds as quat_ds_class
from util import plot_tools, traj_generator, quat_tools, load_tools, process_tools
from util.gmm import gmm as gmm_class

from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter



"""
Load data
"""
q_train, q_init, q_att, index_list = load_tools.load_clfd_dataset(task_id=0, num_traj=1, sub_sample=1)



"""
Process data
"""

q_new, w_new, q_init, q_att, index_list_new = process_tools.pre_process(q_train, q_att, index_list)
 


# # GMM validation
# gmm = gmm_class(q_new, q_new[-1], index_list = index_list_new)
# label = gmm.begin()

# w_train = np.zeros((len(q_new), gmm.K))
# for i in range(len(q_new)):
#     w_train[i, :] = gmm.postLogProb(q_new[i]).T
# plot_tools.plot_gmm_prob(w_train, title="GMM Posterior Probability of Original Data")


"""
Learning
"""
quat_ds = quat_ds_class(q_new, w_new, q_att, index_list = index_list_new)
quat_ds.begin()


"""
Forward Simulation
"""
q_test, w_test = quat_ds.sim(q_init, dt=0.1)
plot_tools.plot_quat(q_test, title='q_test')
plot_tools.plot_gmm_prob(w_test, title="GMM Posterior Probability of Reproduced Data")


plt.show()

