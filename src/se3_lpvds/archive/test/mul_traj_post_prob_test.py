import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


from quat_ds import quat_ds as quat_ds_class
from util import plot_tools, traj_generator, quat_tools, load_tools, process_tools



q_in, index_list         = load_tools.load_clfd_dataset(task_id=2, num_traj=2, sub_sample=1)
q_in, q_out, q_init, q_att, index_list  = process_tools.pre_process(q_in, index_list, opt= "slerp")



quat_ds = quat_ds_class(q_in, q_out, q_att, index_list, K_init=4)
quat_ds._cluster()


gmm = quat_ds.gmm


N = len(index_list[0])
w_train = np.zeros((N, gmm.K))
for i in range(N):
    w_train[i, :] = gmm.postLogProb(q_in[index_list[0][i]]).T
plot_tools.plot_gmm_prob(w_train, title="GMM Posterior Probability of Original Data")





plt.show()