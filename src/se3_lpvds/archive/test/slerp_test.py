import numpy as np
import matplotlib.pyplot as plt
from quat_ds import quat_ds as quat_ds_class
from util import plot_tools, traj_generator, quat_tools, load_tools

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


k1 = 40
k2 = 1
dt=0.1

# q_init, q_att, q_train, _, _, _, _ = traj_generator.generate_traj(q_init=R.random(), K=2, N=50, dt=0.1)
q_train, q_init, q_att, _         = load_tools.load_clfd_dataset(task_id=0, num_traj=1, sub_sample=1)

t_train = [dt*i for i in range(len(q_train))]


q_train_list = [obj.as_quat() for obj in q_train]
idx_list  = np.linspace(0, len(q_train)-1, num= int(len(q_train)/k1), endpoint=True, dtype=int)
q_list    =  [q_train[i].as_quat() for i in idx_list]
key_times =  [t_train[i] for i in idx_list]
key_rots  = R.from_quat(q_list)

slerp = Slerp(key_times, key_rots)


idx_list = np.linspace(0, len(q_train)-1, num= int(len(q_train)/k2), endpoint=True, dtype=int)
key_times  =  [t_train[i] for i in idx_list]

q_interp = slerp(key_times)

# plot_tools.plot_quat(q_train,  title="q_train original")
plot_tools.plot_quat(q_interp, title="q_train interpolated")

plt.show()
