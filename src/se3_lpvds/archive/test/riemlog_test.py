import numpy as np
import matplotlib.pyplot as plt
from quat_ds import quat_ds as quat_ds_class
from util import plot_tools, traj_generator, quat_tools
from scipy.spatial.transform import Rotation as R



"""
Projection test seems to prove that it is more advantageous to project w_train in world frame via riem_log and parallel transport
to attractor. However, the scalue of the previous results seem troublesome, as quaternions (w_train) after projection quickly blow up
"""

# Create 3D example and ensure antipodals points yield the largest norm
x = np.array([0, 0, 1])
y = np.array([0, 0, -1])


u = quat_tools.riem_log(x, y)

"""
Found error in riem_log, forgot to include y-xTyx in calculating the norm; now fixed;
Now the results in projection_test needs to be overturned as they are all different now
"""


# Create 2D example and ensure antipodals points yield the largest norm
q_init, q_att, q_train, w_train_local = traj_generator.generate_traj(K=2, N=60, q_init=R.random())

N = len(q_train)
w_train_global = [q_train[i]*w_train_local[i]*q_train[i].inv() for i in range(N-1)]



x = R.identity()
y = np.array([0,  -1])


u = quat_tools.riem_log(R.identity(), w_train_global)


# u = quat_tools.riem_log(q_att, q_train)

plot_tools.plot_4d_coord(u, title='q_train wrt q_att')

plt.show()

a= 1







