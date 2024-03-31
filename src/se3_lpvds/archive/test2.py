import numpy as np
import matplotlib.pyplot as plt
from quat_ds import quat_ds as quat_ds_class
from util import plot_tools, traj_generator, quat_tools

from scipy.spatial.transform import Rotation as R


# Rx followed by Ry

Rx = R.from_rotvec(np.array([np.pi/3, 0, 0]))

Ry = R.from_rotvec(np.array([0, np.pi/6, 0]))


# Ry in body frame

Rz = Rx * Ry
print(Rz.as_quat())
print(R.from_euler('XYZ', np.array([np.pi/3, np.pi/6, 0])).as_quat())

# Transform Ry to world frame in quaternion
Rz_eqv =  Rx * Ry * Rx.inv() * Rx
print(Rz_eqv.as_quat())

# Transform Ry to world frame in matrix

# Rx_mat = Rx.as_matrix() 
# Ry_mat = Ry.as_matrix()
# print(Rx_mat@Ry_mat@Rx_mat)



# Ry in world frame

# Rz = Ry * Rx 
# Rz_matrix =  Ry.as_matrix() @ Rx.as_matrix()

# print(Rz.as_matrix())
# print(Rz_matrix)