from scipy.spatial.transform import Rotation as R
import numpy as np




q1 = R.random()

q1_q = q1.as_quat()


q2 = R.from_quat(q1_q[np.newaxis, :])

q2_q = q2.as_quat().reshape(-1, 1)

a =1