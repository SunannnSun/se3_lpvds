import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


from quat_ds import quat_ds as quat_ds_class
from util import plot_tools, traj_generator, quat_tools, load_tools, process_tools



"""####### LOAD AND PROCESS DATA ########"""
q_in, p_in, index_list                  = load_tools.load_clfd_dataset(task_id=2, num_traj=3, sub_sample=1)
q_in, q_out, q_init, q_att, index_list  = process_tools.pre_process(q_in, index_list, opt= "slerp")






def angular_velocities(q1, q2, dt):
    return (2 / dt) * np.array([
        q1[0]*q2[1] - q1[1]*q2[0] - q1[2]*q2[3] + q1[3]*q2[2],
        q1[0]*q2[2] + q1[1]*q2[3] - q1[2]*q2[0] - q1[3]*q2[1],
        q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1] - q1[3]*q2[0]])






def compute_ang_vel(q_k, q_kp1, dt=1):
    """
    Compute angular velocity in q_k frame to rotate q_k into q_kp1 given known time difference
    """

    dq = q_k.inv() * q_kp1

    dq = dq.as_rotvec()

    w  = 2 * dq / dt

    return w



dt = 0.1

q_1 = q_in[0]
q_2 = q_in[1]


print(q_2.as_quat())

w = compute_ang_vel(q_1, q_2, dt)



q_2_test = q_1 * R.from_rotvec(w * dt)


q_diff = q_2_test * q_2.inv()
print(np.linalg.norm(( q_2_test * q_2.inv()).as_rotvec()))

















