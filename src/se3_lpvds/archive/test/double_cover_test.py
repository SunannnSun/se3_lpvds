import numpy as np
import matplotlib.pyplot as plt
from quat_ds import quat_ds as quat_ds_class
from util import plot_tools, traj_generator, quat_tools, load_tools

from scipy.spatial.transform import Rotation as R


"""
####### GENERATE RANDOM TRAJECTORY ########
"""

# q_init, q_att, q_train, w_train, dt = load_tools.load_clfd_dataset(task_id=1, num_traj=1, sub_sample=10)


# q_init, q_att, q_train, w_train, t_train, dt = traj_generator.generate_traj(q_init=R.identity(), K=1, N=250, dt=0.1)


from scipy.spatial.transform import Slerp


k1 = 1
k2 = 3


# q_train_list = [obj.as_quat() for obj in q_train]

# idx_list = np.linspace(0, len(q_train)-1, num= int(len(q_train)/k1), endpoint=True, dtype=int)
# q_list   =  [q_train[i].as_quat() for i in idx_list]


# key_times      =  [t_train[i] for i in idx_list]
# key_rots  = R.from_quat(q_list)

# slerp = Slerp(key_times, key_rots)

# idx_list = np.linspace(0, len(q_train)-1, num= int(len(q_train)/k2), endpoint=True, dtype=int)
# key_times  =  [t_train[i] for i in idx_list]

# q_interp = slerp(key_times)

# plot_tools.plot_quat(q_train,  title="q_train original")
# plot_tools.plot_quat(q_interp, title="q_train interpolated")

    
# q_train_att = quat_tools.riem_log(q_att, q_train)

# plot_tools.plot_4d_coord(q_train_att, title='q_train_att')
# plot_tools.plot_4d_coord(quat_tools.list_to_arr(q_train), title="q_train")
# plot_tools.plot_4d_coord(quat_tools.riem_log(q_att, q_train))



# plot_tools.plot_quat(w_train, title="w_train_body_frame")
# plot_tools.plot_4d_coord(quat_tools.list_to_arr(w_train))

# plot_tools.plot_4d_coord(quat_tools.riem_log(R.identity(), w_train))



"""
############ PERFORM QUAT-DS ############
"""
# quat_ds = quat_ds_class(q_train, w_train, q_att)
# quat_ds.begin()
# q_test = quat_ds.sim(q_init, dt=0.1)

# plot_tools.plot_quat(q_test, title="q_test")
# plot_tools.plot_4d_coord(quat_tools.riem_log(q_att, q_test))

"""
############ PLOT RESULTS #############
"""



# plot_tools.animate_rotated_axes(q_train)
# plot_tools.animate_rotated_axes(q_test)
# plot_tools.plot_quat(q_train)

# plot_tools.plot_quat(q_test)
# plot_tools.plot_rot_vec(w_train)

# plot_tools.plot_quat(q_train)

# plot_tools.plot_rotated_axes_sequence(q_train)
# plot_tools.plot_rotated_axes_sequence(q_test)

# plt.show()


q_init, q_att, q_train, w_train, dt, index_list = load_tools.load_clfd_dataset(task_id=3, num_traj=1, sub_sample=3)

# q_init, q_att, q_train, w_train, t_train, dt, index_list= traj_generator.generate_traj(q_init=R.identity(), K=2, N=40, dt=0.1)

# plot_tools.plot_demo(q_train, index_list=index_list)

q_train_att = quat_tools.riem_log(q_att, q_train)

# plot_tools.plot_quat(q_train, title= "Quaternion")
# plot_tools.plot_4d_coord(q_train_att, title="Projected Quaternions in Tangent Space")
# plot_tools.plot_quat(w_train, title="Expected Orientation")


# plt.show()


def compute_ang_vel(q1, q2, dt=1):
    """
    Compute angular velocity in q1 frame to rotate q1 into q2 given known time step
    """

    dq = q1.inv() * q2

    dq = dq.as_rotvec()

    w  = dq / dt

    return w




q1 = R.from_rotvec([1, 0.5, 0])
q2 = R.from_rotvec([1.1, 0.6, 0])

w = compute_ang_vel(q1, q2)
# print(compute_ang_vel(q1, q2))


# verify

# q2_test = q1 * R.from_rotvec(w)

# print(q2_test.as_rotvec())



# q_rec = [q_train[0]] * len(q_train)


# for i in range(len(w_train)):

#     # dq = w_train[i].as_rotvec() * (t_train[i+1] - t_train[i])
#     dq = w_train[i].as_rotvec() * dt

#     q_rec[i+1] = q_rec[i] * R.from_rotvec(dq)

# # plot_tools.plot_quat(q_train, title= "Original Quaternion")

# plot_tools.plot_quat(q_rec, title = "Original Quaternion")


quat_ds = quat_ds_class(q_train, w_train, q_att, index_list = index_list)
quat_ds.begin()
q_test = quat_ds.sim(q_init, dt=0.1)



plot_tools.plot_quat(q_test, title="Reconstructed Quaternion")

plot_tools.animate_rotated_axes(q_train)
plot_tools.animate_rotated_axes(q_test,att=q_att)
# plt.show()

