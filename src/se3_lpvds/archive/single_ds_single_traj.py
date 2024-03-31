import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from util import plot_tools, optimize_tools
from util.quat_tools import *




if __name__ == "__main__":
    """
    :param q_train: list of Rotation objects representing orientation, should be length N
    :param w_train: list of Rotation objects representing angular velocity, should be length N-1


    :note: w_train are all expressed in terms of world coordinate in this case origin
    """

    # Define the identity rotation as origin
    q_id_q = canonical_quat(R.identity().as_quat())  

    # Simulate forward to obtain the sequence of rotations
    N = 40
    dt = 0.1

    q_train = [R.identity()] * N
    w_train = [R.identity()] * (N-1)

    q_init = R.identity()
    q_train[0] = q_init

    for k in range(N-1):
        w_train[k]   = R.from_rotvec(np.pi/6 * np.array([1, 0, 1]) * (N-k)/N)
        q_train[k+1] = R.from_rotvec(w_train[k].as_rotvec() * dt) * q_train[k]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d", proj_type="ortho")
    ax.figure.set_size_inches(10, 8)
    ax.set(xlim=(-2, 2), ylim=(-2, 2), zlim=(-2, 2))
    ax.set_aspect("equal", adjustable="box")
    plot_tools.animate_rotated_axes(ax, q_train)


    #### Learn the matrix A of single linear DS ####

    A = optimize_tools.optimize_single_quat_system(q_train, w_train,  q_train[-1], opt=3)

    
    #### Reproduce the demonstration ####
    q_init = R.random()
    # q_init = R.identity()

    q_test = [q_init]
    q_att_q = canonical_quat(q_train[-1].as_quat())

    for i in range(N+200):
        q_curr   = q_test[i]
        q_curr_q = canonical_quat(q_curr.as_quat())
        q_curr_att = riem_log(q_att_q, q_curr_q)

        w_pred_att = A @ q_curr_att[:, np.newaxis]
        w_pred_id = parallel_transport(q_att_q, q_id_q, w_pred_att)
        w_pred_q  = riem_exp(q_id_q, w_pred_id * dt) # multiplied by dt before projecting back to the quaternion space
        w_pred    = R.from_quat(w_pred_q)

        q_next = w_pred * q_curr
        q_test.append(q_next)

    

    #### Plot the results ####
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d", proj_type="ortho")
    ax.figure.set_size_inches(10, 8)
    ax.set(xlim=(-2, 2), ylim=(-2, 2), zlim=(-2, 2))
    ax.set_aspect("equal", adjustable="box")

    plot_tools.animate_rotated_axes(ax, q_test)

