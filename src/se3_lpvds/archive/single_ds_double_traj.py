import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from util import plot_tools, optimize_tools
from util.quat_tools import *




if __name__ == "__main__":
    """
    Demonstrate learning a single linear quaternion-based dynamical system in a multi-behaviour quaternion trajectory
    """

    ##### Create and plot the synthetic demonstration data ####

    q_id_q = canonical_quat(R.identity().as_quat())  

    N1 = 20
    N2 = 50
    dt = 0.1
    q_init = R.identity()
    q_train = [q_init]

    """
    Change w_init to desired rot_vec configuration
    """

    w_init = np.pi/6 * np.array([1, 0, 0]) 
    w_train = [w_init]
    

    for i in range(N1):
        q_next =  R.from_rotvec(w_train[i] * dt) * q_train[i]
        q_train.append(q_next)
        w_train.append(w_init)  

    w_new = np.pi/6 * np.array([0, 0, 1]) 
    for i in np.arange(N1, N2):
        q_next =  R.from_rotvec(w_train[i] * dt) * q_train[i]
        q_train.append(q_next)
        w_train.append(w_new * (N2-i)/N2) 



    fig = plt.figure()
    ax = fig.add_subplot(projection="3d", proj_type="ortho")
    ax.figure.set_size_inches(10, 8)
    ax.set(xlim=(-2, 2), ylim=(-2, 2), zlim=(-2, 2))
    ax.set_aspect("equal", adjustable="box")
    plot_tools.animate_rotated_axes(ax, q_train)



    #### Learn the matrix A of single linear DS ####

    A = optimize_tools.optimize_single_quat_system(q_train, w_train,  q_train[-1], opt=1)

    
    #### Reproduce the demonstration ####

    q_init = R.random()
    # q_init = R.identity()

    q_test = [q_init]
    q_att_q = canonical_quat(q_train[-1].as_quat())

    for i in range(N1+N2+200):
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