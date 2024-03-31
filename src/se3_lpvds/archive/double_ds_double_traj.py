import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from util import plot_tools, optimize_tools
from util.quat_tools import *

from util.gmm import gmm as gmm_class



if __name__ == "__main__":
    """
    Demonstrate learning a double linear dynamical system in quaternion space
    """


    ##### Create and plot the synthetic demonstration data ####
    rand_seed =  np.random.RandomState(seed=1)
    # rand_seed =  np.random.RandomState()

    q_id_q = canonical_quat(R.identity().as_quat())

    K = 2
    N = 40
    dt = 0.1

    q_train = [R.identity()] * (N * K)
    w_train = [R.identity()] * (N * K -1)
    assignment_arr = np.zeros((N*K, ))

    q_init = R.identity()
    q_train[0] = q_init


    for k in range (N * K -1):

        if (k // N == 0):
            w_train[k] = R.from_rotvec(np.pi/6 * np.array([1, 0, 0]))
        else:
            w_train[k] = R.from_rotvec(np.pi/6 * np.array([0, 0, 1]) * (1 - k/(N*K)))

        # Rotate about the world frame
        q_train[k+1] =  R.from_rotvec(w_train[k].as_rotvec() * dt) * q_train[k]
        assignment_arr[k+1] = k // N
        # Rotate about the body frame
        # w_k_dt = R.from_rotvec(w_train[k].as_rotvec() * dt)
        # q_train[k+1] = R.from_matrix(w_k_dt.apply(q_train[k].as_matrix()))




    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d", proj_type="ortho")
    # ax.figure.set_size_inches(10, 8)
    # ax.set(xlim=(-2, 2), ylim=(-2, 2), zlim=(-2, 2))
    # ax.set_aspect("equal", adjustable="box")
    # plot_tools.animate_rotated_axes(ax, q_train)


    #### Perform pseudo clustering, return assignment_arr and sufficient statistics ####
    gmm = gmm_class(q_train[-1], q_train)
    gmm.begin(assignment_arr)
    gmm.return_norma_class(q_train, assignment_arr)
    postProb = gmm.postLogProb(q_train)


    # A = optimize_tools.optimize_double_quat_system(q_train, w_train, q_train[-1], postProb)
    A = optimize_tools.optimize_quat_system(q_train, w_train, q_train[-1], postProb)


    #### Reproduce the demonstration ####
    # q_init = R.random()
    dt = 0.1
    q_init = R.identity()

    q_test = [q_init]
    q_att_q = canonical_quat(q_train[-1].as_quat())

    for i in range(N+100):
        q_curr   = q_test[i]
        q_curr_q = canonical_quat(q_curr.as_quat())
        q_curr_att = riem_log(q_att_q, q_curr_q)

        h_k = gmm.postLogProb(q_curr_q)
        

        w_pred_att = np.zeros((4, 1))
        for k in range(K):
            h_k_i =  h_k[k, 0]
            w_k_i =  A[k] @ q_curr_att[:, np.newaxis]
            w_pred_att += h_k_i * w_k_i


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

