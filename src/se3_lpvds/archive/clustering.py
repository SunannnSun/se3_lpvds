import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from util import plot_tools, optimize_tools
from util.quat_tools import *

from util.gmm import gmm as gmm_class


if __name__ == "__main__":
    """
    Demonstrate learning a multiple linear dynamical system in quaternion space
    """

    ##### Create and plot the synthetic demonstration data ####
    rand_seed =  np.random.RandomState(seed=1)
    # rand_seed =  np.random.RandomState()

    q_id_q = canonical_quat(R.identity().as_quat())

    K = 3
    N = 30
    dt = 0.1
    q_init = R.identity()
    q_train = [q_init]
    rot_vel = np.pi/6
    w_train = []

    assignment_arr = np.zeros((K*N+1, ), dtype=int)

    for k in range(K):
        if len(w_train) != 0:
            w_train.pop()
        rot_vec = R.random(random_state=rand_seed).as_rotvec()
        # rot_vec = np.array([1, 0, 1])
        w_new = rot_vel * rot_vec/np.linalg.norm(rot_vec)
        w_train.append(w_new)

        for i in np.arange(N*k, N*(k+1)):
            q_next =  R.from_rotvec(w_train[i] * dt) * q_train[i]
            q_train.append(q_next)
            if k == K-1:
                w_train.append(w_new*(N*K-i)/(N*K))
            else:
                w_train.append(w_train[i])
            assignment_arr[i+1] = k


    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d", proj_type="ortho")
    # ax.figure.set_size_inches(10, 8)
    # ax.set(xlim=(-2, 2), ylim=(-2, 2), zlim=(-2, 2))
    # ax.set_aspect("equal", adjustable="box")
    # plot_tools.animate_rotated_axes(ax, q_train)




    # q_att_q = canonical_quat(q_train[-1].as_quat())
    # q_train_q   = list_to_arr(q_train)
    # q_train_att = riem_log(q_att_q, q_train_q)

    # plot_tools.plot_quat(q_train_q)
    # plot_tools.plot_quat(q_train_att)

    # labels2 = GaussianMixture(n_components=2, random_state=0).fit_predict(q_train_att)
    # print(labels3)
    # print(labels2)


    # plot_tools.plot_rotated_axes_sequence(q_train, N = 5)
    # Data = np.hstack((q_train_att, q_train_att)).T

    # DAMM = damm_class(Data)         
    # if DAMM.begin() == 0:
    #     labels = DAMM.result(if_plot=False)

    # print(labels)