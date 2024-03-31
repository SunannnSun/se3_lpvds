import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from .quat_tools import *



def generate_traj(K=2, N=40, dt=0.1, **kwargs):
    """
    :param K: the number of trajectory
    :param N: the number of point per trajectory
    """

    rot_vel  = np.pi/6
    rng_seed =  np.random.RandomState(seed=2)
    w_list   = [R.random(random_state=rng_seed).as_rotvec() for k in range(K)]
    w_list   = [R.from_rotvec(rot_vel * rot_vec / np.linalg.norm(rot_vec)) for rot_vec in w_list]
    # w_list   = [R.from_rotvec(np.array([rot_vel, 0, 0]))]                                           # Rotate about x-axis


    q_train = [R.identity()] * (N * K)
    w_train = [R.identity()] * (N * K)
    t_train = [0] * (N * K)


    if "q_init" in kwargs:
        q_init = kwargs["q_init"]
    else:
        q_init = R.identity()


    q_train[0] = q_init
    for i in range (N * K -1):
        k = i // N
        # """
        if (k != K-1):
            w_train[i] = w_list[k]
        else:
            w_train[i] = R.from_rotvec(w_list[k].as_rotvec() * (1 - i/(N*K)))           # Decaying angular velocity
            # w_train[i] = w_list[k]                                                    # Constant angular velocity
        # """                                          
        q_train[i+1] =   q_train[i] * R.from_rotvec(w_train[i].as_rotvec() * dt)        # Rotate wrt the body frame

        t_train[i+1] = (i+1) * dt
    q_att = q_train[-1]

    # l=0
    for l in range(K):
        w_train[l*N: (l+1)*N-1] = q_train[l*N+1: (l+1)*N]
        w_train[(l+1)*N-1]      =  w_train[(l+1)*N-2]

    index_list = [i for i in range(N*K)]


    return q_init, q_att, q_train, w_train, t_train, dt, index_list