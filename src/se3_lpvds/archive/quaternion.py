import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from util import plot_tools, optimize_tools
from util.gmm import q_gmm as q_gmm_class



def canonical_quat(q):
    """
    Force all quaternions to have positive scalar part; necessary to ensure proper propagation in DS
    """
    if (q[-1] < 0):
        return -q
    else:
        return q



if __name__ == "__main__":

    N = 60
    dt = 0.1  # unit time
    ang_vel = np.pi/6

    w_axis = np.array([1, 0, 0]) 
    q_train = [R.identity()]
    w_train = [w_axis * ang_vel]

    assignment_arr = np.zeros((N+1, ), dtype=int)

    for i in range(int(N/2)):
        assignment_arr[i+1] = 0
        q_next =  R.from_rotvec(w_train[i] * dt) * q_train[i]
        q_train.append(q_next)
        w_train.append(w_axis * ang_vel * (N-i)/N)  

    w_axis = np.array([0, 1, 0]) 
    for i in np.arange(int(N/2), N):
        assignment_arr[i] = 1
        q_next =  R.from_rotvec(w_train[i] * dt) * q_train[i]
        q_train.append(q_next)
        w_train.append(w_axis * ang_vel * (N-i)/N)  #decaying velocity approaching zero near attractor


    q_att = q_train[-1]

    qGMM = q_gmm_class(q_train)
    qGMM.cluster(assignment_arr)
    qGMM.extract_param(q_train, assignment_arr)


    q_normal = qGMM.q_normal_arr[0]

    q_normal.logProb(q_train)



    """
    
    q_test = [R.identity()]
    w_test = []

    for i in range(N):
        q_diff = (q_test[i] * q_att.inv()).as_quat()
        
        w_pred = A @ canonical_quat(q_diff)[0:3]
        w_test.append(w_pred)

        q_next =  R.from_rotvec(w_test[i] * dt) * q_test[i]
        q_test.append(q_next)

    
    # r = R.from_quat(q_test)
    rotations = [canonical_quat(q.as_quat()) for q in q_test]
    r = R.from_quat(rotations)
    q_mean = r.mean()
    # R.mean(q_test)

    """


    fig = plt.figure()
    ax = fig.add_subplot(projection="3d", proj_type="ortho")
    ax.figure.set_size_inches(10, 8)
    ax.set(xlim=(-2, 2), ylim=(-2, 2), zlim=(-2, 2))
    ax.set_aspect("equal", adjustable="box")

    plot_tools.animate_rotated_axes(ax, q_train)

    # plot_tools.plot_rotated_axes(ax, q_train[0])
    # plot_tools.plot_rotated_axes(ax, q_att)
    # plot_tools.plot_rotated_axes(ax, q_mean)
    # plot_tools.plot_rotated_axes(ax, rotations[1])

    plt.show()