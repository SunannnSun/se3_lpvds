import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter
from scipy.spatial.transform import Rotation as R
from .quat_tools import *
import random
from dtw import dtw


font = {'family' : 'Times New Roman',
         'size'   : 18
         }

mpl.rc('font', **font)



def _interp_index_list(q_list, index_list, interp=True, arr=True):

    """
    Parameters:
        index_list: [np.array([0,...,999]), np.array([1000,...1999])]
    
        index_list_interp: [np.array([0,...,999]), np.array([0,...999])]

    Note:
        the input indexList retains the sequence across trajs: good for indexing, bad for plotting when each traj contains different size;
        the output indexList starts at 0 for each traj: bad for indexing, good for plotting
    """



    L = len(index_list)

    index_list_interp = []

    if interp == True:
        ref = index_list[0]
        for l in np.arange(1, L):
            if index_list[l].shape[0] > ref.shape[0]:
                ref = index_list[l]
        N = ref[-1]

        for l in range(L):
            index_list_interp.append(np.linspace(0, N, num=index_list[l].shape[0], endpoint=False, dtype=int))

        if arr==False:
            return index_list_interp

    elif interp == False:
        for l in range(L):
            index_list_interp.append(index_list[l] - index_list[l][0])

    else:
        for l in range(L):
            if l != L-1:
                N = index_list[l+1][0] - index_list[l][0] 
            else:
                N = len(q_list) - index_list[l][0]
            index_list_interp.append(np.arange(0, N))
            

    return np.hstack(index_list_interp)


def _plot_rotated_axes(ax, r , offset=(0, 0, 0), scale=1):
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
    """

    colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB
    loc = np.array([offset, offset])

    for i, (axis, c) in enumerate(zip((ax.xaxis, ax.yaxis, ax.zaxis),
                                      colors)):
        axlabel = axis.axis_name
        axis.set_label_text(axlabel)

        line = np.zeros((2, 3))
        line[1, i] = scale
        line_rot = r.apply(line)
        line_plot = line_rot + loc
        ax.plot(line_plot[:, 0], line_plot[:, 1], line_plot[:, 2], c)

        text_loc = line[1]*1.2
        text_loc_rot = r.apply(text_loc)
        text_plot = text_loc_rot + loc[0]
        ax.text(*text_plot, axlabel.upper(), color=c,
                va="center", ha="center")
    
    


def plot_rotated_axes_sequence(q_list, N=3):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d", proj_type="ortho")
    
    seq = np.linspace(0, len(q_list)-1, N, dtype=int)
    for i in range(N):
        _plot_rotated_axes(ax, q_list[seq[i]],  offset=(3*i, 0, 0))


    ax.set(xlim=(-1.25, 1.25 + 3*N-3), ylim=(-1.25, 1.25), zlim=(-1.25, 1.25))
    ax.set(xticks=range(-1, 2 + 3*N-3), yticks=[-1, 0, 1], zticks=[-1, 0, 1])
    ax.set_aspect("equal", adjustable="box")
    ax.figure.set_size_inches(2*N, 5)
    # plt.tight_layout()
    # plt.show()





def animate_rotated_axes(R_list, scale=1, **argv):
    """
    List of Rotation object
    """



    fig = plt.figure()
    ax = fig.add_subplot(projection="3d", proj_type="ortho")
    ax.figure.set_size_inches(10, 8)
    ax.set(xlim=(-2, 2), ylim=(-2, 2), zlim=(-2, 2))
    ax.set_aspect("equal", adjustable="box")


    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axlabel = axis.axis_name
        axis.set_label_text(axlabel)


    colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB
    lines = [ax.plot([], [], [], c=c)[0] for c in colors]
    
    if "att" in argv:
        lines_att = [ax.plot([], [], [], c=c)[0] for c in colors]


    def _init():
        for line in lines:
            line.set_data_3d([], [], [])
        
        if "att" in argv:
            for line in lines_att:
                line.set_data_3d([], [], [])


    def _animate(i):
        r = R_list[i]

        if "att" in argv:
            att = argv["att"]
            for axis, (line, c) in enumerate(zip(lines_att, colors)):
                line_ = np.zeros((2, 3))
                line_[1, axis] = scale
                line_rot_ = att.apply(line_)
                line.set_data_3d([line_rot_[0, 0], line_rot_[1, 0]], [line_rot_[0, 1], line_rot_[1, 1]], [line_rot_[0, 2], line_rot_[1, 2]])

        for axis, (line, c) in enumerate(zip(lines, colors)):
            line_ = np.zeros((2, 3))
            line_[1, axis] = scale
            line_rot_ = r.apply(line_)
            line.set_data_3d([line_rot_[0, 0], line_rot_[1, 0]], [line_rot_[0, 1], line_rot_[1, 1]], [line_rot_[0, 2], line_rot_[1, 2]])


        fig.canvas.draw()
        ax.set_title(f'Frame: {i}')


    anim = animation.FuncAnimation(fig, _animate, init_func=_init,
                                frames=len(R_list), interval=1000/len(R_list), blit=False, repeat=True)
    
    
    plt.tight_layout()
    plt.show()


def plot_gmm(q_list, index_list, label, interp):

    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.figure.set_size_inches(8, 6)

    index_list_interp = _interp_index_list(q_list, index_list, interp)

    color_mapping = np.take(colors, label)

    q_list_q = list_to_arr(q_list)
    for k in range(4):
        ax.scatter(index_list_interp, q_list_q[:, k], s=1, alpha=0.5, c=color_mapping)

    ax.set_title("GMM results")
    pass


def plot_demo(q_list, index_list, interp, **argv):
    """
    Plot scatter quaternions from demonstrations.
    """
    label_list = ['x', 'y', 'z', 'w']
    colors = ['red', 'blue', 'lime', 'magenta']

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.figure.set_size_inches(12, 6)

    index_list_interp = _interp_index_list(q_list, index_list, interp)

    q_list_q = list_to_arr(q_list)
    for k in range(4):
        ax.scatter(index_list_interp, q_list_q[:, k], s= 1, color=colors[k])

    if "title" in argv:
        ax.set_title(argv["title"])





def plot_quat(q_list, dt=1, **argv):

    if "ax" not in argv:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.figure.set_size_inches(12, 6)
    else:
        ax = argv["ax"]

    q_list_q = list_to_arr(q_list)


    label_list = ['x', 'y', 'z', 'w']
    N = q_list_q.shape[0]


    colors = ['red', 'blue', 'lime', 'magenta']
    for k in range(4):
        ax.plot(np.arange(N)*dt, q_list_q[:, k], color=colors[k], label = label_list[k])

    ax.legend()
    if "title" in argv:
        ax.set_title(argv["title"])

    """
    fig, axs = plt.subplots(4, 1, figsize=(12, 8))

    N = q_list_q.shape[0]
    colors = ['red', 'blue', 'lime', 'magenta']
    for k in range(4):
        axs[k].plot(np.arange(N), q_list_q[:, k], color=colors[k], label = label_list[k])
        axs[k].legend(loc="upper left")
   
    if "title" in argv:
            axs[0].set_title(argv["title"])
    """
    # plt.show()

    return ax


def plot_4d_coord(q_list, **argv):

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.figure.set_size_inches(12, 6)

    label_list = ['x', 'y', 'z', 'w']
    N = q_list.shape[0]


    colors = ['red', 'blue', 'lime', 'magenta']
    for k in range(4):
        ax.plot(np.arange(N), q_list[:, k], color=colors[k], label = label_list[k])
    
    if "title" in argv:
        ax.set_title(argv["title"])

    ax.legend()

    return ax



def plot_rot_vec(w_list, **argv):
    
    N = len(w_list)
    w_arr = np.zeros((N, 3))

    for i in range(N):
        # w_arr[i, :] = w_list[i].as_euler('xyz')
        w_arr[i, :] = w_list[i].as_rotvec()


    fig = plt.figure()
    ax = fig.add_subplot()
    ax.figure.set_size_inches(12, 6)


    label_list = ['w_x', 'w_y', 'w_z']

    colors = ['red', 'blue', 'lime', 'magenta']
    for k in range(3):
        ax.plot(np.arange(N), w_arr[:, k], color=colors[k], label = label_list[k])

    
    if "title" in argv:
        ax.set_title(argv["title"])
        
    ax.legend()

    return ax



def plot_gamma_over_time(w_arr, **argv):

    N, K = w_arr.shape

    fig, axs = plt.subplots(K, 1, figsize=(12, 8))

    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]

    for k in range(K):
        axs[k].scatter(np.arange(N), w_arr[:, k], s=5, color=colors[k])
        axs[k].set_ylim([0, 1])
    
    if "title" in argv:
        axs[0].set_title(argv["title"])




def plot_pose(p_in, p_out, q_out, label=[]):

    sub_sample = 2

    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]

    color_mapping = np.take(colors, label)

    
    # a = Data[0][0]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')

    if len(label) == 0:
        ax.plot(p_in[::sub_sample, 0], p_in[::sub_sample, 1], p_in[::sub_sample, 2], 'o', color='gray', alpha=0.2, markersize=1.5, label="Demonstration")
    else:
        ax.scatter(p_in[::sub_sample, 0], p_in[::sub_sample, 1], p_in[::sub_sample, 2], 'o', color=color_mapping[::sub_sample], s=1, alpha=0.4, label="Demonstration")

    """
    Plot Scatter
    """
    # ax.plot(p_out[:, 0], p_out[:, 1], p_out[:, 2], 'o', color='k',  markersize=1.5, label = "Reproduction")

    """
    Plot Curve
    """
    ax.plot(p_out[:, 0], p_out[:, 1], p_out[:, 2],  color='k', label = "Reproduction")



    ax.scatter(p_out[0, 0], p_out[0, 1], p_out[0, 2], 'o', facecolors='none', edgecolors='magenta',linewidth=2,  s=100, label="Initial")
    ax.scatter(p_out[-1, 0], p_out[-1, 1], p_out[-1, 2], marker=(8, 2, 0), color='k',  s=100, label="Target")

    ax.legend(ncol=4, loc="upper center")

    colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB
    colors = ("r", "g", "b")  # Colorblind-safe RGB


    scale = 0.035
    if  len(q_out) != 0:
        for i in np.linspace(0, p_out.shape[0], num=10, endpoint=False, dtype=int):

            r = q_out[i]
            loc = p_out[i, :]
            for j, (axis, c) in enumerate(zip((ax.xaxis, ax.yaxis, ax.zaxis),
                                                colors)):
                line = np.zeros((2, 3))
                line[1, j] = scale
                line_rot = r.apply(line)
                line_plot = line_rot + loc
                ax.plot(line_plot[:, 0], line_plot[:, 1], line_plot[:, 2], c, linewidth=1)



    # ax.scatter(Data[0], Data[1], Data[2], s=200, c='blue', alpha=0.5)
    ax.axis('equal')
    # ax.set_title('Reference Trajectory')
    ax.tick_params(axis='z', which='major', pad=10)


    ax.set_xlabel(r'$\xi_1(m)$', labelpad=20)
    ax.set_ylabel(r'$\xi_2(m)$', labelpad=20)
    ax.set_zlabel(r'$\xi_3(m)$', labelpad=20)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_title("plot_pose")




def plot_gmm_prob(w_arr, **argv):

    N, K = w_arr.shape

    fig, axs = plt.subplots(K, 1, figsize=(8, 6))

    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]

    for k in range(K):
        axs[k].scatter(np.arange(N), w_arr[:, k], s=5, color=colors[k])
        axs[k].set_ylim([0, 1])
        # axs[k].set_xticks(np.linspace(0, N, 6, endpoint=False, dtype=int))
        # axs[k].set_xticks([0, 25, 50, 75, 100, 125, 150, 175])


    
    if "title" in argv:
        axs[0].set_title(argv["title"])


def plot_gmm_prob_overlay(**argv):
    dt = 10E-3
    w = np.load('w.npy')
    w_pert = np.load('w_perturb.npy')

    K = 4
    
    fig, axs = plt.subplots(K, 1, figsize=(6, 11))

    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]


    for k in range(K):
        if k == 0:
            axs[k].plot(np.arange(w.shape[0])*dt, w[:, k], '--', color=colors[k], linewidth=2, label="Quaternion-DS")
        else:
            axs[k].plot(np.arange(w.shape[0])*dt, w[:, k], '--', color=colors[k], linewidth=2)

        # axs[k].set_xlim([0, 200])

        axs[k].set_ylim([0, 1.1])
        # axs[k].set_xticks(np.linspace(0, N, 6, endpoint=False, dtype=int))
        # axs[k].set_xticks([0, 25, 50, 75, 100, 125, 150, 175])
    

    for k in range(K):
        if k == 0:
            axs[k].plot(np.arange(w_pert.shape[0])*dt, w_pert[:, k], color=colors[k], linewidth=2, label="SE(3) LPV-DS")
        else:
            axs[k].plot(np.arange(w_pert.shape[0])*dt, w_pert[:, k], color=colors[k], linewidth=2)

        # axs[k].set_xlim([0, 200])

        axs[k].set_ylim([0, 1.1])
        # axs[k].set_xticks(np.linspace(0, N, 6, endpoint=False, dtype=int))
        # axs[k].set_xticks([0, 25, 50, 75, 100, 125, 150, 175])

        axs[k].set_ylabel('k = ' + str(k))

    
    axs[0].legend()
    axs[0].set_title(r"$\gamma(\cdot)$ over Time")
    axs[3].set_xlabel("Time (sec)")
    plt.savefig('gmm.png', dpi=600)



def plot_gmm_on_traj(p_in, q_in, gmm):

    label = gmm.assignment_arr

    sub_sample = 2

    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]

    color_mapping = np.take(colors, label)


    # a = Data[0][0]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')


    ax.scatter(p_in[::sub_sample, 0], p_in[::sub_sample, 1], p_in[::sub_sample, 2], 'o', color=color_mapping[::sub_sample], s=1, alpha=0.4, label="Demonstration")

    # ax.plot(p_in[::sub_sample, 0], p_in[::sub_sample, 1], p_in[::sub_sample, 2], color='gray', alpha=0.3, label="Demonstration")
    # ax.plot(p_out[:, 0], p_out[:, 1], p_out[:, 2],  color='k', label = "Reproduction")

    # ax.scatter(p_out[0, 0], p_out[0, 1], p_out[0, 2], 'o', facecolors='none', edgecolors='magenta',linewidth=2,  s=100, label="Initial")
    # ax.scatter(p_out[-1, 0], p_out[-1, 1], p_out[-1, 2], marker=(8, 2, 0), color='k',  s=100, label="Target")

    # ax.legend(ncol=4, loc="upper center")

    colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB
    colors = ("r", "g", "b")  # Colorblind-safe RGB


    scale = 0.035

    K = np.max(label) + 1

    for k in range(K):
        label_k =np.where(label == k)[0]

        p_in_k = p_in[label_k, :]
        loc = np.mean(p_in_k, axis=0)

        r = gmm.q_normal_list[k]["mu"][1]
        for j, (axis, c) in enumerate(zip((ax.xaxis, ax.yaxis, ax.zaxis),
                                            colors)):
            line = np.zeros((2, 3))
            line[1, j] = scale
            line_rot = r.apply(line)
            line_plot = line_rot + loc
            ax.plot(line_plot[:, 0], line_plot[:, 1], line_plot[:, 2], c, linewidth=1)



    # for i in np.linspace(0, len(q_in), num=40, endpoint=False, dtype=int):

    #     r = q_in[i]
    #     loc = p_in[i, :]
    #     for j, (axis, c) in enumerate(zip((ax.xaxis, ax.yaxis, ax.zaxis),
    #                                         colors)):
    #         line = np.zeros((2, 3))
    #         line[1, j] = scale
    #         line_rot = r.apply(line)
    #         line_plot = line_rot + loc
    #         ax.plot(line_plot[:, 0], line_plot[:, 1], line_plot[:, 2], c, linewidth=1)



    # ax.scatter(Data[0], Data[1], Data[2], s=200, c='blue', alpha=0.5)
    ax.axis('equal')
    # ax.set_title('Reference Trajectory')
    ax.tick_params(axis='z', which='major', pad=10)

    ax.set_xlabel(r'$\xi_1(m)$', labelpad=20)
    ax.set_ylabel(r'$\xi_2(m)$', labelpad=20)
    ax.set_zlabel(r'$\xi_3(m)$', labelpad=20)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    ax.set_title("plot_gmm_on_traj")





def plot_train_test(q_train, index_list, q_test, **argv):

    label_list = ['x', 'y', 'z', 'w']
    colors = ['red', 'blue', 'lime', 'magenta']
    # colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon']
    
    fig, axs = plt.subplots(4, 1, figsize=(14, 6))

    q_list_q = list_to_arr(q_train)

    index_list_interp = _interp_index_list(q_train, index_list, interp=True, arr=False)


    q_list_q = list_to_arr(q_train)
    L = len(index_list_interp)
    for l in range(L):
        q_l = q_list_q[index_list[l], :]
        for k in range(4):
            # ax.scatter(index_list_interp, q_list_q[:, k], s=1, c=color_mapping)
            axs[k].plot(index_list_interp[l], q_l[:, k], '--', color=colors[k], alpha=0.3, label = label_list[k])
            if l == 0:
                axs[k].legend(loc="upper left")
            # axs[k].grid(True)
            axs[k].yaxis.set_major_locator(MaxNLocator(nbins=2))
            # axs[k].set_ylabel(label_list[k])



    for ax in axs[:-1]:
        ax.xaxis.set_visible(False)



    N = index_list_interp[0][-1]

    q_test_q = list_to_arr(q_test)


    idx = np.linspace(0, N, num=q_test_q.shape[0], endpoint=True, dtype=int)


    for k in range(4):
        axs[k].plot(idx, q_test_q[:, k], color=colors[k], label = label_list[k])




    if "title" in argv:
            axs[0].set_title(argv["title"])







def overlay_train_test_4d(q_train, index_list, q_test, **argv):

    label_list = ['x', 'y', 'z', 'w']
    colors = ['red', 'blue', 'lime', 'magenta']
    # colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon']
    
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.figure.set_size_inches(8, 6)

    q_list_q = list_to_arr(q_train)

    index_list_interp = _interp_index_list(q_train, index_list, interp=True, arr=False)


    q_list_q = list_to_arr(q_train)
    L = len(index_list_interp)
    
    for l in range(L):
        q_l = q_list_q[index_list[l], :]
        for k in range(4):
            # ax.scatter(index_list_interp, q_list_q[:, k], s=1, c=color_mapping)
            ax.plot(index_list_interp[l], q_l[:, k], linewidth=1, color=colors[k], alpha=0.3)
        # if l == 0:
            # ax.legend(loc="upper left")
            # ax.grid(True)
            # ax.yaxis.set_major_locator(MaxNLocator(nbins=2))
            # axs[k].set_ylabel(label_list[k])

    # ax.legend(loc="upper left")

    # ax.xaxis.set_visible(False)
    ax.spines['top'].set_visible(False)  # Hide top border line
    ax.spines['right'].set_visible(False)  # Hide top border line

    ax.set_xlabel("Sequence Index")

    ax.set_ylabel("Unit Quaternion")


    N = index_list_interp[0][-1]

    q_test_q = list_to_arr(q_test)


    idx = np.linspace(0, N, num=q_test_q.shape[0], endpoint=True, dtype=int)


    for k in range(4):
        ax.plot(idx, q_test_q[:, k], color=colors[k],linewidth=2, label = label_list[k])

    # ax.legend(ncol=4, loc="best")

    # ax.set_title("Reproduction vs. Demonstration")

    # plt.savefig('quaternion.png', dpi=600)


    # if "title" in argv:
    #         axs[0].set_title(argv["title"])


def overlay_train_test_4d_iros(q_train, index_list, q_test, **argv): # used to generate figure in IROS paper
    dt = 10E-3

    label_list = ['q_x ref', 'q_y ref', 'q_z ref', 'q_w ref']
    pred_label_list = ['q_x pred', 'q_y pred', 'q_z pred', 'q_w pred']

    # colors = ['black', 'orchid', 'cornflowerblue', 'seagreen']
    colors = ['coral', 'darkred', 'peru', 'red']

    # colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon']
    
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.figure.set_size_inches(6, 3.5)
    # ax.figure.set_size_inches(17, 10)

    q_list_q = list_to_arr(q_train)

    index_list_interp = _interp_index_list(q_train, index_list, interp=True, arr=False)

    q_list_q = list_to_arr(q_train)
    L = len(index_list_interp)
    
    for l in range(L):
        q_l = q_list_q[index_list[l], :]
        if l == 0:
            for k in range(4):
                ax.plot(index_list_interp[l]*dt/2, q_l[:, k], linewidth=1, color=colors[k], alpha=0.3, label = label_list[k])
        else:
            for k in range(4):
                ax.plot(index_list_interp[l]*dt/2, q_l[:, k], linewidth=1, color=colors[k], alpha=0.3)

    # ax.set_xticks([])
    # ax.set_xticklabels([])

    # ax.xaxis.set_visible(False)
    # ax.spines['top'].set_visible(False)  # Hide top border line
    # ax.spines['right'].set_visible(False)  # Hide top border line

    # ax.set_xlabel("Time (sec)")
    # ax.set_ylabel("Quaternion")

    N = index_list_interp[0][-1]

    q_test_q = list_to_arr(q_test)


    idx = np.linspace(0, N, num=q_test_q.shape[0], endpoint=True, dtype=int)

    for k in range(4):
            ax.plot(idx*dt/2, q_test_q[:, k], color=colors[k], linewidth=2, label = pred_label_list[k])



    # For legend generation only
    """ 
    label_list = ['p_x ref', 'p_y ref', 'p_z ref']
    pred_label_list = ['p_x pred', 'p_y pred', 'p_z pred']

    colors_pos = ['black', 'darkviolet', 'mediumblue']

    for l in range(L):
        q_l = q_list_q[index_list[l], :]
        if l == 0:
            for k in range(3):
                ax.plot(index_list_interp[l]*dt/2, q_l[:, k], linewidth=1, color=colors_pos[k], alpha=0.3, label = label_list[k])
        else:
                ax.plot(index_list_interp[l]*dt/2, q_l[:, k], linewidth=1, color=colors_pos[k], alpha=0.3)

    for k in range(3):
        ax.plot(idx*dt/2, q_test_q[:, k], color=colors_pos[k],linewidth=2, label = pred_label_list[k])
    
    # ax.set_title("Quaternion", fontsize = 30)
        




    ax.legend(ncol=7, loc="best")
    # reorderLegend(ax,['q_x ref', 'q_y ref', 'q_z ref', 'q_w ref', 'q_x pred', 'q_y pred', 'q_z pred', 'q_w pred'])

    reorderLegend(ax,['p_x pred', 'p_x ref', 'p_y pred', 'p_y ref', 'p_z pred', 'p_z ref', 'q_x pred', 'q_x ref', 'q_y pred', 'q_y ref', 'q_z pred', 'q_z ref', 'q_w pred', 'q_w ref'])
    """

    # plt.savefig('quaternion.png', dpi=600)




    
def reorderLegend(ax=None,order=None,unique=False):
    if ax is None: ax=plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0])) # sort both labels and handles by labels
    if order is not None: # Sort according to a given list (not necessarily complete)
        keys=dict(zip(order,range(len(order))))
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t,keys=keys: keys.get(t[0],np.inf)))
    if unique:  labels, handles= zip(*unique_everseen(zip(labels,handles), key = labels)) # Keep only the first of each handle
    ax.legend(handles, labels, ncol=7, bbox_to_anchor=(0., 1.05, 1., .102), loc="upper center", fontsize=20)
    return(handles, labels)


def unique_everseen(seq, key=None):
    seen = set()
    seen_add = seen.add
    return [x for x,k in zip(seq,key) if not (k in seen or seen_add(k))]