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
    else:
        axs[0].set_title(r"$\gamma(\cdot)$ over Time")




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



