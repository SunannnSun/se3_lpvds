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






def plot_gamma(gamma_arr, **argv):

    M, K = gamma_arr.shape

    fig, axs = plt.subplots(K, 1, figsize=(12, 8))

    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]

    for k in range(K):
        axs[k].scatter(np.arange(M), gamma_arr[:, k], s=5, color=colors[k])
        axs[k].set_ylim([0, 1])
    
    if "title" in argv:
        axs[0].set_title(argv["title"])
    else:
        axs[0].set_title(r"$\gamma(\cdot)$ over Time")




def plot_result(p_train, p_test, q_test):

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection='3d')


    ax.plot(p_train[:, 0], p_train[:, 1], p_train[:, 2], 'o', color='gray', alpha=0.2, markersize=1.5, label="Demonstration")
    ax.plot(p_test[:, 0], p_test[:, 1], p_test[:, 2],  color='k', label = "Reproduction")

    ax.scatter(p_test[0, 0], p_test[0, 1], p_test[0, 2], 'o', facecolors='none', edgecolors='magenta',linewidth=2,  s=100, label="Initial")
    ax.scatter(p_test[-1, 0], p_test[-1, 1], p_test[-1, 2], marker=(8, 2, 0), color='k',  s=100, label="Target")


    colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB
    x_min, x_max = ax.get_xlim()
    scale = (x_max - x_min)/4
    
    for i in np.linspace(0, len(q_test), num=10, endpoint=False, dtype=int):

        r = q_test[i]
        loc = p_test[i, :]
        for j, (axis, c) in enumerate(zip((ax.xaxis, ax.yaxis, ax.zaxis),
                                            colors)):
            line = np.zeros((2, 3))
            line[1, j] = scale
            line_rot = r.apply(line)
            line_plot = line_rot + loc
            ax.plot(line_plot[:, 0], line_plot[:, 1], line_plot[:, 2], c, linewidth=1)


    ax.axis('equal')
    ax.legend(ncol=4, loc="upper center")

    ax.set_xlabel(r'$\xi_1$', labelpad=20)
    ax.set_ylabel(r'$\xi_2$', labelpad=20)
    ax.set_zlabel(r'$\xi_3$', labelpad=20)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))





def plot_gmm(p_in, gmm):

    label = gmm.assignment_arr
    K     = gmm.K

    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]

    color_mapping = np.take(colors, label)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(p_in[:, 0], p_in[:, 1], p_in[:, 2], 'o', color=color_mapping[:], s=1, alpha=0.4, label="Demonstration")

    colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB

    x_min, x_max = ax.get_xlim()
    scale = (x_max - x_min)/4
    for k in range(K):
        label_k =np.where(label == k)[0]

        p_in_k = p_in[label_k, :]
        loc = np.mean(p_in_k, axis=0)

        r = gmm.gaussian_list[k]["mu"][1]
        for j, (axis, c) in enumerate(zip((ax.xaxis, ax.yaxis, ax.zaxis),
                                            colors)):
            line = np.zeros((2, 3))
            line[1, j] = scale
            line_rot = r.apply(line)
            line_plot = line_rot + loc
            ax.plot(line_plot[:, 0], line_plot[:, 1], line_plot[:, 2], c, linewidth=1)


    ax.axis('equal')


    ax.set_xlabel(r'$\xi_1$', labelpad=20)
    ax.set_ylabel(r'$\xi_2$', labelpad=20)
    ax.set_zlabel(r'$\xi_3$', labelpad=20)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))




