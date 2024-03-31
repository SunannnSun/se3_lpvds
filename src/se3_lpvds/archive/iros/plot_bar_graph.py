import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


# import matplotlib as mpl 
# mpl.rcParams['text.usetex'] = True


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["font.size"] = 18



bar_colors = ['firebrick', 'white', 'forestgreen', 'white', 'royalblue', 'white']
whisker_colors = ['black', 'firebrick', 'black', 'forestgreen', 'black', 'royalblue']

label_list = ["SE3 LPV-DS", "NODEs", '', '', '', '']

x_list     = [0, 1, 3, 4, 6, 7]

hatch_list = ['', '/', '', '/', '', '/']

whisker_length = 0.2 

fig, axs = plt.subplots(2, 1, figsize=(8, 5))


'''
DTWD
'''

# upper_ex = [5, 5.2, 7, 15, 9, 20]
# q3 = [4, 4, 5, 10, 12, 6, 15]
# means = [3, 3, 4 ,6 , 4.5, 6.5]
# q1 = [2, 2.5, 3, 4, 4, 5]
# lower_ex = [1, 1, 1, 1, 1, 1]
upper_ex    = [900, 1050, 1100, 1700, 1200, 2000]
q3          = [850, 1000, 1000, 1350, 1100, 1750]
means       = [700, 900, 900, 1150, 950, 1400]
q1          = [650, 700, 800, 900, 800, 900]
lower_ex    = [500, 550, 600, 650, 550, 650]

ax = axs[0]

for i in range(6):
    x = x_list[i]

    ax.bar(x, q3[i]-q1[i], bottom = q1[i], capsize=2, linewidth=2, edgecolor=whisker_colors[i], color=bar_colors[i], hatch=hatch_list[i], label=label_list[i])

    ax.hlines(y=means[i], xmin=x-2/5, xmax=x+2/5, linewidth=2, color=whisker_colors[i])

    ax.plot([x, x], [q3[i], upper_ex[i]], color='black', linewidth=2)
    ax.plot([x, x], [q1[i], lower_ex[i]], color='black', linewidth=2)
    ax.plot([x - whisker_length, x + whisker_length], [upper_ex[i], upper_ex[i]], color='black', linewidth=2)
    ax.plot([x - whisker_length, x + whisker_length], [lower_ex[i], lower_ex[i]], color='black', linewidth=2)


ax.legend()
ax.set_xticks([0.5, 3.5, 6.5], [' ', '   ', ' '])
ax.set_ylabel('DTW Error')
ax.grid(axis='y', linestyle='--', alpha=0.7)


"""
quaternion-error
"""


upper_ex = [0.10, 0.10, 0.12, 0.21, 0.20, 0.30] 
q3 = [0.05, 0.06, 0.08, 0.15, 0.12, 0.21]
means = [0.03, 0.045, 0.05, 0.07 , 0.085, 0.13]
q1 = [0.02, 0.025, 0.03, 0.04, 0.05, 0.07]
lower_ex = [0.01, 0.015, 0.01, 0.015, 0.02, 0.03]

ax = axs[1]

for i in range(6):
    x = x_list[i]

    ax.bar(x, q3[i]-q1[i], bottom = q1[i], capsize=2, linewidth=2, edgecolor=whisker_colors[i], color=bar_colors[i], hatch=hatch_list[i])

    ax.hlines(y=means[i], xmin=x-2/5, xmax=x+2/5, linewidth=2, color=whisker_colors[i])

    ax.plot([x, x], [q3[i], upper_ex[i]], color='black', linewidth=2)
    ax.plot([x, x], [q1[i], lower_ex[i]], color='black', linewidth=2)
    ax.plot([x - whisker_length, x + whisker_length], [upper_ex[i], upper_ex[i]], color='black', linewidth=2)
    ax.plot([x - whisker_length, x + whisker_length], [lower_ex[i], lower_ex[i]], color='black', linewidth=2)


ax.set_xticks([0.5, 3.5, 6.5], ['Initial', 'Not Initial', 'Perturbed'])
ax.set_ylabel('Quaternion Error (rad)')
ax.grid(axis='y', linestyle='--', alpha=0.7)





plt.tight_layout()
plt.subplots_adjust(hspace=0.1)
plt.savefig('metric.png', dpi=600, bbox_inches='tight')




################################################################################################################

# """

COLOR_PARAM = "midnightblue"
COLOR_TIME = "darkred"

fig, axs = plt.subplots(1, 2, figsize=(8, 3.5))


data_size = [0, 300, 1000, 3000, 5000, 9000, 18000]

num_param_se3  = [0, 2, 2, 3, 4, 5, 9] 
num_param_se3  = [k * 84 for k in num_param_se3] # 84 parameters incluing A, mu, Sigma for each LTI system
num_param_node = [0, 1, 1, 1, 1, 1, 1]
num_param_node = [k * 1E6 for k in num_param_node]


time_se3  = [0, 0.3, 0.6, 1.2, 1.8, 2, 4]
time_node = [0, 1.5, 5, 8, 15, 26, 55]
time_node = [k * 60 for k in time_node]



ax1 = axs[0]
ax1.plot(data_size, num_param_se3, color=COLOR_PARAM, lw=4, label="SE(3) LPV-DS")
ax1.plot(data_size, num_param_node, '--', color=COLOR_PARAM, lw=4, label="NODEs")



ax2 = axs[1]
ax2.plot(data_size, time_se3, color=COLOR_TIME, lw=4, label="SE(3) LPV-DS")
ax2.plot(data_size, time_node, '--', color=COLOR_TIME, lw=4, label="NODEs")



# ax1.legend(ncol=1, loc ='best')
# ax2.legend(ncol=1, loc ='best')


ax1.set_xlabel("Data Size")
ax2.set_xlabel("Data Size")

ax1.set_ylabel("Number of Parameters")
ax2.set_ylabel("Computation Time (Sec)")


# ax1.yaxis.set_major_locator(MaxNLocator(nbins=3))

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_yticks([100, 1000, 10E5])
ax1.grid(axis='y', linestyle='--', alpha=0.7)

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_yticks([1, 10, 100, 1000])
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('metric3.png', dpi=600)

# """


plt.show()

