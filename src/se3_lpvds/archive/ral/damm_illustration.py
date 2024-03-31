import os
import numpy as np
import matplotlib.pyplot as plt

# Import Packages
from src.damm_lpvds.src.util import load_tools, plot_tools
from src.damm_lpvds.src.damm_lpvds import damm_lpvds as damm_lpvds_class


# choose input option
input_message = '''
Please choose a data input option:
1. PC-GMM benchmark data
2. LASA benchmark data
3. DAMM demo data
Enter the corresponding option number: '''


input_opt = input(input_message)
input_data = load_tools.load_data(int(input_opt))

Data, Data_sh, att, x0_all, dt, _, traj_length = load_tools.processDataStructure(input_data)

# plot_tools.plot_reference_trajectories_DS(input_data, att, 100, 20)



from ral import plot_ral
plot_ral.plot_reference_trajectories_DS2(input_data, att, 100, 20)

# plot_ral.plot_damm_vis(input_data)
# plot_ral.plot_damm_vis_blue(input_data)
# plot_ral.plot_damm_vis_red(input_data)

# plot_ral.plot_PS_vis(input_data)
# plot_ral.plot_PS_launch_init(input_data)
# plot_ral.plot_PS_launch(input_data)


# damm_lpvds = damm_lpvds_class(Data[:, ::3], Data_sh[:, ::3], att, x0_all, dt, traj_length)

# damm_lpvds.begin()


plt.show()