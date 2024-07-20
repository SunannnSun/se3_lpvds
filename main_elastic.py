import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.linalg import null_space

from src.util import load_tools, process_tools, quat_tools
from src.se3_class import se3_class
from src.se3_elastic.src.util import plot_tools
from src.se3_elastic.elastic_pos_class import elastic_pos_class
from src.se3_elastic.elastic_ori_class import elastic_ori_class



'''Load data'''
p_raw, q_raw, t_raw, dt = load_tools.load_UMI()



'''Process data'''
p_in, q_in, t_in             = process_tools.pre_process(p_raw, q_raw, t_raw, opt= "savgol")
p_out, q_out                 = process_tools.compute_output(p_in, q_in, t_in)

p_init, q_init, p_att, q_att = process_tools.extract_state(p_in, q_in)
p_in, q_in, p_out, q_out     = process_tools.rollout_list(p_in, q_in, p_out, q_out)



'''Run lpvds'''
se3_obj = se3_class(p_in, q_in, p_out, q_out, p_att, q_att, dt, K_init=4)
se3_obj.begin()



'''Run elastic'''
elastic_obj = elastic_pos_class(se3_obj.pos_ds.damm.Prior, se3_obj.pos_ds.damm.Mu, se3_obj.pos_ds.damm.Sigma, p_att, p_in, p_out)
traj_data, old_gmm_struct, gmm_struct, old_anchor, new_anchor = elastic_obj.start_adapting()



# K_pos= gmm_struct['Prior'].shape[0]
# for k in range(K_pos):
#     prior_k = gmm_struct['Prior'][k]
#     mu_k = gmm_struct['Mu'][:, k]
#     print(prior_k)
#     print(mu_k)

#     prior_k = old_gmm_struct['Prior'][k]
#     mu_k = old_gmm_struct['Mu'][:, k]
#     print(prior_k)
#     print(mu_k)
    

# # Sigma = np.zeros((K_ori, 4, 4))
# for k in range(K_pos):
#     print("new Sigma", gmm_struct['Sigma'][k, :, :])
#     print("old Sigma", old_gmm_struct['Sigma'][k, :, :])



elastic_ori_obj = elastic_ori_class(se3_obj.ori_ds.gmm.Prior, se3_obj.ori_ds.gmm.Mu, se3_obj.ori_ds.gmm.Sigma, q_att, q_in, q_out)
traj_data_ori, old_gmm_struct_ori, gmm_struct_ori, old_anchor_ori, new_anchor_ori = elastic_ori_obj.start_adapting()



M = traj_data_ori[0].shape[1]
new_ori = [R.identity()] * M
for i in range(M):
    ori_i_red = traj_data_ori[0][:3, i]
    ori_i = elastic_ori_obj.normal_basis @ ori_i_red + elastic_ori_obj.normal_vec
    ori_i_quat = quat_tools.riem_exp(q_att, ori_i.reshape(1, -1))
    new_ori[i] = R.from_quat(ori_i_quat[0])


new_ori_out = [R.identity()] * M
for i in range(M-1):
    new_ori_out[i] = new_ori[i+1]
new_ori_out[-1] = new_ori[-1]



K_ori= gmm_struct_ori['Prior'].shape[0]
Mu = [R.identity()] * K_ori
for k in range(K_ori):
    # prior_k = gmm_struct_ori['Prior'][k]
    # print(prior_k)

    # prior_k = old_gmm_struct_ori['Prior'][k]
    # print(prior_k)

    mu_k = gmm_struct_ori['Mu'][:, k]
    mu_k = elastic_ori_obj.normal_basis @ mu_k + elastic_ori_obj.normal_vec
    mu_k = quat_tools.riem_exp(q_att, mu_k.reshape(1, -1))
    Mu[k] = R.from_quat(mu_k[0])
    # print(Mu[k].as_quat())



    # mu_k = old_gmm_struct_ori['Mu'][:, k]
    # mu_k = elastic_ori_obj.normal_basis @ mu_k + elastic_ori_obj.normal_vec
    # mu_k = quat_tools.riem_exp(q_att, mu_k.reshape(1, -1))
    
    # Mu[k] = se3_obj.ori_ds.gmm.Mu[k]
    # print(se3_obj.ori_ds.gmm.Mu[k].as_quat())


Sigma = np.zeros((K_ori, 4, 4))
# for k in reversed(range(K_ori)):
for k in range(K_ori):
    Sigma[k, :, :] = elastic_ori_obj.normal_basis @ gmm_struct_ori['Sigma'][k, :, :] @ elastic_ori_obj.normal_basis.T
    # Sigma[k, :, :] = elastic_ori_obj.normal_basis @ old_gmm_struct_ori['Sigma'][k, :, :] @ elastic_ori_obj.normal_basis.T

    # print(Sigma[k, :, :])
    # print(se3_obj.ori_ds.gmm.Sigma[k])



se3_obj.ori_ds.elasticUpdate(new_ori, new_ori_out, gmm_struct_ori['Prior'], Mu, Sigma)



# a = elastic_obj.normal_basis @ new_anchor_ori[2, :]+ elastic_obj.normal_vec
# print(a)

# plot_tools.ori_debug(elastic_ori_obj.data[:, :3], traj_data_ori[0][:3, :].T, old_anchor_ori, new_anchor_ori, old_gmm_struct_ori, gmm_struct_ori)

plot_tools.demo_vs_adjust(p_in, traj_data[0][:3, :].T, old_anchor, new_anchor, q_in, new_ori)
plot_tools.demo_vs_adjust_gmm(p_in, traj_data[0][:3, :].T, se3_obj.pos_ds.damm, old_gmm_struct, new_ori, gmm_struct)



'''Evaluate results'''
p_init = p_init[0]
q_init = R.from_quat(-q_init[0].as_quat())
p_test, q_test, gamma_pos, gamma_ori, v_test, w_test = se3_obj.sim(p_init, q_init, step_size=0.05)


from src.util import plot_tools, load_tools, process_tools

'''Plot results'''
# plot_tools.plot_p_out(p_in, p_out, se3_obj.pos_ds)
# plot_tools.plot_q_out(q_in, q_out, se3_obj.ori_ds)

# plot_tools.plot_gmm_pos(p_in, se3_obj.pos_ds.damm)
# plot_tools.plot_gmm_ori(p_in, se3_obj.ori_ds.gmm)

plot_tools.plot_result(p_in, p_test, q_test)

se3_obj.pos_ds.elasticUpdate(traj_data[0], gmm_struct)


q_init = q_init * R.from_euler('xyz', [2.5, 0.2, 2.1])


p_test, q_test, gamma_pos, gamma_ori, v_test, w_test = se3_obj.sim(p_init, q_init, step_size=0.02)
plot_tools.plot_result(p_in, p_test, q_test)


# plot_tools.plot_gamma(gamma_pos, title="pos")
# plot_tools.plot_gamma(gamma_ori, title="ori")



plt.show()







# v = q_in[10].as_quat()
# normal_vec = q_att.as_quat()
# basis_att = null_space(normal_vec.reshape(1, -1))
# print(basis_att)

# c, residuals, rank, s = np.linalg.lstsq(basis_att, quat_tools.list_to_arr(q_in).T, rcond=None)
# c = np.linalg.solve(basis_att.T @ basis_att, basis_att.T @ v)

# print("The coordinates of the vector on the hyperplane in terms of the basis are:")
# print(c)
# print(v)
# print(basis_att @ c + normal_vec)


# # '''Record Relative Orientation'''
# ori_sigma_old = [R.identity] * se3_obj.K
# for k in range(se3_obj.K):

#     Sigma_k = old_gmm_struct["Sigma"][k, :, :]
#     _, eigen_vectors = np.linalg.eig(Sigma_k)
#     ori_sigma_old[k] = R.from_matrix(eigen_vectors)

#     # ori_sigma_old[k] = R.from_matrix(old_gmm_struct.Sigma[k, :, :])


# assignment_array = se3_obj.gmm.assignment_arr
# M = assignment_array.shape[0]
# relative_ori = [R.identity()] * M
# for i in range(M):
#     k = assignment_array[i]
#     relative_ori[i] = q_in[i] * ori_sigma_old[k].inv() 


# # '''Update new orientation'''
# ori_sigma_new = [R.identity] * se3_obj.K
# for k in reversed(range(se3_obj.K)):
#     Sigma_k = gmm_struct["Sigma"][k, :, :]
#     _, eigen_vectors = np.linalg.eig(Sigma_k)
#     ori_sigma_new[k] = R.from_matrix(eigen_vectors)

#     # ori_sigma_new[k] = R.from_matrix(gmm_struct.Sigma[k, :, :])


# new_ori = [R.identity()] * M
# for i in range(M):
#     k = assignment_array[i]
#     new_ori[i] = relative_ori[i] * ori_sigma_new[k]