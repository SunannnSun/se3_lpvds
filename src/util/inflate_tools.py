import os
import numpy as np
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R




def adjust_cov(cov, tot_scale_fact=2, rel_scale_fact=0.15):
    cov_pos = cov[:3, :3]

    # cov_pos = cov
    
    eigenvalues, eigenvectors = np.linalg.eig(cov_pos)

    idxs = eigenvalues.argsort()
    inverse_idxs = np.zeros((idxs.shape[0]), dtype=int)
    for index, element in enumerate(idxs):
        inverse_idxs[element] = index

    eigenvalues_sorted  = np.sort(eigenvalues)
    eigenvectors_sorted = eigenvectors[:, idxs]
    # L = np.diag(eigenvalues_sorted) * tot_scale_fact

    cov_ratio = eigenvalues_sorted[1]/eigenvalues_sorted[2]
    if cov_ratio < rel_scale_fact:
        lambda_3 = eigenvalues_sorted[2]
        lambda_2 = eigenvalues_sorted[1] + lambda_3 * (rel_scale_fact - cov_ratio)
        lambda_1 = eigenvalues_sorted[0] + lambda_3 * (rel_scale_fact - cov_ratio)
        L = np.diag(np.array([lambda_1, lambda_2, lambda_3])) * tot_scale_fact
    else:
        L = np.diag(eigenvalues_sorted) * tot_scale_fact

    Sigma = eigenvectors_sorted @ L @ eigenvectors_sorted.T
    
    eigenvalues, eigenvectors = np.linalg.eig(Sigma)

    Sigma = Sigma[:, inverse_idxs]

    cov[:3, :3] = Sigma

    # cov = Sigma
 

    return cov






def adjust_Covariances(Priors, Sigma, tot_scale_fact, rel_scale_fact):
    # Check Relative Covariance Matrix Eigenvalues
    est_K = len(Sigma)
    dim = np.shape(Sigma)[1]
    Vs = np.zeros((est_K, dim, dim))
    Ls = np.zeros((est_K, dim, dim))
    p1_eig = []
    p2_eig = []
    p3_eig = []

    baseline_prior = (0.5 / len(Priors))

    for k in np.arange(est_K):
        w, v = np.linalg.eig(Sigma[k])
        Ls[k] = np.diag(w)
        Vs[k] = v.copy()
        if not all(sorted(w) == w):
            ids = w.argsort()
            L_ = np.sort(w)
            Ls[k] = np.diag(L_)
            Vs[k] = Vs[k][:, ids]

        if Priors[k] > baseline_prior:
            Ls[k] = tot_scale_fact * Ls[k]

        # 提取最大的两个特征值
        lambda_1 = Ls[k][0][0]
        lambda_2 = Ls[k][1][1]
        p1_eig.append(lambda_1)
        p2_eig.append(lambda_2)
        if dim == 3:
            lambda_3 = Ls[k][2][2]
            p3_eig.append(lambda_3)
        Sigma[k] = Vs[k] @ Ls[k] @ Vs[k].T

    p1_eig = np.array(p1_eig)
    p2_eig = np.array(p2_eig)
    p3_eig = np.array(p3_eig)

    if dim == 2:
        cov_ratios = np.array(p1_eig / p2_eig)
        for k in np.arange(0, est_K):
            if cov_ratios[k] < rel_scale_fact:
                lambda_1 = p1_eig[k]
                lambda_2 = p2_eig[k]
                lambda_1_ = lambda_1 + lambda_2 * (rel_scale_fact - cov_ratios[k])
                Sigma[k] = Vs[k] @ np.diag([lambda_1_, lambda_2]) @ Vs[k].T
    elif dim == 3:
        cov_ratios = np.array(p2_eig / p3_eig)
        for k in np.arange(0, est_K):
            if cov_ratios[k] < rel_scale_fact:
                lambda_1 = p1_eig[k]
                lambda_2 = p2_eig[k]
                lambda_3 = p3_eig[k]
                lambda_2_ = lambda_2 + lambda_3 * (rel_scale_fact - cov_ratios[k])
                lambda_1_ = lambda_1 + lambda_3 * (rel_scale_fact - cov_ratios[k])
                Sigma[k] = Vs[k] @ np.diag([lambda_1_, lambda_2_, lambda_3]) @ Vs[k].T

    return Sigma