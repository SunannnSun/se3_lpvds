import numpy as np

from scipy.spatial.transform import Rotation as R
from scipy.stats import multivariate_normal
from sklearn.mixture import BayesianGaussianMixture

from .util.quat_tools import *
from .util.plot_tools import *


def adjust_cov(cov, tot_scale_fact_pos=2,  tot_scale_fact_ori=1.2, rel_scale_fact=0.15):
    
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    eigenvalues_pos = eigenvalues[: 3]
    idxs_pos = eigenvalues_pos.argsort()
    inverse_idxs_pos = np.zeros((idxs_pos.shape[0]), dtype=int)
    for index, element in enumerate(idxs_pos):
        inverse_idxs_pos[element] = index

    eigenvalues_sorted_pos  = np.sort(eigenvalues_pos)
    cov_ratio = eigenvalues_sorted_pos[1]/eigenvalues_sorted_pos[2]
    if cov_ratio < rel_scale_fact:
        lambda_3_pos = eigenvalues_sorted_pos[2]
        lambda_2_pos = eigenvalues_sorted_pos[1] + lambda_3_pos * (rel_scale_fact - cov_ratio)
        lambda_1_pos = eigenvalues_sorted_pos[0] + lambda_3_pos * (rel_scale_fact - cov_ratio)

        lambdas_pos = np.array([lambda_1_pos, lambda_2_pos, lambda_3_pos])
        L_pos = np.diag(lambdas_pos[inverse_idxs_pos]) * tot_scale_fact_pos
    else:
        L_pos = np.diag(eigenvalues_pos) * tot_scale_fact_pos


    eigenvalues_ori = eigenvalues[3: ]
    idxs_ori = eigenvalues_ori.argsort()
    inverse_idxs_ori = np.zeros((idxs_ori.shape[0]), dtype=int)
    for index, element in enumerate(idxs_ori):
        inverse_idxs_ori[element] = index

    eigenvalues_sorted_ori  = np.sort(eigenvalues_ori)
    cov_ratio = eigenvalues_sorted_ori[2]/eigenvalues_sorted_ori[3]
    if cov_ratio < rel_scale_fact:
        lambda_4_ori = eigenvalues_sorted_ori[3]
        lambda_3_ori = eigenvalues_sorted_ori[2] + lambda_4_ori * (rel_scale_fact - cov_ratio)
        lambda_2_ori = eigenvalues_sorted_ori[1] + lambda_4_ori * (rel_scale_fact - cov_ratio)
        lambda_1_ori = eigenvalues_sorted_ori[0] + lambda_4_ori * (rel_scale_fact - cov_ratio)

        lambdas_ori = np.array([lambda_1_ori, lambda_2_ori, lambda_3_ori, lambda_4_ori])
        L_ori = np.diag(lambdas_ori[inverse_idxs_ori]) * tot_scale_fact_ori
    else:
        L_ori = np.diag(eigenvalues_ori) * tot_scale_fact_ori


    L = np.zeros((7, 7))
    L[:3, :3] = L_pos
    L[3:, 3:] = L_ori

    Sigma = eigenvectors @ L @ eigenvectors.T

    return Sigma




class gmm_class:
    def __init__(self, p_in:np.ndarray, q_in:list, q_att:R, K_init:int):
        """
        Initialize a GMM class

        Parameters:
        ----------
            p_in (np.ndarray):      [M, N] NumPy array of POSITION INPUT

            q_in (list):            M-length List of Rotation objects for ORIENTATION INPUT

            q_att (Rotation):       Single Rotation object for ORIENTATION ATTRACTOR
        """


        # store parameters
        self.p_in     = p_in
        self.q_in     = q_in
        self.q_att    = q_att
        self.K_init   = K_init

        self.M = len(q_in)
        self.N = 7

        # form concatenated state
        self.pq_in    = np.hstack((p_in, riem_log(q_att, q_in)))  




    def fit(self):
        """ 
        Fit model to data; 
        predict and store assignment label;
        extract and store Gaussian component 
        """

        gmm = BayesianGaussianMixture(n_components=self.K_init, n_init=1, random_state=2).fit(self.pq_in)
        assignment_arr = gmm.predict(self.pq_in)

        self._rearrange_array(assignment_arr)
        self._extract_gaussian()

        dual_gamma = self.logProb(self.p_in, self.q_in) # 2K by M

        return dual_gamma[:self.K, :] # K by M; always remain the first half
    




    def _rearrange_array(self, assignment_arr):
        """ Remove empty components and arrange components in order """
        rearrange_list = []
        for idx, entry in enumerate(assignment_arr):
            if not rearrange_list:
                rearrange_list.append(entry)
            if entry not in rearrange_list:
                rearrange_list.append(entry)
                assignment_arr[idx] = len(rearrange_list) - 1
            else:
                assignment_arr[idx] = rearrange_list.index(entry)   
        
        self.K = int(assignment_arr.max()+1)
        self.assignment_arr = assignment_arr




    def _extract_gaussian(self):
        """
        Extract Gaussian components from assignment labels and data

        Parameters:
        ----------
            Priors(list): K-length list of priors

            Mu(list):     K-length list of tuple: ([3, ] NumPy array, Rotation)

            Sigma(list):  K-length list of [N, N] NumPy array 
        """

        assignment_arr = self.assignment_arr

        Prior   = [0] * (2 * self.K)
        Mu      = [(np.zeros((3, )), R.identity())] * (2 * self.K)
        Sigma   = [np.zeros((self.N, self.N), dtype=np.float32)] * (2 * self.K)

        gaussian_list = [] 
        dual_gaussian_list = []
        for k in range(self.K):
            q_k      = [q for index, q in enumerate(self.q_in) if assignment_arr[index]==k] 
            q_k_mean = quat_mean(q_k)

            p_k      = [p for index, p in enumerate(self.p_in) if assignment_arr[index]==k]
            p_k_mean = np.mean(np.array(p_k), axis=0)

            q_diff = riem_log(q_k_mean, q_k) 
            p_diff = p_k - p_k_mean
            pq_diff = np.hstack((p_diff, q_diff))

            Prior[k]  = len(q_k)/ (2 * self.M)
            Mu[k]     = (p_k_mean, q_k_mean)
            Sigma_k  = pq_diff.T @ pq_diff / (len(q_k)-1)  + 10E-6 * np.eye(self.N)
            Sigma_k  = adjust_cov(Sigma_k)
            Sigma[k] = Sigma_k

            gaussian_list.append(
                {   
                    "prior" : Prior[k],
                    "mu"    : Mu[k],
                    "sigma" : Sigma[k],
                    "rv"    : multivariate_normal(np.hstack((Mu[k][0], np.zeros(4))), Sigma[k], allow_singular=True)
                }
            )


            q_k_dual  = [R.from_quat(-q.as_quat()) for q in q_k]
            q_k_mean_dual     = R.from_quat(-q_k_mean.as_quat())

            q_diff_dual = riem_log(q_k_mean_dual, q_k_dual)
            pq_diff_dual = np.hstack((p_diff, q_diff_dual))

            Prior[self.K + k] = Prior[k]
            Mu[self.K + k]     = (p_k_mean, q_k_mean_dual)
            Sigma_k_dual = pq_diff_dual.T @ pq_diff_dual / (len(q_k_dual)-1)  + 10E-6 * np.eye(self.N)
            Sigma_k_dual  = adjust_cov(Sigma_k_dual)
            Sigma[self.K+k]  = Sigma_k_dual

            dual_gaussian_list.append(
                {   
                    "prior" : Prior[self.K + k],
                    "mu"    : Mu[self.K + k],
                    "sigma" : Sigma[self.K+k],
                    "rv"    : multivariate_normal(np.hstack((Mu[self.K + k][0], np.zeros(4))), Sigma[self.K + k], allow_singular=True)
                }
            )


        self.gaussian_list = gaussian_list
        self.dual_gaussian_list = dual_gaussian_list


        self.Prior  = Prior
        self.Mu     = Mu
        self.Sigma  = Sigma




    def logProb(self, p_in, q_in):
        """ Compute log probability"""
        logProb = np.zeros((2 * self.K, p_in.shape[0]))

        for k in range(self.K):
            prior_k, mu_k, _, normal_k = tuple(self.gaussian_list[k].values())

            q_k  = riem_log(mu_k[1], q_in)
            pq_k = np.hstack((p_in, q_k))

            logProb[k, :] = np.log(prior_k) + normal_k.logpdf(pq_k)

        
        for k in range(self.K):
            prior_k, mu_k, _, normal_k = tuple(self.dual_gaussian_list[k].values())

            q_k  = riem_log(mu_k[1], q_in)
            pq_k = np.hstack((p_in, q_k))

            logProb[k+self.K, :] = np.log(prior_k) + normal_k.logpdf(pq_k)


        maxPostLogProb = np.max(logProb, axis=0, keepdims=True)
        expProb = np.exp(logProb - np.tile(maxPostLogProb, (2 * self.K, 1)))
        postProb = expProb / np.sum(expProb, axis = 0, keepdims=True)

        return postProb
    



'''
def adjust_cov_pos(cov, tot_scale_fact=2, rel_scale_fact=0.15):
    
    cov_pos = cov[:3, :3]

    eigenvalues, eigenvectors = np.linalg.eig(cov_pos)

    idxs = eigenvalues.argsort()
    inverse_idxs = np.zeros((idxs.shape[0]), dtype=int)
    for index, element in enumerate(idxs):
        inverse_idxs[element] = index

    eigenvalues_sorted  = np.sort(eigenvalues)
    cov_ratio = eigenvalues_sorted[1]/eigenvalues_sorted[2]
    if cov_ratio < rel_scale_fact:
        lambda_3 = eigenvalues_sorted[2]
        lambda_2 = eigenvalues_sorted[1] + lambda_3 * (rel_scale_fact - cov_ratio)
        lambda_1 = eigenvalues_sorted[0] + lambda_3 * (rel_scale_fact - cov_ratio)

        lambdas = np.array([lambda_1, lambda_2, lambda_3])

        L = np.diag(lambdas[inverse_idxs]) * tot_scale_fact
    else:
        L = np.diag(eigenvalues) * tot_scale_fact

    Sigma = eigenvectors @ L @ eigenvectors.T

    cov[:3, :3] = Sigma

    return cov




def adjust_cov_quat(cov, tot_scale_fact=1.2, rel_scale_fact=0.15):
    
    cov_quat = cov[3:, 3:]

    eigenvalues, eigenvectors = np.linalg.eig(cov_quat)

    idxs = eigenvalues.argsort()
    inverse_idxs = np.zeros((idxs.shape[0]), dtype=int)
    for index, element in enumerate(idxs):
        inverse_idxs[element] = index

    eigenvalues_sorted  = np.sort(eigenvalues)
    cov_ratio = eigenvalues_sorted[2]/eigenvalues_sorted[3]
    if cov_ratio < rel_scale_fact:
        lambda_4 = eigenvalues_sorted[3]
        lambda_3 = eigenvalues_sorted[2] + lambda_4 * (rel_scale_fact - cov_ratio)
        lambda_2 = eigenvalues_sorted[1] + lambda_4 * (rel_scale_fact - cov_ratio)
        lambda_1 = eigenvalues_sorted[0] + lambda_4 * (rel_scale_fact - cov_ratio)

        lambdas = np.array([lambda_1, lambda_2, lambda_3, lambda_4])

        L = np.diag(lambdas[inverse_idxs]) * tot_scale_fact
    else:
        L = np.diag(eigenvalues) * tot_scale_fact


    Sigma = eigenvectors @ L @ eigenvectors.T

    cov[3:, 3:] = Sigma

    return cov
'''
