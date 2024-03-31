import numpy as np

from scipy.spatial.transform import Rotation as R
from scipy.stats import multivariate_normal
from sklearn.mixture import BayesianGaussianMixture

from .quat_tools import *
from .plot_tools import *


class gmm:
    def __init__(self, p_in, q_in, q_att):
        self.p_in     = p_in

        self.q_in     = q_in
        self.q_att    = q_att


        self.N = len(q_in)
        self.M = 4 + 3



    def fit(self, K_init):
        """
        Fit is done on all observations (prior to filtering); assignment_arr is from the prediction on 
        all observations and is used to extract gaussians
        """

        q_in_att = riem_log(self.q_att, self.q_in)

        pq_in = np.hstack((self.p_in, q_in_att))

        gmm = BayesianGaussianMixture(n_components=K_init, n_init=1, random_state=2).fit(pq_in)

        assignment_arr = gmm.predict(pq_in)
        self.assignment_arr = self._rearrange_array_fit(assignment_arr)

        self.K = int(assignment_arr.max()+1)
        self._return_normal_class(assignment_arr)

        self.gmm = gmm



    def predict(self, p_in, q_in, index_list):
        q_in_att = riem_log(self.q_att, q_in)

        pq_in = np.hstack((p_in, q_in_att))

        assignment_arr = self.gmm.predict(pq_in)
        assignment_arr = self._rearrange_array_predict(assignment_arr)

        # postProb = self.postLogProb(q_in)
        postProb = self.postLogProb(p_in, q_in)

        # plot_gmm(q_in, index_list, assignment_arr, interp=True)
        plot_gmm(self.q_in, index_list, self.assignment_arr, interp="Full")

        return postProb



    def _return_normal_class(self, assignment_arr):
        """
        Parameters:
            q_sod: sum of difference

        Return normal_class: 
            Assuming one distribution if only q_list is given;
            Output a list of normal_class if assignment_arr is provided too;
            Each normal class is defined by Mu, Sigma, (Prior is optional);
            Store the result normal class and Prior in gmm class

        @note verify later if the R.mean() yields the same results as manually computing the Krechet mean of quaternion
        """


        Prior   = [0] * self.K
        Mu      = [(np.zeros((3, )), R.identity())] * self.K
        Sigma   = [np.zeros((self.M, self.M), dtype=np.float32)] * self.K

        q_normal_list = [] 

        for k in range(self.K):
            q_k      = [q for index, q in enumerate(self.q_in) if assignment_arr[index]==k] 
            q_k_mean = quat_mean(q_k)

            p_k      = [p for index, p in enumerate(self.p_in) if assignment_arr[index]==k]
            p_k_mean = np.mean(np.array(p_k), axis=0)

            q_sod = riem_log(q_k_mean, q_k)
            p_sod = p_k - p_k_mean

            pq_sod = np.hstack((p_sod, q_sod))


            Prior[k]  = len(q_k)/self.N
            # Mu[k]     = q_k_mean

            Mu[k]     = (p_k_mean, q_k_mean)

            # Sigma[k]  = riem_cov(q_k_mean, q_k) + 10E-6 * np.eye(4)

            Sigma[k]  = pq_sod.T @ pq_sod / len(q_k)  + 10E-6 * np.eye(self.M)

            # mean = np.hstack((Mu[k][0], np.zeros(4)))
            q_normal_list.append(
                {
                    # "prior" : Prior[k],
                    "mu"    : Mu[k],
                    "sigma" : Sigma[k],
                    "rv"    : multivariate_normal(np.hstack((Mu[k][0], np.zeros(4))), Sigma[k], allow_singular=True)
                }
            )

        self.Prior = Prior
        self.q_normal_list = q_normal_list
    

    def return_param(self):
        K = self.K
        M = self.M 

        Priors = np.zeros((K, ))
        Mu     = np.zeros((K, M))
        Sigma  = np.zeros((K, M, M))

        q_normal_list = self.q_normal_list

        for k in range(K):
            Priors[k] = self.Prior[k]
            Mu[k, :3] = q_normal_list[k]["mu"][0]
            Mu[k, 3:] = q_normal_list[k]["mu"][1].as_quat()
            Sigma[k, :, :] = q_normal_list[k]["sigma"]


        return Priors, Mu, Sigma



    def logProb(self, p_in, q_in):

        """
        vectorized operation
        """


        if isinstance(q_in, R):
            logProb = np.zeros((self.K, 1))
        elif isinstance(q_in, list):
            logProb = np.zeros((self.K, len(q_in)))
        else:
            print("Invalid q_list type")
            sys.exit()


        for k in range(self.K):
            mu_k, _, normal_k = tuple(self.q_normal_list[k].values())

            q_k = riem_log(mu_k[1], q_in)

            pq_k = np.hstack((p_in, q_k))

            logProb[k, :] = normal_k.logpdf(pq_k)
        
        return logProb
    


    def postLogProb(self, p_in, q_in):

        """
        vectorized operation
        """

        logPrior = np.log(self.Prior)
        logProb  = self.logProb(p_in, q_in)

        if isinstance(q_in, R):
            postLogProb  = logPrior[:, np.newaxis] + logProb
        elif isinstance(q_in, list):
            postLogProb  = np.tile(logPrior[:, np.newaxis], (1, len(q_in))) + logProb

        maxPostLogProb = np.max(postLogProb, axis=0, keepdims=True)
        expProb = np.exp(postLogProb - np.tile(maxPostLogProb, (self.K, 1)))
        postProb = expProb / np.sum(expProb, axis = 0, keepdims=True)


        return postProb
    
    

    def _dual_gmm(self):

        """
        Double the normal list and the priors
        """

        K_dual = self.K * 2
        Prior_dual   = [0] * K_dual
        q_normal_list_dual = [] 

        for k in range(int(K_dual/2)):      
            Prior_dual[k]  = self.Prior[k] / 2
            Mu, Sigma, _ = tuple(self.q_normal_list[k].values())
    
            q_normal_list_dual.append(
                {
                    "mu"    : Mu,
                    "sigma" : Sigma,
                    "rv"    : multivariate_normal(np.hstack((Mu[0], np.zeros(4))), Sigma, allow_singular=True)
                }
            )

        for k in np.arange(int(K_dual/2), K_dual):          
            k_prev = k - int(K_dual/2)
            Prior_dual[k]  = Prior_dual[k_prev]
            Mu, Sigma, _ = tuple(self.q_normal_list[k_prev].values())
            Mu     = (Mu[0], R.from_quat(-Mu[1].as_quat()))

            q_normal_list_dual.append(
                {
                    "mu"    : Mu,
                    "sigma" : Sigma,
                    "rv"    : multivariate_normal(np.hstack((Mu[0], np.zeros(4))), Sigma, allow_singular=True)
                }
            )
        
        dual_gmm = gmm(self.p_in, self.q_in, self.q_att)
        dual_gmm.K = K_dual
        dual_gmm.Prior = Prior_dual
        dual_gmm.q_normal_list = q_normal_list_dual

        return dual_gmm




    def _rearrange_array_fit(self, assignment_arr):
        """
        Remove empty components and arrange components in order
        """
        rearrange_list = []
        for idx, entry in enumerate(assignment_arr):
            if not rearrange_list:
                rearrange_list.append(entry)
            if entry not in rearrange_list:
                rearrange_list.append(entry)
                assignment_arr[idx] = len(rearrange_list) - 1
            else:
                assignment_arr[idx] = rearrange_list.index(entry)   
        
        self.rearrange_list = rearrange_list
        return assignment_arr




    def _rearrange_array_predict(self, assignment_arr):
        """
        Match the components from _rearrange_array_fit
        """
        rearrange_list = self.rearrange_list
        for idx, entry in enumerate(assignment_arr):
            assignment_arr[idx] = rearrange_list.index(entry)   
        return assignment_arr
