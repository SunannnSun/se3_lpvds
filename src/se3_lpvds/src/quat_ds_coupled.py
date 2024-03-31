import os, sys, json
import numpy as np
from scipy.spatial.transform import Rotation as R

from .util import optimize_tools, plot_tools
from .util.quat_tools import *
from .util.gmm_coupled import gmm as gmm_class   


def compute_ang_vel(q_k, q_kp1, dt=0.1):
    """
    Compute angular velocity in q_k frame to rotate q_k into q_kp1 given known time difference
    """

    dq = q_k.inv() * q_kp1

    dq = dq.as_rotvec()

    w  = 2 * dq / dt

    return w



class quat_ds:
    def __init__(self, p_in, q_in, q_out, q_att, index_list, K_init) -> None:
        self.p_in  = p_in
        self.q_in  = q_in
        self.q_out = q_out
        self.q_att = q_att
        self.index_list = index_list
        self.K_init = K_init

        self.N = len(q_in)
        
        """
        Mapping q_in to index_list
        """
        p_new = []
        q_new = []
        for l in range(len(index_list)):
            for idx in index_list[l]:
                p_new.append(p_in[idx, :])
                q_new.append(q_in[idx])
        self.p_new =  np.array(p_new)
        self.q_new =  q_new


        self.tol = 10E-3
        self.max_iter = 5000


    def _cluster(self):
        gmm = gmm_class(self.p_in, self.q_in, self.q_att)
        gmm.fit(self.K_init)

        self.postProb = gmm.predict(self.p_new, self.q_new, self.index_list)
        self.K        = gmm.K
        self.gmm      = gmm
        

    def _optimize(self):
        self.A = optimize_tools.optimize_quat_system(self.q_new, self.q_out, self.q_att, self.postProb)


    def begin(self):
        self._cluster()
        self._optimize()


    
    def step(self, p_in, q_in, dt):
        """
        recity awaits to be done

        p_in : 
        q_in : R object
        """

        K = self.K
        A = self.A
        q_att = self.q_att
        gmm   = self.gmm

        q_in_att  = riem_log(q_att, q_in)
        q_out_att = np.zeros((4, 1))

        w_k = gmm.postLogProb(p_in, q_in)
        for k in range(K):
            q_out_att += w_k[k, 0] * A[k] @ q_in_att.reshape(-1, 1)

        q_out_body = parallel_transport(q_att, q_in, q_out_att.T)
        q_out_q    = riem_exp(q_in, q_out_body) 
        q_out      = R.from_quat(q_out_q.reshape(4,))

        w_out      = compute_ang_vel(q_in, q_out, dt)
        q_next     = q_in * R.from_rotvec(w_out * dt)


        return q_next, w_k





    def sim(self, p_init, q_init, dt=0.1):
        """
        Forward simulation given an initial point

        Args:
            q_init: A Rotation object for starting point

        Returns:
            q_test: List of Rotation objects
            w_test: Array of weights (N by K)
        
        Parameters:
            w_out: the output angular velocity
         
        """    

        q_init = self._rectify(p_init, q_init)
        
        K = self.K
        A = self.A
        q_att = self.q_att
        gmm   = self.gmm
        
        q_test = [q_init]
        w_test = []

        i = 0
        while np.linalg.norm((q_test[-1] * self.q_att.inv()).as_rotvec()) >= self.tol:
            if i > self.max_iter:
                print("Simulation failed: exceed max iteration")
                return q_test, np.array(w_test)
                # sys.exit(0)

            p_in = p_init

            q_in      = q_test[i]
            q_in_att  = riem_log(q_att, q_in)
            q_out_att = np.zeros((4, 1))

            w_k = gmm.postLogProb(p_in, q_in)
            for k in range(K):
                q_out_att += w_k[k, 0] * A[k] @ q_in_att.reshape(-1, 1)

            q_out_body = parallel_transport(q_att, q_in, q_out_att.T)
            q_out_q    = riem_exp(q_in, q_out_body) 
            q_out      = R.from_quat(q_out_q.reshape(4,))

            w_out      = compute_ang_vel(q_in, q_out, dt)
            q_next     = q_in * R.from_rotvec(w_out * dt)

            q_test.append(q_next)        
            w_test.append(w_k[:, 0])

            i += 1

        return q_test, np.array(w_test)
        
    

    def _rectify(self, p_init, q_init):
        
        """
        Rectify q_init if it lies on the unmodeled half of the quaternion space
        """
        dual_gmm = self.gmm._dual_gmm()
        w_init = dual_gmm.postLogProb(p_init, q_init).T

        # plot_tools.plot_gmm_prob(w_init, title="GMM Posterior Probability of Original Data")

        index_of_largest = np.argmax(w_init)

        if index_of_largest <= (dual_gmm.K/2 - 1):
            return q_init
        else:
            return R.from_quat(-q_init.as_quat())
        

    def logOut(self):
        js_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output_ori.json')

        # with open(js_path, 'r') as f:
        #     data = json.load(f)
        
        Priors, Mu, Sigma = self.gmm.return_param()

        js = {
            "name": "quat_ds result",
            "K": self.K,
            "M": self.gmm.M,
            "Priors": Priors.tolist(),
            "Mu": Mu.ravel().tolist(),
            "Sigma": Sigma.ravel().tolist(),
            "A": self.A.ravel().tolist(),
            "att": self.q_att.as_quat().tolist()
        }

        with open(js_path, "w") as f:
            json.dump(js, f, indent=4)
    
        pass


        