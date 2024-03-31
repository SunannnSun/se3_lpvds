import os, sys, json
import numpy as np
from scipy.spatial.transform import Rotation as R

from .util import optimize_tools, plot_tools
from .util.quat_tools import *
from .util.gmm import gmm as gmm_class   


def compute_ang_vel(q_k, q_kp1, dt=0.01):
    """
    note: dt here is our "assumed" time difference between each of our data points after filtering... and does not
    bear the same meaning as the time step in simulation.. hence in SE(3) should be set consistently with the dt
    used to generate the linear velocity...
    
    """

    dq = q_k.inv() * q_kp1    # from q_k to q_kp1 in body frame

    # dq = q_kp1 * q_k.inv()    # from q_k to q_kp1 in fixed frame

    dq = dq.as_rotvec() 

    w  = dq / dt

    return w



class se3_lpvds:
    def __init__(self, p_in, q_in, p_out, q_out, p_att, q_att, K_init) -> None:

        # store parameters
        self.p_in  = p_in
        self.q_in  = q_in

        self.p_out = p_out
        self.q_out = q_out

        self.p_att = p_att
        self.q_att = q_att

        self.K_init = K_init
        self.M = len(q_in)


        # simulation parameters
        self.tol = 10E-3
        self.max_iter = 10000


        # define output path
        file_path           = os.path.dirname(os.path.realpath(__file__))  
        self.output_path    = os.path.join(os.path.dirname(file_path), 'output_ori.json')



    def _cluster(self):

        gmm = gmm_class(self.p_in, self.q_in, self.q_att)  
        gmm.fit(self.K_init)
        
        self.postProb = gmm.predict() 
        self.K        = gmm.K
        self.gmm      = gmm
        

    def _optimize(self):

        self.A_pos = optimize_tools.optimize_pos(self.p_in, self.p_out, self.p_att, self.postProb)


        self.A_ori = optimize_tools.optimize_ori(self.q_in, self.q_out, self.q_att, self.postProb)



    def begin(self):
        self._cluster()
        self._optimize()

        # self.logOut()




    def step(self, q_in, dt):
            """
            recity awaits to be done
            """

            K = self.K

            A_pos = self.A_pos
            A_ori = self.A_ori

            p_att = self.p_att
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

            w_out      = compute_ang_vel(q_in, q_out)

            # w_out /= 20

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

        # q_init = self._rectify(q_init)
        
        K = self.K

        A_pos = self.A_pos
        A_ori = self.A_ori

        p_att = self.p_att.reshape(1, -1)
        q_att = self.q_att

        gmm   = self.gmm

        p_test = [p_init.reshape(1, -1)]
        q_test = [q_init]
        w_test = []

        i = 0
        while np.linalg.norm((q_test[-1] * self.q_att.inv()).as_rotvec()) >= self.tol:
            if i > self.max_iter:
                print("Simulation failed: exceed max iteration")
                break
            
            p_in  = p_test[i]
            q_in  = q_test[i]

            p_diff  = p_in - p_att
            q_diff  = riem_log(q_att, q_in)
            
            p_out     = np.zeros((3, 1))
            q_out_att = np.zeros((4, 1))

            w_k = gmm.postLogProb(p_in, q_in)
            for k in range(K):
                p_out     += w_k[k, 0] * A_pos[k] @ p_diff.T
                q_out_att += w_k[k, 0] * A_ori[k] @ q_diff.T

            p_next = p_in + p_out.T * dt

            q_out_body = parallel_transport(q_att, q_in, q_out_att.T)
            q_out_q    = riem_exp(q_in, q_out_body) 
            q_out      = R.from_quat(q_out_q.reshape(4,))

            w_out      = compute_ang_vel(q_in, q_out, dt)
            q_next     = q_in * R.from_rotvec(w_out * dt)   #body frame


            p_test.append(p_next)        
            q_test.append(q_next)        
            w_test.append(w_k[:, 0])

            i += 1

        return p_test, q_test, np.array(w_test)
    

    def _rectify(self, q_init):
        
        """
        Rectify q_init if it lies on the unmodeled half of the quaternion space
        """
        dual_gmm = self.gmm._dual_gmm()
        w_init = dual_gmm.postLogProb(q_init).T

        # plot_tools.plot_gmm_prob(w_init, title="GMM Posterior Probability of Original Data")

        index_of_largest = np.argmax(w_init)

        if index_of_largest <= (dual_gmm.K/2 - 1):
            return q_init
        else:
            return R.from_quat(-q_init.as_quat())



    def logOut(self):

        Priors, Mu, Sigma = self.gmm.return_param()

        js = {
            "name": "quat_ds result",
            "K": self.K,
            "M": self.gmm.M,
            "Priors": Priors.tolist(),
            "Mu": Mu.ravel().tolist(),
            "Sigma": Sigma.ravel().tolist(),
            "A": self.A.ravel().tolist(),
            "att": self.q_att.as_quat().tolist(),
            "q_init": self.q_init.as_quat().tolist()
        }

        with open(self.output_path, "w") as f:
            json.dump(js, f, indent=4)
    
        pass       