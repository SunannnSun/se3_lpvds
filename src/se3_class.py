import os, sys, json
import numpy as np
from scipy.spatial.transform import Rotation as R

from .util import optimize_tools, plot_tools
from .util.quat_tools import *
from .gmm_class import gmm_class


def write_json(data, path):
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def compute_ang_vel(q_k, q_kp1, dt=0.01):
    """ Compute angular velocity """

    dq = q_k.inv() * q_kp1    # from q_k to q_kp1 in body frame
    # dq = q_kp1 * q_k.inv()    # from q_k to q_kp1 in fixed frame

    dq = dq.as_rotvec() 
    w  = dq / dt
    
    return w




class se3_class:
    def __init__(self, p_in:np.ndarray, q_in:list, p_out:np.ndarray, q_out:list, p_att:np.ndarray, q_att:R, K_init:int) -> None:
        """
        Initialize an se3_class object

        Parameters:
        ----------
            p_in (np.ndarray):      [M, N] NumPy array of POSITION INPUT

            q_in (list):            M-length List of Rotation objects for ORIENTATION INPUT

            p_out (np.ndarray):     [M, N] NumPy array of POSITION OUTPUT

            q_out (list):           M-length List of Rotation objects for ORIENTATION OUTPUT

            p_att (np.ndarray):     [1, N] NumPy array of POSITION ATTRACTOR

            q_att (Rotation):       Single Rotation object for ORIENTATION ATTRACTOR
            
            K_init:                 Number of Gaussian Components

            M: Observation size

            N: Observation dimenstion (assuming 3D)
        """

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
        self.max_iter = 5000


        # define output path
        file_path           = os.path.dirname(os.path.realpath(__file__))  
        self.output_path    = os.path.join(os.path.dirname(file_path), 'output_ori.json')



    def _cluster(self):
        gmm = gmm_class(self.p_in, self.q_in, self.q_att, self.K_init)  

        self.gamma    = gmm.fit()  # K by M
        self.K        = gmm.K
        self.gmm      = gmm
        self.dual_gmm = gmm._dual_gmm()



    def _optimize(self):
        self.A_pos = optimize_tools.optimize_pos(self.p_in, self.p_out, self.p_att, self.gamma)
        self.A_ori = optimize_tools.optimize_ori(self.q_in, self.q_out, self.q_att, self.gamma)



    def begin(self):
        self._cluster()
        self._optimize()
        self._logOut()


    def sim(self, p_init, q_init, dt, step_size):
        p_test = [p_init.reshape(1, -1)]
        q_test = [q_init]
        gamma_test = []

        v_test = []
        w_test = []


        i = 0
        while np.linalg.norm((q_test[-1] * self.q_att.inv()).as_rotvec()) >= self.tol or np.linalg.norm((p_test[-1] - self.p_att)) >= self.tol:
            if i > self.max_iter:
                print("Exceed max iteration")
                break
            
            p_in  = p_test[i]
            q_in  = q_test[i]

            p_next, q_next, gamma, v, w = self._step(p_in, q_in, dt, step_size)

            p_test.append(p_next)        
            q_test.append(q_next)        
            gamma_test.append(gamma[:, 0])

            v_test.append(v)
            w_test.append(w)

            i += 1

        return np.vstack(p_test), q_test, np.array(gamma_test), v_test, w_test
        


    def _step(self, p_in, q_in, dt, step_size):
        """ Integrate forward by one time step """
        q_in = self._rectify(p_in, q_in)


        # read parameters
        A_pos = self.A_pos
        A_ori = self.A_ori

        p_att = self.p_att.reshape(1, -1)
        q_att = self.q_att

        K     = self.K
        gmm   = self.gmm

        # compute output
        p_diff  = p_in - p_att
        q_diff  = riem_log(q_att, q_in)
        
        p_out     = np.zeros((3, 1))
        q_out_att = np.zeros((4, 1))

        gamma = gmm.logProb(p_in, q_in)   # gamma value 
        for k in range(K):
            p_out     += gamma[k, 0] * A_pos[k] @ p_diff.T
            q_out_att += gamma[k, 0] * A_ori[k] @ q_diff.T

        p_next = p_in + p_out.T * step_size

        q_out_body = parallel_transport(q_att, q_in, q_out_att.T)
        q_out_q    = riem_exp(q_in, q_out_body) 
        q_out      = R.from_quat(q_out_q.reshape(4,))
        w_out      = compute_ang_vel(q_in, q_out, dt)   #angular velocity
        q_next     = q_in * R.from_rotvec(w_out * step_size)   #compose in body frame


        return p_next, q_next, gamma, p_out, w_out
    


    def _rectify(self, p_in, q_in):
        
        """
        Rectify q_init if it lies on the unmodeled half of the quaternion space
        """
        dual_gmm    = self.dual_gmm
        gamma_dual  = dual_gmm.logProb(p_in, q_in).T

        index_of_largest = np.argmax(gamma_dual)

        if index_of_largest <= (dual_gmm.K/2 - 1):
            return q_in
        else:
            return R.from_quat(-q_in.as_quat())
        
    


    def _logOut(self):

        Prior = self.gmm.Prior
        Mu    = self.gmm.Mu
        Mu_rollout = [np.hstack((p_mean, q_mean.as_quat())) for (p_mean, q_mean) in Mu]
        Sigma = self.gmm.Sigma

        Mu_arr      = np.zeros((self.K, 7)) 
        Sigma_arr   = np.zeros((self.K, 7, 7), dtype=np.float32)

        for k in range(self.K):
            Mu_arr[k, :] = Mu_rollout[k]
            Sigma_arr[k, :, :] = Sigma[k]

        json_output = {
            "name": "SE3-LPVDS result",

            "K": self.K,
            "M": 7,
            "Prior": Prior,
            "Mu": Mu_arr.ravel().tolist(),
            "Sigma": Sigma_arr.ravel().tolist(),

            'A_pos': self.A_pos.ravel().tolist(),
            'A_ori': self.A_ori.ravel().tolist(),
            'att_pos': self.p_att.ravel().tolist(),
            'att_ori': self.q_att.as_quat().ravel().tolist(),
            'q_init': self.q_in[0].as_quat().ravel().tolist(),
            "gripper_open": 0
        }

        js_path =  os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'output.json')
        write_json(json_output, js_path)