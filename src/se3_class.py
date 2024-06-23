import os, sys, json
import numpy as np
from scipy.spatial.transform import Rotation as R

from .util import quat_tools, plot_tools
from .lpvds.src.lpvds_class import lpvds_class
from .quaternion_ds.src.quat_class import quat_class



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


        # init pos and ori lpvds class
        self.pos_ds = lpvds_class(p_in, p_out, p_att)
        self.ori_ds = quat_class(q_in, q_out, q_att, K_init=self.K_init)



    def _cluster(self):
        self.pos_ds._cluster()
        self.ori_ds._cluster()



    def _optimize(self):
        self.pos_ds._optimize()
        self.ori_ds._optimize()

        self.A_pos = self.pos_ds.A
        self.A_ori = self.ori_ds.A_ori


    def _logOut(self):
        # self.pos_ds.log
        pass



    def begin(self):
        self._cluster()
        self._optimize()
        # self._logOut()



    def sim(self, p_init, q_init, dt, step_size):
        p_test = [p_init.reshape(1, -1)]
        q_test = [q_init]

        gamma_pos_list = []
        gamma_ori_list = []

        v_test = []
        w_test = []


        i = 0
        while np.linalg.norm((q_test[-1] * self.q_att.inv()).as_rotvec()) >= self.tol or np.linalg.norm((p_test[-1] - self.p_att)) >= self.tol:
            if i > self.max_iter:
                print("Exceed max iteration")
                break
            
            p_in  = p_test[i]
            q_in  = q_test[i]

            p_next, q_next, gamma_pos, gamma_ori, v, w = self._step(p_in, q_in, dt, step_size)

            p_test.append(p_next)        
            q_test.append(q_next)        
            gamma_pos_list.append(gamma_pos[:, 0])
            gamma_ori_list.append(gamma_ori[:, 0])


            v_test.append(v)
            w_test.append(w)

            i += 1

        return np.vstack(p_test), q_test, np.array(gamma_pos_list),  np.array(gamma_ori_list), v_test, w_test
        


    def _step(self, p_in, q_in, dt, step_size):
        """ Integrate forward by one time step """

        p_out     = np.zeros((3, 1)) # missing

        gamma_pos = self.pos_ds.damm.logProb(p_in)   # gamma value 



        p_next                   = self.pos_ds._step(p_in, dt)
        q_next, gamma_ori, omega = self.ori_ds._step(q_in, dt, step_size)


        return p_next, q_next, gamma_pos, gamma_ori, p_out, omega
    


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
        
    

