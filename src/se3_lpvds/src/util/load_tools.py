import os
import numpy as np
from scipy.spatial.transform import Rotation as R



def _get_sequence(seq_file):
    """
    Returns a list of containing each line of `seq_file`
    as an element

    Args:
        seq_file (str): File with name of demonstration files
                        in each line

    Returns:
        [str]: List of demonstration files
    """
    seq = None
    with open(seq_file) as x:
        seq = [line.strip() for line in x]
    return seq




def load_clfd_dataset(task_id=1, num_traj=1, sub_sample=3):
    """
    Load data from clfd dataset

    Return:
        p_raw:  a LIST of L trajectories, each containing M observations of N dimension, or [M, N] ARRAY;
                M can vary and need not be same between trajectories

        q_raw:  a LIST of L trajectories, each containting a LIST of M (Scipy) Rotation objects;
                need to consistent with M from position

        
    Note:
        NO time stamp available in this dataset!

        [num_demos=9, trajectory_length=1000, data_dimension=7] 
        A data point consists of 7 elements: px,py,pz,qw,qx,qy,qz (3D position followed by quaternions in the scalar first format).
    """

    L = num_traj
    T = 10.0            # pick a time duration to hand engineer an equal-length time stamp

    file_path           = os.path.dirname(os.path.realpath(__file__))  
    dir_path            = os.path.dirname(file_path)
    data_path           = os.path.dirname(dir_path)

    seq_file    = os.path.join(data_path, "dataset", "pos_ori", "robottasks_pos_ori_sequence_4.txt")
    filenames   = _get_sequence(seq_file)
    datafile    = os.path.join(data_path, "dataset", "pos_ori", filenames[task_id])
    
    data        = np.load(datafile)[:, ::sub_sample, :]

    p_raw = []
    q_raw = []
    t_raw = []

    for l in range(L):
        M = data[l, :, :].shape[0]

        data_ori = np.zeros((M, 4))         # convert to scalar last format, consistent with Scipy convention
        w        = data[l, :, 3 ].copy()  
        xyz      = data[l, :, 4:].copy()
        data_ori[:, -1]  = w
        data_ori[:, 0:3] = xyz

        p_raw.append(data[l, :, :3])
        q_raw.append([R.from_quat(q) for q in data_ori.tolist()])
        t_raw.append(np.linspace(0, T, M, endpoint=False))


    return p_raw, q_raw, t_raw
