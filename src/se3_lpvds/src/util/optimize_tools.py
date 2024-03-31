import numpy as np
import cvxpy as cp

from .quat_tools import *
from .plot_tools import *



def optimize_quat_system(p_in, q_in, p_out, q_out, p_att, q_att, postProb):
    """
    :param q_in:  list of Rotation objects representing orientation, should be length N
    :param q_out:  list of Rotation objects representing angular velocity, should be length N-1
    :param q_att:    single Rotation object represent the target attractor
    :param postProb: posterior probability of each observation, shape (K, N), where K is number of components and N is the number of observations
    """

    p_diff = p_in - np.tile(p_att.reshape(1, -1), (p_in.shape[0], 1))
    # pq_in    = np.hstack((p_diff, riem_log(q_att, q_in)))
    # pq_in    = riem_log(q_att, q_in)
    pq_in    = p_diff

    q_out_body = riem_log(q_in, q_out)            
    q_out_att  = parallel_transport(q_in, q_att, q_out_body)

    # pq_out = np.hstack((p_out, q_out_att))
    # pq_out =  q_out_att
    pq_out =  p_out

    K, _ = postProb.shape
    N = 3


    max_norm = 0.5
    A_vars = []
    constraints = []
    for k in range(K):
        A_vars.append(cp.Variable((N, N), symmetric=False))

        constraints += [A_vars[k].T + A_vars[k] << np.zeros((N, N))]

        # constraints += [cp.norm(A_vars[k], 'fro') <= max_norm]


    for k in range(K):
        pq_pred_k = A_vars[k] @ pq_in.T
        if k == 0:
            pq_pred  = cp.multiply(np.tile(postProb[k, :], (N, 1)), pq_pred_k)
        else:
            pq_pred += cp.multiply(np.tile(postProb[k, :], (N, 1)), pq_pred_k)
    pq_pred = pq_pred.T

    
    objective = cp.norm(pq_out-pq_pred, 'fro')


    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.MOSEK, verbose=True)


    A_res = np.zeros((K, N, N))
    for k in range(K):
        A_res[k, :, :] = A_vars[k].value
        print(A_vars[k].value)
        print(np.linalg.norm(A_vars[k].value, 'fro'))

    return A_res