import numpy as np
from cvxopt import matrix, solvers


def MPC_Controller_noConstraints(x, F, H ,p):
    H = matrix(H)
    Fx = F@x
    Fx = matrix(Fx)
    solution = solvers.qp(H, Fx)
    solution = np.array(solution['x'])
    u = solution[0:p, 0]
    u = u.reshape(-1, 1)
    return u

def MPC_Controller_Constraints(x,F,H,M,Beta_bar,b,p):
    H = matrix(H)
    Fx = F@x
    Fx = matrix(Fx)
    solution = solvers.qp(H, Fx, M, Beta_bar + b@x)
    solution = np.array(solution['x'])
    u = solution[0:p, 0]
    u = u.reshape(-1, 1)
    return u