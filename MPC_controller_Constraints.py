import numpy as np
from scipy.linalg import block_diag

def M_F_generator(x_low,x_high,u_low,u_high,N_P,Phi,Gamma):
    n = np.size(x_low, 0)
    p = np.size(u_low,0)
    M = np.concatenate((np.zeros([n, p]), np.zeros([n, p]), -np.eye(n),np.eye(n)), axis=0)
    F = np.concatenate((-np.eye(n), np.eye(n), np.zeros([n, p]), np.zeros([n, p])), axis=0)
    Beta = np.concatenate((-u_low,u_high,-x_low, x_high),axis=0)
    M_Np = np.concatenate((-np.eye(n), np.eye(n)),axis=0)
    Beta_N = np.concatenate((-x_low,x_high), axis=0)

    M_bar = np.zeros([(2*n+2*p)*N_P+2*n,n])
    M_bar[np.arange(0,(2*n+2*p)), :] = M
    Beta_bar = np.concatenate((np.tile(Beta, (N_P, 1)), Beta_N), axis=0)

    M_2bar = M
    F_2bar = F

    for i in range(N_P-2):
        M_2bar = block_diag(M_2bar, M)
        F_2bar = block_diag(F_2bar, F)
    M_2bar = block_diag(M_2bar, M_Np)
    M_2bar = np.concatenate((np.zeros([2*n+2*p, n*N_P]),M_2bar), axis=0)

    F_2bar = block_diag(F_2bar, F)
    F_2bar = np.concatenate((F_2bar, np.zeros(2*n, p*N_P)),axis=0)

    b = -(M_bar + M_2bar@Phi)
    M = M_2bar @ Gamma + F_2bar

    return M, Beta_bar, b
