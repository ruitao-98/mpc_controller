import numpy as np
from scipy.linalg import block_diag

def MPC_matrics(A, B, Q, R, S, N_P):
    #计算系统矩阵维度
    n = np.size(A, 0)
    p = np.size(B, 1)


    #初始化phi矩阵
    Phi = np.zeros([N_P*n, n])
    Gamma = np.zeros([N_P*n, N_P*p])

    #临时对角矩阵
    tmp = np.eye(n)

    #构建phi和gamma矩阵
    rows = np.arange(0, n)

    #当i = 0
    i = 0
    Phi[i * n:(i + 1) * n, :] = np.power(A, i + 1)
    Gamma[rows, :] = np.concatenate((tmp @ B, Gamma[np.arange(0,3), 0:-p]), axis=1)
    rows = n + rows
    tmp = A
    print(rows)
    print(tmp @ B)
    for i in range(1, N_P):
        Phi[i*n:(i+1)*n, :] = np.power(A, i+1)
        Gamma[rows,:]=  np.concatenate((tmp @ B, Gamma[rows - n, 0:-p]), axis=1)
        rows = n + rows
        tmp = A@tmp


    # 构建omega矩阵
    Omega = np.kron(np.eye(N_P-1), Q)
    Omega = block_diag(Omega, S)
    Psi = np.kron(np.eye(N_P), R)

    #计算二次规划矩阵FH
    F = Gamma.T @ Omega @ Phi
    H = Psi + Gamma.T @ Omega @Gamma

    return Phi, Gamma, Omega, Psi, F, H

def MPC_matrics_single_prediction(A, B, Q, R, X):


    B_temp = B.T @ Q
    B_t = B_temp @ B

    Q_bar = 2 * B_t + 2 * R
    p_ = 2 * (B.T @ Q @ A @ X)
    c_ = X.T @ A.T @ Q @ A @ X

    # print(Q_bar)
    # print(p_)
    # print(c_)

    return Q_bar, p_, c_
