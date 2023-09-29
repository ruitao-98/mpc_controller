import numpy as np
import MPC_Matrics_pm as pm
import MPC_controller as contro
import matplotlib as mpl
import matplotlib.pyplot as plt
########系统建模########



########权重设计########
Q = np.array([[1,0,0],
              [0,1,0],
              [0,0,1]])
S = np.array([[1,0,0],
              [0,1,0],
              [0,0,1]])
# 输入权重
R = np.array([[1,0,0],
              [0,1,0],
              [0,0,1]])

# 状态初始值
x_0 = np.array([[1],[1],[2]])
x = x_0

#系统矩阵
T = 0.01  # 控制周期
A = np.array([[0,1,0],
             [0,0,0],
             [0,0,0]])

n = np.size(A, 0)
I = np.eye(n)
A = np.multiply(A, T) - I

#输入矩阵
B_1 = np.array([[0], [1], [0]])
B_2 = x.T

B = B_1 @ B_2
p = np.size(B, 1)
B = np.multiply(B, T)


# 定义系统运行步数
k_steps = 20
# 定义x_history零矩阵，用于储存系统状态结果，维度n x k_steps
x_history = np.zeros([n, k_steps])
# 初始状态存入状态向量第一个位置
x_history [:,0] = x[:, 0]


# 定义u_history零矩阵，用于储存系统输入结果，维度p x k_steps
u_history = np.zeros([p,k_steps])

# 定义预测区间，预测区间要小于系统运行步数
N_P = 4
#计算二次规划需用到的矩阵
Phi, Gamma, Omega, Psi, F, H = pm.MPC_matrics(A, B, Q, R, S, N_P)
#不含二次规划需用到的矩阵


for k in range(k_steps-1):
    u = contro.MPC_Controller_noConstraints(x, F, H, p)

    #更新B
    B_2 = x.T
    B = B_1 @ B_2
    B = np.multiply(B, T)

    x = A @ x + B @ u

    # 更新矩阵
    Phi, Gamma, Omega, Psi, F, H = pm.MPC_matrics(A, B, Q, R, S, N_P)

    x_history[:, k + 1] = x[:, 0]
    u_history[:, k] = u[:,0]
    print("第{}步的状态变量为 {:.2f},{:.2f},{:.2f}".format(k+1, x[0,0], x[1,0], x[2,0]),
          "输入为 {:.2f} {:.2f} {:.2f}".format(u[0,0],u[1,0],u[2,0]))




###############画图#################
mpl.rcParams.update({'font.size': 10})
font = {'family': 'serif',
        'serif': 'Times New Roman',
        'weight': 'normal',
        'size': 15}
config = {
    "font.family":'serif',
    "font.size": 25,
    "mathtext.fontset":'stix',
}
plt.rcParams.update(config)
plt.rc('font',  **font)


fig = plt.figure(figsize=(10, 20))
ax = fig.add_subplot(2,1,1)
ax.plot(np.arange(0, x_history.shape[1]), x_history[0,:],  linestyle='-', label="x1",linewidth = 3.5, c='red')
ax.plot(np.arange(0, x_history.shape[1]), x_history[1,:],  linestyle='-', label="x2",linewidth = 3.5, c='blue')
ax.plot(np.arange(0, x_history.shape[1]), x_history[2,:],  linestyle='-', label="x3",linewidth = 3.5, c='green')
plt.legend(loc = 'lower right', markerscale = 0.5, fontsize='medium',shadow=False,framealpha=0.5)

ax1 = fig.add_subplot(2,1,2)
ax1.step(np.arange(0, u_history.shape[1]), u_history[0,:],  linestyle='-', label="u1",linewidth = 3.5, c='red')
ax1.step(np.arange(0, u_history.shape[1]), u_history[1,:],  linestyle='-', label="u2",linewidth = 3.5, c='blue')
ax.plot(np.arange(0, u_history.shape[1]), u_history[2,:],  linestyle='-', label="u3",linewidth = 3.5, c='green')
plt.legend(loc = 'lower right', markerscale = 0.5, fontsize='medium',shadow=False,framealpha=0.5)
plt.show()