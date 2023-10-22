import numpy as np
import MPC_Matrics_pm as pm
import MPC_controller as contro
import matplotlib as mpl
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
import math
########系统建模########


# 建立一个2维的系统模型用于验证，即状态X只包括e e' f，每个均为标量 单步预测
K = np.zeros((2,2))
M = np.eye(2)
M_inv = np.linalg.inv(M)  #矩阵的逆
D = np.zeros((2,2))
K_e = 1500

# 约束
K_min = np.array([[100, 0], [0, 100]])
K_max = np.array([[1500, 0], [0, 1500]])
D_min = np.array([[20, 0], [0, 20]])
D_max = np.array([[50, 0], [0, 50]])

c = M
G = np.zeros((8, 6))
G[0:2, 0:2] = -c
G[2:4, 0:2] = c
G[4:6, 2:4] = -c
G[6:8, 2:4] = c

h = np.zeros((8, 2))
h[0:2, :] = K_max
h[2:4, :] = -K_min
h[4:6, :] = D_max
h[6:8, :] = -D_min

At = np.zeros((6,2))
At[4:6,:] = M
b = np.eye(2)

omega = 2
raduis = 10

########权重设计########

#状态权重
# Q = np.array([[50,0,0],
#               [0,0.1,0],
#               [0,0,0.1]])  # 过程损失

Q = np.eye(6)
Qvalues = [50,50,0.1,0.1,0.1,0.1]
np.fill_diagonal(Q, Qvalues)

# S = np.array([[1,0,0],
#               [0,1,0],
#               [0,0,1]])  # 终端损失，这一版方案中不要这个
# 输入权重
# R = np.array([[10**(-6), 0, 0],
#               [0, 10**(-6), 0],
#               [0, 0, 10**(-6)]])  #输入损失

R = np.eye(6)
Rvalues = [10**(-6), 10**(-6), 10**(-6), 10**(-6), 10**(-6), 10**(-6)]
np.fill_diagonal(R, Rvalues)

# 状态初始值
"""
x_1 = -k/m
x_2 = -d/m
x_2 = 1/m
"""

#系统矩阵
# A_1 = np.array([[0,1,0],
#              [0,0,0],
#              [0,0,0]])

T = 0.01  # 控制周期
A_temp = np.zeros((6,6))  #定义一个空矩阵A
I_2x2 = np.eye(2)  #系统是二维系统
A_temp[0:2, 2:4] = I_2x2
A_1 = np.copy(A_temp)  #不带K的系统矩阵 6*6

K_temp = I_2x2 * K_e
A_temp[4:6, 2:4] = K_temp
A_2 = A_temp   #带K的系统矩阵 6*6

n = np.size(A_2, 0)  #维度 6
I = np.eye(n)
A_2 = np.multiply(A_2, T) + I  #离散化以后的系统矩阵 6*6
A_1 = np.multiply(A_1, T) + I  #离散化以后的系统矩阵 6*6
# 输入矩阵，根据状态更新矩阵
B_1 = np.zeros((6, 2))
B_1[2:4,:] = I_2x2  # 6*2

#更改输入矩阵B为恒定值
# B = np.array([[0], [1], [0]])
# p = np.size(B, 1)

# 定义系统运行步数
k_steps = 100


######记录矩阵#####
x_history = np.zeros([n, k_steps])  #记录状态变量
trajectory_history = np.zeros([2, k_steps])  #记录轨迹历史
force_history = np.zeros([1, k_steps])   #记录力轨迹

# 定义u_history零矩阵，用于储存系统输入结果，维度1 x k_steps
u_history = np.zeros([n, k_steps-1])   #记录输入矩阵历史
#################

# 定义预测区间，预测区间要小于系统运行步数
N_P = 1  # 单步预测，保持系统的凸的

#不含二次规划需用到的矩阵
noise_mean = 0
noise_stddev = 1



# 机器人初始位置以及力信息
x_c = math.cos(0) * raduis
y_c = math.sin(0) * raduis  #位置
x_c_ = 0
y_c_ = 0   #速度
x_c__ = 0
y_c__ = 0  #加速度

X_c = np.array([[x_c, 0], [0, y_c]])
X_c_ = np.array([[x_c_, 0], [0, y_c_]])
X_c__ = np.array([[x_c__, 0], [0, y_c__]])  #状态矩阵
F_ext = np.array([[0, 0],
                  [0, 0]])  #传感器外力值

for k in range(k_steps):

    # 在k时刻机器人的期望轨迹
    t_k = T * k
    x_d = math.cos(omega * t_k) * raduis
    y_d = math.sin(omega * t_k) * raduis  #期望轨迹
    x_d_ = -raduis * omega * math.sin(omega * t_k)
    y_d_ = raduis * omega * math.cos(omega * t_k)  #期望轨迹一阶导
    x_d__ = -raduis * (omega ** 2) * math.cos(omega * t_k)
    y_d__ = -raduis * (omega ** 2) * math.sin(omega * t_k)   #期望轨迹二阶导

    X_d = np.array([[x_d, 0], [0, y_d]])
    X_d_ = np.array([[x_d_, 0], [0, y_d_]])
    X_d__ = np.array([[x_d__, 0], [0, y_d__]]) # 2*2
    #############################

    E = X_c - X_d  # 2*2 对角矩阵
    E_ = X_c_ - X_d_
    E__ = X_c__ - X_d__
    X = np.vstack((E, E_, F_ext))  # 状态空间描述下的机器人状态 6*2

    B_2 = X.T #2*6
    B = B_1 @ B_2  # B随状态变量x会变化 6*6
    p = np.size(B, 1)
    B = np.multiply(B, T)  # 最终的输入矩阵B

    # 维度1和维度2分别求解


    #维度2
    Q_bar, p_, c_ = pm.MPC_matrics_single_prediction(A_2, B, Q, R, X)

    u = contro.MPC_single_qpsolver(Q_bar, p_, c_, p, G, h, At, b)
    print(u)
    x = A_1 @ X + B @ u




    # 系统转移方程，添加噪声模拟不确定性
    noise = np.random.normal(noise_mean, noise_stddev, (3, 1))
    x = A_1 @ x + B @ u + noise
    # 更新计算二次规划需用到的矩阵



    x_history[:, k] = x[:, 0]
    trajectory_history[:,k] = np.array([[x_c], [y_c]])
    # force_history[:,k] = np.array(f_ext)
    print("第{}步的状态变量为 {:.2f},{:.2f},{:.2f}".format(k + 1, x[0,0], x[1,0], x[2,0]),
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


fig = plt.figure(figsize=(10, 12))
ax = fig.add_subplot(2,1,1)
ax.plot(np.arange(0, x_history.shape[1]), x_history[0,:],  linestyle='-', label="x1",linewidth = 3.5, c='red')
ax.plot(np.arange(0, x_history.shape[1]), x_history[1,:],  linestyle='-', label="x2",linewidth = 3.5, c='blue')
ax.plot(np.arange(0, x_history.shape[1]), x_history[2,:],  linestyle='-', label="x3",linewidth = 3.5, c='green')
plt.legend(loc = 'lower right', markerscale = 0.5, fontsize='medium',shadow=False,framealpha=0.5)

ax1 = fig.add_subplot(2,1,2)
ax1.step(np.arange(0, u_history.shape[1]), u_history[0,:],  linestyle='-', label="u1",linewidth = 3.5, c='red')
ax1.step(np.arange(0, u_history.shape[1]), u_history[1,:],  linestyle='-', label="u2",linewidth = 3.5, c='blue')
ax1.step(np.arange(0, u_history.shape[1]), u_history[2,:],  linestyle='-', label="u3",linewidth = 3.5, c='green')
plt.legend(loc = 'lower right', markerscale = 0.5, fontsize='medium',shadow=False,framealpha=0.5)
plt.show()