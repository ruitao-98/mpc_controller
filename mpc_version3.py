import numpy as np
import MPC_Matrics_pm as pm
import MPC_controller as contro
import matplotlib as mpl
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
import math
########系统建模########


"""
x_1 = -k/m
x_2 = -d/m
x_2 = 1/m
"""

# 建立一个2维的系统模型用于验证，即状态X只包括e e' f，每个均为标量 单步预测
k_1 = 0
k_2 = 0
m_1 = 1
m_2 = 1
d_1 = 0
d_2 = 0
k_e = 1500

# 约束 两个维度设计的相同的约束条件
k_min = 100
k_max = 1500
d_min = 20
d_max = 50
c = m_1

G_1 = np.zeros((4, 3))
G_1[0, 0] = -c
G_1[1, 0] = c
G_1[2, 1] = -c
G_1[3, 1] = c
G_2 = np.zeros((4,3))
G_2[0, 0] = -c
G_2[1, 0] = c
G_2[2, 1] = -c
G_2[3, 1] = c

h_1 = np.zeros((4, 1))
h_1[0, :] = k_max
h_1[1, :] = -k_min
h_1[2, :] = d_max
h_1[3, :] = -d_min
h_2 = np.copy(h_1)

At_1 = np.zeros((1,3))
At_1[0,2] = m_1
At_2 = np.zeros((1,3))
At_2[0,2] = m_2
b_1 = 1.0
b_2 = 1.0
######约束



omega = 2
raduis = 10

########权重设计########

#状态权重
Q_1 = np.array([[50,0,0],
              [0,0.1,0],
              [0,0,0.1]])  # 过程损失
Q_2 = np.array([[50,0,0],
              [0,0.1,0],
              [0,0,0.1]])  # 过程损失


# S = np.array([[1,0,0],
#               [0,1,0],
#               [0,0,1]])  # 终端损失，这一版方案中不要这个
# 输入权重
R_1 = np.array([[10**(-6), 0, 0],
              [0, 10**(-6), 0],
              [0, 0, 10**(-6)]])  #输入损失
R_2 = np.array([[10**(-6), 0, 0],
              [0, 10**(-6), 0],
              [0, 0, 10**(-6)]])  #输入损失




#系统矩阵，维度1和维度2是同样的系统矩阵
A_temp = np.array([[0,1,0],
             [0,0,0],
             [0,0,0]])

T = 0.01  # 控制周期

A_1 = np.copy(A_temp)  #不带K的矩阵
A_temp[2, 1] = k_e
A_2 = A_temp   #带K的系统矩阵 3*3
#离散
n = np.size(A_2, 0)  #维度 3
I = np.eye(n)
A_K = np.multiply(A_2, T) + I  #离散化以后的系统矩阵, with K 3*3
A_ = np.multiply(A_1, T) + I  #离散化以后的系统矩阵, without K 3*3
# 输入矩阵，根据状态更新矩阵
_B = np.zeros((3, 1))
_B[2,:] = 1  # 3*1

################

# 定义系统运行步数
k_steps = 200


######记录矩阵#####
x_history1 = np.zeros([n, k_steps])  #记录状态变量，维度1
x_history2 = np.zeros([n, k_steps])  #记录状态变量，维度2
trajectory_history = np.zeros([2, k_steps])  #记录轨迹历史
trajectory_history_d = np.zeros([2, k_steps])
print(trajectory_history)
force_history = np.zeros([2, k_steps])   #记录力轨迹

# 定义u_history零矩阵，用于储存系统输入结果，维度1 x k_steps
u_history1 = np.zeros([n, k_steps])   #记录输入矩阵历史，维度1
u_history2 = np.zeros([n, k_steps])   #记录输入矩阵历史，维度2
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

# X_c = np.array([[x_c, 0], [0, y_c]])
# X_c_ = np.array([[x_c_, 0], [0, y_c_]])
# X_c__ = np.array([[x_c__, 0], [0, y_c__]])  #状态矩阵
f_x = 0
f_y = 0

for k in range(k_steps):

    # 在k时刻机器人的期望轨迹
    t_k = T * k
    x_d = math.cos(omega * t_k) * raduis
    y_d = math.sin(omega * t_k) * raduis  #期望轨迹
    x_d_ = -raduis * omega * math.sin(omega * t_k)
    y_d_ = raduis * omega * math.cos(omega * t_k)  #期望轨迹一阶导
    x_d__ = -raduis * (omega ** 2) * math.cos(omega * t_k)
    y_d__ = -raduis * (omega ** 2) * math.sin(omega * t_k)   #期望轨迹二阶导

    # X_d = np.array([[x_d, 0], [0, y_d]])
    # X_d_ = np.array([[x_d_, 0], [0, y_d_]])
    # X_d__ = np.array([[x_d__, 0], [0, y_d__]]) # 2*2
    #############################

    e_1 = x_c - x_d
    e_2 = y_c - y_d
    e_1_ = x_c_ - x_d_
    e_2_ = y_c_ - y_d_
    e_1__ = x_c__ - x_d__
    e_2__ = y_c__ - y_d__  #计算偏差

    X_1 = np.array([[e_1], [e_1_], [f_x]])
    X_2 = np.array([[e_2], [e_2_], [f_y]])

    __B_1 = X_1.T  # 1*3
    __B_2 = X_2.T  # 1*3
    B_1_temp = _B @ __B_1  # B随状态变量x会变化 3*3
    B_2_temp = _B @ __B_2  # B随状态变量x会变化 3*3

    B_1 = np.multiply(B_1_temp, T)  # 最终的输入矩阵B
    B_2 = np.multiply(B_2_temp, T)  # 最终的输入矩阵B
    p = np.size(B_1, 0)


    Q_bar_1, p_1, c_1 = pm.MPC_matrics_single_prediction(A_K, B_1, Q_1, R_1, X_1)
    Q_bar_2, p_2, c_2 = pm.MPC_matrics_single_prediction(A_K, B_2, Q_2, R_2, X_2)

    u_1 = contro.MPC_single_qpsolver(Q_bar_1, p_1, c_1, p, G_1, h_1, At_1, b_1)
    u_2 = contro.MPC_single_qpsolver(Q_bar_2, p_2, c_2, p, G_2, h_2, At_2, b_2)

    K_1 = np.array([[-(1 / u_1[2, 0]) * u_1[0, 0]], [-(1 / u_1[2, 0]) * u_1[1, 0]], [1 / u_1[2, 0]]])  # k d m
    K_2 = np.array([[-(1 / u_2[2, 0]) * u_2[0, 0]], [-(1 / u_2[2, 0]) * u_2[1, 0]], [1 / u_2[2, 0]]])  # k d m
    k_1 = K_1[0,0]
    d_1 = K_1[1,0]
    m_1 = K_1[2,0]
    k_2 = K_2[0,0]
    d_2 = K_2[1,0]
    m_2 = K_2[2,0]

##
    noise_1 = np.random.normal(noise_mean, noise_stddev)
    noise_2 = np.random.normal(noise_mean, noise_stddev)

    # 此时计算得到的刚度矩阵，用于下一个时刻的状态转移使用，记住，状态是x_c - x_d，并不是真实的位姿
    # 计算加速度
    # e_1__ = (1/m_1)(f_x - b_1*e_1_ - k_1*e_1)
    # e_1_ = e_1_ + e_1__*T
    # e_1 = e_1 + e_1_*T
    # x_c = x_c
    #
    # e_2__ = (1/m_2)(f_y - b_2*e_2_ - k_2*e_2)
    # e_2_ = e_2_ + e_2__*T
    # e_2 = e_2 + e_2_*T
    print(m_1)
    print(b_1)
    print(k_1)
    print(f_x)
    print(e_1_)
    f_x = noise_1+f_x
    f_y = noise_2
    x_c__ = (1/m_1)*(f_x - b_1*e_1_ - k_1*e_1) + x_d__
    x_c_ = x_c_ + x_c__*T
    x_c = x_c + x_c_*T

    y_c__ = (1/m_2)*(f_y - b_2*e_2_ - k_2*e_2) + y_d__
    y_c_ = y_c_ + y_c__*T
    y_c = y_c + y_c_*T



    # x = A_ @ X_1 + B_1 @ u_1 + noise_1
    # y = A_ @ X_2 + B_1 @ u_2 + noise_2

    # x_history1[:, k] = x[:, 0]
    # x_history2[:, k] = x[:, 0]
    trajectory_history[0, k] = x_c
    trajectory_history[1, k] = y_c
    force_history[0,k] = f_x
    force_history[1,k] = f_y
    X_d = np.array([[x_d],[y_d]])
    trajectory_history_d[:, k] = X_d[:, 0]

    #刚度矩阵解算
    """
    x_1 = -k/m
    x_2 = -d/mgai 
    x_2 = 1/m
    """



    u_history1[:, k] = K_1[:,0]
    u_history2[:, k] = K_2[:,0]
    # print("第{}步的x维度状态变量为 {:.2f},{:.2f},{:.2f}".format(k, x[0,0], x[1,0], x[2,0]),
    #       "x维度的输入为 {:.2f} {:.2f} {:.2f}".format(u_1[0,0],u_1[1,0],u_2[2,0]))

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


fig = plt.figure(figsize=(10, 10))
plt.figure()  # 创建第一个画布
plt.plot(trajectory_history[0,:], trajectory_history[1,:],  linestyle='-', label="tranjectory",linewidth = 2, c='red')
plt.plot(trajectory_history_d[0,:], trajectory_history_d[1,:],  linestyle='--', label="tranjectory_d",linewidth = 2, c='blue')
plt.legend(loc = 'lower right', markerscale = 0.5, fontsize='medium',shadow=False,framealpha=0.5)
plt.xlim(-15, 15)  # x轴范围从1到4
plt.ylim(-15, 15)  # y轴范围从1到5

plt.figure()  # 创建第二个画布
plt.step(np.arange(0, u_history1.shape[1]), u_history1[0,:],  linestyle='-', label="u1_k",linewidth = 2, c='red')
plt.step(np.arange(0, u_history1.shape[1]), u_history1[1,:],  linestyle='-', label="u1_d",linewidth = 2, c='blue')
plt.step(np.arange(0, u_history1.shape[1]), u_history1[2,:],  linestyle='-', label="u1_m",linewidth = 2, c='green')
plt.legend(loc = 'lower right', markerscale = 0.5, fontsize='medium',shadow=False,framealpha=0.5)

plt.figure()  # 创建第三个画布
plt.step(np.arange(0, u_history2.shape[1]), u_history2[0,:],  linestyle='-', label="u2_k",linewidth = 2, c='red')
plt.step(np.arange(0, u_history2.shape[1]), u_history2[1,:],  linestyle='-', label="u2_d",linewidth = 2, c='blue')
plt.step(np.arange(0, u_history2.shape[1]), u_history2[2,:],  linestyle='-', label="u2_m",linewidth = 2, c='green')
plt.legend(loc = 'lower right', markerscale = 0.5, fontsize='medium',shadow=False,framealpha=0.5)

plt.figure()  # 创建第四个画布
plt.step(np.arange(0, force_history.shape[1]), force_history[0,:],  linestyle='-', label="f_x",linewidth = 2, c='red')
plt.step(np.arange(0, force_history.shape[1]), force_history[1,:],  linestyle='-', label="f_y",linewidth = 2, c='blue')
plt.legend(loc = 'lower right', markerscale = 0.5, fontsize='medium',shadow=False,framealpha=0.5)

plt.show()