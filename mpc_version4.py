import numpy as np
import MPC_Matrics_pm as pm
import MPC_controller as contro
import matplotlib as mpl
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
import math
import casadi as ca
import time
########系统建模########


"""
x_1 = -k/m
x_2 = -d/m
x_2 = 1/m
"""
# 建立一个2维的系统模型用于验证，即状态X只包括e e' f，每个均为标量 多步预测 序列化求解
k_1 = 0
k_2 = 0
m_1 = 1
m_2 = 1
d_1 = 0
d_2 = 0
k_e = 1500

# 约束 两个维度设计的相同的约束条件
k_min = 60
k_max = 1500
d_min = 20
d_max = 100
c = m_1

#旋转的角速度和旋转半径
omega = 2
raduis = 10

#系统矩阵，维度1和维度2是同样的系统矩阵
A_temp = np.array([[0,1,0],
             [0,0,0],
             [0,0,0]])

T = 0.01  # 控制周期
N = 5  # 需要预测的步长【超参数】

A_1 = np.copy(A_temp)  #不带K的矩阵
A_temp[2, 1] = k_e
A_2 = A_temp   #带K的系统矩阵 3*3
print(A_2)
#离散
n = np.size(A_2, 0)  #维度 3
I = np.eye(n)
A_K = np.multiply(A_2, T) + I  #离散化以后的系统矩阵, with K 3*3
A_ = np.multiply(A_1, T) + I  #离散化以后的系统矩阵, without K 3*3
# 输入矩阵，根据状态更新矩阵
_B = np.zeros((3, 1))
_B[1,:] = 1  # 3*1




## 系统状态，分为维度1和维度2
e_1 = ca.SX.sym('e_1')  # 位置差
e_1_ = ca.SX.sym('e_1_')  # 速度差
f_x = ca.SX.sym('fx')  # 力
states1 = ca.vertcat(e_1, e_1_)  # 构建位置速度差
states1 = ca.vertcat(states1, f_x)  # 包括力增广
# states = ca.vertcat(*[x, y, theta])
# 或者 ca.vcat([x, y, theta)一步实现
n_states = states1.size()[0]  # 获得系统状态的尺寸，向量以（n_states, 1）的格式呈现

e_2 = ca.SX.sym('e_2')  # 位置差
e_2_ = ca.SX.sym('e_2_')  # 速度差
f_y = ca.SX.sym('fy')  # 力
states2 = ca.vertcat(e_2, e_2_)  # 构建位置速度差
states2 = ca.vertcat(states2, f_y)  # 包括力增广

#控制系统输入
u1_1 = ca.SX.sym('u1_1')  # 输入1
u1_2 = ca.SX.sym('u1_2')  # 输入2
u1_3 = ca.SX.sym('u1_3')  # 输入3
controls1 = ca.vertcat(u1_1, u1_2)  # 控制向量
controls1 = ca.vertcat(controls1, u1_3)  # 控制向量
n_controls = controls1.size()[0]  # 控制向量尺寸

u2_1 = ca.SX.sym('u2_1')  # 输入1
u2_2 = ca.SX.sym('u2_2')  # 输入2
u2_3 = ca.SX.sym('u2_3')  # 输入3
controls2 = ca.vertcat(u2_1, u2_2)  # 控制向量
controls2 = ca.vertcat(controls2, u2_3)  # 控制向量

# K_1 = np.array([[-(1 / u_1[2, 0]) * u_1[0, 0]], [-(1 / u_1[2, 0]) * u_1[1, 0]], [1 / u_1[2, 0]]])  # k d m
# x_c__ = (1/m_1)*(f_x - d_1*e_1_ - k_1*e_1) + x_d__
# 运动学模型
rhs1 = ca.vertcat( e_1_, (1/u1_3)*(f_x - (-u1_2/u1_3) * e_1_ - (-u1_1/u1_3) * e_1))
rhs1 = ca.vertcat(rhs1, 0)
rhs2 = ca.vertcat( e_2_, (1/u2_3)*(f_y - (-u2_2/u2_3) * e_2_ - (-u2_1/u2_3) * e_2))
rhs2 = ca.vertcat(rhs2, 0)
# 利用CasADi构建一个函数
f1 = ca.Function('f', [states1, controls1], [rhs1], ['input_state1', 'control_input1'], ['rhs1'])
f2 = ca.Function('f', [states2, controls2], [rhs2], ['input_state2', 'control_input2'], ['rhs2'])

# 开始构建MPC
## 相关变量，格式(状态长度， 步长)
U1 = ca.SX.sym('U1', n_controls, N)  # N步内的控制输出
X1 = ca.SX.sym('X1', n_states, N + 1)  # N+1步的系统状态，通常长度比控制多1
P1 = ca.SX.sym('P1', n_states + n_states)  # 构建问题的相关参数
U2 = ca.SX.sym('U2', n_controls, N)  # N步内的控制输出
X2 = ca.SX.sym('X2', n_states, N + 1)  # N+1步的系统状态，通常长度比控制多1
P2 = ca.SX.sym('P2', n_states + n_states)  # 构建问题的相关参数
# 在这里每次只需要给定当前/初始位置和目标终点位置

## NLP问题
### 惩罚矩阵
Q = np.array([[20,0,0],
                [0,0.1,0],
                [0,0,0.1]])  # 过程损失
R = np.array([[10**(-6), 0, 0],
              [0, 10**(-6), 0],
              [0, 0, 10**(-6)]])  #输入损失
### 优化目标
obj1 = 0  # 初始化优化目标值
obj2 = 0  # 初始化优化目标值
for i in range(N):
    # 在N步内对获得优化目标表达式
    obj1 = obj1 + ca.mtimes([(X1[:, i]).T, Q, X1[:, i]]) + ca.mtimes([U1[:, i].T, R, U1[:, i]])
    obj2 = obj2 + ca.mtimes([(X2[:, i]).T, Q, X2[:, i]]) + ca.mtimes([U2[:, i].T, R, U2[:, i]])

#### 在预测周期内的状态和输入的变化情况
# 初始状态
# 机器人初始位置以及力信息
x_c = math.cos(0) * raduis
y_c = math.sin(0) * raduis  #位置
x_c_ = 0
y_c_ = 0   #速度
x_c__ = 0
y_c__ = 0  #加速度
f_x = 0
f_y = 0
fd_x = 0
fd_y = 0
x_d = math.cos(omega * 0) * raduis
y_d = math.sin(omega * 0) * raduis  # 期望轨迹
x_d_ = -raduis * omega * math.sin(omega * 0)
y_d_ = raduis * omega * math.cos(omega * 0)  # 期望轨迹一阶导
x_d__ = -raduis * (omega ** 2) * math.cos(omega * 0)
y_d__ = -raduis * (omega ** 2) * math.sin(omega * 0)  # 期望轨迹二阶导
e_1 = x_c - x_d
e_2 = y_c - y_d
e_1_ = x_c_ - x_d_
e_2_ = y_c_ - y_d_
e_1__ = x_c__ - x_d__
e_2__ = y_c__ - y_d__  # 计算偏差

#初始状态
X1[0, 0] = e_1_
X1[1, 0] = e_1
X1[2, 0] = 0
X2[0, 0] = e_2_
X2[1, 0] = e_2
X2[2, 0] = 0
for i in range(N):
    f_value1 = f1(X1[:, i], U1[:, i])
    f_value2 = f2(X2[:, i], U2[:, i])
    x_c__ = f_value1 + x_d__
    x_c_ = x_c_ + x_c__ * T
    x_c = x_c_ + x_c_ * T
    y_c__ = f_value2 + y_d__
    y_c_ = y_c_ + y_c__ * T
    y_c = y_c_ + y_c_ * T

    # 更新参数，此时到达i+1时刻
    x_d = math.cos(omega * i + 1) * raduis
    y_d = math.sin(omega * i + 1) * raduis  #期望轨迹
    x_d_ = -raduis * omega * math.sin(omega * i + 1)
    y_d_ = raduis * omega * math.cos(omega * i + 1)  #期望轨迹一阶导
    x_d__ = -raduis * (omega ** 2) * math.cos(omega * i + 1)
    y_d__ = -raduis * (omega ** 2) * math.sin(omega * i + 1)   #期望轨迹二阶导

    e_1 = x_c - x_d
    e_2 = y_c - y_d
    e_1_ = x_c_ - x_d_
    e_2_ = y_c_ - y_d_
    e_1__ = x_c__ - x_d__
    e_2__ = y_c__ - y_d__  #计算偏差
    # 更新机器人状态
    X1[0, i + 1] = e_1_
    X1[1, i + 1] = e_1
    X1[2, i + 1] = 0
    X2[0, i + 1] = e_2_
    X2[1, i + 1] = e_2
    X2[2, i + 1] = 0

ff = ca.Function('ff', [U1, U2], [X1, X2], ['input_u1','input_u2'], ['states1', 'states2'])

g1 = []  # 用list来存储优化目标的向量
g2 = []  # 用list来存储优化目标的向量
for i in range(N + 1):
    # 这里的约束条件只有小车的坐标（x,y）必须在-2至2之间
    # 由于xy没有特异性，所以在这个例子中顺序不重要（但是在更多实例中，这个很重要）
    g1.append(X1[0, i] + -raduis * omega * math.sin(omega * i + 1))  #x_c'
    g1.append(X1[1, i] + math.cos(omega * i + 1) * raduis)    #x_c
    g2.append(X2[0, i] + raduis * omega * math.cos(omega * i + 1))   #y_c'
    g2.append(X2[1, i] + math.sin(omega * i + 1) * raduis)   #y_c


opt_variables1 = ca.vertcat(ca.reshape(U1, -1, 1), ca.reshape(X1, -1, 1))
opt_variables2 = ca.vertcat(ca.reshape(U2, -1, 1), ca.reshape(X2, -1, 1))
nlp_prob1 = {'f': obj1, 'x': opt_variables1, 'p': P1, 'g': ca.vertcat(*g1)}
nlp_prob2 = {'f': obj2, 'x': opt_variables2, 'p': P2, 'g': ca.vertcat(*g2)}
opts_setting = {'ipopt.max_iter': 100, 'ipopt.print_level': 5   , 'print_time': 0,
                'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6}

solver1 = ca.nlpsol('solver', 'ipopt', nlp_prob1, opts_setting)
solver2 = ca.nlpsol('solver', 'ipopt', nlp_prob2, opts_setting)

# 开始仿真
## 定义约束条件，实际上CasADi需要在每次求解前更改约束条件。不过我们这里这些条件都是一成不变的
## 因此我们定义在while循环外，以提高效率
### 状态约束

### 控制约束
lbg = []
ubg = []
lbx = []  # 最低约束条件
ubx = []  # 最高约束条件
for _ in range(N):
    x_d = math.cos(omega * _) * raduis
    y_d = math.sin(omega * _) * raduis  #期望轨迹

    #### 记住这个顺序，不可搞混！
    #### U是以(n_controls, N)存储的，但是在定义问题的时候被改变成(n_controlsxN,1)的向量
    #### 实际上，第一组控制v0和omega0的index为U_0为U_1，第二组为U_2和U_3
    #### 因此，在这里约束必须在一个循环里连续定义。
    lbx.append(-1500) #-k/m
    lbx.append(-100)   #-d/m
    lbx.append(1)    #1/m
    lbg.append(-np.inf)  #x_c'
    lbg.append(-12 - x_d)  #x_c
    lbg.append(-np.inf)  #f_x

    ubg.append(np.inf)
    ubg.append(9 - y_d)
    ubg.append(-np.inf)
    ubx.append(-100)
    ubx.append(-20)
    ubx.append(1)
for _ in range(N+1):
    x_d = math.cos(omega * _) * raduis
    y_d = math.sin(omega * _) * raduis  #期望轨迹

    lbg.append(-np.inf)  #x_c'
    lbg.append(-12 - x_d)  #x_c
    lbg.append(-np.inf)  #f_x
    ubg.append(np.inf)
    ubg.append(9 - y_d)
    ubg.append(-np.inf)
## 仿真条件和相关变量
t0 = 0.0  # 仿真时间
x0 = np.array([0.0, 0.0, 0.0]).reshape(-1, 1)  # 机器人初始状态
u0 = np.array([-1000, -60, 0] * N).reshape(-1, 3)  # 系统初始控制状态，为了统一本例中所有numpy有关
# 变量都会定义成（N,状态）的形式方便索引和print
x_c = []  # 存储系统的状态
u_c = []  # 存储控制全部计算后的控制指令
t_c = []  # 保存时间
xx = []  # 存储每一步机器人位置
sim_time = 20.0  # 仿真时长
index_t = []  # 存储时间戳，以便计算每一步求解的时间
## 开始仿真
mpciter = 0  # 迭代计数器
start_time = time.time()  # 获取开始仿真时间
### 终止条件为小车和目标的欧式距离小于0.01或者仿真超时
while (mpciter - sim_time / T < 0.0):
    ### 初始化优化参数

    ### 初始化优化目标变量
    init_control = ca.reshape(u0, -1, 1)
    ### 计算结果并且
    t_ = time.time()
    res = solver1(x0=init_control, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
    index_t.append(time.time() - t_)
    ### 获得最优控制结果u
    u_sol = ca.reshape(res['x'], n_controls, N)  # 记住将其恢复U的形状定义
    ###
    ff_value = ff(u_sol, c_p)  # 利用之前定义ff函数获得根据优化后的结果
    # 小车之后N+1步后的状态（n_states, N+1）
    ### 存储结果
    x_c.append(ff_value)
    u_c.append(u_sol[:, 0])
    t_c.append(t0)
    ### 根据数学模型和MPC计算的结果移动小车并且准备好下一个循环的初始化目标
    t0, x0, u0 = shift_movement(T, t0, x0, u_sol, f)
    ### 存储小车的位置
    x0 = ca.reshape(x0, -1, 1)
    xx.append(x0.full())
    ### 计数器+1
    mpciter = mpciter + 1


################

# 定义系统运行步数
k_steps = 500


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
noise_mean = 1
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
    step = 0.1
    if ((k > 20) & (k<100)):
        f_x = 100
        f_y = 100
        x_c__ = (1/m_1)*(f_x - d_1*e_1_ - k_1*e_1) + x_d__
        x_c_ = x_c_ + x_c__*T
        x_c = x_c + x_c_*T

        # x_c = x_c + step
        # f_x = (x_c - x_d) * k_1

        y_c__ = (1/m_2)*(f_y - d_2*e_2_ - k_2*e_2) + y_d__
        y_c_ = y_c_ + y_c__*T
        y_c = y_c + y_c_*T
        # y_c = y_c + step
        # f_y = (y_c - y_d) * k_2
    elif (k>50)&(k<70):
        f_x = f_y = 0
        x_c__ = (1/m_1)*(f_x - d_1*e_1_ - k_1*e_1) + x_d__
        x_c_ = x_c_ + x_c__*T
        x_c = x_c + x_c_*T

        # x_c = x_c - step
        # f_x = (x_c - x_d) * k_1

        y_c__ = (1/m_2)*(f_y - d_2*e_2_ - k_2*e_2) + y_d__
        y_c_ = y_c_ + y_c__*T
        y_c = y_c + y_c_*T
        # y_c = y_c - step
        # f_y = (y_c - y_d) * k_2

    else:
        f_x = f_y = 0
        x_c__ = (1/m_1)*(f_x - d_1*e_1_ - k_1*e_1) + x_d__
        x_c_ = x_c_ + x_c__*T
        x_c = x_c + x_c_*T
        # f_x = -k_e * T * e_1_ + f_x

        y_c__ = (1/m_2)*(f_y - d_2*e_2_ - k_2*e_2) + y_d__
        y_c_ = y_c_ + y_c__*T
        y_c = y_c + y_c_*T
        # f_y = -k_e * T * e_2_ + f_y


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
    x_2 = -d/m
    x_3 = 1/m
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


fig = plt.figure(figsize=(10, 10), dpi=600)
plt.figure()  # 创建第一个画布
plt.plot(trajectory_history[0,:], trajectory_history[1,:],  linestyle='-', label="tranjectory",linewidth = 2, c='red')
plt.plot(trajectory_history_d[0,:], trajectory_history_d[1,:],  linestyle='--', label="tranjectory_d",linewidth = 2, c='blue')
plt.legend(loc = 'lower right', markerscale = 0.5, fontsize='medium',shadow=False,framealpha=0.5)
plt.xlim(-15, 15)  # x轴范围从1到4
plt.ylim(-15, 15)  # y轴范围从1到5
plt.savefig("trajectory.png", dpi=600)

plt.figure(figsize=(8, 8))  # 创建第二个画布
plt.step(np.arange(0, u_history1.shape[1]), u_history1[0,:],  linestyle='-', label="u1_k",linewidth = 2, c='red')
plt.step(np.arange(0, u_history1.shape[1]), u_history1[1,:],  linestyle='-', label="u1_d",linewidth = 2, c='blue')
plt.step(np.arange(0, u_history1.shape[1]), u_history1[2,:],  linestyle='-', label="u1_m",linewidth = 2, c='green')
plt.legend(loc = 'lower right', markerscale = 0.5, fontsize='medium',shadow=False,framealpha=0.5)

plt.figure(figsize=(8, 8))  # 创建第三个画布
plt.step(np.arange(0, u_history2.shape[1]), u_history2[0,:],  linestyle='-', label="u2_k",linewidth = 2, c='red')
plt.step(np.arange(0, u_history2.shape[1]), u_history2[1,:],  linestyle='-', label="u2_d",linewidth = 2, c='blue')
plt.step(np.arange(0, u_history2.shape[1]), u_history2[2,:],  linestyle='-', label="u2_m",linewidth = 2, c='green')
plt.legend(loc = 'lower right', markerscale = 0.5, fontsize='medium',shadow=False,framealpha=0.5)

plt.figure(figsize=(8, 8))  # 创建第四个画布
plt.plot(np.arange(0, force_history.shape[1]), force_history[0,:],  linestyle='-', label="f_x",linewidth = 2, c='red')
plt.plot(np.arange(0, force_history.shape[1]), force_history[1,:],  linestyle='-', label="f_y",linewidth = 2, c='blue')
plt.legend(loc = 'lower right', markerscale = 0.5, fontsize='medium',shadow=False,framealpha=0.5)

plt.show()