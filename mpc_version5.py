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


#系统矩阵，维度1和维度2是同样的系统矩阵
T = 0.01  # 控制周期
N = 5  # 需要预测的步长【超参数】

## 系统状态，分为维度1和维度2
x_c_ = ca.SX.sym('x_c_')  # 速度
x_c = ca.SX.sym('x_c')  # 位置
f_x = ca.SX.sym('fx')  # 力
states1 = ca.vertcat(x_c_, x_c)  # 构建位置速度差
states1 = ca.vertcat(states1, f_x)  # 包括力增广
# states = ca.vertcat(*[x, y, theta])
# 或者 ca.vcat([x, y, theta)一步实现
n_states = states1.size()[0]  # 获得系统状态的尺寸，向量以（n_states, 1）的格式呈现

e_2 = ca.SX.sym('y_c')  # 位置
e_2_ = ca.SX.sym('y_c_')  # 速度
f_y = ca.SX.sym('fy')  # 力
states2 = ca.vertcat(e_2_, e_2)  # 构建位置速度差
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
rhs1 = ca.vertcat( e_1_, (u1_3)*(f_x - (-u1_2/u1_3) * e_1_ - (-u1_1/u1_3) * e_1))
rhs1 = ca.vertcat(rhs1, 0)
rhs2 = ca.vertcat( e_2_, (u2_3)*(f_y - (-u2_2/u2_3) * e_2_ - (-u2_1/u2_3) * e_2))
rhs2 = ca.vertcat(rhs2, 0)
# 利用CasADi构建一个函数
f1 = ca.Function('f', [states1, controls1], [rhs1], ['input_state1', 'control_input1'], ['rhs1'])
f2 = ca.Function('f', [states2, controls2], [rhs2], ['input_state2', 'control_input2'], ['rhs2'])


# 开始构建MPC
## 相关变量，格式(状态长度， 步长)
U1 = ca.SX.sym('U1', n_controls, N)  # N步内的控制输出
X1 = ca.SX.sym('X1', n_states, N + 1)  # N+1步的系统状态，通常长度比控制多1
P1 = ca.SX.sym('P1', n_states + 1)  # 构建问题的相关参数
U2 = ca.SX.sym('U2', n_controls, N)  # N步内的控制输出
X2 = ca.SX.sym('X2', n_states, N + 1)  # N+1步的系统状态，通常长度比控制多1
P2 = ca.SX.sym('P2', n_states + 1)  # 构建问题的相关参数
# 在这里每次只需要给定当前/初始位置和目标终点位置
omega = 2
raduis = 10
#### 在预测周期内的状态和输入的变化情况
# 机器人初始位置以及力信息
I = P1[-1] = P2[-1]  #此时处于的时间刻度
e_1 = P1[1]
e_1_ = P1[0]
f_x = P1[2]

e_2 = P2[1]
e_2_ = P2[0]
f_y = P2[2]
##############################
x_d = ca.cos(omega * I*T) * raduis
y_d = ca.sin(omega * I*T) * raduis  # 期望轨迹
x_d_ = -raduis * omega * ca.sin(omega * I*T)
y_d_ = raduis * omega * ca.cos(omega * I*T)  # 期望轨迹一阶导
x_d__ = -raduis * (omega ** 2) * ca.cos(omega * I*T)
y_d__ = -raduis * (omega ** 2) * ca.sin(omega * I*T)  # 期望轨迹二阶导

#逆向结算机器人的位置和速度
x_c_ = e_1_ + x_d_
x_c = e_1 + x_d
y_c_ = e_2_ + y_d_
y_c = e_2 + y_d

#初始状态
X1[0, 0] = e_1_
X1[1, 0] = e_1
X1[2, 0] = f_x
X2[0, 0] = e_2_
X2[1, 0] = e_2
X2[2, 0] = f_y
for i in range(N):
    f_value1 = f1(X1[:, i], U1[:, i])
    f_value2 = f2(X2[:, i], U2[:, i])
    x_c__ = f_value1[1] + x_d__
    x_c_ = x_c_ + x_c__ * T
    x_c = x_c_ + x_c_ * T
    y_c__ = f_value2[1] + y_d__
    y_c_ = y_c_ + y_c__ * T
    y_c = y_c_ + y_c_ * T

    # 更新参数，此时到达i+I+1时刻
    x_d = ca.cos(omega * (i + I + 1)*T) * raduis
    y_d = ca.sin(omega * (i + I + 1)*T) * raduis  #期望轨迹
    x_d_ = -raduis * omega * ca.sin(omega * (i + I + 1)*T)
    y_d_ = raduis * omega * ca.cos(omega * (i + I + 1)*T)  #期望轨迹一阶导
    x_d__ = -raduis * (omega ** 2) * ca.cos(omega * (i + I + 1)*T)
    y_d__ = -raduis * (omega ** 2) * ca.sin(omega * (i + I + 1)*T)   #期望轨迹二阶导

    e_1 = x_c - x_d
    e_2 = y_c - y_d
    e_1_ = x_c_ - x_d_
    e_2_ = y_c_ - y_d_
    # 更新机器人状态
    X1[0, i + 1] = e_1_
    X1[1, i + 1] = e_1
    X1[2, i + 1] = 0
    X2[0, i + 1] = e_2_
    X2[1, i + 1] = e_2
    X2[2, i + 1] = 0

ff1 = ca.Function('ff1', [U1, P1], [X1], ['input_u1','param1'], ['states1'])
ff2 = ca.Function('ff2', [U2, P2], [X2], ['input_u2','param2'], ['states2'])


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

# g1_func = ca.Function('g', [X1_sym, P1_sym], [g1])
# g2_func = ca.Function('g', [X2_sym, P2_sym], [g2])

# g1 = ca.SX.sym('g1', 2*(N + 1))  # 用list来存储优化目标的向量
# g2 = ca.SX.sym('g1', 2*(N + 1))  # 用list来存储优化目标的向量
# X1_sym = ca.SX.sym('X1_sym', X1.shape)
# X2_sym = ca.SX.sym('X2_sym', X2.shape)
# P1_sym = ca.SX.sym('P1_sym', P1.shape)
# P2_sym = ca.SX.sym('P2_sym', P2.shape)
g1 = []
g2 = []
for i in range(N + 1):
    # g1[2*i] = X1[0, i] + (-raduis * omega * ca.sin(omega * (i + P1[-1])))  #x_c'
    # g1[2*i + 1] = X1[1, i] + ca.cos(omega * (i + P1[-1])) * raduis    #x_c
    # g2[2*i] = X2[0, i] + raduis * omega * ca.cos(omega * (i + P2[-1]))   #y_c'
    # g2[2*i + 1] = X2[1, i] + ca.sin(omega * (i + P2[-1])) * raduis   #y_c
    g1.append(X1[0, i] + (-raduis * omega * ca.sin(omega * (i + P1[-1]))))  #x_c'
    g1.append(X1[1, i] + ca.cos(omega * (i + P1[-1])) * raduis)    #x_c
    g1.append(X1[2, i])  # x_c
    g2.append(X2[0, i] + raduis * omega * ca.cos(omega * (i + P2[-1])))   #y_c'
    g2.append(X2[1, i] + ca.sin(omega * (i + P2[-1])) * raduis)   #y_c
    g2.append(X2[2, i])  # y_c


# X1_sym = ca.SX.sym('X1_sym', X1.shape)
# X2_sym = ca.SX.sym('X2_sym', X2.shape)
# P1_sym = ca.SX.sym('P1_sym', P1.shape)
# P2_sym = ca.SX.sym('P2_sym', P2.shape)
#
# g1_func = ca.Function('g1', [X1_sym, P1_sym], [g1])
# g2_func = ca.Function('g2', [X2_sym, P2_sym], [g2])

# opt_variables1 = ca.vertcat(ca.reshape(U1, -1, 1), ca.reshape(X1, -1, 1))
# opt_variables2 = ca.vertcat(ca.reshape(U2, -1, 1), ca.reshape(X2, -1, 1))
nlp_prob1 = {'f': obj1, 'x': ca.reshape(U1, -1, 1), 'p': P1, 'g': ca.vertcat(*g1)}
nlp_prob2 = {'f': obj2, 'x': ca.reshape(U2, -1, 1), 'p': P2, 'g': ca.vertcat(*g2)}
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
    #### 记住这个顺序，不可搞混！
    #### U是以(n_controls, N)存储的，但是在定义问题的时候被改变成(n_controlsxN,1)的向量
    #### 实际上，第一组控制v0和omega0的index为U_0为U_1，第二组为U_2和U_3
    #### 因此，在这里约束必须在一个循环里连续定义。
    lbx.append(-1500) #-k/m
    lbx.append(-100)   #-d/m
    lbx.append(1)    #1/m
    ubx.append(-100)
    ubx.append(-20)
    ubx.append(1)

for _ in range(N+1):
    lbg.append(-np.inf)  #x_c'
    lbg.append(-12 )  #x_c
    lbg.append(-np.inf)  #f_x
    ubg.append(np.inf)
    ubg.append(9)
    ubg.append(np.inf)
## 仿真条件和相关变量
t0 = 0.0  # 仿真时间
x_d_r = ca.cos(omega * 0) * raduis
y_d_r = ca.sin(omega * 0) * raduis  # 期望轨迹
x_d_r_ = -raduis * omega * ca.sin(omega * 0)
y_d_r_ = raduis * omega * ca.cos(omega * 0)
x_d_r__ = -raduis * (omega ** 2) * ca.cos(omega * 0)
y_d_r__ = -raduis * (omega ** 2) * ca.sin(omega * 0)  # 期望轨迹二阶导
x01 = np.array([0.0-x_d_r_, 10.0-x_d_r, 0.0]).reshape(-1, 1)  # 机器人初始状态
x02 = np.array([0.0-y_d_r_, 0.0-y_d_r, 0.0]).reshape(-1, 1)  # 机器人初始状态
u01 = np.array([-1000, -60, 1] * N).reshape(-1, 3)  # 系统初始控制状态，为了统一本例中所有numpy有关
u02 = np.array([-1000, -60, 1] * N).reshape(-1, 3)  # 系统初始控制状态，为了统一本例中所有numpy有关
# 变量都会定义成（N,状态）的形式方便索引和print
x_c = []  # 存储系统的状态
u_c = []  # 存储控制全部计算后的控制指令
t_c = []  # 保存时间
xx1 = []
xx2 = []
trajectory_history_d1 = []
trajectory_history_d2 = []
sim_time = 2  # 仿真时长
index_t = []  # 存储时间戳，以便计算每一步求解的时间
## 开始仿真
mpciter = 0  # 迭代计数器
start_time = time.time()  # 获取开始仿真时间
def shift_movement(T, t0, x0, u, x_d_r, x_d_r_, x_d_r__):
    e_1__ = (u[2,0]) * (x0[2] - (-u[1,0] / u[2,0]) * x0[0] - (-u[0,0] / u[2,0]) * x0[1])
    x_c__ = e_1__ + x_d_r__
    x_c_ = x_c__ + x_c__*T
    x_c = x_c_ + x_c_*T

    # 时间增加
    t = t0 + T
    # 准备下一个估计的最优控制，因为u[:, 0]已经采纳，我们就简单地把后面的结果提前
    u_end = ca.horzcat(u[:, 1:], u[:, -1])
    return t, x_c, x_c_, x_c__, u_end.T
### 终止条件为小车和目标的欧式距离小于0.01或者仿真超时
while (mpciter - sim_time / T < 0.0):
    ### 初始化优化参数
    trajectory_history_d1.append(x_d_r)
    trajectory_history_d2.append(y_d_r)

    c_p1 = np.concatenate((x01, np.array([[mpciter]])))
    c_p2 = np.concatenate((x02, np.array([[mpciter]])))

    ### 初始化优化目标变量
    init_control1 = ca.reshape(u01, -1, 1)
    init_control2 = ca.reshape(u02, -1, 1)
    ### 计算结果并且
    t_ = time.time()
    res1 = solver1(x0=init_control1, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
    res2 = solver2(x0=init_control2, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
    index_t.append(time.time() - t_)
    ### 获得最优控制结果u
    u_sol1 = ca.reshape(res1['x'], n_controls, N)  # 记住将其恢复U的形状定义
    u_sol2 = ca.reshape(res2['x'], n_controls, N)  # 记住将其恢复U的形状定义
    ###
    ff_value1 = ff1(u_sol1, c_p1)  # 利用之前定义ff函数获得根据优化后的结果
    ff_value2 = ff2(u_sol2, c_p2)  # 利用之前定义ff函数获得根据优化后的结果
    # 小车之后N+1步后的状态（n_states, N+1）
    ### 存储结果
    # x_c.append(ff_value1)
    # u_c.append(u_sol1[:, 0])
    # t_c.append(t0)
    ### 根据数学模型和MPC计算的结果移动小车并且准备好下一个循环的初始化目标
    t0, x_c, x_c_, x_c__, u01 = shift_movement(T, t0, x01, u_sol1, x_d_r, x_d_r_, x_d_r__)
    t0, y_c, y_c_, y_c__, u02 = shift_movement(T, t0, x02, u_sol2, x_d_r, x_d_r_, y_d_r__)

    #计算下一次期望轨迹
    x_d_r = math.cos(omega * (mpciter + 1)*T) * raduis
    y_d_r = math.sin(omega * (mpciter + 1)*T) * raduis  #期望轨迹
    x_d_r_ = -raduis * omega * math.sin(omega * (mpciter + 1)*T)
    y_d_r_ = raduis * omega * math.cos(omega * (mpciter + 1)*T)  #期望轨迹一阶导
    x_d_r__ = -raduis * (omega ** 2) * ca.cos(omega * (mpciter + 1)*T)
    y_d_r__ = -raduis * (omega ** 2) * ca.sin(omega * (mpciter + 1)*T)  # 期望轨迹二阶导
    ### 存储小车的位置
    print(type(x_c_), np.shape(x_c_))
    print(type(x_d_r_), np.shape(x_d_r_))
    print(type(x_c), np.shape(x_c))
    print(type(x_d_r), np.shape(x_d_r))

    x01 = np.array([float(x_c_.full())- x_d_r_, float(x_c.full()) - x_d_r, 0.0]).reshape(-1, 1)
    x02 = np.array([float(y_c.full())- y_d_r_, float(y_c.full()) - y_d_r, 0.0]).reshape(-1, 1)


    xx1.append(x01[1])
    xx2.append(x02[1])

    ### 计数器+1
    mpciter = mpciter + 1


################
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
plt.plot(xx1, xx2,  linestyle='-', label="tranjectory",linewidth = 2, c='red')
# plt.plot(trajectory_history_d1, trajectory_history_d2,  linestyle='--', label="tranjectory_d",linewidth = 2, c='blue')
plt.legend(loc = 'lower right', markerscale = 0.5, fontsize='medium',shadow=False,framealpha=0.5)
# plt.xlim(-15, 15)  # x轴范围从1到4
# plt.ylim(-15, 15)  # y轴范围从1到5
plt.savefig("trajectory.png", dpi=600)

plt.show()

