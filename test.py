import numpy as np
import math
x_history = np.zeros([2, 10])
print(x_history)
B = np.array([[0], [1], [0]])
p = np.size(B, 0)
print(p)
n = 5
tmp = np.eye(n)
print(tmp[1,1])
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# 创建一个一维数组，用于索引
indices = np.array([0, 2])

# 使用一维数组作为索引获取矩阵的元素
result = matrix[indices, 0:2]

print(result)
n = np.size(B)
rows = np.arange(1, n+1)
print(rows)
vector = np.array([1, 2, 3, 4, 5])

# 将向量中的所有元素减去1
result = rows - 1
print(matrix[result, 0:-1])

N_P = 3 # 假设N_P是一个常数

Q = np.array([[1, 2], [3, 4]]) # 假设Q是一个2x2的矩阵

Omega = np.kron(np.eye(N_P-1), Q)
print(Omega)

from cvxopt import matrix, solvers
P = matrix([[2., -1.], [-1., 4.]])
q = matrix([-1., -1.])
G = matrix([[-1., 0.], [0., -1.]])
h = matrix([0., 0.])
A = matrix([1., 1.], (1, 2))
b = matrix(1.)
solution = solvers.qp(P, q, G, h, A, b)
print(np.array(solution['x']))
A = np.array([[0,1,0],
             [0,0,0],
             [0,0,0]])

n = np.size(A, 0)
T = 0.01
I = np.eye(n)
print(np.multiply(A, T) - I)
# print(np.array([0, 1, 0]).T)

print(2**3)

x_0 = np.array([[1],[1],[2]])
x = x_0




noise_mean = 0
noise_stddev = 0.5
noise = np.random.normal(noise_mean, noise_stddev, (2, 1))
print(noise)

print(2**5)

print(math.cos(1)*10)
print(np.zeros((6,6)))
print(np.eye(2))
K = 1500
A_temp = np.zeros((6,6))  #定义一个空矩阵A
I_2x2 = np.eye(2)  #系统是二维系统
A_temp[0:2, 2:4] = I_2x2
A_1 = np.copy(A_temp)  #不带K的系统矩阵
K_temp = I_2x2 * K
A_temp[4:6, 2:4] = K_temp
A_2 = A_temp   #带K的系统矩阵
print(A_1)
print(A_2)
B_1 = np.zeros((2, 6))
B_1[:,2:4] = I_2x2
print(B_1)

k = np.array([[1, 0],
              [0, 1]])

m = np.array([[2, 0],
              [0, 3]])

# 计算 m 的逆矩阵
m_inv = np.linalg.inv(m)

# 计算 k * m^(-1)
result = np.dot(m_inv, k)
# result_2 = k/m

print("Result (k * m^(-1)):")
print(result)

print(np.zeros((2, 2)))
# print(result_2)

# 创建一个6x1的列向量
column_vector = np.array([[1],
                          [2],
                          [3],
                          [4],
                          [5],
                          [6]])

# 创建一个1x6的行向量
row_vector = np.array([[1, 2, 3, 4, 5, 6]])

# 计算外积
result = np.dot(column_vector, row_vector)

print("Result (6x6 matrix):")
print(result)
x_c = 5
y_c = 0
X_c = np.eye(2) @ np.array([[x_c], [y_c]])
print(X_c)

matrix = np.zeros((6, 6))
# 赋予对角线上的值
values = [1, 2, 3, 4, 5, 6]
np.fill_diagonal(matrix, values)
print(matrix)
# 创建一个列向量
column_vector = np.array([[1],
                          [2],
                          [3],
                          [4],
                          [5],
                          [6]])

# 使用矩阵乘法计算结果
result = np.dot(matrix, column_vector)

print("Result (6x1 column vector):")
print(result)

R = np.eye(6)
Rvalues = [10**(-6), 10**(-6), 10**(-6), 10**(-6), 10**(-6), 10**(-6)]
np.fill_diagonal(R, Rvalues)
print(R)


M = np.eye(2)

c = M
G = np.zeros((8, 6))
G[0:2, 0:2] = -c
G[2:4, 0:2] = c
G[4:6, 2:4] = -c
G[6:8, 2:4] = c

print(G)