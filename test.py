import numpy as np
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
B = np.array([[0], [1], [0]])
A @B
print(B[:,0])
print(2**3)

x_0 = np.array([[1],[1],[2]])
x = x_0

# 输入矩阵
B_1 = np.array([[0], [1], [0]])
B_2 = x.T

B = B_1 @ B_2
B = np.multiply(T, B)  #

print('B', B)
