import numpy as np
import math
from cvxopt import matrix, solvers

import cvxopt

# 定义二次规划问题的参数
P = cvxopt.matrix([[2.0, 1.0], [1.0, 2.0]])
q = cvxopt.matrix([1.0, 1.0])

# 不等式约束
G = cvxopt.matrix([[-1.0, 0.0], [0.0, -1.0]])
h = cvxopt.matrix([0.0, 0.0])

# 等式约束
A = cvxopt.matrix([[-1.0], [-1.0]])
b = cvxopt.matrix([-1.0])

# 解决二次规划问题
sol = cvxopt.solvers.qp(P, q, G, h, A, b)
p = 1
# 输出结果
print('...............')
print("Solution:")
print(np.array(sol['x']))
sol = np.array(sol['x'])
u = sol[0:2, 0]
print(u)
u = u.reshape(-1, 1)
print(u)

import cvxopt

# 定义二次规划问题的参数
P = cvxopt.matrix([[2.0, 1.0], [1.0, 2.0]])
q = cvxopt.matrix([1.0, 1.0])

# 不等式约束
G = cvxopt.matrix([[-1.0, 0.0], [0.0, -1.0]])
h = cvxopt.matrix([0.0, 0.0])

# 等式约束
A = cvxopt.matrix([[-1.0, -1.0]])
b = cvxopt.matrix([-1.0])

# 解决二次规划问题
sol = cvxopt.solvers.qp(P, q, G, h, A, b)

# 输出结果
print("Solution:")
print(sol['x'])