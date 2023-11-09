import numpy as np
import math
from cvxopt import matrix, solvers
from casadi import *
import cvxopt


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.transforms as transforms

#初始化画布
fig = plt.figure()
plt.grid(ls='--')

#绘制一条正弦函数曲线
x = np.linspace(0,2*np.pi,100)
y = np.sin(x)

crave_ani = plt.plot(x,y,'red',alpha=0.5)[0]


#绘制曲线上的切点
point_ani = plt.plot(0,0,'r',alpha=0.4,marker='o')[0]


#绘制x、y的坐标标识
xtext_ani = plt.text(5,0.8,'',fontsize=12)
ytext_ani = plt.text(5,0.6,'',fontsize=12)
ktext_ani = plt.text(5,0.4,'',fontsize=12)

#计算切线的函数
def tangent_line(x0,y0,k):
	xs = np.linspace(x0 - 0.5,x0 + 0.5,100)
	ys = y0 + k * (xs - x0)
	return xs,ys

#计算斜率的函数
def slope(x0):
	num_min = np.sin(x0 - 0.05)
	num_max = np.sin(x0 + 0.05)
	k = (num_max - num_min) / 0.1
	return k

#绘制切线
k = slope(x[0])
xs,ys = tangent_line(x[0],y[0],k)
tangent_ani = plt.plot(xs,ys,c='blue',alpha=0.8)[0]


#更新函数
def updata(num):
	k=slope(x[num])
	xs,ys = tangent_line(x[num],y[num],k)
	tangent_ani.set_data(xs,ys)
	point_ani.set_data([x[num]],[y[num]])
	xtext_ani.set_text('x=%.3f'%x[num])
	ytext_ani.set_text('y=%.3f'%y[num])
	ktext_ani.set_text('k=%.3f'%k)
	return [point_ani,xtext_ani,ytext_ani,tangent_ani,ktext_ani]

ani = animation.FuncAnimation(fig=fig,func=updata,frames=np.arange(0,100),interval=50)
# ani.save('sin_x.gif')
plt.show()
x = SX.sym('x',5)
y = SX.sym('y',5)
print(x)
print(horzcat(x,y))
U = SX.sym('U', 5, 4)
print(U)
u0 = np.array([0.0, 0.0]*5).reshape(2, 5)
print(u0)



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()

# 创建一个正方形
square = patches.Rectangle((0.5 - 0.1 / 2, 0.5 - 0.1 / 2), 0.1, 0.1, fc='b') # 确保方块在图像的中心
ax.add_patch(square)

# 设置坐标范围和隐藏坐标轴
ax.axis('equal')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.axis('off')

# 执行动画的函数
def animate(frame_number):
	# # 每帧更新角度
	# square.angle = frame_number % 360

	# 创建以方块中心为原点的旋转矩阵
	t = transforms.Affine2D().rotate_deg_around(0.5, 0.5, frame_number)
	# 更新旋转矩阵
	square.set_transform(t + ax.transData)
	return [square]

# 创建动画对象，frames是帧数，interval是每帧之间的延迟(以毫秒为单位)，blit是一种优化选项，可以提高动画的渲染速度
ani1 = FuncAnimation(fig, animate, frames=360, interval=20, blit=True)

# 保存动画
# ani1.save('rotating_square.gif', writer='imagemagick')

plt.show()

x0 = np.array([0.0, 0.0, 0.0]).reshape(-1, 1)
c_p = np.concatenate((x0,  np.arange(0, 0 + 5).reshape(-1, 1)), axis=0)

print(c_p)


# rect = plt.Rectangle((0, 0), 1, 1,  angle=80, fc='blue')
# ax.add_patch(rect)
# ax.set_xlim(0, 50)
# ax.set_ylim(0, 10)
#
# def update1(frame):
# 	x = frame / 10  # 每一帧水平运动的距离
# 	rect.set_x(x)  # 设置小方块的横坐标
# 	return [rect]
#
# def init():
# 	return [rect]
#
#
# ani1 = animation.FuncAnimation(fig, update1, frames=range(10), init_func=init, blit=True)
# ani1.save('rect.gif')
# plt.show()
