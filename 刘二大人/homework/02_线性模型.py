# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# 真实原函数 y = 2 * x + 2

x_data = [1.0, 2.0, 3.0]
y_data = [4.0, 6.0, 8.0]


def forward(x):
    return x * w + b


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)**2


w_list = np.arange(0., 4., .01)
b_list = np.arange(0., 4., .01)
X_w, Y_b = np.meshgrid(w_list, b_list) # 画三维图需要这个函数转化为坐标平面上的散点


mse_list = []

for w in w_list:
    for b in b_list:
        print('w =', w, 'b =', b)
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            y_pred = forward(x_val)
            loss_val = loss(x_val, y_val)
            l_sum += loss_val
            print('\t', x_val, y_val, y_pred, loss_val)
        print('MSE=', l_sum/3)
        mse_list.append(l_sum/3)

Z_mse = np.array(mse_list).reshape((400, 400))


# 绘图
fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(X_w, Y_b, Z_mse, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)

# ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_title('W, b distribution in 3-D picture')
ax.set_xlabel('W')
ax.set_ylabel('b')
ax.set_zlabel('Mse')

plt.show()
