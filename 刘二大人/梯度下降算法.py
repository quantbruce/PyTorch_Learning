###刘二大人
#
###1. 批量梯度下降法

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0

def forward(x):
    return x * w

def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y)**2
    return cost/len(xs)

def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (w * x - y)
    return grad/len(xs)


print('Predicted before training', 4, forward(4))
for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val
    print('epoch:', epoch, 'w:', w, 'loss=:', cost_val)
print('Predicted after training', 4, forward(4))




# 随机梯度下降算法

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)**2

def gradient(x, y):
    return 2 * x * (w * x - y)


print('Predicted before training', 4, forward(4))
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w -= 0.01 * grad
        print('\tgrad:', x, y, grad)
        l = loss(x, y)
print('Predicted after training', 4, forward(4))






