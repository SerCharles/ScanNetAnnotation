import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

# 数据点分布在这条曲线附近
def func(x):
    return 2*np.sin(2*np.pi*x)

# 误差函数， 计算拟合曲线与真实数据点之间的差 ，作为leastsq函数的输入
def residuals(p, x, y):
    fun = np.poly1d(p)    # poly1d（）函数可以按照输入的列表p返回一个多项式函数
    loss = y - fun(x)
    print(np.sum(loss**2,axis=0))
    return loss

# 拟合函数
def fitting(p):
    pars = np.random.rand(p+1)  # 生成p+1个随机数的列表，这样poly1d函数返回的多项式次数就是p
    r = leastsq(residuals, pars, args=(X, Y))   # 三个参数：误差函数、函数参数列表、数据点
    return r

# 要进行拟合的数据点
X = np.linspace(0, 1, 10)
Y = [np.random.normal(0, 0.1)+num for num in func(X)]  # 添加噪声

# 方便绘制曲线，所以创建
x_ = np.linspace(0, 1, 100)
y_ = func(x_)

# print(fitting(3))   可以看一下返回的是什么
fit_pars = fitting(3)[0]

plt.plot(x_, y_, label='real line')
plt.scatter(X, Y, label='real points')
plt.plot(x_, np.poly1d(fit_pars)(x_), label='fitting line')
plt.legend()
plt.show()
