# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:10:44 2024

@author: Qianchen Gong
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import matplotlib.pyplot as plt
from MeanFieldModel import MeanFieldModel
from StandardRecurrentConnection import StandardRecurrentConnection
from ExternalInput import ExternalInput
import seaborn as sns
from matplotlib.gridspec import GridSpec

np.set_printoptions(precision=15) #设置NumPy数组输出精度为13位

# 创建 MeanFieldModel 对象
n = [100, 100]
O = MeanFieldModel(n)

# 创建圆形掩膜
xx, yy = np.meshgrid(np.linspace(-O.n[1]/2 + 0.5, O.n[1]/2 - 0.5, O.n[1]), np.linspace(-O.n[0]/2 + 0.5, O.n[0]/2 - 0.5, O.n[0]))
xx = xx / O.n[1]  # 归一化坐标
yy = yy / O.n[0]  # 归一化坐标
rr = np.sqrt(xx**2 + yy**2)
mask = (rr < 0.5).astype(int)
# 修改激活函数
f_original = O.f
O.f = lambda v: mask * f_original(v)

num_neurons_in_mask = np.sum(mask) # 掩膜内的神经元个数

# 创建递归投影
O.Proj_In, O.Proj_Out = StandardRecurrentConnection(O)



# 定义外部输入函数
Ic = 200
stim_t = [2, 5]
stim_x = [0.5, 0.1]
stim_r = 0.05
O.Ext = ExternalInput()
O.Ext.Target = O

# 定义神经元网络的外部输入函数
def Deterministic_func(x, t):
    # x 是神经元的位置索引矩阵，t 是时间，单位毫秒
    norm_x1 = (x[:, 0] - 1) / (O.n[1] - 1)  # 归一化第一维度的坐标
    norm_x2 = (x[:, 1] - 1) / (O.n[0] - 1)  # 归一化第二维度的坐标
    
    # 计算神经元到刺激中心的距离
    distance = np.sqrt((norm_x1 - stim_x[0]) ** 2 + (norm_x2 - stim_x[1]) ** 2)
    
    # 判断是否在刺激半径范围内，并且时间是否在刺激时间范围内
    is_within_radius = distance < stim_r
    is_within_time = (stim_t[1] * 1000 > t) & (t > stim_t[0] * 1000)
    
    # 计算外部输入电流
    return is_within_radius * is_within_time * Ic

# 将确定性输入函数赋值给外部输入对象
O.Ext.Deterministic = Deterministic_func

# 模拟设置
dt = 1
write_cycle = 10
Capacity = 8500
R = O.CreateRecorder( Capacity )
T_end = write_cycle * R.Capacity - 1 * write_cycle
# 模拟的结束时间。- 1 * write_cycle 是减去一次记录周期，以确保模拟在记录周期的结束时刻停止
# T_end 的最终值为 100000 ms - 10 ms = 99990 ms

# 实时绘图设置
flag_realtime_plot = 1
T_plot_cycle = 1000



# 模拟循环
while True:
    # 结束条件
    if O.t >= T_end:
        break
        
    '''O.Update( dt = dt )'''
    O.IndividualModelUpdate( dt = dt );
    for i in range(len(O.Proj_Out)):
        O.Project(O.Proj_Out[i])
        

    if O.t == 1:
        # 创建热图的初始绘图
        h1 = plt.imshow(O.R_f, cmap='PRGn', vmin=0, vmax=0.12,)
        plt.colorbar(h1)
        
    
    # 实时绘图  
    if O.t % T_plot_cycle == 0:

        h1.set_data(O.R_f) 
        plt.draw()
        plt.pause(0.001)
        
   

