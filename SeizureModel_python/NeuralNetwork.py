# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 20:14:31 2024

@author: Qianchen Gong
"""

import numpy as np
from Projection import Projection
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib as mpl
from Recorder import Recorder
import psutil  #获取系统的可用内存

np.set_printoptions(precision=15) #设置NumPy数组输出精度为13位

class NeuralNetwork(object): 
    def __init__(self, n, t=0):
        self.n = n
        self.t = t
        # self.Proj_In = [] # projection的实例
        # self.Proj_Out = []
        # self.Proj_In = 1
        # self.Proj_Out = 1
        self.Proj_In =[
            Projection(self, self, Type='E', Topology='linear', Method = 'convolution'),
            Projection(self, self, Type='E', Topology='linear', Method = 'convolution'),
            Projection(self, self, Type='E', Topology='linear', Method = 'convolution')
        ]
        self.Proj_Out =[
            Projection(self, self, Type='E', Topology='linear', Method = 'convolution'),
            Projection(self, self, Type='E', Topology='linear', Method = 'convolution'),
            Projection(self, self, Type='E', Topology='linear', Method = 'convolution')
        ]
        self.Ext = None
        self.Recorder = None
        self.Graph = None
        self.UserData = {}  # 字典
    
    @property#将dim方法声明为一个属性，使得可以像访问属性一样使用它，而不是调用方法。例如，可以用obj.dim而不是obj.dim()。
    def dim(self):
        s = (self.n).shape
        if len(s) > 2:
            return len(s)
        else:
            return 1 if np.prod(self.n) == np.max(self.n) else 2
        
    def conv2_field(self, R, ker, nT, topology): # (value, w, target.n, topology)
        # R = np.transpose(R)  # 转置操作来匹配MATLAB的行列顺序
        # ker = np.transpose(ker)  # 同样转置卷积核
        
        # Check kernel size
        if ker.shape[0] % 2 == 0 or ker.shape[1] % 2 == 0: #检查行数和列数是否为奇数，若为偶数，抛出错误
            raise ValueError('Kernel size needs to be odd')

        n = R.shape

        # Check if n - nt can be evenly divided, if not, zero-pad the input layer
        # 在 odd 向量中存在非零元素时对矩阵 R 进行零填充，然后通过循环移位和平滑处理来调整矩阵 R 的边缘
        odd = np.mod(np.array(n) - np.array(nT), 2)
        if np.any(odd):
            R = np.pad(R, [(0, odd[0]), (0, odd[1])], mode='constant') #在 R 的行和列的末尾增加 odd[0] 行和 odd[1] 列的零填充。
            R = (R + np.roll(R, odd, axis=(0, 1))) / 2 #原始的 R 与循环移位后的 R 相加并取平均值，通过移位和平滑处理调整矩阵 R 的边缘
        n = R.shape

        # Convolution
        O = convolve2d(R, ker, mode='full')

        # Check size of O matrix & change O into a correct size
        nO = O.shape
        nRound = np.ceil(np.array(nO) / np.array(nT)).astype(int) #逐元素相除后，向上取整
        if not np.all(np.array(nO) == np.array(nT)):
            O = np.pad(O, ((0, nRound[0] * nT[0] - nO[0]), (0, nRound[1] * nT[1] - nO[1])), mode='constant')
        dshift = (np.array(nO) - np.array(nT)) // 2
        O = np.roll(O, -dshift, axis=(0, 1)) # axis=(0, 1)参数指定沿着行和列两个轴进行移位操作

        if topology.lower() == 'circular':
            Output = np.zeros(nT)
            for iter1 in range(nRound[0]):
                for iter2 in range(nRound[1]):
                    start1 = (iter1 - 1) * nT[0]
                    end1 = iter1 * nT[0]-1
                    start2 = (iter2 - 1) * nT[1]
                    end2 = iter2 * nT[1]-1
                    Output += O[start1:end1, start2:end2]
        elif topology.lower() == 'linear':
            Output = O[:nT[0], :nT[1]]

        # Output = np.transpose(Output)

        return Output
    
    # def conv2_field(self, R, ker):
    # # 获取输入矩阵 R 和卷积核 ker 的形状
    #     m, n = R.shape
    #     km, kn = ker.shape
        
    #     # 输出矩阵初始化为零
    #     output_matrix = np.zeros((m, n))
        
    #     # 卷积核的半尺寸偏移量
    #     dshift = km // 2
        
    #     # 遍历输出矩阵的每个位置
    #     for i in range(m):
    #         for j in range(n):
    #             # 获取区域的行索引和列索引，进行周期性处理
    #             rows = np.mod(np.arange(i - dshift, i + dshift + 1), m)  # 行周期性处理
    #             cols = np.mod(np.arange(j - dshift, j + dshift + 1), n)  # 列周期性处理
                
    #             # 提取区域并计算卷积
    #             # region = R[rows, cols]
    #             region = R[rows[:, None], cols]  # rows[:, None] 是为了将 rows 转换为列向量
    #             output_matrix[i, j] = np.sum(region * ker)
    
    #     return output_matrix
    
    def Project(self, P):
        #for i in range(len(P)):
            # 每个突触前神经元的强度
        if P.WPre is not None and np.any(P.WPre.flatten() != 1):
            P.Value = P.WPre * P.Value
        # 如果 WPre 不为空且存在不为 1 的元素，
        # 则说明预突触神经元的权重是非均匀的，需要对连接对象的值进行调整。
        
        # 根据突触投影的分布情况，使用相应的方法
        if P.Method.lower() == 'convolution':
            P.Value = self.conv2_field(P.Value, P.W, P.Target.n, P.Topology)
            # P.Value = self.conv2_field(P.Value, P.W)
            # P.Value = P.Value + 0.1
        elif P.Method.lower() == 'multiplication':
            P.Value = P.W * P.Value.flatten()
            P.Value = P.Value.reshape(P.Target.n)
        elif P.Method.lower() == 'function':
            P.Value = P.W(P.Value)
        
        # 每个突触后受体的强度
        # if P.WPost is not None and np.any(P.WPost.flatten() != 1):
        if P.WPost is not None and np.any(P.WPost != 1):
            P.Value = P.Value * P.WPost

        # test
        test_P_Value = P.Value
        
    # def update(self, dt):
    #     if dt <= 0:
    #         print('The model is running backward.')
    #     for i in range(len(self.Proj_Out)):
    #         self.Project(self.Proj_Out[i])
        
    # def plot(self):
    #     self.IndividualModelPlot()
    
    def IndividualModelPlot(self):
        # 判断是否创建图
        VarName = ['V', 'phi', 'Cl_in', 'g_K']
        if self.Graph is None or not plt.fignum_exists(self.Graph.number): #plt.fignum_exists检查图形窗口是否存在
            # self.Graph.number 可以获取图形窗口的编号
            
            self.Graph, Ax = plt.subplots(len(VarName), 1, figsize=(4, 3))
            # len(VarName)行，1列
            # figsize=(10, 7.5)设置图形窗口的大小，单位为英寸
            
            # self.Graph = plt.figure(figsize=(10, 7.5))
            
            manager = plt.get_current_fig_manager() #获取当前图形窗口的管理器对象 manager，用于设置窗口的几何属性
            manager.window.setGeometry(
                                       int(0.25*manager.window.screen().size().width()), #左边距
                                       int(0.05*manager.window.screen().size().height()), #右变距
                                       int(0.5*manager.window.screen().size().width()), #宽度
                                       int(0.85*manager.window.screen().size().height()) #高度
            )      
            for j in range(len(VarName) - 1, -1, -1): #反向循环，从 len(VarName) - 1 开始，到 -1 结束（不包括 -1），步长为 -1
                Ax[j].set_title(VarName[j])  # 设置子图的标题
                plt.draw()
            # Ax[0].set_title('V')
        else:
            Ax = self.Graph.get_axes() # 返回当前图形窗口（即 Figure 对象）中所有轴（Axes 对象）的列表
            for j in range(len(VarName) - 1, -1, -1):
                Ax[j].set_title(VarName[j])
            # Ax[0].set_title('V')
                
        # # 定义每幅图的颜色条范围
        # norms = [
        #     mpl.colors.Normalize(vmin=-80, vmax=-20),  # 第1幅图
        #     mpl.colors.Normalize(vmin=-80, vmax=-20),  # 第2幅图
        #     mpl.colors.Normalize(vmin=0, vmax=50),     # 第3幅图
        #     mpl.colors.Normalize(vmin=0, vmax=np.mean(self.g_K_max) * np.mean(self.f_max))       # 第4幅图
        # ]
        # # norms  = [
        # #     mpl.colors.Normalize(vmin=-63, vmax=-53),  # 第1幅图
        # #     mpl.colors.Normalize(vmin=-50, vmax=-40),  # 第2幅图
        # #     mpl.colors.Normalize(vmin=1, vmax=11),     # 第3幅图
        # #     mpl.colors.Normalize(vmin=-5, vmax=5)      # 第4幅图
        # # ] 
        
        # 判断模型的维度并选择相应的方法来绘图
        if np.any(self.n == 1): # one-dimensional case
            for j in range(len(VarName)):
                if len(Ax[j].lines) == 0:   # 使用 Ax[j].lines 检查轴上是否有线对象
                    # line, =Ax[j].plot(self.__dict__[VarName[j]], label=f"Data_{VarName[j]}")
                    line = Ax[j].plot(self.__dict__[VarName[j]], label=f"Data_{VarName[j]}")[0]
                    # self.__dict__[VarName[j]]获取VarName[j]数据，绘制折线图并设置标签
                    # [0]: 获取 plot 返回的第一个 Line2D 对象，因为 plot 返回的是一个 Line2D 对象的列表
                else:
                    for line in Ax[j].lines: #遍历Ax[j]的所有轴对象
                        if line.get_label() == f"Data_{VarName[j]}":
                            line.set_ydata(self.__dict__[VarName[j]])
                            break
                # 设置每个子图的y轴范围
                if VarName[j] == 'V':
                    Ax[j].set_ylim([-80, -20])  # 设置 V 的 Y 轴范围
                elif VarName[j] == 'phi':
                    Ax[j].set_ylim([-80, -20])  # 设置 phi 的 Y 轴范围
                elif VarName[j] == 'Cl_in':
                    Ax[j].set_ylim([0, 40])  # 设置氯离子浓度的 Y 轴范围
                elif VarName[j] == 'g_K':
                    Ax[j].set_ylim([0, np.mean(self.g_K_max * self.f_max)])  # 设置 g_K 的 Y 轴范围
            
                # 设置 x 轴范围
                Ax[j].set_xlim([0, np.max(self.n)])  # 设置 X 轴范围
                
        elif len(self.n) == 2 and not np.any(self.n == 1):  # two-dimensional case
        
            # 定义每幅图的颜色条范围
            norms = [
                mpl.colors.Normalize(vmin=-80, vmax=-20),  # 第1幅图
                mpl.colors.Normalize(vmin=-80, vmax=-20),  # 第2幅图
                mpl.colors.Normalize(vmin=0, vmax=50),     # 第3幅图
                mpl.colors.Normalize(vmin=0, vmax=np.mean(self.g_K_max * self.f_max))       # 第4幅图
            ]
            # norms  = [
            #     mpl.colors.Normalize(vmin=-63, vmax=-53),  # 第1幅图
            #     mpl.colors.Normalize(vmin=-50, vmax=-40),  # 第2幅图
            #     mpl.colors.Normalize(vmin=1, vmax=11),     # 第3幅图
            #     mpl.colors.Normalize(vmin=-5, vmax=5)      # 第4幅图
            # ] 
            for j in range(len(VarName)):
                if len(Ax[j].images) == 0:  # Ax[j].images 检查轴上是否有图像对象
                    im= Ax[j].imshow(self.__dict__[VarName[j]], aspect = 1, cmap = 'gist_ncar', norm=norms[j]) #imshow 没有 label 属性
                    im.set_label(f"Data_{VarName[j]}")
                    plt.draw()
                    self.Graph.colorbar(im, ax=Ax[j])
                else:
                    for im in Ax[j].images:
                        if im.get_label() == f"Data_{VarName[j]}":
                            im.set_data(self.__dict__[VarName[j]])
                            break      
        return self.Graph
    
        
    def CreateRecorder(self, N):
        if N is None:
            # Use 5% of RAM to record
            available_memory = psutil.virtual_memory().available
            N = 0.05 * available_memory / 8  # Available slots
            N = N / np.prod(self.n)
            N = 1000 * (N // 1000)  # Make sure the Capacity is counted in 'thousands'
            # N // 1000表示对N进行整数除法，并将结果向下取整到最接近的千位整数
        self.Recorder = Recorder(self, int(N))
        # (self.Recorder).append(self.Recorder) # python中没有vertcat对应的函数，可以将所有的Recorder对象收集到一个列表中
        return self.Recorder

    def WriteToRecorder(self):
            # If .Recorder has not been set up, create it
            if not hasattr(self, 'Recorder') or self.Recorder is None:
                self.CreateRecorder(N=None)
            # hasattr(object, 'attribute_name')  :如果对象具有该属性，则返回True，否则返回False
            Recorder.Record(self.Recorder)

    # *args传递任意数量的位置参数，**kwargs传递任意数量的关键字参数     
    def link(Oi, Oj, *args, **kwargs):  # link 可以接收任意数量的额外位置参数和关键字参数
        return Projection(Oi, Oj, *args, **kwargs)
        