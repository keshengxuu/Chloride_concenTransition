# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 15:32:33 2024

@author: Qianchen Gong
"""

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.stats import norm
from tkinter import simpledialog

np.set_printoptions(precision=15) #设置NumPy数组输出精度为13位

class ExternalInput:
    def __init__(self):
        # Properties
        # self._Target = None  # Must be an instance of NeuralNetwork
        self.Target = None
        self.Random = {
            'tau_x': None,
            'tau_t': None,
            'sigma': 0,
            'Iz': 0
        }
        self.Deterministic = None  # Function for deterministic input
        self.Tmax = None  # Expiration time of ExternalInput
        self.UserData = None  # For any additional user-defined data
    
    # # Property getter and setter for 'Target'  
    # @property #访问
    # def Target(self):
    #     return self._Target

    # @Target.setter # 设置
    # def Target(self, O):
    #     if not isinstance(O, NeuralNetwork):
    #         raise ValueError('The Target of ExternalInput must be an instance of NeuralNetwork or its subclasses')
    #     self._Target = O
    
    # Method to evaluate the ExternalInput at time `t` with timestep `dt`
    def Evaluate(self, t, dt):
        np.random.seed(42) # 设置相同的随机种子
        
        O = self.Target
        n = O.n
        x2, x1 = np.meshgrid(np.arange(1, n[1]+1), np.arange(1, n[0]+1))
        I = 0
        
        # Check validity (delete if expired)
        self.CheckValidity()
        if self.Target is None:
            return I
        
        # Deterministic part
        if self.Deterministic is not None:
            I_d = self.Deterministic(np.column_stack((x1.flatten(), x2.flatten())), t)
            # np.column_stack()函数，将多个一维数组按列拼接起来，产生一个二维数组
            I_d = np.reshape(I_d, n)
            I += I_d
        
        # Random part - Ornstein-Uhlenbeck process
        z = np.random.randn(*n)
        if self.Random['tau_x'] is not None:
            z = self.SpaceFilter(z, self.Random['tau_x'])
        
        # Time - Ornstein-Uhlenbeck process
        if self.Random['tau_t'] is not None and self.Random['tau_t'] > 0:
            if np.all(self.Random['Iz'] == 0):
                z0 = np.random.randn(*n)
                if self.Random['tau_x'] is not None:
                    z0 = self.SpaceFilter(z0, self.Random['tau_x'])
                self.Random['Iz'] = z0

            self.Random['Iz'] = (self.Random['Iz'] * np.exp(-dt / self.Random['tau_t']) +
                                 np.sqrt(2 * dt / self.Random['tau_t']) * z)
        else:
            self.Random['Iz'] = z

        # Calculate result
        I = I.astype(np.float64)  # 将 I 转换为 float64
        I += self.Random['sigma'] * self.Random['Iz']
        return I

    # Space filter for random noise in Ornstein-Uhlenbeck process
    def SpaceFilter(self, z, tau_x):
        dx = 1
        for i in range(len(tau_x)):
            if tau_x[i] > 0:
                z = np.moveaxis(z, i, 0) # 将当前维度 i 移动到第一个维度
                nz = z.shape
                z = np.reshape(z, (z.shape[0], -1)) #矩阵重整形为二维数组
                b = np.sqrt(2 * dx / tau_x[i])
                a = [1, (dx - tau_x[i]) / tau_x[i]]
                z = signal.lfilter(b, a, z, axis=0, zi=(1 - b) * z[0, :])[0] #滤波
                z = np.reshape(z, nz) # 恢复原始形状
                z = np.moveaxis(z, 0, i) # 将第一个维度移回原始位置 i
        return z

    # Check validity of ExternalInput
    # 检查此ExternalInput实例是否仍在其有效时间范围内 
    # 如果不在，则将其删除，以节省计算能力
    def CheckValidity(self):
        if isinstance(self.Target, list):
            for E in self.Target:
                E.CheckValidity()
            return

        if self.Tmax is not None and self.Target.t > self.Tmax:
            self.delete()
            print("An ExternalInput instance has expired & deleted.")

    # Destructor function to delete ExternalInput
    def delete(self):
        if self.Target is not None:
            O = self.Target
            O.Ext.remove(self)

    @staticmethod
    def ring_pattern():
        # 使用 Matplotlib 确定圆的三个点
       print('Select three points to determine a circle.')
       plt.figure()
       ax = plt.gca()
       X, Y = zip(*plt.ginput(3))  # 获取用户点击的三个点
       ax.plot(X, Y, '.', markersize=20)
       plt.show()

       # 计算确定圆的参数
       M = np.array([
           [X[0]**2 + Y[0]**2, X[0], Y[0], 1],
           [X[1]**2 + Y[1]**2, X[1], Y[1], 1],
           [X[2]**2 + Y[2]**2, X[2], Y[2], 1]
       ])
       
       A = np.linalg.det(M[:, 1:4])
       B = -np.linalg.det(M[:, [0, 2, 3]])
       C = np.linalg.det(M[:, [0, 1, 3]])
       D = -np.linalg.det(M[:, 0:3])
       
       x0 = -B / (2 * A)  # 圆心的x坐标
       y0 = -C / (2 * A)  # 圆心的y坐标
       r = np.sqrt((B**2 + C**2 - 4 * A * D) / (4 * A**2))  # 圆的半径

       # 绘制刺激环
       theta = np.linspace(0, 2 * np.pi, 360)
       plt.plot(x0 + r * np.cos(theta), y0 + r * np.sin(theta), 'g--', linewidth=4)
       plt.show()

       # 使用 Tkinter 获取用户输入
       questions = [
           'Stimulation width (unit: # of neurons)',
           'Stimulation strength (pA)',
           'Stimulation duration (ms)',
           'Stimulation sinusoidal frequency (Hz) - if empty or 0, it will be just a square wave'
       ]
       default_answers = ['2', '300', '20', '']
       
       answers = []
       for question, default in zip(questions, default_answers):
           answer = simpledialog.askstring("Stimulation setting", question, initialvalue=default)
           answers.append(answer)
       
       stim_width, strength, Tmax = map(float, answers[:3])
       freq = float(answers[3]) if answers[3] else 0
       delta_theta = 0 if freq > 0 else np.pi / 2

       # 定义刺激函数
       func = lambda x, t: (((r + stim_width / 2) > np.sqrt((x[:, 0] - y0) ** 2 + (x[:, 1] - x0) ** 2)) & \
                     (np.sqrt((x[:, 0] - y0) ** 2 + (x[:, 1] - x0) ** 2) > (r - stim_width / 2))) * \
                     strength * np.sin(2 * np.pi * freq * t + delta_theta)

       # 构建 ExternalInput 实例
       ext_input = ExternalInput()
       ext_input.Deterministic = func
       ext_input.Tmax = Tmax
       
       return ext_input, func

    @staticmethod
    def focal_pattern():
        # 使用 Matplotlib 确定刺激中心
        print('Please determine stimulation center.')
        plt.figure()
        ax = plt.gca()
        x0, y0 = plt.ginput(1)[0]  # 获取用户点击的坐标
        scatter = ax.scatter(x0, y0, s=50, color='g')
        plt.show()

        # 使用 Tkinter 获取用户输入
        questions = [
            'Stimulation radius (unit: # of neurons)',
            'Stimulation strength (pA)',
            'Stimulation duration (ms)',
            'Stimulation sinusoidal frequency (kHz) - if empty or 0, it will be just a square wave',
            'Shape (sharp or Gaussian)'
        ]
        default_answers = ['5', '300', '20', '', 'sharp']
        
        answers = []
        for question, default in zip(questions, default_answers):
            answer = simpledialog.askstring("Stimulation setting", question, initialvalue=default)
            answers.append(answer)
        
        stim_radius, strength, Tmax = map(float, answers[:3])
        freq = float(answers[3]) if answers[3] else 0
        stim_type = answers[4]
        delta_theta = 0 if freq > 0 else np.pi / 2

        # 定义刺激函数
        if stim_type.lower() == 'gaussian':
            func = lambda x, t: norm.pdf(np.sqrt((x[:, 0] - y0) ** 2 + (x[:, 1] - x0) ** 2), 0, stim_radius) / norm.pdf(0) * \
                                strength * np.sin(2 * np.pi * freq * t + delta_theta)
        else:  # sharp case
            func = lambda x, t: (np.sqrt((x[:, 0] - y0) ** 2 + (x[:, 1] - x0) ** 2) < stim_radius) * \
                                strength * np.sin(2 * np.pi * freq * t + delta_theta)

        # 构建 ExternalInput 实例
        ext_input = ExternalInput()
        ext_input.Deterministic = func
        ext_input.Tmax = Tmax
        
        return ext_input, func
