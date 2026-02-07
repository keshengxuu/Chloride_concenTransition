# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:28:44 2024

@author: Qianchen Gong
"""
import numpy as np


np.set_printoptions(precision=15) #设置NumPy数组输出精度为13位


class Projection(object):
    def ResamplePMF(self, Dist, N):
        # seed=None
        # np.random.seed(seed)  # 设置随机数种子
        # #设置随机数种子只是为了检查代码是否正确，后续应该删除随机数种子
        
        np.random.seed(42) # 设置相同的随机种子
        
        # Normalization
        Dist = Dist.flatten() / np.sum(Dist)
        D_cumsum = np.cumsum(np.concatenate(([0], Dist))) 
        #计算累积分布函数，并在最前面添加0元素，np.concatenate用于连接[0], Dist两个数组
        
        # Generate samples
        dist, _ = np.histogram(np.random.rand(N), bins=D_cumsum)
        # np.histogram用于计算直方图，bins是区间
        
        # Calculate sample category sequence if needed
        x = None
        x = np.full(N, np.nan)
        idx_event = np.where(dist > 0)[0]
        pointer = 0
        for i in idx_event:
            x[pointer:pointer + dist[i]] = i+1
            pointer += dist[i]
        np.random.shuffle(x) #随机打乱（shuffle）一个数组的顺序
        
        # Normalize sample distribution
        dist = dist / np.sum(dist)
        
        # Reshape sample distribution to the original shape
        dist = dist.reshape(Dist.shape) 
        
        return dist, x
    
    
    def __init__(self, Oi, Oj, Topology, Type ,Method):
        self.Source = Oi  # Source (Pre-synaptic) NeuralNetwork
        self.Target = Oj  # Target (Post-synaptic) NeuralNetwork
    
        
        self.Method = Method  # 'multiplication', 'convolution', or 'function'
        self.WPre = 1  # Strength of the projection, indexed by pre-synaptic neurons
        self.WPost = 1  # Strength of the projection, indexed by post-synaptic neurons
        self.W = 1  # Connectivity information
        self.Topology = Topology  # Only effective when method is 'convolution'
        self.Type = Type  # Determine what type of projection this is, commonly used 'E' or 'I'
        self.Value = 0  # The value of this projection
        self.Userdata = {}  # Stored user data
        
        
        n = self.Source.n  # Number of neurons in the source neural network
        self.WPre = np.ones(n)

    
    def Kernelize(self, func, **kwargs):
        N = kwargs.get('N', 0)
        KerSize = kwargs.get('KerSize', self.Target.n)

        # Check whether func has the right format
        # if func.__code__.co_varnames != 1:  # .__code__.co_argcount返回函数func参数的个数
        #     raise ValueError("The kernel function can only have one input (relative position).") #如果func的参数个数不等于1，则表示函数定义有误
    
        # Sample from the function 'func'
        x1, x2 = np.meshgrid(np.arange(-KerSize[1]+1, KerSize[1]), # 注意：np.arange终点不包含在内
                             np.arange(-KerSize[0]+1, KerSize[0]))
        positions = np.stack((x2.ravel(), x1.ravel()), axis=1) #.ravel()展平为一维数组，np.stack沿着新的轴连接一系列数组
        K = func(positions)
        K /= np.sum(K)  # Normalization
    
        if N > 0:
            K = self.ResamplePMF(K, N)
    
        K = K.reshape(x1.shape)  # Change it back to its shape
    
        # Set the corresponding parameters in P
        self.Method = 'convolution'
        self.W = K
        #self.UserData['Wfunc'] = func  # Save the information
    
        return K # 输出K为取样核函数、卷积核函数,就是P.W
    
     
    
# if __name__ == '__main__': 
#     n = np.array([100, 50])
#     O = MeanFieldModel(n)
#     Oi = O
#     Oj = O
#     P_E = Projection(Oi, Oj, Type='E', Topology='linear', Method = 'convolution')
#     Sigma_E = np.diag(O.n) * 0.02  # percentage of the field ，，np.diag生成对角矩阵
#     kk = P_E.Kernelize(lambda x: multivariate_normal.pdf(x, mean=[0, 0], cov=np.diag(Sigma_E**2)), 
#                   KerSize = np.ceil(2.5 * np.diag(Sigma_E)))
    # Oi.Proj_Out = []  
    # Oj.Proj_In = []   
    # Oi.Proj_Out.append(P_E)
    # Oj.Proj_In.append(P_E)
      
    
    
    
    
    
    
    
    
    
    
        