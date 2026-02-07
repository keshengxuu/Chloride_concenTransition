# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 17:21:09 2024

@author: Qianchen Gong

"""

import numpy as np
from scipy.signal import convolve

class Recorder(object):
    """
    Recorder 是一个记录 NeuralNetwork 仿真结果的类。
    Recorder is a class that records the simulation results of NeuralNetwork.
    To create a Recorder that is associated with an instance of NeuralNetwork, use create_recorder(NeuralNetwork, Capacity).
    To write to a Recorder, use write_to_recorder(NeuralNetwork).
    """
    # 返回给定正整数的质因数
    def factor(n): 
        """Returns the prime factors of a given number n."""
        i = 2
        factors = []
        while i * i <= n:
            if n % i:
                i += 1
            else:
                n //= i
                factors.append(i)
        if n > 1:
            factors.append(n)
        return factors
    
    def __init__(self, O, N):
        # Initialize other properties
        self.VarName = ['V', 'phi', 'Cl_in', 'g_K']
        self._Idx = 1
        self.S = None
        self.Graph = {'Figure': None, 'Raster': None}
        self.Listener = []
        self.Children = None
        self.UserData = {}
        
        
        # O must be an instance of NeuralNetwork
        # if not isinstance(O, NeuralNetwork):
        #     raise TypeError('Recorder can only be created and associated with a NeuralNetwork.')
        
        # Register them together
        O.Recorder = self
        self.Parent = O
        
        # Capacity & time vector
        if N is None:
            N = int(np.ceil(10e6 / np.prod(O.n)))
        self._Capacity = N
        self.T = np.full((1, N), np.nan)
        
        # Initialize dynamic variables
        self.Var = {}
        for i in self.VarName:
            self.Var[i] = np.full((np.prod(O.n), N), np.nan)
            # shape = (np.prod(O.n), N)
            # # 创建稀疏矩阵，默认为0
            # aa_sparse = sparse.lil_matrix(shape, dtype=np.float32)
            # aa_sparse[:] = np.nan 
            # self.Var[i] = aa_sparse[:]
        
        # Initialize spike buffer
        self.SBuffer = [None] * N
        
        self.Listener.append(('Idx', self._Idx))
        self.Listener.append(('Capacity', self._Capacity))
    
    @property
    def Idx(self):
        return self._Idx

    @Idx.setter
    def Idx(self, value):
        old_value = self._Idx
        self._Idx = value
        if old_value != value:
            self.AutoCompressCheck()

    @property
    def Capacity(self):
        return self._Capacity

    @Capacity.setter
    def Capacity(self, value):
        old_value = self._Capacity
        self._Capacity = value
        if old_value != value:
            self.CapacityChange()
    
    @property
    def Ancestor(self):
        S = self.Parent
        from NeuralNetwork import NeuralNetwork
        while not isinstance(S, NeuralNetwork):
            S = S.Parent
        return S
    
    def Record(self):
        """
        Record from its Parent NeuralNetwork.
        """
        O = self.Parent
        self.T[0, self.Idx - 1] = O.t
        for var in self.VarName:
            self.Var[var][:, self.Idx - 1] = getattr(O, var).flatten() # getattr获取对象的属性值
        self.Idx += 1
        
    def TransferSpikeRecord(self):
        # If there are multiple instances, call the function recursively
       if isinstance(self, list) and len(self) > 1:
           for recorder in self:
               recorder.TransferSpikeRecord()
           return
       
       if not self.SBuffer:
           print('No spike has been recorded')
           return

       s = [x for x in self.SBuffer if x is not None]  # Filter out None values 过滤掉所有值为None的元素
       s = [np.ravel(x).tolist() for x in s]  # Flatten and convert to lists
       # ravel函数将数组x展平成一维数组
       # .tolist()：将展平后的一维数组转换为Python的列表。
       n = [len(x) for x in s] # 计算列表s中每一个元素的长度，并将这些长度存储在列表n中。
       I = [i for i, length in enumerate(n) if length > 0] # 找出列表n中长度大于0的元素的索引，并将这些索引存储在列表I中。
       # enumerate(n)：生成一个包含索引和值的枚举对象，其中i是索引，length是对应的列表n中的值。
       
       if not I:
           return  # No spikes detected at all
       
       tp = [np.full((len(n[I[i]]),), self.T[I[i]]) for i in reversed(range(len(I)))]
       # 创建一个矩阵，其元素是 self.T(I(i))，尺寸为 [1, n(I(i))]
       tp = np.concatenate(tp) # 将列表tp中的所有数组沿着一个轴连接起来，形成一个一维数组。
       s = np.concatenate([np.array(x) for x in s]) # 将列表s中的所有元素转换为NumPy数组后连接起来，形成一个一维数组。
       self.S.extend(tp.tolist() + s.tolist()) # 将数组tp和s转换为列表后，添加到对象属性self.S的末尾。
       self.SBuffer = [None] * self.Capacity  
    
    def AddVar(self, *var_names):
        for var_name in var_names:
            if var_name in self.VarName:
                print(f'Warning: The variable {var_name} has been defined. No need to redefine it.')
                continue
            self.VarName.append(var_name)
            self.Var[var_name] = np.full((np.prod(self.Parent.n), self.Capacity), np.nan)
            
    def CapacityChange(self):
        C = self.Capacity  # Target Capacity
        c = (self.T).size  # Original Capacity
        if c > C:
            print('The new capacity is smaller than original capacity, this operation may cause loss of data.')
            ANS = input('Are you sure to proceed? If you do, ensure you have safely transferred the original data first. (1/0) ')
            if ANS == '1':
                self.T = self.T[:C]
                self.SBuffer = self.SBuffer[:C]
                for m in self.VarName:
                    self.Var[m] = self.Var[m][:, :C]
                if self.Idx > C:
                    print('Warning: Writing Idx has been changed.')
                    self.Idx = C + 1
            else:
                self.Capacity = c  # Restore the original capacity
        elif c < C:
            self.T = np.append(self.T, np.full((C - c,), np.nan))
            self.SBuffer.extend([None] * (C - c))
            for m in self.VarName:
                self.Var[m] = np.hstack((self.Var[m], np.full((self.Var[m].shape[0], C - c), np.nan)))


    def compress(self, r):
        # % 如果R中有多个实例，则调用递归函数
        if isinstance(self, list) and len(self) > 1:
            Rcom = []
            for R in self:
                Rcom.append(self.compress(R, r) if r is not None else self.compress(R))
            return Rcom
        
        # Check whether self.Children has been created
        if self.Children is None or not getattr(self.Children, 'is_valid', False):
            Original_Recorder = self.Ancestor.Recorder
            self.Children = Recorder(self.Ancestor, self.Capacity)
            ExtraVarName = list(set(self.VarName) - set(self.Children.VarName)) # set用于去除列表中的重复元素
            if ExtraVarName:
                self.Children.AddVar(*ExtraVarName)
            self.Children.Parent = self
            self.Ancestor.Recorder = Original_Recorder
        
        Rcom = self.Children
    
        # Check whether the compression ratio 'r' can avoid interpolation
        N = self.Capacity
        if r is None:
            pf = self.factor(N)
            upf = list(set(pf))
            d = [1]
            for f in upf:
                d = [val * (f ** exp) for val in d for exp in range(sum(pf == f) + 1)]
                # d = [val * (f ** exp) for val in d for exp in range(pf.count(f) + 1)]
            r_goal = 10
            d_array = np.array(d)
            idx_choice = np.argmin(np.abs(d_array - r_goal))
            r = d_array[idx_choice]
            print(f'Compression ratio is set to be {r}')
        
        if N % r != 0:
            print('Warning: Compression ratio needs to be divisible by Capacity to prevent error or loss of data.')
    
        # Compress time vector
        Tvec = self.T[r-1::r]
        n = len(Tvec)
        Rcom.T[Rcom.Idx:Rcom.Idx+n] = Tvec
    
        # Compress each dynamical variable & extrafields
        width = r + (r % 2) - 1
        for name in self.VarName:
            x = self.Var[name]
            x = convolve(x, np.ones(width) / width, mode='valid')
            x_begin = np.cumsum(x[:width-1:2], axis=0) / np.arange(1, width, 2)[:, None]
            x_end = np.cumsum(x[:width-1:-2], axis=0) / np.arange(width-1, 0, -2)[:, None]
            x = np.vstack((x_begin, x[width-1::r], x_end))
            x = x[:, r-1::r]
            Rcom.Var[name][:, Rcom.Idx:Rcom.Idx+n] = x.T
        
        Rcom.Idx += n
    
        # Deal with spiking data if there is spiking records
        self.TransferSpikeRecord(self)
        Rcom.S = np.hstack((Rcom.S, self.S)) # np.hstack将数组沿第二个轴（即列方向）连接
    
        return Rcom
    

    def Refresh(self):
        if isinstance(self, list) and len(self) > 1:
            for recorder in self:
                recorder.Refresh()
            return
        
        self.T = np.full_like(self.T, np.nan) # np.full_like创建一个与给定数组形状和数据类型相同的新数组
        for name in self.VarName:
            self.Var[name] = np.full_like(self.Var[name], np.nan)
        self.SBuffer = [None] * self.Capacity
        self.S = []
        self.Idx = 1

    def AutoCompressCheck(self):
        if self.Idx > self.Capacity:
            print('Capacity is full ... Compressing data to Recorder.Children')
            self.Compress(r=None)
            self.Refresh()

# 示例用法
# if __name__ == '__main__': 
#     n =[100, 100]
#     O = MeanFieldModel(n)
#     RR = Recorder(O, 5000)


