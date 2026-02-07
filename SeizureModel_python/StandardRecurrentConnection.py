# -*- coding: utf-8 -*-
"""
Created on Tue May  7 18:54:05 2024

@author: Qianchen Gong
"""
import numpy as np
from scipy.stats import multivariate_normal
from Projection import Projection

np.set_printoptions(precision=15) #设置NumPy数组输出精度为13位

def StandardRecurrentConnection(O):
    # Build recurrent excitation
    P_E = Projection(O, O, Type='E', Topology='linear', Method = 'function')
    Sigma_E = np.diag(O.n) * 0.02  # percentage of the field
    P_E.Kernelize(lambda x: multivariate_normal([0, 0], Sigma_E ** 2).pdf(x),
              KerSize=np.ceil(2.5 * np.diag(Sigma_E)))
    # P_E.W = lambda x: np.sum(x) / np.prod(O.n)
    P_E.WPost *= 100  # Projection strength

    # Build recurrent inhibition
    P_I1 = Projection(O, O, Type='I', Topology='linear', Method = 'function')
    Sigma_I = np.diag(O.n) * 0.03  # percentage of the field
    P_I1.Kernelize(lambda x: multivariate_normal([0, 0], Sigma_I ** 2).pdf(x), 
              KerSize=np.ceil(2.5 * np.diag(Sigma_I)))
    # P_I1.W = lambda x: np.sum(x) / np.prod(O.n)
    P_I1.WPost *= 250  # Projection strength

    # Build the global recurrent inhibition
    P_I2 = Projection(O, O, Type='I', Topology='linear', Method = 'function')
    P_I2.W = lambda x: np.sum(x) / np.prod(O.n)  # uniform distribution
    P_I2.WPost *= 50  # Projection strength
    # 注意这里输出的
    
    O.Proj_In = [ P_E, P_I1, P_I2 ]
    O.Proj_Out = [ P_E, P_I1, P_I2 ]
    
    return O.Proj_In, O.Proj_Out

# if __name__ == '__main__': 
#     n = np.array([100, 100])
#     O = MeanFieldModel(n)
#     [ P_E, P_I1, P_I2 ] = StandardRecurrentConnection( O );
#     O.Proj_Out = [ P_E, P_I1, P_I2 ]  
#     O.Proj_In = [ P_E, P_I1, P_I2 ]   