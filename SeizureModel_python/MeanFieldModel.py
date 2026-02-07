# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 15:59:53 2024

@author: Qianchen Gong
"""
from NeuralNetwork import NeuralNetwork
import numpy as np

np.set_printoptions(precision=15) #设置NumPy数组输出精度为13位

class MeanFieldModel(NeuralNetwork):
    
    def gen_param(self, mu, sigma, m, flag):
        np.random.seed(42) # 设置相同的随机种子
        
        if sigma == 0:
            output = np.full(m, mu)
        elif sigma > 0:
            if flag == 'normal' or flag == 0:
                #output = mu + sigma * np.random.randn(*m)
                output = np.random.normal(mu, sigma, m)
            elif flag == 'gamma' or flag == 1:
                shape = (mu ** 2) / (sigma ** 2)
                scale = (sigma ** 2) / mu
                output = np.random.gamma(shape, scale, size=m)
            # elif flag == 'lognormal' or flag == 2:
            #     MU = lambda MEAN, DELTA: np.log(MEAN) - np.log(1 + DELTA ** 2) / 2
            #     STD = lambda MEAN, DELTA: np.sqrt(np.log(1 + DELTA ** 2))
            #     output = np.exp(MU(mu, sigma/mu) + STD(mu, sigma/mu) * np.random.randn(*m))
            else:
                raise ValueError("Invalid flag value")
        else:
            raise ValueError("Standard deviation needs to be positive")
        
        return output


    def __init__(self, n, t = 0):
        super().__init__(n) # 调用父类的构造函数
        
        self.n = n
        self.t = t
        self.V = self.gen_param(-58., 0, self.n, 0)  # 初始膜电位
        self.C = self.gen_param(100., 0, self.n, 1)  # 细胞电容
        self.g_L = self.gen_param(4., 0, self.n, 1)  # 泄漏电导
        self.E_L = self.gen_param(-58, 0, self.n, 0)  # 静息膜电位
        self.T_refractory = self.gen_param(5., 0, self.n, 1)  # 绝对不应期时间
        self.f_max = 1. / self.T_refractory
        self.f0 = self.gen_param(0.001, 0, self.n, 1)
        self.beta = self.gen_param(2.5, 0, self.n, 1)
        self.f = lambda u: self.f_max / (1 + np.exp(-u / self.beta))
        self.tau_syn_E = self.gen_param(15., 0, self.n, 0)
        self.tau_syn_I = self.gen_param(15., 0, self.n, 0)
        self.E_Esyn = self.gen_param(0., 0, self.n, 0)
        self.flag_STP = False
        self.tau_D = self.gen_param(0., 0, self.n, 1)
        self.tau_F = self.gen_param(0., 0, self.n, 1)
        self.U = self.gen_param(0.2, 0, self.n, 1)
        self.phi = self.gen_param(-45., 0, self.n, 0)
        self.phi_0 = self.gen_param(-45., 0, self.n, 0)
        self.phi_inf = lambda u: self.phi_0 + u * 60
        self.tau_phi = self.gen_param(100., 0, self.n, 1)
        self.Vd_Cl = 0.25 * np.sqrt(2) / 12 * 20**3 / 1000 * np.ones(self.n)
        self.Cl_ex = 110.
        self.Cl_in_eq = self.gen_param(6., 0, self.n, 1)
        self.tau_Cl = self.gen_param(5000., 0, self.n, 1)
        self.Cl_in = self.Cl_in_eq
        self.tau_K = self.gen_param(5000., 0, self.n, 1)
        self.g_K_max = self.gen_param(40., 0, self.n, 1)
        self.E_K = self.gen_param(-90., 0, self.n, 0)
        self.g_K = np.zeros(self.n)
        self.Input_E = np.zeros(n)
        self.Input_I = np.zeros(n)
        self.R_f = np.zeros(n)
        self.tau_syn = [self.tau_syn_E,  self.tau_syn_I,  self.tau_syn_I]
        self.E_Cl = None
        self.I_synE = None
        self.I_synI = None
        self.I_K = None
        self.w = None
        

    def IndividualModelUpdate(self, dt = 1):
        #super().__init__( self.n )  #调用父类NeuralNetwork的构造函数
        
        I_ext = 0
        I_ext = self.Ext.Evaluate(self.t, dt) + I_ext;
        
        # tau_syn = [self.tau_syn_E,  self.tau_syn_I,  self.tau_syn_I]
        # Input   = [self.Input_E,    self.Input_I,    self.Input_I]

        # # Collect input from 'Projection'
        # for i in range(len(tau_syn)):
        #     if np.all(tau_syn[i] == 0):
        #         Input[i] += self.Proj_In[i].Value
        #     else: # Apply synaptic filtering
        #         Input[i] += self.Proj_In[i].Value / tau_syn[i]

        # self.Input_E = Input[0]
        # self.Input_I = Input[2]
        
        self.Input_E += self.Proj_In[0].Value / self.tau_syn_E
        self.Input_I0 = self.Input_I + self.Proj_In[1].Value / self.tau_syn_I
        # test_Input_I0 = self.Input_I0
        self.Input_I = self.Input_I0 + self.Proj_In[2].Value / self.tau_syn_I
        # test_Input_I = self.Input_I
        
        
        # # test
        # test_Input_E = self.Input_E  
        # test_Input_I = self.Input_I
        
        # Update equation 1 (membrane potential)
        g_sum = (self.g_L + 
                 self.Input_E / self.f_max + 
                 self.Input_I / self.f_max + 
                 self.g_K / self.f_max)
          
        self.E_Cl = 26.7 * np.log(self.Cl_in / self.Cl_ex)
        V_inf = ((self.g_L * self.E_L + 
                  self.Input_E / self.f_max * self.E_Esyn + 
                  self.Input_I / self.f_max * self.E_Cl + 
                  self.g_K / self.f_max * self.E_K + 
                  I_ext) / g_sum)
        
        tau_V_eff = self.C / g_sum
        self.V = V_inf + (self.V - V_inf) * np.exp(-dt / tau_V_eff)
        
        self.I_synE = self.Input_E * (self.V - self.tau_syn_E)
        self.I_synI = self.Input_I * (self.V - self.E_Cl)
        
        # test
        # test_V = self.V
        
        # Update equation 2 (threshold dynamics)
        self.phi += (self.phi_inf(self.R_f / self.f_max) - self.phi) * (1 - np.exp(-dt / self.tau_phi))
          
        # test
        # test_phi = self.phi 
        
        # Update equation 3 (chloride dynamics)
        Faraday = 96500
        # Cl_in_inf = (self.tau_Cl / self.Vd_Cl / Faraday * self.Input_I * (self.V - self.E_Cl) + self.Cl_in_eq)
        # self.Cl_in = Cl_in_inf + (self.Cl_in - Cl_in_inf) * np.exp(-dt / self.tau_Cl)
        up = self.Input_I * (self.V - self.E_Cl) / (self.Vd_Cl * Faraday)
        # # Cl_in_avg = np.mean(self.Cl_in)
        # total_elements = self.Cl_in.size 
        # half_elements = total_elements // 2  
        # random_indices = np.random.choice(total_elements, half_elements, replace=False)
        # random_elements = self.Cl_in.flat[random_indices]
        # Cl_in_avg = np.nanmean(random_elements)
        
        # th = 12  # Concentration threshold
        # k = 0.5  # Transition steepness
        # # w = (1 - np.tanh(k * (Cl_in_avg - th))) / 2
        # self.w = 1 / (1 + np.exp(k * (Cl_in_avg - th)))  # Sigmoid 
        # # u = 6.7
        # # sigma = 0.1
        # # w = 1 / (1 + np.exp((Cl_in_avg - u) / sigma))
        self.w = 0.94 # Wcon =0.94, generate spiral waves
        adaptive_up = up * self.w
        Cl_in_inf = adaptive_up * self.tau_Cl + self.Cl_in_eq
        self.Cl_in = Cl_in_inf + (self.Cl_in - Cl_in_inf) * np.exp(-dt / self.tau_Cl)
        
        # test
        # test_Cl_in = self.Cl_in
          
        # Update equation 4 (sAHP dynamics)
        self.g_K = self.g_K * np.exp(-dt / self.tau_K) + self.g_K_max * self.R_f * dt / self.tau_K
        
        self.I_K = self.g_K * (self.V - self.E_K)
        
        # test
        # test_g_K = self.g_K
          
        # Generate rate
        self.R_f = self.f(self.V - self.phi)
        
        # test_R_f = self.R_f
 
        ######### Short-term plasticity variables ########
        for i in range(len(self.Proj_Out)):
            self.Proj_Out[i].Value = self.R_f * dt

        #test
        # test_Proj_Out0_Value = self.Proj_Out[0].Value
        # test_Proj_Out1_Value = self.Proj_Out[0].Value
        # test_Proj_Out2_Value = self.Proj_Out[0].Value
        
        # # Filter the synaptic input
        # if np.all(tau_syn[i] == 0):
        #     self.Input_E = 0
        #     self.Input_I = 0
        # else:
        self.Input_E *= np.exp(-dt / self.tau_syn_E)
        self.Input_I *= np.exp(-dt / self.tau_syn_I)

        # test
        # test_Input_E = self.Input_E  
        # test_Input_I = self.Input_I

        # Update time
        self.t += dt
        
        # Test
        # test_t = self.t
    
    
# the main function to testify the class is correct or not 
# if __name__ == '__main__': 
#     # n = np.array([[100, 100]])
#     n =[100, 100]
#     model = MeanFieldModel(n)
#     xx = model.gen_param(-58, 0.01, n, 0)
#     yy = model.IndividualModelUpdate()
#     R = model.CreateRecorder( N = 10000 )
    
