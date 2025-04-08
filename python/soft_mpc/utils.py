import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy

FILTER_ORDER = 2
N_FILTER     = 50

dir_path = os.path.dirname(os.path.realpath(__file__))
with open(dir_path+'/butterworth_table_order_2.yml', 'r') as file:
    TABLE = np.array(yaml.safe_load(file)['table'])



class ExpMovingAvg:
    def __init__(self, fc=100, fs=1000):
        
        self.fc     = fc
        self.fs     = fs
        self.poly_coef = 2*(1-np.cos(2*np.pi*self.fc/self.fs))
        self.alpha = ( -self.poly_coef + np.sqrt(self.poly_coef**2 + 4*self.poly_coef) ) / 2
        # very close to 1-np.exp(-2*np.pi*self.fc/self.fs)
        # see https://blog.mbedded.ninja/programming/signal-processing/digital-filters/exponential-moving-average-ema-filter/#fn:2
        print('alpha = ', self.alpha)
        self.raw  = np.zeros(2)
        self.filt = 0.

    def filter(self, raw):
        self.filt = self.alpha * raw + (1 - self.alpha) * self.filt
        # self.prev = raw.copy()
        return self.filt.copy()
    
    
class LPFButterOrder1:
    def __init__(self, fc=100, fs=1000):
        
        self.fc     = fc
        self.fs     = fs
        self.cutoff = 2*self.fc/self.fs

        b, a = scipy.signal.iirfilter(1, Wn=self.cutoff, btype="low", ftype="butter")
        
        self.a1 = a[1]
        self.b0 = b[0]
        self.b1 = b[1]
        
        self.raw  = np.zeros(2)
        self.prev = 0
        self.filt = 0.

    def filter(self, raw):
        self.filt = self.b0 * raw + self.b1 * self.prev - self.a1 * self.filt
        self.prev = raw.copy()
        return self.filt.copy()
    

class LPFButterOrder2:
    def __init__(self, fc=100, fs=1000):

        # see derivation of the digital filter + relation between cutoff and coefs here : http://www.kwon3d.com/theory/filtering/fil.html
        # see also Section 2.2.4.4, p.38 of David A. Winter, BIOMECHANICS AND MOTOR CONTROL OF HUMAN MOVEMENT (4th Edition)
        self.fc     = fc
        self.fs     = fs
        self.cutoff = 2*self.fc/self.fs

        self.filt   = np.zeros(2)
        self.raw    = np.zeros(3)

        self.table = TABLE 
        # coef can also be found with scipy.signal.iir(2, Wn=1./self.cutoff, btype="low", ftype="butter")
        # matches the table in the following order : ( a[0] a[1] a[2] b[0] b[1] b[2] ) 
        b, a = scipy.signal.iirfilter(2, Wn=self.cutoff, btype="low", ftype="butter")
        
        self.a1 = a[1] 
        self.a2 = a[2] 
        self.b0 = b[0] 
        self.b1 = b[1] 
        self.b2 = b[2] 

    def filter(self, raw):
        self.raw[2] = raw.copy()
        self.filt[1] = self.b0 * self.raw[2] + \
                       self.b1 * self.raw[1] + \
                       self.b2 * self.raw[0] - \
                       self.a1 * self.filt[1] - \
                       self.a2 * self.filt[0]
        self.raw[0]  = self.raw[1].copy()
        self.raw[1]  = self.raw[2].copy()
        self.filt[0] = self.filt[1].copy()
        return self.filt[1].copy()
    

class LPFButterOrder3:
    def __init__(self, fc=100, fs=1000):
        
        self.fc     = fc
        self.fs     = fs
        self.cutoff = 2*self.fc/self.fs

        self.filt   = np.zeros(3)
        self.raw    = np.zeros(4)

        self.table = TABLE 
        b, a = scipy.signal.iirfilter(3, Wn=self.cutoff, btype="low", ftype="butter")
        
        self.a1 = a[1] 
        self.a2 = a[2] 
        self.a3 = a[3] 
        self.b0 = b[0] 
        self.b1 = b[1] 
        self.b2 = b[2] 
        self.b3 = b[3] 

    def filter(self, raw):
        self.raw[3] = raw.copy()
        self.filt[2] = self.b0 * self.raw[3] + \
                       self.b1 * self.raw[2] + \
                       self.b2 * self.raw[1] + \
                       self.b3 * self.raw[0] - \
                       self.a1 * self.filt[2] - \
                       self.a2 * self.filt[1] - \
                       self.a3 * self.filt[0]
        self.raw[0]  = self.raw[1].copy()
        self.raw[1]  = self.raw[2].copy()
        self.raw[2]  = self.raw[3].copy()
        self.filt[0] = self.filt[1].copy()
        self.filt[1] = self.filt[2].copy()
        return self.filt[2].copy()
    
    
    
# # sin wave with noise
# N_SAMPLES = 1000
# DT        = 1e-3
# t = np.linspace(0, (N_SAMPLES-1)*DT, N_SAMPLES)
# sig = np.sin(2*2*np.pi*t)
# noise =  0.4*np.random.rand(N_SAMPLES) + 0.1*np.sin(50*2*np.pi*t) + 0.1*np.cos(15*2*np.pi*t)   
# data = sig + noise


# CUTOFF = 4.1
# CUTOFF2 = 20 #7
# CUTOFF3 = 20
# FILTER0 = ExpMovingAvg(fc=CUTOFF, fs=1./DT)
# FILTER1 = LPFButterOrder1(fc=CUTOFF, fs=1./DT)
# FILTER2 = LPFButterOrder2(fc=CUTOFF2, fs=1./DT)
# FILTER3 = LPFButterOrder3(fc=CUTOFF3, fs=1./DT)
# data_filtered0 = np.zeros(data.shape)
# data_filtered1 = np.zeros(data.shape)
# data_filtered2 = np.zeros(data.shape)
# data_filtered3 = np.zeros(data.shape)
# for i in range(N_SAMPLES):
#     data_filtered0[i] = FILTER0.filter(data[i])
#     data_filtered1[i] = FILTER1.filter(data[i])
#     data_filtered2[i] = FILTER2.filter(data[i])
#     data_filtered3[i] = FILTER3.filter(data[i])
# plt.plot(data, label='Unfiltered', color = 'b', alpha=0.3)
# plt.plot(data_filtered0, label='Exponential MA', color='y', linewidth=3)
# plt.plot(data_filtered1, label='1st order ButterLPF', color='r', linewidth=3)
# plt.plot(data_filtered2, label='2nd order ButterLPF', color='g', linewidth=3)
# plt.plot(data_filtered3, label='3rd order ButterLPF', color='k', linestyle='-', linewidth=3)
# plt.legend(fontsize=20)
# plt.show()