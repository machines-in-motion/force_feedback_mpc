import numpy as np
import matplotlib.pyplot as plt

# Smoothing factor
dt = 1e-2
fsmp = 1./dt   # Sampling frequency


fmax = fsmp # Cutoff frequency (constrained by Shannon ?)
fc = np.linspace(0, fmax, 1000)
# EMA
alpha_0 = np.exp(-2*np.pi*fc*dt) # Exp MA
# Euler
alpha_1 = 1./(1+2*np.pi*fc*dt)   # Wiki
# Exact
y = np.cos(2*np.pi*fc*dt)
alpha_2 = 1-(y-1+np.sqrt(y**2 - 4*y +3)) # Stack
fig = plt.figure()
plt.plot(fc, alpha_0, label="EMA", linewidth=2.)
plt.plot(fc, alpha_1, label="Euler", linewidth=2.)
plt.plot(fc, alpha_2, label="Exact", linewidth=2.)
plt.xlabel('Cut-off frequency $f_s$ ($Hz$)', size=16)
plt.ylabel('Smoothing coefficient '+r'$\alpha$', size=16)
plt.title(r'$\alpha$'+' coefficient vs cut-off frequency at $f_s = 1000Hz$', size=16)
plt.grid()
plt.legend(prop={'size': 16})
plt.show()

# # Filtering 
# N = 1000
# t = np.linspace(0,1,N)
# freq_in = 20
# input_signal = np.sin(t*2*np.pi*freq_in)

# # Choose filter type 
# FC = 1
# # alpha = np.exp(-2*np.pi*FC*dt) # Exp MA
# # alpha = 1./(1+2*np.pi*FC*dt)   # Wiki
# y = np.cos(2*np.pi*FC*dt)
# alpha = 1-(y-1+np.sqrt(y**2 - 4*y +3)) # Stack
# output_signal = np.zeros(N)
# output_signal[0] = input_signal[0]
# for i in range(N-1):
#   output_signal[i+1] = alpha*output_signal[i] + (1.-alpha)*input_signal[i]

# plt.plot(t, input_signal, label='input')
# plt.plot(t, output_signal, label='output')
# plt.show()