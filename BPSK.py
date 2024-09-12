import numpy as np
import matplotlib.pyplot as plot
import math

Nbits = 10
Nsamp = 20
M = 2

#1. Generate data bits
np.random.seed(10)
a = np.random.randint(0,2,Nbits)
#a = np.array([1,0,0,1,1,0,1,0,0,0])
b = 2*a - 1
print(a)
print(b)
f = 2;
t = np.arange(0, 1, 1/(f*Nsamp))
cos_t = np.cos(2*np.pi*f*t)
plot.plot(t, cos_t)
# Modulate
x_t=[]
for i in range(Nbits):
   x_t.extend(b[i]*cos_t)
#  if(a[i]==1)
#    x_t.extend(cos_t)
#  else:
#    x_t.extend(-1*cos_t)

#print(x_t)
tt = np.arange(0, Nbits, 1/(f*Nsamp))
plot.plot(tt, x_t)
