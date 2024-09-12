import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math


Nbits = 10
Nsamp = 100
M = 2
f = 2;
t = np.arange(0, 1, 1/(f*Nsamp))
cos_t = np.cos(2*np.pi*f*t)

SV = {
   1: [1, 5],
   2: [2, 6],
   3: [3, 7],
   4: [4, 8],
   5: [0, 8],
   6: [1, 9],
   7: [0, 7],
   8: [1, 8],
   9: [2, 9],
  10: [1, 2]
}

def generate_G1():
    OutputBit = []  
    G1 = [1] *10   
    for i in range(10):
        NewBits = G1[2] ^ G1[9]  # XOR operation
        G1.insert(0, NewBits)    
        OutputBit.append(G1.pop())  
    return OutputBit
    
def generate_G2(tap):
    OutputBit = []
    G2 = [1]*10
    for i in range(10):
        NewBits = G2[1] ^ G2[2] ^ G2[5] ^ G2[7] ^ G2[8] ^ G2[9]
        G2.insert(0, NewBits)
        G2.pop()
        OutputBit.append(G2[SV[tap][0]] ^ G2[SV[tap][1]])  # PRN code generation
    return OutputBit

def generate_GoldCode(tapA):
    G1 = generate_G1()
    G2 = generate_G2(tapA)
    CA = []
    for i in range(len(G1)):
        CA.append(G1[i] ^ G2[i])
    return CA

gold_code_1 = generate_GoldCode(1)
gold_code_2 = generate_GoldCode(2)
gold_code_3 = generate_GoldCode(3)

# Input sequences
a1 = np.array([1,0,0,1])
a2 = np.array([1,1,0,1])
a3 = np.array([1,0,1,1])

a1_New = []
a2_New = []
a3_New = []

def compute_values(input_array, gold_code):
    values = []
    for i in range(len(input_array)):
        for j in range(len(gold_code)):
          #print(input_array[i] ^ gold_code[j])
          values.append((-1)**(1+(input_array[i] ^ gold_code[j])))
    return values

a1_New = compute_values(a1, gold_code_1)
a2_New = compute_values(a2, gold_code_2)
a3_New = compute_values(a3, gold_code_3)

def modulate_signal(input_array, cos_t, Nbits, f, Nsamp):
    x_t = []
    for i in range(Nbits):
        x_t.extend(input_array[i] * cos_t)
    return x_t

tt = np.arange(0, Nbits, 1/(f*Nsamp))

x_t1 = []
x_t2 = []
x_t3 = []

x_t1 = modulate_signal(a1_New, cos_t, Nbits, f, Nsamp)
x_t2 = modulate_signal(a2_New, cos_t, Nbits, f, Nsamp)
x_t3 = modulate_signal(a3_New, cos_t, Nbits, f, Nsamp)

CDMA_signal = np.array(x_t1) + np.array(x_t2) + np.array(x_t3)

plt.figure(figsize=(10, 8))

ax1 = plt.subplot(411)
ax1.plot(tt, modulate_signal(a1_New, cos_t, Nbits, f, Nsamp))
plt.ylabel('User1 BPSK')
plt.grid(True)
ax1.set_xticks(np.arange(0, Nbits + 1, 1))   

ax2 = plt.subplot(412, sharex=ax1)
ax2.plot(tt, modulate_signal(a2_New, cos_t, Nbits, f, Nsamp))
plt.ylabel('User2 BPSK')
plt.grid(True)
ax2.set_xticks(np.arange(0, Nbits + 1, 1))  

ax3 = plt.subplot(413, sharex=ax1)
ax3.plot(tt, modulate_signal(a3_New, cos_t, Nbits, f, Nsamp))
plt.ylabel('User3 BPSK')
plt.grid(True)
ax3.set_xticks(np.arange(0, Nbits + 1, 1))  

ax4 = plt.subplot(414, sharex=ax1)
ax4.plot(tt, CDMA_signal)
plt.xlabel('Nbit')
plt.ylabel('CMDA')
plt.grid(True)
ax3.set_xticks(np.arange(0, Nbits + 1, 1))  

plt.tight_layout()
plt.show()