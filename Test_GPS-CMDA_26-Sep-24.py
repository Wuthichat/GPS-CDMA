print("Test GPS-CMDA 26-Sep-24....")

import numpy as np
import matplotlib.pyplot as plt
import math

Nbits = 13
Nsamp = 10
M = 2
f = 2
t = np.arange(0, 1, 1/(f*Nsamp))
cos_t = np.cos(2*np.pi*f*t)

SNR_dB_range = np.linspace(0, 20, 10)  # SNR range in dB
BER_user1 = []
BER_user2 = []
BER_user3 = []

num = 20

SV = {
   1: [1, 5],
   2: [2, 6],
   3: [3, 7]
}

# generate_GoldCode And CDMA signal for 3user. *-------------------*
def generate_G1():
    OutputBit = []
    G1 = [1] *10
    for i in range(num):
        NewBits = G1[2] ^ G1[9]  # XOR operation
        G1.insert(0, NewBits)
        OutputBit.append(G1.pop())
    return OutputBit

def generate_G2(tap):
    OutputBit = []
    G2 = [1]*10
    for i in range(num):
        NewBits = G2[1] ^ G2[2] ^ G2[5] ^ G2[7] ^ G2[8] ^ G2[9]
        G2.insert(0, NewBits)
        G2.pop()
        OutputBit.append(G2[SV[tap][0]] ^ G2[SV[tap][1]])
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
a1 = np.random.randint(0, 2, Nbits)
a2 = np.random.randint(0, 2, Nbits)
a3 = np.random.randint(0, 2, Nbits)

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

CDMA_signal = np.array(a1_New) + np.array(a2_New) + np.array(a3_New)  #TX

# plot signal user1,2,3 and CDMA signal *-------------------*
ax = plt.subplot(4, 1, 1)
ax.step(np.arange(11), a1_New[:11], where='pre')
plt.xticks(np.arange(0, 11, 1))
plt.title("Transmitted Signal")
plt.xlabel("bits")
plt.ylabel("USER_1")
plt.grid(True)

ax2 = plt.subplot(4, 1, 2)
ax2.step(np.arange(11), a2_New[:11], where='pre')
plt.xticks(np.arange(0, 11, 1))
plt.xlabel("bits")
plt.ylabel("USER_2")
plt.grid(True)

ax3 = plt.subplot(4, 1, 3)
ax3.step(np.arange(11), a3_New[:11], where='pre')
plt.xticks(np.arange(0, 11, 1))
plt.xlabel("bits")
plt.ylabel("USER_3")
plt.grid(True)

ax4 = plt.subplot(4, 1, 4)
ax4.step(np.arange(11), CDMA_signal[:11], where='pre')
plt.xticks(np.arange(0, 11, 1))
plt.xlabel("bits")
plt.ylabel("CDMA")
plt.grid(True)
plt.tight_layout()
plt.show()

# generate_Noise(AWGN) to CH. *-------------------*
mu = 0
sigma = 5
n_t = np.random.normal(mu, sigma, np.size(CDMA_signal) )
r_t = CDMA_signal + n_t

# plot Transmitted Signal Noise And receiver *-------------------*
ax5 = plt.subplot(3, 1, 1)
ax5.step(np.arange(11), CDMA_signal[:11], where='pre')
plt.xticks(np.arange(0, 11, 1))
plt.title("Transmitted Signal")
plt.xlabel("Sample Index")
plt.ylabel("Bit Value")
plt.grid(True)

ax6 = plt.subplot(3, 1, 2)
ax6.plot(n_t[:11])
plt.title("Noise Signal")

ax7 = plt.subplot(3, 1, 3)
ax7.step(np.arange(11), r_t[:11],where = 'pre')
plt.xticks(np.arange(0, 11, 1))
plt.title("Received Signal")
plt.xlabel("Sample Index")
plt.ylabel("Bit Value")
plt.grid(True)

plt.tight_layout()
plt.show()

# Receiver Correlator. *-------------------*
def receiver(signal, PRN):
    signal_new = []
    a_hat = []
    for i in range(len(signal)):
        signal_new.append(signal[i] * PRN[i%len(PRN)])
    for j in range(0,len(signal_new),num):
        a_hat.append(sum(signal_new[j:j+num]))

    return a_hat
receiver1 = receiver(r_t, gold_code_1)
receiver2 = receiver(r_t, gold_code_2)
receiver3 = receiver(r_t, gold_code_3)

#print(receiver1)
#print(receiver2)
#print(receiver3)
ax8 = plt.subplot(3, 1, 1)
x1 = np.linspace(0, 10, len(receiver1))
ax8.stem(x1, receiver1)
plt.title("Receiver 1")
plt.ylabel("Bit Value")
plt.grid(True)

ax9 = plt.subplot(3,1,2)
x2 = np.linspace(0, 10, len(receiver2))
ax9.stem(x2, receiver2)
plt.title("Receiver 2")
plt.ylabel("Bit Value")
plt.grid(True)

ax10 = plt.subplot(3,1,3)
x3 = np.linspace(0, 10, len(receiver3))
ax10.stem(x3, receiver3)
plt.title("Receiver 3")
plt.ylabel("Bit Value")
plt.grid(True)

plt.tight_layout()
plt.show()

# Decision. *-------------------*
def decision(a_hat):
    decision = []
    for i in range(len(a_hat)):
        if a_hat[i] >= 0:
            decision.append(0)
        else:
            decision.append(1)
    return decision

decision_1 = decision(receiver1)
decision_2 = decision(receiver2)
decision_3 = decision(receiver3)

print((f"user1 demod --> {decision_1}"))
print((f"user2 demod --> {decision_2}"))
print((f"user3 demod --> {decision_3}"))

# Error Rate Cal. *-------------------*
def error_rate(a, a_hat):
    error = 0
    errors = []
    for i in range(len(a)):
        if a[i] != a_hat[i]:
            error += 1
        errors.append(error / (i + 1))  
    error_rate = error / len(a)
    return error_rate

print(f"Error rate user 1: {error_rate(a1, decision_1)}")
print(f"Error rate user 2: {error_rate(a2, decision_2)}")
print(f"Error rate user 3: {error_rate(a3, decision_3)}")

#Autocorrelate. *-------------------*
def autocorrelate():
    for i in SV.keys():
        CA = generate_GoldCode(i)
        for j in range(len(CA)):
            CA[j] = (-1)**(1 + CA[j])

        # Autocorrelation
        Correlate = np.correlate(CA, CA, mode='full')

        # zoom in
        mid_point = len(Correlate) // 2
        zoom_range = 50 

        # Slice ข้อมูล
        zoomed_Correlate = Correlate[mid_point - zoom_range:mid_point + zoom_range]
        zoomed_lag = np.arange(-zoom_range, zoom_range)
        #plot
        plt.plot(zoomed_lag, zoomed_Correlate)
        plt.title(f'Zoomed-in Autocorrelation Plot: {i}')
        plt.xlabel('Lag')
        plt.ylabel('Correlation')
        plt.grid(True)
        plt.show()


#autocorrelate()

