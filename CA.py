SV = {
   1: [1,5],
   2: [2,6],
   3: [3,7],
   4: [4,8],
   5: [0,8],
   6: [1,9],
   7: [0,7],
   8: [1,8],
   9: [2,9],
  10: [1,2]
  }

def generate_G1():
    OutputBit = []  
    G1 = [1] * 10   
    for i in range(1023):
        NewBits = G1[2] ^ G1[9]  # "^" --> XOR 
        G1.insert(0, NewBits)    
        OutputBit.append(G1.pop())  
    return OutputBit
    
def generate_G2(tap):
    OutputBit = []
    G2 = [1]*10
    for i in range(1023):
        NewBits = G2[1] ^ G2[2] ^ G2[5] ^ G2[7] ^ G2[8] ^ G2[9]
        G2.insert(0,NewBits)
        G2.pop()
        OutputBit.append(G2[SV[tap][0]]^G2[SV[tap][1]]) #PRN --> N
    return OutputBit

def generate_GoldCode(tapA):
    G1 = generate_G1()
    G2 = generate_G2(tapA)
    CA = []
    for i in range(len(G1)):
        CA.append(G1[i] ^ G2[i])
    return CA

for i in range(1,len(SV)+1):
    print(f"generate_GoldCode:{i}",generate_GoldCode(i))


#print(f"generate_GoldCode {6}",generate_GoldCode(6))
