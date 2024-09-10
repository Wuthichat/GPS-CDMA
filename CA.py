SV = {
   1: [2,6],
   2: [3,7],
   3: [4,8],
   4: [5,9],
   5: [1,9],
   6: [2,10],
   7: [1,8],
   8: [2,9],
   9: [3,10],
  10: [2,3]
  }

def generate_G1():
    OutputBit = []  
    G1 = [1] * 10   
    for i in range(100):
        NewBits = G1[2] ^ G1[9]  # "^" --> XOR 
        G1.insert(0, NewBits)    
        OutputBit.append(G1.pop())  
    return OutputBit
    
def generate_G2():
    OutputBit = []
    G2 = [1]*10
    for i in range( 100):
        NewBits = G2[1] ^ G2[2] ^ G2[5] ^ G2[7] ^ G2[8] ^ G2[9]
        G2.insert(0,NewBits)
        G2.pop()
        OutputBit.append(G2[SV[2][0]]^G2[SV[2][1]]) #PRN --> N
    return OutputBit

def generate_GoldCode():
    G1 = generate_G1()
    G2 = generate_G2()
    CA = []
    for i in range(len(G1)):
        CA.append(G1[i] ^ G2[i])
    return CA
    
print(generate_GoldCode())
#print(SV[1][0])