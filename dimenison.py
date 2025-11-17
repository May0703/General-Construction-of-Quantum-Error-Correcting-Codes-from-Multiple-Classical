# This code calculates the code dimension of the four cases defined in the paper:
# General Construction of Quantum Error-Correcting Codes from Multiple Classical Codes
# Author: Yue Wu
# Date: November 2025

import numpy as np
import galois
import math

def calculate_code_dimension(L1,L2,L3,case):
    '''
    subtract the ranks of HX from the kernel dimension of the HZ to get the code dimension
    '''
    
    HZ, HX = generate_HX_HZ(L1,L2,L3,case)
    GF = galois.GF(2)
    HZ = GF(HZ)
    LZ = HZ.null_space()
    dKZ = np.shape(LZ)[0]

    nX = np.shape(HX)[0]
    HXT = GF(np.transpose(HX))
    LXT = HXT.null_space()
    dKXT = np.shape(LXT)[0]

    k = dKZ - (nX -dKXT)

    return k

def generate_HX_HZ(L1,L2,L3,case):
    nc1 = nb1 = L1 
    nc2 = nb2 = L2
    nc3 = nb3 = L3
    H1 = np.identity(L1)
    H1 = H1 + np.roll(H1,-1,axis=1)
    H2 = np.identity(L2)
    H2 = H2 + np.roll(H2,-1,axis=1)
    H3 = np.identity(L3)
    H3 = H3 + np.roll(H3,-1,axis=1)
    I1c = np.identity(nc1)
    I2c = np.identity(nc2)
    I3c = np.identity(nc3)
    I1b = np.identity(nb1)
    I2b = np.identity(nb2)
    I3b = np.identity(nb3)  

    if case == 1:
        HZ = np.zeros((nc1*nc2*nb3+nc1*nb2*nc3+nb1*nc2*nc3,nb1*nb2*nc3+nc1*nb2*nb3+nb1*nc2*nb3),dtype=int)
        HZ[:nc1*nc2*nb3,nb1*nb2*nc3:(nb1*nb2*nc3+nc1*nb2*nb3)] = np.kron(np.kron(I1c,H2),I3b)
        HZ[:nc1*nc2*nb3,(nb1*nb2*nc3+nc1*nb2*nb3):(nb1*nb2*nc3+nc1*nb2*nb3+nb1*nc2*nb3)] = np.kron(np.kron(H1,I2c),I3b)

        HZ[nc1*nc2*nb3:(nc1*nc2*nb3+nc1*nb2*nc3),:nb1*nb2*nc3] = np.kron(np.kron(H1,I2b),I3c)
        HZ[nc1*nc2*nb3:(nc1*nc2*nb3+nc1*nb2*nc3),nb1*nb2*nc3:(nb1*nb2*nc3+nc1*nb2*nb3)] = np.kron(np.kron(I1c,I2b),H3)

        HZ[(nc1*nc2*nb3+nc1*nb2*nc3):(nc1*nc2*nb3+nc1*nb2*nc3+nb1*nc2*nc3),:nb1*nb2*nc3] = np.kron(np.kron(I1b,H2),I3c)
        HZ[(nc1*nc2*nb3+nc1*nb2*nc3):(nc1*nc2*nb3+nc1*nb2*nc3+nb1*nc2*nc3),(nb1*nb2*nc3+nc1*nb2*nb3):(nb1*nb2*nc3+nc1*nb2*nb3+nb1*nc2*nb3)] = np.kron(np.kron(I1b,I2c),H3)

        HX = np.zeros((nb1*nb2*nb3,nb1*nb2*nc3+nc1*nb2*nb3+nb1*nc2*nb3),dtype=int)
        HX[:,:nb1*nb2*nc3] = np.kron(np.kron(I1b,I2b),np.transpose(H3))
        HX[:,nb1*nb2*nc3:(nb1*nb2*nc3+nc1*nb2*nb3)] = np.kron(np.kron(np.transpose(H1),I2b),I3b)
        HX[:,(nb1*nb2*nc3+nc1*nb2*nb3):(nb1*nb2*nc3+nc1*nb2*nb3+nb1*nc2*nb3)] = np.kron(np.kron(I1b,np.transpose(H2)),I3b)
    elif case == 2:
        HZ = np.zeros((nc1*nc2*nc3+nb1*nb2*nc3,nb1*nb2*nb3+nb1*nc2*nc3+nc1*nb2*nc3+nc1*nc2*nb3),dtype=int)
        HZ[:nc1*nc2*nc3,nb1*nb2*nb3:(nb1*nb2*nb3+nb1*nc2*nc3)] = np.kron(np.kron(H1,I2c),I3c)
        HZ[:nc1*nc2*nc3,(nb1*nb2*nb3+nb1*nc2*nc3):(nb1*nb2*nb3+nb1*nc2*nc3+nc1*nb2*nc3)] = np.kron(np.kron(I1c,H2),I3c)
        HZ[:nc1*nc2*nc3,(nb1*nb2*nb3+nb1*nc2*nc3+nc1*nb2*nc3):(nb1*nb2*nb3+nb1*nc2*nc3+nc1*nb2*nc3+nc1*nc2*nb3)] = np.kron(np.kron(I1c,I2c),H3)

        HZ[nc1*nc2*nc3:(nc1*nc2*nc3+nb1*nb2*nc3),:nb1*nb2*nb3] = np.kron(np.kron(I1b,I2b),H3)
        HZ[nc1*nc2*nc3:(nc1*nc2*nc3+nb1*nb2*nc3),nb1*nb2*nb3:(nb1*nb2*nb3+nb1*nc2*nc3)] = np.kron(np.kron(I1b,np.transpose(H2)),I3c)
        HZ[nc1*nc2*nc3:(nc1*nc2*nc3+nb1*nb2*nc3),(nb1*nb2*nb3+nb1*nc2*nc3):(nb1*nb2*nb3+nb1*nc2*nc3+nc1*nb2*nc3)] = np.kron(np.kron(np.transpose(H1),I2b),I3c)

        HX = np.zeros((nc1*nb2*nb3+nb1*nc2*nb3,nb1*nb2*nb3+nb1*nc2*nc3+nc1*nb2*nc3+nc1*nc2*nb3),dtype=int)
        HX[:nc1*nb2*nb3,:nb1*nb2*nb3] = np.kron(np.kron(H1,I2b),I3b)
        HX[:nc1*nb2*nb3,(nb1*nb2*nb3+nb1*nc2*nc3):(nb1*nb2*nb3+nb1*nc2*nc3+nc1*nb2*nc3)] = np.kron(np.kron(I1c,I2b),np.transpose(H3))
        HX[:nc1*nb2*nb3,(nb1*nb2*nb3+nb1*nc2*nc3+nc1*nb2*nc3):(nb1*nb2*nb3+nb1*nc2*nc3+nc1*nb2*nc3+nc1*nc2*nb3)] = np.kron(np.kron(I1c,np.transpose(H2)),I3b)

        HX[nc1*nb2*nb3:(nc1*nb2*nb3+nb1*nc2*nb3),:nb1*nb2*nb3] = np.kron(np.kron(I1b,H2),I3b)
        HX[nc1*nb2*nb3:(nc1*nb2*nb3+nb1*nc2*nb3),nb1*nb2*nb3:(nb1*nb2*nb3+nb1*nc2*nc3)] = np.kron(np.kron(I1b,I2c),np.transpose(H3))
        HX[nc1*nb2*nb3:(nc1*nb2*nb3+nb1*nc2*nb3),(nb1*nb2*nb3+nb1*nc2*nc3+nc1*nb2*nc3):(nb1*nb2*nb3+nb1*nc2*nc3+nc1*nb2*nc3+nc1*nc2*nb3)] = np.kron(np.kron(np.transpose(H1),I2c),I3b)
    elif case == 3:
        HZ = np.zeros((nc1*nc2*nb3+nc1*nb2*nc3+nb1*nc2*nc3,nb1*nb2*nc3+nc1*nb2*nb3+nb1*nc2*nb3+nc1*nc2*nc3),dtype=int)
        HZ[:nc1*nc2*nb3,:nb1*nb2*nc3] = np.kron(np.kron(H1,H2),np.transpose(H3))
        HZ[:nc1*nc2*nb3,nb1*nb2*nc3:(nb1*nb2*nc3+nc1*nb2*nb3)] = np.kron(np.kron(I1c,H2),I3b)
        HZ[:nc1*nc2*nb3,(nb1*nb2*nc3+nc1*nb2*nb3):(nb1*nb2*nc3+nc1*nb2*nb3+nb1*nc2*nb3)] = np.kron(np.kron(H1,I2c),I3b)
        HZ[:nc1*nc2*nb3,(nb1*nb2*nc3+nc1*nb2*nb3+nb1*nc2*nb3):(nb1*nb2*nc3+nc1*nb2*nb3+nb1*nc2*nb3+nc1*nc2*nc3)] = np.kron(np.kron(I1c,I2c),np.transpose(H3))

        HZ[nc1*nc2*nb3:(nc1*nc2*nb3+nc1*nb2*nc3),:nb1*nb2*nc3] = np.kron(np.kron(H1,I2b),I3c)
        HZ[nc1*nc2*nb3:(nc1*nc2*nb3+nc1*nb2*nc3),nb1*nb2*nc3:(nb1*nb2*nc3+nc1*nb2*nb3)] = np.kron(np.kron(I1c,I2b),H3)
        HZ[nc1*nc2*nb3:(nc1*nc2*nb3+nc1*nb2*nc3),(nb1*nb2*nc3+nc1*nb2*nb3):(nb1*nb2*nc3+nc1*nb2*nb3+nb1*nc2*nb3)] = np.kron(np.kron(H1,np.transpose(H2)),H3)
        HZ[nc1*nc2*nb3:(nc1*nc2*nb3+nc1*nb2*nc3),(nb1*nb2*nc3+nc1*nb2*nb3+nb1*nc2*nb3):(nb1*nb2*nc3+nc1*nb2*nb3+nb1*nc2*nb3+nc1*nc2*nc3)] = np.kron(np.kron(I1c,np.transpose(H2)),I3c)

        HZ[(nc1*nc2*nb3+nc1*nb2*nc3):(nc1*nc2*nb3+nc1*nb2*nc3+nb1*nc2*nc3),:nb1*nb2*nc3] = np.kron(np.kron(I1b,H2),I3c)
        HZ[(nc1*nc2*nb3+nc1*nb2*nc3):(nc1*nc2*nb3+nc1*nb2*nc3+nb1*nc2*nc3),nb1*nb2*nc3:(nb1*nb2*nc3+nc1*nb2*nb3)] = np.kron(np.kron(np.transpose(H1),H2),H3)
        HZ[(nc1*nc2*nb3+nc1*nb2*nc3):(nc1*nc2*nb3+nc1*nb2*nc3+nb1*nc2*nc3),(nb1*nb2*nc3+nc1*nb2*nb3):(nb1*nb2*nc3+nc1*nb2*nb3+nb1*nc2*nb3)] = np.kron(np.kron(I1b,I2c),H3)
        HZ[(nc1*nc2*nb3+nc1*nb2*nc3):(nc1*nc2*nb3+nc1*nb2*nc3+nb1*nc2*nc3),(nb1*nb2*nc3+nc1*nb2*nb3+nb1*nc2*nb3):(nb1*nb2*nc3+nc1*nb2*nb3+nb1*nc2*nb3+nc1*nc2*nc3)] = np.kron(np.kron(np.transpose(H1),I2c),I3c)

        HX = np.zeros((nb1*nb2*nb3,nb1*nb2*nc3+nc1*nb2*nb3+nb1*nc2*nb3+nc1*nc2*nc3),dtype=int)
        HX[:,:nb1*nb2*nc3] = np.kron(np.kron(I1b,I2b),np.transpose(H3))
        HX[:,nb1*nb2*nc3:(nb1*nb2*nc3+nc1*nb2*nb3)] = np.kron(np.kron(np.transpose(H1),I2b),I3b)
        HX[:,(nb1*nb2*nc3+nc1*nb2*nb3):(nb1*nb2*nc3+nc1*nb2*nb3+nb1*nc2*nb3)] = np.kron(np.kron(I1b,np.transpose(H2)),I3b)
        HX[:,(nb1*nb2*nc3+nc1*nb2*nb3+nb1*nc2*nb3):(nb1*nb2*nc3+nc1*nb2*nb3+nb1*nc2*nb3+nc1*nc2*nc3)] = np.kron(np.kron(np.transpose(H1),np.transpose(H2)),np.transpose(H3))
    elif case == 4:
        HZ = np.zeros((nc1*nc2*nc3+nb1*nb2*nc3,nb1*nb2*nb3+nb1*nc2*nc3+nc1*nb2*nc3+nc1*nc2*nb3),dtype=int)
        HZ[:nc1*nc2*nc3,:nb1*nb2*nb3] = np.kron(np.kron(H1,H2),H3)
        HZ[:nc1*nc2*nc3,nb1*nb2*nb3:(nb1*nb2*nb3+nb1*nc2*nc3)] = np.kron(np.kron(H1,I2c),I3c)
        HZ[:nc1*nc2*nc3,(nb1*nb2*nb3+nb1*nc2*nc3):(nb1*nb2*nb3+nb1*nc2*nc3+nc1*nb2*nc3)] = np.kron(np.kron(I1c,H2),I3c)
        HZ[:nc1*nc2*nc3,(nb1*nb2*nb3+nb1*nc2*nc3+nc1*nb2*nc3):(nb1*nb2*nb3+nb1*nc2*nc3+nc1*nb2*nc3+nc1*nc2*nb3)] = np.kron(np.kron(I1c,I2c),H3)

        HZ[nc1*nc2*nc3:(nc1*nc2*nc3+nb1*nb2*nc3),:nb1*nb2*nb3] = np.kron(np.kron(I1b,I2b),H3)
        HZ[nc1*nc2*nc3:(nc1*nc2*nc3+nb1*nb2*nc3),nb1*nb2*nb3:(nb1*nb2*nb3+nb1*nc2*nc3)] = np.kron(np.kron(I1b,np.transpose(H2)),I3c)
        HZ[nc1*nc2*nc3:(nc1*nc2*nc3+nb1*nb2*nc3),(nb1*nb2*nb3+nb1*nc2*nc3):(nb1*nb2*nb3+nb1*nc2*nc3+nc1*nb2*nc3)] = np.kron(np.kron(np.transpose(H1),I2b),I3c)
        HZ[nc1*nc2*nc3:(nc1*nc2*nc3+nb1*nb2*nc3),(nb1*nb2*nb3+nb1*nc2*nc3+nc1*nb2*nc3):(nb1*nb2*nb3+nb1*nc2*nc3+nc1*nb2*nc3+nc1*nc2*nb3)] = np.kron(np.kron(np.transpose(H1),np.transpose(H2)),H3)

        HX = np.zeros((nc1*nb2*nb3+nb1*nc2*nb3,nb1*nb2*nb3+nb1*nc2*nc3+nc1*nb2*nc3+nc1*nc2*nb3),dtype=int)
        HX[:nc1*nb2*nb3,:nb1*nb2*nb3] = np.kron(np.kron(H1,I2b),I3b)
        HX[:nc1*nb2*nb3,nb1*nb2*nb3:(nb1*nb2*nb3+nb1*nc2*nc3)] = np.kron(np.kron(H1,np.transpose(H2)),np.transpose(H3))
        HX[:nc1*nb2*nb3,(nb1*nb2*nb3+nb1*nc2*nc3):(nb1*nb2*nb3+nb1*nc2*nc3+nc1*nb2*nc3)] = np.kron(np.kron(I1c,I2b),np.transpose(H3))
        HX[:nc1*nb2*nb3,(nb1*nb2*nb3+nb1*nc2*nc3+nc1*nb2*nc3):(nb1*nb2*nb3+nb1*nc2*nc3+nc1*nb2*nc3+nc1*nc2*nb3)] = np.kron(np.kron(I1c,np.transpose(H2)),I3b)

        HX[nc1*nb2*nb3:(nc1*nb2*nb3+nb1*nc2*nb3),:nb1*nb2*nb3] = np.kron(np.kron(I1b,H2),I3b)
        HX[nc1*nb2*nb3:(nc1*nb2*nb3+nb1*nc2*nb3),nb1*nb2*nb3:(nb1*nb2*nb3+nb1*nc2*nc3)] = np.kron(np.kron(I1b,I2c),np.transpose(H3))
        HX[nc1*nb2*nb3:(nc1*nb2*nb3+nb1*nc2*nb3),(nb1*nb2*nb3+nb1*nc2*nc3):(nb1*nb2*nb3+nb1*nc2*nc3+nc1*nb2*nc3)] = np.kron(np.kron(np.transpose(H1),H2),np.transpose(H3))
        HX[nc1*nb2*nb3:(nc1*nb2*nb3+nb1*nc2*nb3),(nb1*nb2*nb3+nb1*nc2*nc3+nc1*nb2*nc3):(nb1*nb2*nb3+nb1*nc2*nc3+nc1*nb2*nc3+nc1*nc2*nb3)] = np.kron(np.kron(np.transpose(H1),I2c),I3b)

    return HZ, HX


pairs = []
for x in range(2, 10):
    for y in range(x, 10):  # y从x开始，避免重复
        pairs.append((x, y))

for L3 in range(2,10):
# 循环遍历这些对
    for L1, L2 in pairs:
        k = calculate_code_dimension(L1,L2,L3,4)
        k_theoretical = 4 * math.gcd(L1, L2) + 2 * (L3 - 1) * (math.gcd(L1,3) - 1) * (math.gcd(L2,3) - 1)
        print(f"L1={L1}, L2={L2}, L3={L3}, code dimension k={k}, theoretical k={k_theoretical}")
        if k != k_theoretical:
            print(f"Mismatch found for L1={L1}, L2={L2}, L3={L3}: calculated k={k}, theoretical k={k_theoretical}")