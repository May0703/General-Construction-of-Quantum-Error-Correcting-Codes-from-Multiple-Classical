# This code checks whether the weight-4 and weight-5 operators are logical by checking whether they belong to the stabilizer group.

import numpy as np
import galois

HX0 = np.zeros((3,14,4),int)
for i in range(3):
    ic = np.array(list(range(0,i))+list(range(i+1,3)))
    HX0[i,0,3] = 3
    for j in range(2):
        HX0[i,j+1,3] = ic[j]
    HX0[i,3,i] = -1
    HX0[i,3,3] = 3
    for j in range(2):
        HX0[i,4+j,ic[j]] = 1
        HX0[i,4+j,3] = ic[1-j]
    for j in range(8):
        HX0[i,6+j,i] = -(j//4)
        HX0[i,6+j,ic[0]] = j%2
        HX0[i,6+j,ic[1]] = (j//2)%2
        HX0[i,6+j,3] = i

def HX_f(Lx,Ly,Lz):
    HX = np.zeros((Lx,Ly,Lz,3,Lx,Ly,Lz,4),int)
    for i in range(Lx):
        for j in range(Ly):
            for k in range(Lz):
                for l in range(3):
                    for m in range(14):
                        di,dj,dk,n = HX0[l,m]
                        i1,j1,k1 = (np.array([i,j,k])+[di,dj,dk]) % [Lx,Ly,Lz]
                        HX[i,j,k,l,i1,j1,k1,n] = 1
    return HX.reshape((Lx*Ly*Lz*3,-1))

#Lx,Ly,Lz,n_sol = np.array(sys.argv[1:],int)
GF2 = galois.GF(2)
res = []
for Lx0 in range(2,21):
    for Ly0 in range(Lx0,21):
        for Lz0 in range(Ly0,21):
            print(Lx0,Ly0,Lz0,flush=True)
            Lx,Ly,Lz = Lx0,Ly0,Lz0
            if np.gcd(Lx0,Ly0) != 1:
                Lx,Ly,Lz = Lx0,Lz0,Ly0
            HX = GF2(HX_f(Lx,Ly,Lz)).T
            ker0 = HX.null_space().reshape((-1,Lx,Ly,Lz,3))

            print(ker0.shape[0])
            #np.save(f"data/ker_Lx_{Lx}_Ly_{Ly}_Lz_{Lz}",ker)

            logical = np.zeros((Lx,Ly,Lz,4),int)
            if np.gcd(Lx,Lz) != 1 or np.gcd(Ly,Lz) != 1:
                logical[0,0,0,0] = 1
                logical[0,0,0,1] = 1
                logical[1,0,0,1] = 1
                logical[0,1,0,0] = 1
            else:
                logical[0,0,0,2] = 1
                logical[0,0,0,3] = 1
                logical[1,0,0,2] = 1
                logical[0,1,0,2] = 1
                logical[1,1,0,2] = 1

            logical = GF2(logical.reshape((-1,1)))
            HX_log = np.hstack([logical,HX])
            #print(np.linalg.matrix_rank(HX_log))
            ker1 = HX_log.null_space()
            print(ker1.shape[0])
            res.append([Lx,Ly,Lz,ker0.shape[0],ker1.shape[0]])
np.save("data_logical2",np.array(res,int))
