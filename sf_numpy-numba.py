import numpy as np
import numba
from numba import jit,prange


def Vpqrs(file,m):
    V = np.zeros([m,m,m,m])
    f = open(file, 'r')
    lines = f.readlines()
    line = 0
    for i in range(m):
        for j in range(m):
            for k in range(m):
                for l in range(m):
                    V[i,j,k,l] = float(lines[line].replace(" ", "").replace('(','').replace(')',"").split(',')[0])
                    line = line +1
    return V

def Hstar(file,m):
    H = np.zeros([m,m])
    f = open(file, 'r')
    lines = f.readlines()
    line = 0
    for i in range(m):
        for j in range(m):
            # print(float(lines[i+j][0:-21].strip()[1:]))
            H[i,j] = float(lines[line].replace(" ", "").replace('(','').replace(')',"").split(',')[0])
            line = line +1
            
    return H

@jit(nopython=True,fastmath=True)
def createInitialState(n,m,excite):
    state = np.zeros(n)
    if excite == 'Singlet' :
        for i in range(n):
            state[i] = i+1
    if excite == 'Triplet' :
        for i in range(n-1):
            state[i] = i+1
        state[n-1] = n+1
    return state
@jit(nopython=True,fastmath=True)
def odometer(state,n,m):
    newState = state.copy()
    l = 0
    for j in range(n-1,-1,-1):
        if newState[j] < m - n + j  :
            l = newState[j]
            for k in range(j,n):
                if newState[k]%2 == 0 and (l + 2 + k - j)%2 == 0 and (l + 2 + k - j) not in newState:
                    newState[k] = l + 2 + k - j 
                if newState[k]%2 != 0 and (l + 2 + k - j)%2 != 0 and (l + 2 + k - j) not in newState:
                    newState[k] = l + 2 + k - j 
            if newState[j] != l:
                return newState
    newState = np.zeros(n)
    return newState

def createBinaryState(state,n,m):
    binState = np.zeros(m+1,dtype=np.bool)
    for i in state.astype(int):
        binState[i-1] = True
    return binState

def createSlaterDeterminants(n,m,excite):
    state = createInitialState(n,m,excite).copy()
    binStates = []
    binStates.append(createBinaryState(state,n,m))
    while True :
        state = odometer(state,n,m)
        if np.sum(state) == 0:
            break
        binStates.append(createBinaryState(state,n,m))
    return binStates


def saveSlaterDeterminantsToDisk(binStates):
    filename = 'SlaterDeterminants.sd'
    SlaterDeterminants = open(filename,"w+") 
    for i in binStates:
        bitStr = str(i)[10:-2]
        SlaterDeterminants.write(bitStr+'\n')
    return

def load(filename):
    SlaterDeterminants = open(filename,"r")
    binStates=[]
    for i in SlaterDeterminants.readlines():
        binStates.append(bitarray(i))
    return binStates

@jit(nopython=True,fastmath=True)
def sign(n,binstate):
    if binstate[-1] == False:
        s = 1
        for i in range(n):
            if binstate[i] !=0 :
                s *= -1
        return s
    return 1
@jit(nopython=True,fastmath=True)
def addParticle(n,binstate):
    if binstate[-1] == False:
        a = binstate.copy()
        a[:] = False
        a[n-1] = True
        comp = np.bitwise_and(a,binstate)
        if np.sum(comp) == False :
            binstate[n-1] = True
        else :
            binstate[len(binstate)-1] = True
    return 
@jit(nopython=True,fastmath=True)
def removeParticle(n,binstate):
    if binstate[-1] == False:
        a = binstate.copy()
        a[:] = False
        a[n-1] = True
        comp = np.bitwise_and(a,binstate)
        if np.sum(comp) == True :
            binstate[n-1] = False
        else :
            binstate[len(binstate)-1] = True
    return 


@jit(nopython=True,fastmath=True)
def secondQuantizationOneBodyOperator(p,q,state1,state2):
    phase = 1

    removeParticle(q,state1)
    if state1[-1] == True:
        return 0
    phase *= sign(q,state1)

    addParticle(p,state1)
    if state1[-1]== True:
        return 0
    phase *= sign(p,state1)

    if state2 != state1:
        phase = 0
    return phase
@jit(nopython=True,fastmath=True)
def secondQuantizationTwoBodyOperator(p,q,r,s,state1,state2):
    phase = 1
    #  Removing r
    removeParticle(r,state1)
    if state1[-1]== True:
        return 0
    phase *= sign(r,state1)
    # Removing s
    removeParticle(s,state1)
    if state1[-1]== True:
        return 0
    phase *= sign(s,state1)
    # Adding q
    addParticle(q,state1)
    if state1[-1]== True:
        return 0
    phase *= sign(q,state1)
    # Adding p
    addParticle(p,state1)
    if state1[-1]== True:
        return 0
    phase *= sign(p,state1)
    if state1 != state2:
        return 0
    return phase

@jit(nopython=True,fastmath=True,parallel=True)
def computeHamiltonianMatrix(slaterDeterminants,V,Hone,M):
    nSlaterDeterminants = int(len(slaterDeterminants))
    nOrbitals = M
    H = np.zeros((nSlaterDeterminants,nSlaterDeterminants))

    for m in prange(nSlaterDeterminants):
        for n in prange(m,nSlaterDeterminants):
            #
            for p in prange(nOrbitals):
                for q in prange(nOrbitals):
                    # One body part

                    phase = 0
                    phase += secondQuantizationOneBodyOperator(2*p,2*q,slaterDeterminants[n],slaterDeterminants[m])
                    phase += secondQuantizationOneBodyOperator(2*p+1,2*q+1,slaterDeterminants[n],slaterDeterminants[m])
                    H[m,n] += Hone[p,q]*phase

                    # Two-body interaction

                    for r in prange(nOrbitals):
                        for s in prange(nOrbitals):
                            if V[p,q,r,s] != 0 :
                                phase = 0
                                phase += secondQuantizationTwoBodyOperator(2*p, 2*q, 2*r, 2*s,slaterDeterminants[n],slaterDeterminants[m])

                                phase += secondQuantizationTwoBodyOperator(2*p+1, 2*q+1, 2*r+1, 2*s+1,slaterDeterminants[n],slaterDeterminants[m])

                                phase += secondQuantizationTwoBodyOperator(2*p, 2*q+1, 2*r, 2*s+1,slaterDeterminants[n],slaterDeterminants[m])

                                phase += secondQuantizationTwoBodyOperator(2*p+1, 2*q, 2*r+1, 2*s,slaterDeterminants[n],slaterDeterminants[m])
                                H[m, n] += 0.5*V[p,q,r,s]*phase
            # H(n, m) = std::conj(H(m,n)); 
            if m != n :
                H[n,m] = np.conj(H[m,n])
    return H




### Config ###
n = 2
m = 30
excite = 'Singlet' # Singlet or Triplet
twoelectron_file = 'trnseemat.dat'
oneelectron_file = 'trnsh.dat'

######
# print(len(createSlaterDeterminants(n,m,excite)))

Hamiltonian = computeHamiltonianMatrix(createSlaterDeterminants(n,m,excite),Vpqrs(twoelectron_file,15),Hstar(oneelectron_file,15),15)
print(Hamiltonian)
# np.save('H.npy', Hamiltonian)
# Hamiltonian = np.load('H.npy')
# w,v = np.linalg.eigh(Hamiltonian)
# print(w,v)