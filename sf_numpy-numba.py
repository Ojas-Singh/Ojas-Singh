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
            H[i,j] = float(lines[line].replace(" ", "").replace('(','').replace(')',"").split(',')[0])
            line = line +1
            
    return H

@jit(nopython=True,fastmath=True)
def createInitialState(n,m,excite):
    if excite == 'Singlet' :
        if n%2 == 0 :
            stateUp = np.zeros(int(n/2))
            stateDown = np.zeros(int(n/2))
            for i in range(int(n/2)):
                stateUp[i] = i+1
                stateDown[i] = i+1
            return stateUp,stateDown
        else:
            stateUp = np.zeros(int(n/2)+1)
            stateDown = np.zeros(int(n/2))
            for i in range(int(n/2)):
                stateUp[i] = i+1
                stateDown[i] = i+1
            stateUp[-1] = int(n/2+1)
            return stateUp,stateDown
    # if excite == 'Triplet' :
    #     for i in range(n-1):
    #         state[i] = i+1
    #     state[n-1] = n+1

    return stateUp,stateDown

@jit(nopython=True,fastmath=True)
def odometer(state,n,m):
    n=n-1
    newState = state.copy()
    l = 0
    for j in range(n,-1,-1):
        if newState[j] < m - n + j  :
            l = newState[j]
            for k in range(j,n+1):
                newState[k] = l + 1 + k - j    
            if newState[j] != l:
                return newState
    newState = np.zeros(n)
    return newState

@jit(nopython=True,fastmath=True)
def createBinaryState(state,n,m,binState):
    # for i in state.astype(int):
    for i in state:
        binState[int(i)-1] = True
    return binState

def mix(Up,Down):
    state = [2*i-1 for i in Up] + [2*i for i in Down]
    return state
# @jit(nopython=False,fastmath=True)
def createSlaterDeterminants(n,m,excite):
    if n%2 == 0:
        N = int(n/2) 
    else: 
        N = int(n/2)+1
    stateUp = createInitialState(n,m,excite)[0].copy()
    stateDown = createInitialState(n,m,excite)[1].copy()
    statesUp = [stateUp.copy()]
    statesDown = [stateDown.copy()]
    binStates = []
    # binStates.append(createBinaryState(state,n,m))
    up = True
    down = True
    state = stateUp
    while up :
        state = odometer(state,N,m)
        if np.sum(state) == 0:
            up = False
        else:
            statesUp.append(state)
        # binStates.append(createBinaryState(state,n,m))
    state = stateDown
    while down :
        state = odometer(state,int(n/2),m)
        if np.sum(state) == 0:
            down = False
        else:
            statesDown.append(state)
    for i in statesUp:
        for j in statesDown:
            binStates.append(createBinaryState(mix(i,j),n,m,np.zeros(int(2*m+1),dtype=np.bool)))

    return binStates


def saveSlaterDeterminantsToDisk(binStates):
    filename = 'SlaterDeterminants.sd'
    SlaterDeterminants = open(filename,"w+") 
    for i in binStates:
        bitStr = str(np.array(i,dtype=int)).replace(" ","")[1:-1]
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
        a[n] = True
        comp = np.bitwise_and(a,binstate)
        # comp = a & binstate
        if comp.any() == False :
            binstate[n] = True
        else :
            binstate[-1] = True
    return 
@jit(nopython=True,fastmath=True)
def removeParticle(n,binstate):
    if binstate[-1] == False:
        a = binstate.copy()
        a[:] = False
        a[n] = True
        comp = np.bitwise_and(a,binstate)
        # comp = a & binstate
        if comp.any() == True :
            binstate[n] = False
        else :
            binstate[-1] = True
    return 


@jit(nopython=True,fastmath=True)
def secondQuantizationOneBodyOperator(p,q,state,state2):
    state1 = state.copy()
    phase = 1

    removeParticle(q,state1)
    if state1[-1] == True:
        return 0
    phase *= sign(q,state1)

    addParticle(p,state1)
    if state1[-1]== True:
        return 0
    phase *= sign(p,state1)

    if (state1!=state2).any():
        phase = 0
    return phase
@jit(nopython=True,fastmath=True)
def secondQuantizationTwoBodyOperator(p,q,r,s,state,state2):
    state1 = state.copy()
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
    if (state1!=state2).any():
        return 0
    return phase

@jit(nopython=True,parallel=True,fastmath=True)
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
m = 15
excite = 'Singlet' # Singlet or Triplet
twoelectron_file = 'trnseemat.dat'
oneelectron_file = 'trnsh.dat'



######

saveSlaterDeterminantsToDisk(createSlaterDeterminants(n,m,excite))
# saveSlaterDeterminantsToDisk(computeHamiltonianMatrix(createSlaterDeterminants(n,m,excite),Vpqrs(twoelectron_file,15),Hstar(oneelectron_file,15),15)[1])
# print(len(createSlaterDeterminants(n,m,excite)))

Hamiltonian = computeHamiltonianMatrix(createSlaterDeterminants(n,m,excite),Vpqrs(twoelectron_file,15),Hstar(oneelectron_file,15),15)
np.save('H.npy', Hamiltonian)
np.savetxt('H.txt',Hamiltonian)
# Hamiltonian = np.load('H.npy')
w,v = np.linalg.eigh(Hamiltonian)
np.savetxt('w.txt',w)
np.savetxt('v.txt',v)
