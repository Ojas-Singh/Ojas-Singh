# For Creating SlaterDeterminants  
import numba
from numba import jit
import numpy 
import bitarray
from bitarray import bitarray

# State [1,2,3,4] 
# binState |111100000...0m> 

@jit(nopython=True,fastmath=True)
def createInitialState(n,m,excite):
    state = numpy.zeros(n)
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
    newState = numpy.zeros(n)
    return newState

def createBinaryState(state,n,m):
    binState = bitarray(m)
    binState.setall(False)
    for i in state.astype(int):
        binState[i-1] = True
    return binState

def createSlaterDeterminants(n,m,excite):
    state = createInitialState(n,m,excite).copy()
    binStates = []
    binStates.append(createBinaryState(state,n,m))
    while True :
        state = odometer(state,n,m)
        if numpy.sum(state) == 0:
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

saveSlaterDeterminantsToDisk(createSlaterDeterminants(10,30,'Singlet'))
# print(load('SlaterDeterminants.sd'))

# print(odometer([3,5],2,6))
# print(createSlaterDeterminants(2,6,'Singlet'))
