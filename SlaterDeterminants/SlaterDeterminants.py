# For Creating SlaterDeterminants  
import numpy 
import bitarray
from bitarray import bitarray

# State [1,2,3,4] 
# binState |111100000...0m> 

def createInitialState(n,m):
    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i+1
    return state

def odometer(state,n,m):
    newState = state.copy()
    l = 0
    for j in range(n-1,-1,-1):
        if newState[j] < m - n + j + 1 :
            l = newState[j]
            for k in range(j,n):
                newState[k] = l + 1 + k - j 
            return newState
    newState = numpy.zeros(n)
    return newState

def createBinaryState(state,n,m):
    binState = bitarray(m)
    binState.setall(False)
    for i in state.astype(int):
        binState[i-1] = True
    return binState

def createSlaterDeterminants(n,m):
    state = createInitialState(n,m).copy()
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


saveSlaterDeterminantsToDisk(createSlaterDeterminants(10,25))# (10,25) 45Sec
