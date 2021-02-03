import bitarray
from bitarray import bitarray
import numpy 

#binstate = bitarray([1,0,1,1])

def sign(n,binstate):
    if binstate[-1] == False:
        s = 1
        for i in range(n):
            if binstate[i] !=0 :
                s *= -1
        return s
    return 1
def addParticle(n,binstate):
    if binstate[-1] == False:
        a = bitarray(len(binstate))
        a.setall(False)
        a[n-1] = True
        comp = a & binstate
        if comp.count() == False :
            binstate[n-1] = True
        else :
            binstate[len(binstate)-1] = True
    return 

def removeParticle(n,binstate):
    if binstate[-1] == False:
        a = bitarray(len(binstate))
        a.setall(False)
        a[n-1] = True
        comp = a & binstate
        if comp.count() == True :
            binstate[n-1] = False
        else :
            binstate[len(binstate)-1] = True
    return 

# binstate = bitarray([0,0,1,1,0,0,1,1,0,0,0])
# print(binstate)
# print(addParticle(1,binstate))
# print(binstate)
# print(len(binstate))
# print(removeParticle(2,binstate))
# print(binstate)
# print(removeParticle(1,binstate))
# print(binstate)
