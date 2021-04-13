"""
A Psi4 input script to compute Full Configuration Interaction from a SCF reference

Requirements:
SciPy 0.13.0+, NumPy 1.7.2+

References:
Equations from [Szabo:1996]
"""

__authors__ = "Tianyuan Zhang"
__credits__ = ["Tianyuan Zhang", "Jeffrey B. Schriber", "Daniel G. A. Smith"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2017-05-26"

import time
import numpy as np
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4

# Check energy against psi4?
compare_psi4 = True

# Memory for Psi4 in GB
# psi4.core.set_memory(int(2e9), False)
psi4.core.set_output_file('output.dat', False)

# Memory for numpy in GB
numpy_memory = 2

mol = psi4.geometry("""
He 
symmetry c1
""")


psi4.set_options({'basis': 'cc-pvdz',
                  'scf_type': 'pk',
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8})

print('\nStarting SCF and integral build...')
t = time.time()

# First compute SCF energy using Psi4
scf_e, wfn = psi4.energy('SCF', return_wfn=True)

# Grab data from wavfunction class
C = wfn.Ca()
ndocc = wfn.doccpi()[0]
nmo = wfn.nmo()

# Compute size of Hamiltonian in GB
from scipy.special import comb
nDet = comb(nmo, ndocc)**2
H_Size = nDet**2 * 8e-9
print('\nSize of the Hamiltonian Matrix will be %4.2f GB.' % H_Size)
if H_Size > numpy_memory:
    clean()
    raise Exception("Estimated memory utilization (%4.2f GB) exceeds numpy_memory \
                    limit of %4.2f GB." % (H_Size, numpy_memory))

# Integral generation from Psi4's MintsHelper
t = time.time()
mints = psi4.core.MintsHelper(wfn.basisset())
H = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())

print('\nTotal time taken for ERI integrals: %.3f seconds.\n' % (time.time() - t))

# AO Kinetic

print("Kinetic")
print(np.asarray(mints.ao_kinetic()).shape)
print(np.asarray(mints.ao_kinetic()))
# AO Potential
print("Potential")
print(np.asarray(mints.ao_potential()).shape)
print(np.asarray(mints.ao_potential()))

#Make spin-orbital MO
print('Starting AO -> spin-orbital MO transformation...')
t = time.time()
MO = np.asarray(mints.mo_spin_eri(C, C))
Vee = np.asarray(mints.mo_eri(C, C,C,C))
print(MO.shape)
print(np.asarray(C))
# print(kk)
# print(MO)
# Update H, transform to MO basis and tile for alpha/beta spin
H = np.einsum('uj,vi,uv', C, C, H)
Hone = H.copy()
H = np.repeat(H, 2, axis=0)
H = np.repeat(H, 2, axis=1)
print(H.shape,"okk")
# Make H block diagonal
spin_ind = np.arange(H.shape[0], dtype=np.int) % 2
H *= (spin_ind.reshape(-1, 1) == spin_ind)
print(H.shape)
print('..finished transformation in %.3f seconds.\n' % (time.time() - t))

from helper_CI import Determinant, HamiltonianGenerator
from itertools import combinations

print('Generating %d Full CI Determinants...' % (nDet))
t = time.time()
detList = []
for alpha in combinations(range(nmo), ndocc):
    for beta in combinations(range(nmo), ndocc):
        detList.append(Determinant(alphaObtList=alpha, betaObtList=beta))

print('..finished generating determinants in %.3f seconds.\n' % (time.time() - t))

print('Generating Hamiltonian Matrix...')

t = time.time()
Hamiltonian_generator = HamiltonianGenerator(H, MO)
Hamiltonian_matrix = Hamiltonian_generator.generateMatrix(detList)

print('..finished generating Matrix in %.3f seconds.\n' % (time.time() - t))

print('Diagonalizing Hamiltonian Matrix...')

t = time.time()

e_fci, wavefunctions = np.linalg.eigh(Hamiltonian_matrix)
print(e_fci[0])
print('..finished diagonalization in %.3f seconds.\n' % (time.time() - t))

fci_mol_e = e_fci[0] + mol.nuclear_repulsion_energy()

print('# Determinants:     % 16d' % (len(detList)))

print('SCF energy:         % 16.10f' % (scf_e))
print('FCI correlation:    % 16.10f' % (fci_mol_e - scf_e))
print('Total FCI energy:   % 16.10f' % (fci_mol_e))

if compare_psi4:
    psi4.compare_values(psi4.energy('FCI'), fci_mol_e, 6, 'FCI Energy')


import numpy as np
import numba
from numba import jit


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

# @jit(nopython=True,fastmath=True)
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

# @jit(nopython=True,parallel=True,fastmath=True)
def computeHamiltonianMatrix(slaterDeterminants,V,Hone,M):
    nSlaterDeterminants = int(len(slaterDeterminants))
    nOrbitals = M
    H = np.zeros((nSlaterDeterminants,nSlaterDeterminants))

    for m in range(nSlaterDeterminants):
        for n in range(m,nSlaterDeterminants):
            #
            for p in range(nOrbitals):
                for q in range(nOrbitals):
                    # One body part

                    phase = 0
                    phase += secondQuantizationOneBodyOperator(2*p,2*q,slaterDeterminants[n],slaterDeterminants[m])
                    phase += secondQuantizationOneBodyOperator(2*p+1,2*q+1,slaterDeterminants[n],slaterDeterminants[m])
                    H[m,n] += Hone[p,q]*phase

                    # Two-body interaction

                    for r in range(nOrbitals):
                        for s in range(nOrbitals):
                            # if V[p,q,r,s] != 0 :
                            phase = 0
                            phase += secondQuantizationTwoBodyOperator(2*p, 2*q, 2*r, 2*s,slaterDeterminants[n],slaterDeterminants[m])

                            phase += secondQuantizationTwoBodyOperator(2*p+1, 2*q+1, 2*r+1, 2*s+1,slaterDeterminants[n],slaterDeterminants[m])

                            phase += secondQuantizationTwoBodyOperator(2*p, 2*q+1, 2*r, 2*s+1,slaterDeterminants[n],slaterDeterminants[m])

                            phase += secondQuantizationTwoBodyOperator(2*p+1, 2*q, 2*r+1, 2*s,slaterDeterminants[n],slaterDeterminants[m])
                            H[m, n] += 0.5*V[p,r,q,s]*phase
                                
            if m != n :
                H[n,m] = np.conj(H[m,n])
    return H




### Config ###
n = 2
m = 5
excite = 'Singlet' # Singlet or Triplet

Hamiltonian = computeHamiltonianMatrix(createSlaterDeterminants(n,m,excite),Vee,Hone,m)

# np.save('H.npy', Hamiltonian)

# Hamiltonian = np.load('H.npy')
w,v = np.linalg.eigh(Hamiltonian)
print("Energy By Our Ci code :",w[0])
print("Psi4 Eigen Value ")
print(e_fci)
print("our Eigen Value ")
print(w)