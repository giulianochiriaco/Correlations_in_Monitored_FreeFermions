import numpy as np
import time
import random
import scipy.linalg as spl

# Enrico: Set the random seed for reproducibility
np.random.seed(42)
random.seed(42)

'''
 Enrico: to my understanding np.random.seed() might not be guaranteed to work 
 if two different threads are executing the function at the same time. 
 Let's see and in case we can use only random! 
'''

# Rmn defines the evolution operator for the correlation matrix Dij
# s is the evolution time of one cycle. It can be set to 1, as it always multiplies t1, t2 and t12


def Rmn(s, t1, t2, t12, L):
    Rmn = np.array(np.zeros((2*L, 2*L)), dtype=complex)  # evolution matrix
    ta = t1-t2
    # Rmn has a simple expression in Fourier space: k ranges from -L/2 to L/2-1. (Assumes periodic Boundary conditions)
    # It is simpler to write Rmn in Fourier space and then sum over k to reconstruct it in real space
    # This operation is time consuming (~L^3) and is performed only once at the start of the evolution, since the evolution matrix only depends on fixed parameters of the Hamiltonian
    ck = np.array([np.cos(2*np.pi*k/L) for k in range(-int(L/2), int(L/2))],
                  dtype=complex)  # vector of cosine cos(2*np.pi*k/L)
    sk = np.array([np.sin(2*np.pi*k/L) for k in range(-int(L/2),
                  int(L/2))], dtype=complex)  # vector of sine
    # common prefactor coming from the exponential of the identity matrix
    expf = np.exp(-1j*(t1+t2)*s*ck)
    # introduce very small cut-off to avoid divisions by zero
    sqr = np.sqrt(t12**2+ta**2*ck**2+10**(-16))
    for m in range(L):
        for n in range(L):
            expmn = np.array([np.exp(2*np.pi*1j*(m-n)*k/L)
                             for k in range(-int(L/2), int(L/2))], dtype=complex)  # common Fourier weight
            # perform the Fourier summation over k
            f1 = np.sum(expf*expmn*np.cos(sqr*s))/L
            fz = -1j*np.sum(expf*expmn*ta*ck*np.sin(sqr*s)/sqr)/L
            fx = -1j*np.sum(expf*expmn*t12*np.sin(sqr*s)/sqr)/L
            Rmn[m, n] = f1+fz  # R11 part of the evolution operator
            Rmn[m+L, n] = fx  # R21 part
            Rmn[m, n+L] = fx  # R12 part
            Rmn[m+L, n+L] = f1-fz  # R22 part
    return Rmn


def KronD(i, j):  # kronecker delta
    if i == j:
        return 1
    else:
        return 0
    # return int(i==j)


def evolt(Dij, Rmn, p2, Ncycles, p1=0):  # evolves the correlation matrix Dij using the evolution operator Rmn for Nsteps cycles, with measurement probability p2 on the second chain and with measurement probability p1 on the first chain
    time_counter = 0
    Dij_t = Dij
    # gets number of sites (divided by 2 because Dij is 2L x 2L
    L = int(len(Dij)/2)
    Id = np.eye(len(Dij_t))
    while time_counter < Ncycles:  # iterates over number of cycles
        Dij_t = np.dot(np.conjugate(np.transpose(Rmn)),
                       np.dot(Dij_t, Rmn))  # unitary evolution
        # measurement part, performed for every site of the second chain (from L+1 to 2*L)
        for k in np.random.permutation(np.arange(L)):
            p1 = random.random()
            if p1 < p1:  # if random number smaller than p perform measurement, otherwise do nothing
                Dkk = Dij_t[k, k]  # get occupation nk of site k
                p2 = random.random()
                if p2 < Dkk:  # if second random number is smaller than nk, then apply nk=c_k^dag*c_k operator, otherwise apply 1-nk
                    # correlation matrix changes this way
                    Dij_t = Dij_t - \
                        np.outer(Dij_t[:, k], Dij_t[k, :]) / \
                        Dij_t[k, k]+np.outer(Id[:, k], Id[k, :])
                else:
                    D1 = Id-Dij_t
                    Dij_t = Dij_t + \
                        np.outer(D1[:, k], D1[k, :])/(1-Dij_t[k, k]
                                                      )-np.outer(Id[:, k], Id[k, :])
        # measurement part, performed for every site of the second chain (from L+1 to 2*L)
        for k in np.random.permutation(np.arange(L, 2*L)):
            p1 = random.random()
            if p1 < p2:  # if random number smaller than p perform measurement, otherwise do nothing
                Dkk = Dij_t[k, k]  # get occupation nk of site k
                p2 = random.random()
                if p2 < Dkk:  # if second random number is smaller than nk, then apply nk=c_k^dag*c_k operator, otherwise apply 1-nk
                    # correlation matrix changes this way
                    Dij_t = Dij_t - \
                        np.outer(Dij_t[:, k], Dij_t[k, :]) / \
                        Dij_t[k, k]+np.outer(Id[:, k], Id[k, :])
                else:
                    D1 = Id-Dij_t
                    Dij_t = Dij_t + \
                        np.outer(D1[:, k], D1[k, :])/(1-Dij_t[k, k]
                                                      )-np.outer(Id[:, k], Id[k, :])
        time_counter += 1
    return Dij_t  # return correlation matrix


def Transient_Cr_avg(L, t1, t2, t12, p2, p1, Nmax, t_step=1, R=0):
    # This function evolves the system in time until Nmax cycles.
    # Nmax is the maximum number of cycles we want to evolve. t_step is how many cycles between one correlation computation and the next, default is 1, if change it then write t_step=x with x the number of cycles.ALWAYS MAKE SURE Nmax IS DIVISIBLE BY t_step
    # For each time step we compute the correlation matrix before and after the measurements.
    # We extract the space averaged correlation function vs r for both chains and for inter-chain correlations.
    # Output is six correlations vs r (before/after and 11/12/22 chains)
    if len(np.shape(R)) == 0:
        R = Rmn(0.5, t1, t2, t12, L)
    v1 = np.random.permutation(np.concatenate(
        [np.zeros(int(L/2)), np.ones(int(L/2))]))
    v2 = np.random.permutation(np.concatenate(
        [np.zeros(int(L/2)), np.ones(int(L/2))]))
    # Enrico: I found out dtype=complex is equivalent to dtype=np.complex128
    D0 = np.array(np.diag(np.concatenate((v1, v2))), dtype=complex)
    NT = int(Nmax/t_step)
    Cr11A = np.zeros((NT, L//2+1), dtype=complex)
    Cr12A = np.zeros((NT, L), dtype=complex)
    Cr22A = np.zeros((NT, L//2+1), dtype=complex)
    Cr12B = np.zeros((NT, L), dtype=complex)
    Cr11B = np.zeros((NT, L//2+1), dtype=complex)
    Cr22B = np.zeros((NT, L//2+1), dtype=complex)
    tvec = t_step*np.arange(NT)
    r_1 = np.arange(L//2+1)
    r_12 = np.arange(L)
    j_values = np.arange(L)[:, None]  # Column vector for broadcasting
    jp_1 = (j_values + r_1) % L  # Compute indices with periodic BCs
    jp_12 = (j_values + r_12) % L  # Compute indices with periodic BCs

    # loop over the number of cycles
    for i in range(Nmax):
        # compute correlations BEFORE measurements (but only every t_step cycles)
        if i % t_step == 0:
            D1 = np.abs(D0.copy())**2
            D11 = D1[:L, :L]
            D12 = D1[:L, L:]
            D22 = D1[L:, L:]
            Cr11B[i, :] = np.mean(D11[j_values, jp_1], axis=0)
            Cr12B[i, :] = np.mean(D12[j_values, jp_12], axis=0)
            Cr22B[i, :] = np.mean(D22[j_values, jp_1], axis=0)
        # evolve for half a cycle with unitary evolution + measurements at the end
        D0 = evolt(D0, R, p2, 1, p1)
        # compute correlations AFTER measurements (but only every t_step cycles)
        if i % t_step == 0:
            D1 = np.abs(D0.copy())**2
            D11 = D1[:L, :L]
            D12 = D1[:L, L:]
            D22 = D1[L:, L:]
            Cr11A[i, :] = np.mean(D11[j_values, jp_1], axis=0)
            Cr12A[i, :] = np.mean(D12[j_values, jp_12], axis=0)
            Cr22A[i, :] = np.mean(D22[j_values, jp_1], axis=0)
        # evolve for the remaining half a cycle with unitary evolution but no measurement
        D0 = evolt(D0, R, 0, 1, p1=0)

    return Cr11A, Cr12A, Cr22A, Cr11B, Cr12B, Cr22B, tvec
