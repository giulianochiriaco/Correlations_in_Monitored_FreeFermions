import numpy as np
import random
# import time
# import scipy.linalg as spl

# Enrico: Set the random seed for reproducibility
# Instead of globally seeding, create independent random generator instances
rng = np.random.default_rng(42)
py_rng = random.Random(42)

'''
 Enrico: to my understanding np.random.seed() might not be guaranteed to work 
 if two different threads are executing the function at the same time. 
 Let's see and in case we can use only random! 
'''


def Rmn(s, t1, t2, t12, L):
    """Unitary time evolution matrix of the correlation matrix.
    The function computes the unitary time evolution matrix Rmn for a system of size L, given the parameters s, t1, t2, and t12. The matrix is constructed in Fourier space and then transformed to real space.

    Args:
        s (float): time evolution parameter.
        t1 (float): hopping in chain 1.
        t2 (float): hopping in chain 2.
        t12 (float): inter-chain hopping.
        L (int): system size (better be even).

    Returns:
        ndarray: unitary time evolution matrix Rmn of shape (2L, 2L).
    """
    Rmn = np.array(np.zeros((2*L, 2*L)), dtype=complex)  # evolution matrix
    ta = t1-t2
    # Rmn has a simple expression in Fourier space: k ranges from -L/2 to L/2-1. (Assumes periodic Boundary conditions)
    # It is simpler to write Rmn in Fourier space and then sum over k to reconstruct it in real space
    # This operation is time consuming (~L^3) and is performed only once at the start of the evolution, since the evolution matrix only depends on fixed parameters of the Hamiltonian

    kvec = np.arange(-int(L/2), int(L/2))  # length L

    # vector of cosine cos(2*pi*k/L)
    ck = np.array(np.cos(2*np.pi*kvec/L), dtype=complex)
    # common prefactor coming from the exponential of the identity matrix
    expf = np.exp(-1j*(t1+t2)*s*ck)
    # introduce very small cut-off to avoid divisions by zero
    sqr = np.sqrt(t12**2+ta**2*ck**2+1.e-16)
    # Precompute factors for the three sums (length L arrays)
    factor1 = expf * np.cos(sqr*s)
    factor2 = expf * ta * ck * np.sin(sqr*s)/sqr
    factor3 = expf * t12 * np.sin(sqr*s)/sqr

    # Rearrange factors into standard FFT order
    factor1_shifted = np.fft.ifftshift(factor1)
    factor2_shifted = np.fft.ifftshift(factor2)
    factor3_shifted = np.fft.ifftshift(factor3)

    # Compute DFTs which give 1/L * sum_k (factor * exp(2πi u k / L))
    f1_array = np.fft.ifft(factor1_shifted)
    fz_array = -1j * np.fft.ifft(factor2_shifted)
    fx_array = -1j * np.fft.ifft(factor3_shifted)

    # Build a difference index matrix: for each m, n compute u = (m - n) mod L
    idx = np.arange(L)
    diff_mod = np.mod(idx[:, None] - idx[None, :], L)  # shape (L,L)

    # Create the 2D arrays from the FFT results, indexing using diff_mod
    f1_matrix = f1_array[diff_mod]
    fz_matrix = fz_array[diff_mod]
    fx_matrix = fx_array[diff_mod]

    # Fill the evolution operator Rmn using block structure
    Rmn[:L, :L] = f1_matrix + fz_matrix      # R11
    Rmn[L:, :L] = fx_matrix                  # R21
    Rmn[:L, L:] = fx_matrix                  # R12
    Rmn[L:, L:] = f1_matrix - fz_matrix       # R22

    return Rmn


def KronD(i, j):  # kronecker delta
    return i == j


def evolt(Dij, Rmn, pR, Ncycles, pL=0):
    """
    Evolves the correlation matrix Dij using the evolution operator Rmn for Nsteps cycles, with measurement probability pR on the second chain and with measurement probability pL on the first chain.
    The function performs a unitary evolution followed by measurements on the correlation matrix. The measurements are performed with probabilities pR and pL on the second and first chains, respectively.

    Args:
        Dij (ndarray): correlation matrix to be evolved.
        Rmn (ndarray): unitary evolution operator.
        pR (float): measurement probability on the second chain.
        Ncycles (int): number of cycles to evolve.
        pL (float, optional): measurement probability on the first chain. Defaults to 0.

    Returns:
        ndarray: time-evolved correlation matrix.
    """
    Dij_t = Dij
    # gets number of sites (divided by 2 because Dij is 2L x 2L
    L = Dij.shape[0]//2
    Id = np.eye(Dij_t.shape[0], dtype=Dij.type)  # identity matrix
    for _ in range(Ncycles):
        Dij_t = Rmn.conj().T @ Dij @ Rmn
        # Dij_t = np.dot(np.conjugate(np.transpose(Rmn)),
        #                np.dot(Dij_t, Rmn))  # unitary evolution
        # measurement part, performed for every site of the second chain (from L+1 to 2*L)
        for k in np.random.permutation(np.arange(L)):
            if random.random() < pL:  # if random number smaller than p perform measurement, otherwise do nothing
                Dkk = Dij_t[k, k]  # get occupation nk of site k
                if random.random() < Dkk:  # if second random number is smaller than nk, then apply nk=c_k^dag*c_k operator, otherwise apply 1-nk
                    # correlation matrix changes this way
                    Dij_t = Dij_t \
                        + np.outer(Id[:, k], Id[k, :]) \
                        - np.outer(Dij_t[:, k], Dij_t[k, :]) / Dkk
                else:
                    D1 = Id-Dij_t
                    Dij_t = Dij_t \
                        - np.outer(Id[:, k], Id[k, :]) \
                        + np.outer(D1[:, k], D1[k, :])/(1-Dkk)
        # measurement part, performed for every site of the second chain (from L+1 to 2*L)
        for k in np.random.permutation(np.arange(L, 2*L)):
            if random.random() < pR:  # if random number smaller than p perform measurement, otherwise do nothing
                Dkk = Dij_t[k, k]  # get occupation nk of site k
                if random.random() < Dkk:  # if second random number is smaller than nk, then apply nk=c_k^dag*c_k operator, otherwise apply 1-nk
                    # correlation matrix changes this way
                    Dij_t = Dij_t \
                        - np.outer(Dij_t[:, k], Dij_t[k, :]) / Dkk \
                        + np.outer(Id[:, k], Id[k, :])
                else:
                    D1 = Id-Dij_t
                    Dij_t = Dij_t \
                        + np.outer(D1[:, k], D1[k, :])/(1-Dkk) \
                        - np.outer(Id[:, k], Id[k, :])
    return Dij_t  # return correlation matrix


def Transient_Cr_avg(L, t1, t2, t12, pR, pL, Nmax, t_step=1, R=0):
    """This function evolves the system in time until Nmax cycles.
    Nmax is the maximum number of cycles we want to evolve. t_step is how many cycles between one correlation computation and the next, default is 1, if change it then write t_step=x with x the number of cycles.ALWAYS MAKE SURE Nmax IS DIVISIBLE BY t_step
    For each time step we compute the correlation matrix before and after the measurements.
    We extract the space averaged correlation function vs r for both chains and for inter-chain correlations.

    Args:
        L (int): system size.
        t1 (float): chain 1 hopping.
        t2 (float): chain 2 hopping.
        t12 (float): inter-chain hopping.
        pR (float): measurement probability on the second chain.
        pL (float): measurement probability on the first chain.
        Nmax (int): maximum number of cycles to evolve.
        t_step (int, optional): cycles between correlation computation. Defaults to 1.
        R (ndarray, optional): unitary time evolution matrix. Defaults to 0.

    Returns:
        ndarray: correlations after and before measurement vs radius r (Cr11A, Cr12A, Cr22A, Cr11B, Cr12B, Cr22B).
        ndarray: time vector tvec.
    """

    if len(np.shape(R)) == 0:
        R = Rmn(0.5, t1, t2, t12, L)
    v1 = np.random.permutation(np.concatenate(
        [np.zeros(int(L/2)), np.ones(int(L/2))]))
    v2 = np.random.permutation(np.concatenate(
        [np.zeros(int(L/2)), np.ones(int(L/2))]))
    D0 = np.array(np.diag(np.concatenate((v1, v2))), dtype=complex)
    NT = int(Nmax/t_step)

    Cr11A = np.zeros((NT, L//2+1), dtype=complex)
    Cr22A = np.copy(Cr11A)

    Cr12A = np.zeros((NT, L), dtype=complex)
    Cr12B = np.copy(Cr12A)

    Cr11B = np.zeros((NT, L//2+1), dtype=complex)
    Cr22B = np.copy(Cr11B)

    tvec = t_step*np.arange(NT)

    # + Precompute indices for correlation distance r
    r_1 = np.arange(L//2+1)
    r_12 = np.arange(L)
    j_values = np.arange(L)[:, None]  # Column vector for broadcasting
    jp_1 = (j_values + r_1) % L  # Compute indices with periodic BCs
    jp_12 = (j_values + r_12) % L  # Compute indices with periodic BCs

    # loop over the number of cycles
    for i in range(Nmax):
        if i % t_step == 0:
            # compute correlations BEFORE measurements (but only every t_step cycles)
            D1 = np.abs(D0.copy())**2
            D11 = D1[:L, :L]
            D12 = D1[:L, L:]
            D22 = D1[L:, L:]
            Cr11B[i, :] = np.mean(D11[j_values, jp_1], axis=0)
            Cr12B[i, :] = np.mean(D12[j_values, jp_12], axis=0)
            Cr22B[i, :] = np.mean(D22[j_values, jp_1], axis=0)
        # evolve for half a cycle with unitary evolution + measurements at the end
        D0 = evolt(D0, R, pR=pR, Ncycles=1, pL=pL)

        if i % t_step == 0:
            # compute correlations AFTER measurements (but only every t_step cycles)
            D1 = np.abs(D0.copy())**2
            D11 = D1[:L, :L]
            D12 = D1[:L, L:]
            D22 = D1[L:, L:]
            Cr11A[i, :] = np.mean(D11[j_values, jp_1], axis=0)
            Cr12A[i, :] = np.mean(D12[j_values, jp_12], axis=0)
            Cr22A[i, :] = np.mean(D22[j_values, jp_1], axis=0)

        # evolve for the remaining half a cycle with unitary evolution but no measurement
        D0 = evolt(D0, R, pR=0, Ncycles=1, pL=0)

    return Cr11A, Cr12A, Cr22A, Cr11B, Cr12B, Cr22B, tvec
