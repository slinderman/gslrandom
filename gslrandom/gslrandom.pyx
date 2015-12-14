# distutils: language = c++
# distutils: sources = gslrandom/cgslrandom.cpp
# distutils: libraries = stdc++ gsl gslcblas
# distutils: library_dirs = /usr/local/lib
# distutils: include_dirs =  /usr/local/include gslrandom/
# distutils: extra_compile_args = -O3 -w -std=c++0x -fopenmp
# distutils: extra_link_args = -fopenmp
# cython: boundscheck = False
# cython: wraparound = True
# cython: cdivision = True

import numpy as np
cimport numpy as np
from cython.parallel import parallel, prange

from libcpp.vector cimport vector
from openmp cimport omp_get_num_threads, omp_get_thread_num, omp_get_max_threads


cpdef int get_omp_num_threads():
    # This might not be kosher
    cdef int num_threads = omp_get_max_threads()
    return num_threads


###############################################################################
#                      Random number generator                                #
###############################################################################

# Expose the RNG class to Python
cdef class PyRNG:

    def __cinit__(self, unsigned long seed=0):
        self.thisptr = new BasicRNG(seed)

    def __dealloc__(self):
        del self.thisptr

###############################################################################
#                                 Multinomial                                 #
###############################################################################

cdef extern from "cgslrandom.h":
    double sample_multinomial(BasicRNG* brng,
                              const int K,
                              const unsigned int N,
                              const double p[],
                              unsigned int[] n) nogil

    double seeded_sample_multinomial(unsigned long seed,
                                     int K,
                                     unsigned int N,
                                     const double p[],
                                     unsigned int[] n) nogil

cpdef _multinomial(PyRNG rng,
                   unsigned int N,
                   double[::1] p_K,
                   unsigned int[::1] n_K):
    """
    Draw a single vector of multinomial distributed counts.
    """
    cdef size_t K = p_K.size
    assert n_K.size == K
    sample_multinomial(rng.thisptr, K, N, &p_K[0], &n_K[0])


cpdef _vec_multinomial(PyRNG rng,
                       unsigned int[::1] N_I,
                       double[:,::1] p_IK,
                       unsigned int[:,::1] n_IK):
    """
    Draw multiple independent vectors of multinomial distributed counts.
    """
    cdef size_t I = N_I.shape[0]
    cdef size_t K = p_IK.shape[1]
    assert p_IK.shape[0] == I
    assert n_IK.shape[0] == I
    assert n_IK.shape[1] == K

    cdef np.intp_t i
    for i in xrange(I):
        _multinomial(rng, N_I[i], p_IK[i,:], n_IK[i,:])


cpdef _par_vec_multinomial(list rngs,
                           unsigned int[::1] N_I,
                           double[:,::1] p_IK,
                           unsigned int[:,::1] n_IK):
    """
    Draw multiple independent vectors of multinomial distributed counts in parallel.

    Assumes len(rngs) is number of threads.
    """
    cdef size_t I = N_I.shape[0]
    cdef size_t K = p_IK.shape[1]
    assert p_IK.shape[0] == I
    assert n_IK.shape[0] == I
    assert n_IK.shape[1] == K

    cdef vector[BasicRNG*] rngsv
    for rng in rngs:
        rngsv.push_back((<PyRNG>rng).thisptr)
    
    cdef np.int_t i, thread_num
    with nogil:
        for i in prange(I, schedule='static'):
            thread_num = omp_get_thread_num()
            sample_multinomial(rngsv[thread_num], K, N_I[i], &p_IK[i, 0], &n_IK[i,0])


cpdef _seeded_par_vec_multinomial(unsigned long[::1] seeds_I,
                                  unsigned int[::1] N_I,
                                  double[:,::1] p_IK,
                                  unsigned int[:,::1] n_IK):
    """
    Draw multiple independent vectors of multinomial distributed counts in parallel.

    Does not take a PyRNG object. 
    Instead, takes a list of seeds, each of which seed a new BasicRNG object.
    """
    cdef size_t I = N_I.shape[0]
    cdef size_t K = p_IK.shape[1]
    assert p_IK.shape[0] == I
    assert n_IK.shape[0] == I
    assert n_IK.shape[1] == K
    assert seeds_I.shape[0] == I

    cdef np.int_t i
    cdef int n_threads = get_omp_num_threads()  # apparently this might not be kosher
    for i in prange(I, nogil=True, num_threads=n_threads):
        seeded_sample_multinomial(seeds_I[i], K, N_I[i], &p_IK[i, 0], &n_IK[i,0])


def multinomial(rng, N, p, out=None):
    """
    Sample from a multinomial.

    Accepts arrays of multiple N and/or p arguments and broadcasts output.

    If multiple N and/or p arguments are given and multiple PyRNG objects are given,
    this will parallelize draws across a number of threads equal to the number of PyRNGs.
    ----------
    rng : PyRNG object or list of PyRNG objects
          If a list is given, this will call _par_vec_multinomial.
    N : int or numpy array of ints
        N parameter of the multinomial
    p : ndarray of floats
        if p.ndim == 1 then this is the standard K-length vector of probabilities 
        if p.ndim > 1, then the method assumes that the LAST dimension is K.
    out : ndarray, optional
          Must be same shape as the broadcasted N, p arguments.
          If given, output will be copied in.
    Returns
    -------
    n : ndarray
        Sub-counts that sum to N, drawn from multinomial.
        Shape will match broadcasted N, p arguments.
        If out is provided, dtype will match out, otherwise np.uint32.
    """
    if isinstance(rng, list):
        assert isinstance(rng[0], PyRNG)
    else:
        assert isinstance(rng, PyRNG)
    assert isinstance(N, (int, np.integer)) or isinstance(N, np.ndarray)
    assert isinstance(p, np.ndarray) and p.dtype == np.float

    K = p.shape[-1]

    # One N and one p vector
    if isinstance(N, (int, np.integer)) and p.ndim == 1:
        if isinstance(rng, list):
            rng = rng[0] # only needs one RNG in this case
        N = np.uint32(N)
        if (out is None) or (out.dtype != np.uint32):
            n_K = np.zeros(K, dtype=np.uint32)
            _multinomial(rng, N, p, n_K)
            if out is not None:
                assert out.shape == (K,)
                out[:] = n_K
                return out
            return n_K
        else:
            assert out.shape == (K,)
            _multinomial(rng, N, p, out)
            return out

    # One N and multiple p vectors
    elif isinstance(N, (int, np.integer)) and p.ndim > 1:
        shp = p.shape[:-1]
        N = N * np.ones(shp, dtype=np.uint32)

    # Multiple N and one p vector
    elif isinstance(N, np.ndarray) and p.ndim == 1:
        shp = N.shape
        p = p * np.ones(shp + (K,), dtype=np.float)
        if N.dtype != np.uint32:
            N = N.astype(np.uint32)

    # Multiple N and multiple p vectors
    elif isinstance(N, np.ndarray) and p.ndim > 1:
        assert N.shape == p.shape[:-1]
        shp = N.shape
        if N.dtype != np.uint32:
            N = N.astype(np.uint32)
    else:
        raise ValueError()

    if out is not None:
        assert out.shape == shp + (K,)

    if len(shp) == 1:
        if (out is None) or (out.dtype != np.uint32):
            n_IK = np.zeros_like(p, dtype=np.uint32)
        else:
            assert out.shape == shp + (K,)
            n_IK = out
        if isinstance(rng, list):
            _par_vec_multinomial(rng, N, p, n_IK)
        else:
            _vec_multinomial(rng, N, p, n_IK)
        if (out is not None) and (out.dtype != np.uint32):
            out[:] = n_IK.reshape(shp + (K,))
            return out
        return n_IK

    else:
        # Flatten input
        I = np.prod(shp)
        N_I = N.reshape((I,))
        p_IK = p.reshape((I, K))
        n_IK = np.zeros_like(p_IK, dtype=np.uint32)
        if isinstance(rng, list):
            _par_vec_multinomial(rng, N_I, p_IK, n_IK)
        else:
            _vec_multinomial(rng, N_I, p_IK, n_IK)
        if out is not None:
            out[:] = n_IK.reshape(shp + (K,))
            return out
        else:
            return n_IK.reshape(shp + (K,))


def seeded_multinomial(N, p, out=None):
    """
    Dummy version of multinomial.  Used only for timing comparisons.

    Calls _seeded_par_vec_multinomial which creates/deletes a RNG everytime.

    Gives _seeded_par_vec_multinomial a list of random seeds.
    """
    assert isinstance(N, (int, np.integer)) or isinstance(N, np.ndarray)
    assert isinstance(p, np.ndarray) and p.dtype == np.float

    K = p.shape[-1]

    # One N and one p vector
    if isinstance(N, (int, np.integer)) and p.ndim == 1:
        shp = ()
        N_I = np.asarray([N], dtype=np.uint32)
        p_IK = p.reshape((1, K))
        I = N_I.size

    # One N and multiple p vectors
    elif isinstance(N, (int, np.integer)) and p.ndim > 1:
        shp = p.shape[:-1]
        if p.ndim == 2:
            p_IK = p
        else:
            p_IK = p.reshape((-1, K))
        I = p_IK.shape[0]
        N_I = N * np.ones(I, dtype=np.uint32)

    # Multiple N and one p vector
    elif isinstance(N, np.ndarray) and p.ndim == 1:
        shp = N.shape
        if N.ndim > 1:
            N_I = N.ravel()
        else:
            N_I = N
        if N_I.dtype != np.uint32:
            N_I = N_I.astype(np.uint32)
        I = N_I.size
        p_IK = p * np.ones((I, K), dtype=np.float)

    # Multiple N and multiple p vectors
    elif isinstance(N, np.ndarray) and p.ndim > 1:
        shp = p.shape[:-1]
        assert N.shape == shp
        if p.ndim == 2:
            p_IK = p
        else:
            p_IK = p.reshape((-1, K))

        if N.ndim > 1:
            N_I = N.ravel()
        else:
            N_I = N
        if N_I.dtype != np.uint32:
            N_I = N_I.astype(np.uint32)
        I = N_I.size
    else:
        raise ValueError()

    seeds_I = np.random.randint(2**16, size=I).astype(np.uint64)

    if (out is None) or (out.dtype != np.uint32) or (len(shp) > 1) or (len(shp) == 0):
        n_IK = np.zeros((I, K), dtype=np.uint32)
        _seeded_par_vec_multinomial(seeds_I, N_I, p_IK, n_IK)
        if out is not None:
            assert out.shape == shp + (K,)
            out[:] = n_IK.reshape(shp + (K,))
            return out
        return n_IK.reshape(shp + (K,))
    else:
        assert out.shape == (I, K)
        _seeded_par_vec_multinomial(seeds_I, N_I, p_IK, out)
        return out

###############################################################################
#                                 Dirichlet                                   #
###############################################################################

cdef extern from "cgslrandom.h":
    void sample_dirichlet (BasicRNG* brng,
                           const size_t K,
                           const double alpha[],
                           const double theta[]) nogil

cpdef _dirichlet(PyRNG rng,
                 double[::1] alpha_K,
                 double[::1] theta_K):
    """
    Draw a single vector of probabilities from a Dirichlet.
    """
    cdef size_t K = alpha_K.size
    assert theta_K.size == K
    sample_dirichlet(rng.thisptr, K, &alpha_K[0], &theta_K[0])

cpdef _vec_dirichlet(PyRNG rng,
                     double[:,::1] alpha_IK,
                     double[:,::1] theta_IK):
    """
    Draw a multiple vectors of probabilities from independent Dirichlets.
    """
    cdef size_t I = alpha_IK.shape[0]
    cdef size_t K = alpha_IK.shape[1]
    assert theta_IK.shape[0] == I
    assert theta_IK.shape[1] == K
    cdef np.intp_t i
    for i in xrange(I):
        sample_dirichlet(rng.thisptr, K, &alpha_IK[i, 0], &theta_IK[i, 0])

cpdef _par_vec_dirichlet(list rngs,
                         double[:,::1] alpha_IK,
                         double[:,::1] theta_IK):
    """
    Draw a multiple vectors of probabilities from independent Dirichlets in parallel.

    Assumes len(rngs) is number of threads.
    """
    cdef size_t I = alpha_IK.shape[0]
    cdef size_t K = alpha_IK.shape[1]
    assert theta_IK.shape[0] == I
    assert theta_IK.shape[1] == K

    cdef vector[BasicRNG*] rngsv
    for rng in rngs:
        rngsv.push_back((<PyRNG>rng).thisptr)

    cdef np.intp_t i, thread_num
    with nogil:
        for i in prange(I, schedule='static'):
            thread_num = omp_get_thread_num()
            sample_dirichlet(rngsv[thread_num], K, &alpha_IK[i, 0], &theta_IK[i, 0])

def dirichlet(rng, alpha, out=None):
    """
    Sample from a Dirichlet.

    Parallelizes if multiple alpha vectors and multiple PyRNGs are given.
    ----------
    rng : PyRNG object or list of PyRNG objects
          If a list is given, this will call _par_vec_dirichlet.
    alpha : ndarray of floats
            if alpha.ndim == 1 then this is the standard K-length vector of parameters 
            if alpha.ndim > 1, then the method assumes that the LAST dimension is K.
    out : ndarray, optional
          Must be same shape as alpha.  If given, output will be copied in.
    Returns
    -------
    theta : ndarray
            K-length vector of probabilities (sums to 1).
            If alpha.ndim > 1, the last dimension is K and sums to 1.
    """
    if isinstance(rng, list):
        assert isinstance(rng[0], PyRNG)
    else:
        assert isinstance(rng, PyRNG)
    assert isinstance(alpha, np.ndarray) and alpha.dtype == np.float

    if out is not None:
        assert alpha.shape == out.shape
        assert out.dtype == np.float

    K = alpha.shape[-1]

    if alpha.ndim == 1:
        theta = out
        if theta is None:
            theta = np.empty_like(alpha)
        _dirichlet(rng, alpha, theta)
        return theta

    if alpha.ndim == 2:
        alpha_IK = alpha
        theta_IK = out
        if theta_IK is None:
            theta_IK = np.empty_like(alpha_IK)

    elif alpha.ndim > 2:
         alpha_IK = alpha.reshape((-1, K))
         theta_IK = np.empty_like(alpha_IK)

    if isinstance(rng, list):
        _par_vec_dirichlet(rng, alpha_IK, theta_IK)
    else:
        _vec_dirichlet(rng, alpha_IK, theta_IK)

    if alpha.ndim == 1:
        return theta_IK
    else:
        return theta_IK.reshape(alpha.shape)

###############################################################################
#                     Chinese Restaurant Table (CRT)                          #
###############################################################################

cdef extern from "cgslrandom.h":
    unsigned int sample_crt (const unsigned int m,
                             const double r) nogil

cpdef unsigned int _crt(unsigned int m, double r):
    return sample_crt(m, r) 

cpdef _vec_crt(unsigned int[::1] m_I, double[::1] r_I, unsigned int[::1] l_I):
    cdef size_t I = m_I.shape[0]
    assert r_I.shape[0] == I
    assert l_I.shape[0] == I

    cdef np.intp_t i
    with nogil:
        for i in prange(I, schedule='static'):
            l_I[i] = sample_crt(m_I[i], r_I[i])

cpdef unsigned int _sumcrt(unsigned int[::1] m_I, double[::1] r_I):
    cdef size_t I = m_I.size
    assert r_I.size == I

    cdef unsigned int l = 0

    cdef np.intp_t i
    with nogil:
        for i in prange(I, schedule='dynamic'):
            l += sample_crt(m_I[i], r_I[i])
    return l

def crt(m, r, out=None):
    """
    Sample from a Chinese Restaurant Table (CRT) distribution [1].

    l ~ CRT(m, r) can be sampled as the sum of indep. Bernoullis:

            l = \sum_{n=1}^m Bernoulli(r/(r + n-1))

    where m >= 0 is integer and r >=0 is real.

    This method broadcasts the parameters m, r if ndarrays are given.
    Also will parallelize if multiple inputs are given.

    No PyRNG needed.  Randomness comes from rand() in stdlib.h.
    ----------
    m : int or ndarray of ints
    r : float or ndarray of floats
    out : ndarray, optional
          Must be same shape as m or r.
    Returns
    -------
    l : int or ndarray of ints, the sample from the CRT

    References
    ----------
    [1] M. Zhou & L. Carin. Negative Binomial Count and Mixture Modeling. 
        In IEEE (2012).
    """
    if np.isscalar(m) and np.isscalar(r):
        assert m >= 0
        assert r >= 0
        assert out is None
        return np.uint32(_crt(np.uint32(m), float(r)))  # why is _crt returning longs?

    if isinstance(m, np.ndarray) and np.isscalar(r):
        assert (m >= 0).all()
        assert r >= 0
        shp = m.shape
        m_I = m
        if m_I.dtype != np.uint32:
            m_I = m_I.astype(np.uint32)
        if len(shp) > 1:
            m_I = m_I.ravel()
        I = m_I.size
        r_I = r * np.ones(I)

    elif np.isscalar(m) and isinstance(r, np.ndarray):
        assert m >= 0
        assert (r >= 0).all()
        shp = r.shape
        r_I = r
        if r_I.dtype != float:
            r_I = r_I.astype(float)
        if len(shp) > 1:
            r_I = r_I.ravel()
        I = r_I.size
        m_I = m * np.ones(I, dtype=np.uint32)

    elif isinstance(m, np.ndarray) and isinstance(r, np.ndarray):
        assert (m >= 0).all()
        assert (r >= 0).all()
        assert m.shape == r.shape
        shp = m.shape
        m_I = m
        if m_I.dtype != np.uint32:
            m_I = m_I.astype(np.uint32)
        r_I = r
        if r_I.dtype != float:
            r_I = r_I.astype(float)
        if len(shp) > 1:
            m_I = m_I.ravel()
            r_I = r_I.ravel()

    l_I = out
    if (l_I is None) or (l_I.dtype != np.uint32) or (len(shp) > 1):
        l_I = np.empty_like(m_I, dtype=np.uint32)

    _vec_crt(m_I, r_I, l_I)

    if out is not None:
        if len(shp) > 1:
            out[:] = l_I.reshape(shp)
        elif out.dtype != np.uint32:
            out[:] = l_I
        return out
    return l_I


def sumcrt(m, r):
    """
    Sample a sum of independent CRTs.

    Avoids creating an extra array before summing. Possibly unnecessary.
    ----------
    m : int or ndarray of ints
    r : float or ndarray of floats

    Returns
    -------
    l : int, the sample of the sum of CRTs
    """
    if np.isscalar(m) and np.isscalar(r):  # crt is a special case
        assert m >= 0
        assert r >= 0
        return _crt(np.uint32(m), float(r))

    if isinstance(m, np.ndarray) and np.isscalar(r):
        assert (m >= 0).all()
        assert r >= 0
        shp = m.shape
        m_I = m
        if m_I.dtype != np.uint32:
            m_I = m_I.astype(np.uint32)
        if len(shp) > 1:
            m_I = m_I.ravel()
        I = m_I.size
        r_I = r * np.ones(I)

    elif np.isscalar(m) and isinstance(r, np.ndarray):
        assert m >= 0
        assert (r >= 0).all()
        shp = r.shape
        r_I = r
        if r_I.dtype != float:
            r_I = r_I.astype(float)
        if len(shp) > 1:
            r_I = r_I.ravel()
        I = r_I.size
        m_I = m * np.ones(I, dtype=np.uint32)

    elif isinstance(m, np.ndarray) and isinstance(r, np.ndarray):
        assert (m >= 0).all()
        assert (r >= 0).all()
        assert m.shape == r.shape
        shp = m.shape
        m_I = m
        if m_I.dtype != np.uint32:
            m_I = m_I.astype(np.uint32)
        r_I = r
        if r_I.dtype != float:
            r_I = r_I.astype(float)
        if len(shp) > 1:
            m_I = m_I.ravel()
            r_I = r_I.ravel()

    return _sumcrt(m_I, r_I)


###############################################################################
#                                 Bincount                                    #
###############################################################################

cdef extern from "cgslrandom.h":
    void cbincount(size_t n_obs,
                  const unsigned int obs[],
                  const unsigned int weights[],
                  size_t minlength,
                  unsigned int bins[],
                  int reset_bins) nogil

cpdef _bincount(unsigned int[::1] obs,
                unsigned int[::1] weights,
                unsigned int[::1] bins,
                int reset_bins):
    
    cdef:
        size_t n_obs, minlength

    n_obs = obs.size
    minlength = bins.size
    cbincount(n_obs, &obs[0], &weights[0], minlength, &bins[0], reset_bins)

def bincount(obs, weights=None, minlength=None, bins=None, reset_bins=True):
    if not isinstance(obs, np.ndarray) or obs.dtype != np.uint32:
        obs = np.ndarray(obs, dtype=np.uint32)
    if minlength is None:
        minlength = obs.max() + 1
    if weights is None:
        weights = np.ones(obs.size, dtype=np.uint32)
    if bins is None:
        bins = np.zeros(minlength, dtype=np.uint32)
    assert bins.size == minlength
    assert obs.size == weights.size
    _bincount(obs, weights, bins, int(reset_bins))
    return bins
