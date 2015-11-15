# distutils: language = c++
# distutils: sources = gslrandom/cgslrandom.cpp
# distutils: libraries = stdc++ gsl gslcblas
# distutils: library_dirs = /usr/local/lib
# distutils: include_dirs =  /usr/local/include gslrandom/
# distutils: extra_compile_args = -O3 -w -std=c++11 -fopenmp
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

cdef extern from "cgslrandom.h":
    cdef cppclass BasicRNG:
        BasicRNG(unsigned long seed) except +

# Expose the RNG class to Python
cdef class PyRNG:
    cdef BasicRNG *thisptr

    def __cinit__(self, unsigned long seed=0):
        self.thisptr = new BasicRNG(seed)

    def __dealloc__(self):
        del self.thisptr

###############################################################################
#                                 Multinomial                                 #
###############################################################################

cdef extern from "cgslrandom.h":
    double sample_multinomial(BasicRNG* brng,
                              int K,
                              unsigned int N,
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
