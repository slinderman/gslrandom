# distutils: language = c++
# distutils: sources = gslrandom/cgslrandom.cpp
# distutils: libraries = stdc++ gsl gslcblas
# distutils: library_dirs = /usr/local/lib
# distutils: include_dirs =  /usr/local/include gslrandom/
# distutils: extra_compile_args = -O3 -w -std=c++11
# cython: boundscheck = False
# cython: wraparound = True
# cython: cdivision = True

import numpy as np
cimport numpy as np
from cython.parallel import parallel, prange


###############################################################################
#                      Random number generator                                #
###############################################################################
cdef extern from "cgslrandom.h":
    cdef cppclass BasicRNG:
        BasicRNG() except +

# Expose the RNG class to Python
cdef class PyRNG:
    cdef BasicRNG *thisptr

    def __cinit__(self):
        self.thisptr = new BasicRNG()

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

    double dumb_sample_multinomial(int K,
                                   unsigned int N,
                                   const double p[],
                                   unsigned int[] n) nogil


cpdef _typed_multinomial(PyRNG rng,
                         unsigned int N,
                         double[::1] p,
                         unsigned int[::1] n ):
    """
    Draw a vector of multinomial distributed counts
    """
    cdef int K = p.size
    sample_multinomial(rng.thisptr, K, N, &p[0], &n[0])

cpdef _vec_typed_multinomial(PyRNG rng,
                             unsigned int[::1] N,
                             double[:,::1] p,
                             unsigned int[:,::1] n):
    cdef int L = N.shape[0]
    cdef int K = p.shape[1]
    assert p.shape[0] == L
    assert n.shape[0] == L
    assert n.shape[1] == K

    # Sample n's one at a time
    cdef int l
    for l in xrange(L):
        _typed_multinomial(rng, N[l], p[l,:], n[l,:])

cpdef _dumb_vec_typed_multinomial(unsigned int[::1] N,
                                 double[:,::1] p,
                                 unsigned int[:,::1] n):
    cdef int L = N.shape[0]
    cdef int K = p.shape[1]
    assert p.shape[0] == L
    assert n.shape[0] == L
    assert n.shape[1] == K

    # Sample n's one at a time
    cdef int l
    for l in prange(L, nogil=True, num_threads=2):
        dumb_sample_multinomial(K, N[l], &p[l,0], &n[l,0])

def multinomial(rng, N, p, out=None):
    assert isinstance(rng, PyRNG)
    assert isinstance(N, (int, np.integer)) or isinstance(N, np.ndarray)
    assert isinstance(p, np.ndarray) and p.dtype == np.float

    K = p.shape[-1]

    if isinstance(N, (int, np.integer)) and p.ndim == 1:
        # Solve simple case immediately
        N = np.uint32(N)
        if out is None or out.dtype != np.uint32:
            out_tmp = np.zeros(K, dtype=np.uint32)
            _typed_multinomial(rng, N, p, out_tmp)
            if out is not None: 
                assert out.shape == (K,)
                out[:] = out_tmp 
            return out_tmp
        else:
            _typed_multinomial(rng, N, p, out)
            return out

    elif isinstance(N, (int, np.integer)) and p.ndim > 1:
        shp = p.shape[:-1]
        N = N * np.ones(shp, dtype=np.uint32)

    elif isinstance(N, np.ndarray) and p.ndim == 1:
        shp = N.shape
        p = p * np.ones(shp + (K,), dtype=np.float)
        if N.dtype != np.uint32:
            N = N.astype(np.uint32)

    elif isinstance(N, np.ndarray) and p.ndim > 1:
        assert N.shape == p.shape[:-1]
        shp = N.shape
        if N.dtype != np.uint32:
            N = N.astype(np.uint32)
    else:
        raise ValueError()

    # Check the output
    if out is not None:
        # TODO: Make this work for out arrays with wrong dtype (but right shape)
        assert out.shape == shp + (K,) and out.dtype == np.uint32

    if len(shp) > 1:
        # Flatten outputs to 2D
        L = np.prod(shp)
        N1d = N.reshape((L,))
        p2d = p.reshape((L,K))
        out2d = np.zeros((L,K), dtype=np.uint32)
        _vec_typed_multinomial(rng, N1d, p2d, out2d)
        if out is None:
            return out2d.reshape(shp + (K,))
        else:
            out[:] = out2d.reshape(shp + (K,))
            return out
    else:
        if out is None:
            out = np.zeros(shp + (K,), dtype=np.uint32)
        _vec_typed_multinomial(rng, N, p, out)
        return out

def dumb_multinomial(N_L, P_LK, N_LK):
    # TODO: Add checks/asserts/reshapes/casts etc.
    # Currently this assumes inputs are all correct
    _dumb_vec_typed_multinomial(N_L, P_LK, N_LK)

