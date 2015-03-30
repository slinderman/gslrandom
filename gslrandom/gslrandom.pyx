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

def multinomial(rng, N, p, out=None):
    assert isinstance(rng, PyRNG)
    assert isinstance(N, int)
    assert isinstance(p, np.ndarray) and p.dtype == np.float

    # Calculate the shape of the output
    # N int, p is a >1d ndarray -> shape = p.shape[:-1]
    if isinstance(N, int) and isinstance(p, np.ndarray) and p.ndim == 1:
        shp = (1,)

    elif isinstance(N, int) and isinstance(p, np.ndarray) and p.ndim > 1:
        shp = p.shape[:-1]

    # N is D-array, p is 1d array -> shape = N.shape
    elif isinstance(N, np.ndarray) and isinstance(p, np.ndarray) and p.ndim == 1:
        shp = N.shape

    # N is D-array, p is (D+1)-array and N.shape == p.shape[:-1]
    #   -> shape = N.shape
    elif isinstance(N, np.ndarray) and isinstance(p, np.ndarray) and p.ndim > 1:
        assert N.shape == p.shape[:-1]
        shp = N.shape

    else:
        raise NotImplementedError()

    # Get K
    K = p.shape[-1]

    # Populate N and p
    N = N * np.ones(shp, dtype=np.uint32)
    p = p * np.ones(shp + (K,), dtype=np.float)

    # Cast N and p to the right types
    if N.dtype is not np.uint32:
        N = N.astype(np.uint32)

    if p.dtype is not np.float:
        p = p.astype(np.float)

    # Check the output
    if out is not None:
        assert out.shape == shp + (K,) and out.dtype == np.uint32
    else:
        out = np.zeros(shp + (K,), dtype=np.uint32)

    # Flatten the inputs to 2D
    L = np.prod(shp)
    N1d = N.reshape((L,))
    p2d = p.reshape((L,K))
    out2d = out.reshape((L,K))

    _vec_typed_multinomial(rng, N1d, p2d, out2d)

    # Reshape the outputs
    out = out2d.reshape(shp + (K,))

    return out