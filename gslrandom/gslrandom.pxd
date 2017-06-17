# distutils: language = c++
# distutils: sources = gslrandom/cgslrandom.cpp
# distutils: libraries = gsl gslcblas
# distutils: library_dirs = /usr/local/lib
# distutils: include_dirs =  /usr/local/include gslrandom/
# distutils: extra_compile_args = -O3 -w -fopenmp
# distutils: extra_link_args = -fopenmp
# cython: boundscheck = False
# cython: wraparound = True
# cython: cdivision = True

cdef extern from "cgslrandom.h":
    cdef cppclass BasicRNG:
        BasicRNG(unsigned long seed) except +

cdef class PyRNG:
    cdef BasicRNG *thisptr