from gslrandom import PyRNG, multinomial, dumb_multinomial
import numpy as np
import numpy.random as rn
import time

I = 100000
K = 1000

N_I = rn.poisson(rn.gamma(0.8, 100, size=I)).astype(np.uint32)
N_I = np.ones(I, dtype=np.uint32)*1000
P_IK = 1./K * np.ones((I, K), dtype=np.float)

N_IK = np.ones((I,K), dtype=np.uint32)
s = time.time()
dumb_multinomial(N_I, P_IK, N_IK)
print '%fs: PyGSL with many RNGs'%(time.time() - s)
assert (N_IK.sum(axis=1) == N_I).all()

rng = PyRNG()
N_IK = np.ones((I,K), dtype=np.uint32)
s = time.time()
multinomial(rng, N_I, P_IK, N_IK)
print '%fs: PyGSL w/o asserts/checks/reshapes'%(time.time() - s)
assert (N_IK.sum(axis=1) == N_I).all()

rng = PyRNG()
N_IK = np.ones((I,K), dtype=np.uint32)
s = time.time()
N_IK[:] = multinomial(rng, N_I.astype(np.uint64), P_IK, None)
print '%fs: PyGSL with asserts/checks/reshapes'%(time.time() - s)
assert (N_IK.sum(axis=1) == N_I).all()

rng = PyRNG()
N_IK = np.ones((I,K), dtype=np.uint32)
s = time.time()
for i in xrange(I):
    N_IK[i,:] = rn.multinomial(N_I[i], P_IK[i,:])
print '%fs: Numpy'%(time.time() - s)
assert (N_IK.sum(axis=1) == N_I).all()





