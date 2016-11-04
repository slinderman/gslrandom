from gslrandom import PyRNG, multinomial, seeded_multinomial, get_omp_num_threads
import numpy as np
import numpy.random as rn
import time

I = 100000
K = 1000

N_I = rn.poisson(rn.gamma(2, 500, size=I)).astype(np.uint32)
P_IK = 1./K * np.ones((I, K), dtype=np.float)

N_IK = np.ones((I, K), dtype=np.uint32)
s = time.time()
seeded_multinomial(N_I, P_IK, N_IK)
print(('%fs: No PyRNG parallel version with %d cores' % (time.time() - s, get_omp_num_threads())))
assert (N_IK.sum(axis=1) == N_I).all()

rngs = [PyRNG(rn.randint(2**16)) for _ in range(get_omp_num_threads())]
N_IK = np.ones((I, K), dtype=np.uint32)
s = time.time()
multinomial(rngs, N_I, P_IK, N_IK)
print(('%fs: PyRNG parallel version with %d cores' % (time.time() - s, len(rngs))))
assert (N_IK.sum(axis=1) == N_I).all()

rng = PyRNG(rn.randint(2**16))
N_IK = np.ones((I, K), dtype=np.uint32)
s = time.time()
multinomial(rng, N_I, P_IK, N_IK)
print(('%fs: PyRNG version with 1 core' % (time.time() - s)))
assert (N_IK.sum(axis=1) == N_I).all()

rng = PyRNG()
N_IK = np.ones((I, K), dtype=np.uint32)
s = time.time()
for i in range(I):
    N_IK[i, :] = rn.multinomial(N_I[i], P_IK[i, :])
print(('%fs: Numpy version with 1 cores' % (time.time() - s)))
assert (N_IK.sum(axis=1) == N_I).all()
