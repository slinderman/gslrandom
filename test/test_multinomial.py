from gslrandom import PyRNG, multinomial
import numpy as np

### Simple test
N = 10
K = 5
p = 1./K * np.ones(K)

rng = PyRNG()
n = multinomial(rng, N, p)
assert n.sum() == N
print "n 1D:", n

### N int p 2D?
L = 100000
N = 1000
K = 10
p = 1./K * np.ones((L,K))

rng = PyRNG()
n = multinomial(rng, N, p)
assert n.shape == (L,K)
assert np.all(n.sum(axis=-1) == N)
p_emp = n.mean(0) / N
err = np.amax(abs(p-p_emp))
print "n:    ",   n
print "E[p]: ", p_emp
assert err < 1e-3
