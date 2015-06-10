"""
A simple Geweke test. I'm afraid there might be a slight bias toward
sampling the first entry in the multinomial.
"""
import os
import numpy as np
np.random.seed(1234)
from scipy.stats import probplot, beta
import matplotlib.pyplot as plt

from pybasicbayes.distributions import Multinomial
from pybasicbayes.util.text import progprint_xrange

from gslrandom import multinomial_par, multinomial,PyRNG, get_omp_num_threads
if "OMP_NUM_THREADS" in os.environ:
    num_threads = int(os.environ["OMP_NUM_THREADS"])
else:
    num_threads = get_omp_num_threads()
assert num_threads > 0

import ipdb; ipdb.set_trace()

# Choose random seeds
seeds = np.random.randint(2**16, size=num_threads)
pyrngs = [PyRNG(seed) for seed in seeds]

alpha = 1.
K = 3
N = 100
Ns = np.random.poisson(10, size=N).astype(np.uint32)
# Ns = np.ones(N).astype(np.uint32)

# Sample model
dirichlet = Multinomial(alphav_0=alpha*np.ones(K), K=K)
X = np.zeros((N, K), dtype=np.uint32)
multinomial_par(pyrngs, Ns, dirichlet.weights * np.ones((N,K)), X)

N_iter = 50000
samplers = ["numpy", "multinomial", "multinomial_par"]
fig = plt.figure()
for i,sampler in enumerate(samplers):
    print "Testing ", sampler
    ps = []
    for itr in progprint_xrange(N_iter, perline=50):
        # Resample the dirichlet
        dirichlet.resample(X)

        # Resample X
        if sampler == "numpy":
            for n,x in zip(Ns,X):
                x[:] = np.random.multinomial(n, dirichlet.weights)
        elif sampler == "multinomial":
            multinomial(pyrngs[0], Ns, dirichlet.weights * np.ones((N,K)), out=X)
        elif sampler == "multinomial_par":
            multinomial_par(pyrngs, Ns, dirichlet.weights * np.ones((N,K)), X)
        else:
            raise Exception("invalid sampler")

        # Get sample
        ps.append(dirichlet.weights.copy())

    ps = np.array(ps)
    print np.mean(ps, axis=0)
    print np.std(ps, axis=0)

    for k in xrange(K):
        ax = fig.add_subplot(K,len(samplers),i*K+k+1)
        marg_p = beta(alpha, (K-1)*alpha)
        probplot(ps[:,k], dist=marg_p, plot=ax)
        ax.set_title(sampler + "_%d" % k)

plt.show()