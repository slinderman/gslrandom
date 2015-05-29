from gslrandom import PyRNG, multinomial, dumb_multinomial, multinomial_par, get_omp_num_threads
import numpy as np

def test_simple():
    # One N count, one p array, out structure NOT provided
    N = 10
    K = 5
    p = 1./K * np.ones(K)
    rng = PyRNG()

    n_iter = 10000
    z = np.zeros(K)
    for _ in xrange(n_iter):
        n = multinomial(rng, N, p)
        assert n.sum() == N
        z += n
    assert (np.abs(z/z.sum() - p) < 1e-2).all()

def test_simple_with_out():
    # One N count, one p array, out structure provided
    N = 10
    K = 5
    p = 1./K * np.ones(K)
    rng = PyRNG()

    n_iter = 10000
    z = np.zeros(K)
    for _ in xrange(n_iter):
        out = np.zeros(K, dtype=np.float) 
        multinomial(rng, N, p, out)
        assert out.sum() == N
        z += out
    assert (np.abs(z/z.sum() - p) < 1e-2).all()

def test_multi_N_single_p():
    # Multiple N counts, one p array, out structure NOT provided
    L = 10
    N = np.arange(L) + 10
    K = 5
    p = 1./K * np.ones(K)
    rng = PyRNG()

    n_iter = 10000
    z = np.zeros((L,K))
    for _ in xrange(n_iter):
        n = multinomial(rng, N, p)
        assert n.shape == (L,K)
        assert (n.sum(axis=1) == N).all()
        z += n
    assert (np.abs(z/z.sum(axis=1)[:,np.newaxis] - p) < 1e-2).all()

def test_multi_N_single_p_with_out():
    # Multiple N counts, one p array, out structure provided
    L = 10
    N = np.arange(L) + 10
    K = 5
    p = 1./K * np.ones(K)
    rng = PyRNG()

    n_iter = 10000
    z = np.zeros((L,K))
    for _ in xrange(n_iter):
        out = np.zeros((L,K), dtype=np.uint32)
        multinomial(rng, N, p, out)
        assert (out.sum(axis=1) == N).all()
        z += out
    assert (np.abs(z/z.sum(axis=1)[:,np.newaxis] - p) < 1e-2).all()

def test_single_N_multi_p():
    # One N count, multiple p arrays, out structure NOT provided
    N = 10
    K = 5
    p = np.zeros((2, K))
    p[0,:] = 1./K * np.ones(K)
    p[1,:] = np.asarray([0.5, 0.25, 0.05, 0.1, 0.1])
    rng = PyRNG()

    n_iter = 10000
    z = np.zeros((2,K))
    for _ in xrange(n_iter):
        n = multinomial(rng, N, p)
        assert n.shape == (2,K)
        assert (n.sum(axis=1) == N).all()
        z += n
    assert (np.abs(z/z.sum(axis=1)[:,np.newaxis] - p) < 1e-2).all()

def test_single_N_multi_p_with_out():
    # One N count, multiple p arrays, out structure provided
    N = 10
    K = 5
    p = np.zeros((2, K))
    p[0,:] = 1./K * np.ones(K)
    p[1,:] = np.asarray([0.5, 0.25, 0.05, 0.1, 0.1])
    rng = PyRNG()

    n_iter = 10000
    z = np.zeros((2,K))
    for _ in xrange(n_iter):
        out = np.zeros((2,K), dtype=np.uint32)
        multinomial(rng, N, p, out)
        assert out.shape == (2,K)
        assert (out.sum(axis=1) == N).all()
        z += out
    assert (np.abs(z/z.sum(axis=1)[:,np.newaxis] - p) < 1e-2).all()

def test_multi_N_multi_p():
    # Multiple N counts, multiple p arrays, out structure NOT provided
    L = 10
    N = np.arange(L) + 10
    K = 5
    p = np.zeros((L, K))
    p[:5] = 1./K * np.ones(K)
    p[5:] = np.asarray([0.5, 0.25, 0.05, 0.1, 0.1])
    rng = PyRNG()

    n_iter = 10000
    z = np.zeros((L,K))
    for _ in xrange(n_iter):
        n = multinomial(rng, N, p)
        assert n.shape == (L,K)
        assert (n.sum(axis=1) == N).all()
        z += n
    assert (np.abs(z/z.sum(axis=1)[:,np.newaxis] - p) < 1e-2).all()

def test_multi_N_multi_p_with_out():
    # Multiple N counts, multiple p arrays, out structure provided
    L = 10
    N = np.arange(L) + 10
    K = 5
    p = np.zeros((L, K))
    p[:5] = 1./K * np.ones(K)
    p[5:] = np.asarray([0.5, 0.25, 0.05, 0.1, 0.1])
    rng = PyRNG()

    n_iter = 10000
    z = np.zeros((L,K))
    for _ in xrange(n_iter):
        out = np.zeros((L,K), dtype=np.uint32)
        multinomial(rng, N, p, out)
        assert out.shape == (L,K)
        assert (out.sum(axis=1) == N).all()
        z += out
    assert (np.abs(z/z.sum(axis=1)[:,np.newaxis] - p) < 1e-2).all()

def test_dumb_multi_N_multi_p_with_out():
    # Test the "dumb" implementation (no PyRNGs)
    # Currently failing: seeding new RNGs everytime leads to biases
    # Multiple N counts, multiple p arrays, out structure provided
    L = 10
    N = np.arange(L, dtype=np.uint32) + 10
    K = 5
    p = np.zeros((L, K))
    p[:5] = 1./K * np.ones(K)
    p[5:] = np.asarray([0.5, 0.25, 0.05, 0.1, 0.1])

    n_iter = 10000
    z = np.zeros((L,K))
    for _ in xrange(n_iter):
        out = np.zeros((L,K), dtype=np.uint32)
        dumb_multinomial(N, p, out)
        assert out.shape == (L,K)
        assert (out.sum(axis=1) == N).all()
        z += out
    print z/z.sum(axis=1)[:,np.newaxis]
    assert (np.abs(z/z.sum(axis=1)[:,np.newaxis] - p) < 1e-2).all()


def test_parallel_multi_N_multi_p_with_out():
    # Multiple N counts, multiple p arrays, out structure provided
    L = 10
    N = np.arange(L, dtype=np.uint32) + 10
    K = 5
    p = np.zeros((L, K))
    p[:5] = 1./K * np.ones(K)
    p[5:] = np.asarray([0.5, 0.25, 0.05, 0.1, 0.1])

    # Create some RNGs
    rngs = [PyRNG() for _ in xrange(get_omp_num_threads())]

    n_iter = 10000
    z = np.zeros((L,K))
    for _ in xrange(n_iter):
        out = np.zeros((L,K), dtype=np.uint32)
        multinomial_par(rngs, N, p, out)
        assert out.shape == (L,K)
        assert (out.sum(axis=1) == N).all()
        z += out

    print z/z.sum(axis=1)[:,np.newaxis]
    assert (np.abs(z/z.sum(axis=1)[:,np.newaxis] - p) < 1e-2).all()

if __name__ == '__main__':
    # test_simple()
    # test_simple_with_out()
    # test_multi_N_single_p()
    # test_multi_N_single_p_with_out()
    # test_single_N_multi_p()
    # test_single_N_multi_p_with_out()
    # test_multi_N_multi_p()
    # test_multi_N_multi_p_with_out()
    # test_dumb_multi_N_multi_p_with_out() # FAILS 
    test_parallel_multi_N_multi_p_with_out()