from gslrandom import PyRNG, get_omp_num_threads, multinomial, seeded_multinomial
import numpy as np
import numpy.random as rn

"""
Tests for the different multinomial methods in gslrandom.pyx.

The Python method multinomial calls the Cython methods:
    _multinomial (for the one N, one p case)
    _vec_multinomial (for the multi N and/or multi p case and only one PyRNG is given)
    -par_vec_multinomial(for the multi N and/or multi p case and multiple PyRNGs are given)

These tests cover all the above cases, so calling multinomial tests all the Cython methods.

The Python method seeded_multinomial calls _seeded_par_vec_multinomial.
"""

#################################################
#               multinomial tests               #
#################################################


def test_one_rng_one_N_one_p_no_out():
    K = 5
    N = 10
    p_K = np.ones(K) / K
    rng = PyRNG(rn.randint(2**16))

    n_iter = 10000
    z_K = np.zeros(K)
    for _ in xrange(n_iter):
        n_K = multinomial(rng, N, p_K)
        assert n_K.sum() == N
        z_K += n_K
    assert np.allclose(z_K / z_K.sum(), p_K, atol=1e-2)


def test_one_rng_one_N_one_p_with_out():
    K = 5
    N = 10
    p_K = np.ones(K) / K
    rng = PyRNG(rn.randint(2**16))

    n_iter = 10000
    z_K = np.zeros(K)
    for _ in xrange(n_iter):
        n_K = np.zeros(K, dtype=np.float)
        multinomial(rng, N, p_K, out=n_K)
        assert n_K.sum() == N
        z_K += n_K
    assert np.allclose(z_K / z_K.sum(), p_K, atol=1e-2)


def test_one_rng_multi_N_one_p_no_out():
    K = 5
    I = 10
    N_I = np.ones(I) * 10
    p_K = np.ones(K) / K
    rng = PyRNG(rn.randint(2**16))

    n_iter = 10000
    z_IK = np.zeros((I, K))
    for _ in xrange(n_iter):
        n_IK = multinomial(rng, N_I, p_K)
        assert n_IK.shape == (I, K)
        assert np.allclose(n_IK.sum(axis=1), N_I)
        z_IK += n_IK
    norm_z_IK = z_IK.astype(float) / np.sum(z_IK, axis=1, keepdims=True)
    p_IK = np.ones((I, K)) * p_K
    assert np.allclose(norm_z_IK, p_IK, atol=1e-2)


def test_one_rng_multi_N_one_p_with_out():
    K = 5
    I = 10
    N_I = np.ones(I) * 10
    p_K = np.ones(K) / K
    rng = PyRNG(rn.randint(2**16))

    n_iter = 10000
    z_IK = np.zeros((I, K))
    for _ in xrange(n_iter):
        n_IK = np.zeros((I, K))
        multinomial(rng, N_I, p_K, out=n_IK)
        assert np.allclose(n_IK.sum(axis=1), N_I)
        z_IK += n_IK
    norm_z_IK = z_IK.astype(float) / np.sum(z_IK, axis=1, keepdims=True)
    p_IK = np.ones((I, K)) * p_K
    assert np.allclose(norm_z_IK, p_IK, atol=1e-2)


def test_one_rng_one_N_multi_p_no_out():
    K = 5
    I = 3
    N = 10
    p_IK = np.ones((I, K)) / K
    p_IK[0, :] = [0.5, 0.25, 0.05, 0.1, 0.1]  # make one non-uniform
    rng = PyRNG(rn.randint(2**16))

    n_iter = 10000
    z_IK = np.zeros((I, K))
    for _ in xrange(n_iter):
        n_IK = multinomial(rng, N, p_IK)
        assert n_IK.shape == (I, K)
        assert (n_IK.sum(axis=1) == N).all()
        z_IK += n_IK
    norm_z_IK = z_IK.astype(float) / np.sum(z_IK, axis=1, keepdims=True)
    assert np.allclose(norm_z_IK, p_IK, atol=1e-2)


def test_one_rng_one_N_multi_p_with_out():
    K = 5
    I = 3
    N = 10
    p_IK = np.ones((I, K)) / K
    p_IK[0, :] = [0.5, 0.25, 0.05, 0.1, 0.1]  # make one non-uniform
    rng = PyRNG(rn.randint(2**16))

    n_iter = 10000
    z_IK = np.zeros((I, K))
    for _ in xrange(n_iter):
        n_IK = np.zeros((I, K))
        multinomial(rng, N, p_IK, out=n_IK)
        assert (n_IK.sum(axis=1) == N).all()
        z_IK += n_IK
    norm_z_IK = z_IK.astype(float) / np.sum(z_IK, axis=1, keepdims=True)
    assert np.allclose(norm_z_IK, p_IK, atol=1e-2)


def test_one_rng_multi_N_multi_p_no_out():
    K = 5
    I = 3
    N_I = np.arange(1, I+1) * 10
    p_IK = np.ones((I, K)) / K
    p_IK[0, :] = [0.5, 0.25, 0.05, 0.1, 0.1]  # make one non-uniform
    rng = PyRNG(rn.randint(2**16))

    n_iter = 10000
    z_IK = np.zeros((I, K))
    for _ in xrange(n_iter):
        n_IK = multinomial(rng, N_I, p_IK)
        assert n_IK.shape == (I, K)
        np.allclose(n_IK.sum(axis=1), N_I)
        z_IK += n_IK
    norm_z_IK = z_IK.astype(float) / np.sum(z_IK, axis=1, keepdims=True)
    assert np.allclose(norm_z_IK, p_IK, atol=1e-2)


def test_one_rng_multi_N_multi_p_with_out():
    K = 5
    I = 3
    N_I = np.arange(1, I+1) * 10
    p_IK = np.ones((I, K)) / K
    p_IK[0, :] = [0.5, 0.25, 0.05, 0.1, 0.1]  # make one non-uniform
    rng = PyRNG(rn.randint(2**16))

    n_iter = 10000
    z_IK = np.zeros((I, K))
    for _ in xrange(n_iter):
        n_IK = np.zeros((I, K))
        multinomial(rng, N_I, p_IK, out=n_IK)
        np.allclose(n_IK.sum(axis=1), N_I)
        z_IK += n_IK
    norm_z_IK = z_IK.astype(float) / np.sum(z_IK, axis=1, keepdims=True)
    assert np.allclose(norm_z_IK, p_IK, atol=1e-2)


def test_multi_rng_multi_N_multi_p_no_out():
    K = 5
    A = 3
    B = 2
    N_AB = (np.arange(1, A*B+1) * 10).reshape((A, B))
    p_ABK = np.ones((A, B, K)) / K
    p_ABK[0, 1, :] = [0.5, 0.25, 0.05, 0.1, 0.1]    # make one non-uniform
    p_ABK[1, 0, :] = [0.9, 0.05, 0.03, 0.01, 0.01]  # make one really non-uniform
    rngs = [PyRNG(rn.randint(2**16)) for _ in xrange(get_omp_num_threads())]

    n_iter = 10000
    z_ABK = np.zeros((A, B, K))
    for _ in xrange(n_iter):
        n_ABK = multinomial(rngs, N_AB, p_ABK)
        assert n_ABK.shape == (A, B, K)
        np.allclose(n_ABK.sum(axis=-1), N_AB)
        z_ABK += n_ABK
    norm_z_ABK = z_ABK.astype(float) / np.sum(z_ABK, axis=-1, keepdims=True)
    assert np.allclose(norm_z_ABK, p_ABK, atol=1e-2)


def test_multi_rng_multi_N_multi_p_with_out():
    K = 5
    A = 3
    B = 2
    N_AB = (np.arange(1, A*B+1) * 10).reshape((A, B))
    p_ABK = np.ones((A, B, K)) / K
    p_ABK[0, 1, :] = [0.5, 0.25, 0.05, 0.1, 0.1]  # make one non-uniform
    p_ABK[1, 0, :] = [0.9, 0.05, 0.03, 0.01, 0.01]  # make one really non-uniform
    rngs = [PyRNG(rn.randint(2**16)) for _ in xrange(get_omp_num_threads())]

    n_iter = 10000
    z_ABK = np.zeros((A, B, K))
    for _ in xrange(n_iter):
        n_ABK = np.zeros((A, B, K), dtype=np.uint32)
        multinomial(rngs, N_AB, p_ABK, out=n_ABK)
        np.allclose(n_ABK.sum(axis=-1), N_AB)
        z_ABK += n_ABK
    norm_z_ABK = z_ABK.astype(float) / np.sum(z_ABK, axis=-1, keepdims=True)
    assert np.allclose(norm_z_ABK, p_ABK, atol=1e-2)

#################################################
#               seeded_multinomial tests        #
#################################################


def test_seeded_one_N_one_p_no_out():
    K = 5
    N = 10
    p_K = np.ones(K) / K

    n_iter = 10000
    z_K = np.zeros(K)
    for _ in xrange(n_iter):
        n_K = seeded_multinomial(N, p_K)
        assert n_K.sum() == N
        z_K += n_K
    assert np.allclose(z_K / z_K.sum(), p_K, atol=1e-2)


def test_seeded_one_N_one_p_with_out():
    K = 5
    N = 10
    p_K = np.ones(K) / K

    n_iter = 10000
    z_K = np.zeros(K)
    for _ in xrange(n_iter):
        n_K = np.zeros(K, dtype=np.float)
        seeded_multinomial(N, p_K, out=n_K)
        assert n_K.sum() == N
        z_K += n_K
    assert np.allclose(z_K / z_K.sum(), p_K, atol=1e-2)


def test_seeded_multi_N_one_p_no_out():
    K = 5
    I = 10
    N_I = np.ones(I) * 10
    p_K = np.ones(K) / K

    n_iter = 10000
    z_IK = np.zeros((I, K))
    for _ in xrange(n_iter):
        n_IK = seeded_multinomial(N_I, p_K)
        assert n_IK.shape == (I, K)
        assert np.allclose(n_IK.sum(axis=1), N_I)
        z_IK += n_IK
    norm_z_IK = z_IK.astype(float) / np.sum(z_IK, axis=1, keepdims=True)
    p_IK = np.ones((I, K)) * p_K
    assert np.allclose(norm_z_IK, p_IK, atol=1e-2)


def test_seeded_multi_N_one_p_with_out():
    K = 5
    I = 10
    N_I = np.ones(I) * 10
    p_K = np.ones(K) / K

    n_iter = 10000
    z_IK = np.zeros((I, K))
    for _ in xrange(n_iter):
        n_IK = np.zeros((I, K))
        seeded_multinomial(N_I, p_K, out=n_IK)
        assert np.allclose(n_IK.sum(axis=1), N_I)
        z_IK += n_IK
    norm_z_IK = z_IK.astype(float) / np.sum(z_IK, axis=1, keepdims=True)
    p_IK = np.ones((I, K)) * p_K
    assert np.allclose(norm_z_IK, p_IK, atol=1e-2)


def test_seeded_one_N_multi_p_no_out():
    K = 5
    I = 3
    N = 10
    p_IK = np.ones((I, K)) / K
    p_IK[0, :] = [0.5, 0.25, 0.05, 0.1, 0.1]  # make one non-uniform

    n_iter = 10000
    z_IK = np.zeros((I, K))
    for _ in xrange(n_iter):
        n_IK = seeded_multinomial(N, p_IK)
        assert n_IK.shape == (I, K)
        assert (n_IK.sum(axis=1) == N).all()
        z_IK += n_IK
    norm_z_IK = z_IK.astype(float) / np.sum(z_IK, axis=1, keepdims=True)
    assert np.allclose(norm_z_IK, p_IK, atol=1e-2)


def test_seeded_one_N_multi_p_with_out():
    K = 5
    I = 3
    N = 10
    p_IK = np.ones((I, K)) / K
    p_IK[0, :] = [0.5, 0.25, 0.05, 0.1, 0.1]  # make one non-uniform

    n_iter = 10000
    z_IK = np.zeros((I, K))
    for _ in xrange(n_iter):
        n_IK = np.zeros((I, K))
        seeded_multinomial(N, p_IK, out=n_IK)
        assert (n_IK.sum(axis=1) == N).all()
        z_IK += n_IK
    norm_z_IK = z_IK.astype(float) / np.sum(z_IK, axis=1, keepdims=True)
    assert np.allclose(norm_z_IK, p_IK, atol=1e-2)


def test_seeded_multi_N_multi_p_no_out():
    K = 5
    A = 3
    B = 2
    N_AB = (np.arange(1, A*B+1) * 10).reshape((A, B))
    p_ABK = np.ones((A, B, K)) / K
    p_ABK[0, 1, :] = [0.5, 0.25, 0.05, 0.1, 0.1]  # make one non-uniform
    p_ABK[1, 0, :] = [0.9, 0.05, 0.03, 0.01, 0.01]  # make one really non-uniform

    n_iter = 10000
    z_ABK = np.zeros((A, B, K))
    for _ in xrange(n_iter):
        n_ABK = seeded_multinomial(N_AB, p_ABK)
        assert n_ABK.shape == (A, B, K)
        np.allclose(n_ABK.sum(axis=-1), N_AB)
        z_ABK += n_ABK
    norm_z_ABK = z_ABK.astype(float) / np.sum(z_ABK, axis=-1, keepdims=True)
    assert np.allclose(norm_z_ABK, p_ABK, atol=1e-2)


def test_seeded_multi_N_multi_p_with_out():
    K = 5
    A = 3
    B = 2
    N_AB = (np.arange(1, A*B+1) * 10).reshape((A, B))
    p_ABK = np.ones((A, B, K)) / K
    p_ABK[0, 1, :] = [0.5, 0.25, 0.05, 0.1, 0.1]  # make one non-uniform
    p_ABK[1, 0, :] = [0.9, 0.05, 0.03, 0.01, 0.01]  # make one really non-uniform

    n_iter = 10000
    z_ABK = np.zeros((A, B, K))
    for _ in xrange(n_iter):
        n_ABK = np.zeros((A, B, K), dtype=np.uint32)
        seeded_multinomial(N_AB, p_ABK, out=n_ABK)
        np.allclose(n_ABK.sum(axis=-1), N_AB)
        z_ABK += n_ABK
    norm_z_ABK = z_ABK.astype(float) / np.sum(z_ABK, axis=-1, keepdims=True)
    assert np.allclose(norm_z_ABK, p_ABK, atol=1e-2)


if __name__ == '__main__':
    test_one_rng_one_N_one_p_no_out()
    test_one_rng_one_N_one_p_with_out()
    test_one_rng_multi_N_one_p_no_out()
    test_one_rng_multi_N_one_p_with_out()
    test_one_rng_one_N_multi_p_no_out()
    test_one_rng_one_N_multi_p_with_out()
    test_one_rng_multi_N_multi_p_no_out()
    test_one_rng_multi_N_multi_p_with_out()
    test_multi_rng_multi_N_multi_p_no_out()
    test_multi_rng_multi_N_multi_p_with_out()

    test_seeded_one_N_one_p_no_out()
    test_seeded_one_N_one_p_with_out()
    test_seeded_multi_N_one_p_no_out()
    test_seeded_multi_N_one_p_with_out()
    test_seeded_one_N_multi_p_no_out()
    test_seeded_one_N_multi_p_with_out()
    test_seeded_multi_N_multi_p_no_out()
    test_seeded_multi_N_multi_p_with_out()
    test_seeded_multi_N_multi_p_no_out()
    test_seeded_multi_N_multi_p_with_out()
