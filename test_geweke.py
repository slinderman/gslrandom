import os
import numpy as np
import numpy.random as rn
rn.seed(1234)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

from gslrandom import seeded_multinomial, multinomial, PyRNG, get_omp_num_threads

if "OMP_NUM_THREADS" in os.environ:
    num_threads = int(os.environ["OMP_NUM_THREADS"])
else:
    num_threads = get_omp_num_threads()
assert num_threads > 0


def pp_plot(F_dict, G_dict, title=None, xlabel=None, ylabel=None, file_name=None):
    """Generates a P-P plot for given functions F and G.

    Arguments:
        F -- List of samples from function F.
        G -- List of samples from function G.
    """
    n_plots = len(F_dict.keys())
    x = int(np.ceil(np.sqrt(n_plots)))
    fig = plt.figure()
    if title is not None:
        fig.suptitle(title, size=20)
    gs = gridspec.GridSpec(x, x, wspace=0.0, hspace=0.0)
    # if xlabel is None:
    #     xlabel = 'CDF of generative samples'
    # if ylabel is None:
    #     ylabel = 'CDF of inferential samples'

    for n, (i, j) in enumerate(np.ndindex(x, x)):
        if n < n_plots:
            key = F_dict.keys()[n]
            F = np.array(F_dict[key])
            G = np.array(G_dict[key])
            F.sort()

            F_cdf = [np.mean(F < f) for f in F]
            G_cdf = [np.mean(G < f) for f in F]
            ax = fig.add_subplot(gs[i, j])
            ax.set_title(key)
            ax.plot(F_cdf, G_cdf, 'b.', lw=0.005)
            ax.plot([-0.05, 1.05], [-0.05, 1.05], 'g--', lw=1.5)

        if i == x - 1:
            # ax.set_xlabel(xlabel)
            plt.setp(ax.get_xticklabels(), fontsize=8)
        else:
            plt.setp(ax.get_xticklabels(), visible=False)

        if j == 0:
            # ax.set_ylabel(ylabel)
            plt.setp(ax.get_yticklabels(), fontsize=8)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)

    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name)
        plt.close()


def forward_sample(N_I, alpha_K):
    I = N_I.size
    K = alpha_K.size
    p_IK = rn.dirichlet(alpha_K, size=I)
    n_IK = np.ones((I, K), dtype=np.uint32)
    for i in xrange(I):
        n_IK[i] = rn.multinomial(N_I[i], p_IK[i])
    return p_IK, n_IK


def backward_sample(n_IK, alpha_K, n_gibbs_iter=10, version='multinomial', rngs=None):
    I, K = n_IK.shape
    N_I = n_IK.sum(axis=1)
    p_IK = np.zeros((I, K))
    for itn in xrange(n_gibbs_iter):
        for i in xrange(I):
            p_IK[i] = rn.dirichlet(alpha_K + n_IK[i])

        if version == 'multinomial':
            assert rngs is not None
            multinomial(rngs, N_I, p_IK, out=n_IK)

        elif version == 'seeded_multinomial':
            seeded_multinomial(N_I, p_IK, out=n_IK)

        elif version == 'numpy':
            for i in xrange(I):
                n_IK[i] = rn.multinomial(N_I[i], p_IK[i])
        else:
            raise TypeError
    return p_IK, n_IK


def calc_funcs(n_IK, p_IK, func_dict):
    func_dict['n_var'].append(n_IK.var())
    func_dict['n_max'].append(n_IK.max())
    func_dict['n_median'].append(np.median(n_IK))
    func_dict['p_entropy'].append(-(p_IK * np.log(p_IK)).sum())
    func_dict['p_var'].append(p_IK.var())
    func_dict['p_max'].append(p_IK.max())


def geweke_test(n_iter=20000, n_gibbs_iter=10, version='multinomial'):
    I = 10
    K = 4
    alpha_K = np.ones(K)
    N_I = np.random.poisson(100, size=I).astype(np.uint32)

    rngs = None
    if version == 'multinomial':
        rngs = [PyRNG(rn.randint(2**16)) for _ in xrange(num_threads)]

    forward_dict = defaultdict(list)
    backward_dict = defaultdict(list)

    for itn in xrange(n_iter):
        if itn % 100 == 0:
            print 'ITERATION %d' % itn
        p_IK, n_IK = forward_sample(N_I, alpha_K)
        calc_funcs(n_IK, p_IK, forward_dict)

        p_IK, n_IK = backward_sample(n_IK,
                                     alpha_K,
                                     n_gibbs_iter=n_gibbs_iter,
                                     version=version,
                                     rngs=rngs)
        calc_funcs(n_IK, p_IK, backward_dict)

    pp_plot(forward_dict, backward_dict)

if __name__ == '__main__':
    geweke_test(version='seeded_multinomial')
