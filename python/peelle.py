from numpy.random import lognormal, rand, randint, multivariate_normal, normal, poisson

import numpy as np
from numpy import linalg as la
import matplotlib.pylab as plt
# from matplotlib import pylab as plt
from scipy.special import comb

NSAMP = 200_000
def correl(sigmaN, n, nsamp=NSAMP, sigma=0.02, corr=1):
    nsamp = nsamp // n 
    alpha = normal(0, sigma, size=(nsamp, n))
    Sigma = sigmaN**2 * ((1-corr) * np.eye(n) + corr)
    beta = multivariate_normal(np.zeros((n,)), Sigma, size=nsamp)
    y = 1 + alpha + beta
    s2 = sigma ** 2 + (1-corr) * sigmaN**2 * y**2
    one = np.ones_like(y)
    S0 = np.sum(y ** 0 / s2, axis=1)
    S1 = np.sum(y ** 1 / s2, axis=1)
    S2 = np.sum(y ** 2 / s2, axis=1)
    ym = np.average(y, weights=1/s2, axis=1) 
    yfit = S1 / (S0 + corr * sigmaN**2 * (S2 * S0 - S1 ** 2))
    return yfit.mean() 

def peelle(sigmaN, n, nsamp=NSAMP, sigma=0.02):
    return correl(sigmaN, n, nsamp=nsamp, sigma=sigma, corr=1)

def uncorr(sigmaN, n, nsamp=NSAMP, sigma=0.02):
    return correl(sigmaN, n, nsamp=nsamp, sigma=sigma, corr=0)

def plot_approx(fig, corr, axnum=(1, 1), nsigma=30):
    ax = fig.axes[axnum[1]]
    M = [2, 5, 20, 100]
    FMT = ['k:', 'b:', 'm:', 'r:']
    sigma = np.linspace(1e-6, 0.5, nsigma)
    for m, fmt in zip(M, FMT):
        K = (1 - 1 / m) * (2 + corr * (m - 2))
        bias = 1 - K * sigma ** 2
        print(m, fmt, K, bias)
        ax.plot(sigma[keep], bias[keep], fmt)


def plot_bias(fig, corr, label, axnum=(1,1,1), nsigma=25, plot_taylor=False):
    ax = fig.add_subplot(*axnum)
    M = [2, 5, 20, 100]
    FMT = ['k-', 'k--', 'k-.', 'k:']
    sigma = np.linspace(1e-6, 0.25, nsigma)
    for m, fmt in zip(M, FMT):
        print(m, fmt)
        bias = [correl(s, m, corr=corr) for s in sigma]
        ax.plot(sigma, bias, fmt, label='{}'.format(m))
        if plot_taylor:
            K = (1 - 1 / m) * (2 + corr * (m - 2))
            bias = 1 - K * sigma ** 2
            keep = bias > 0.6
            ax.plot(sigma[keep], bias[keep], ':', alpha=0.5)
    ax.set_ylabel('$\\mu^\\star$')
    if axnum[2] == axnum[1]*axnum[0]:
        ax.set_xlabel('$\\sigma_\\tau$')
        ax.legend(title='\\# of points', loc=4, ncol=2)
    else:
        ax.set_xticklabels([])
    ax.set_xlim(0, 0.2499)
    ax.set_ylim(0, 1.1)
    ax.text(0.005, 0.07, label, ha='left', va='bottom', fontsize=8)

# Plot uncorrelated
def plot_figure2():
    plt.style.use('mnras')
    fig = plt.figure(2)
    fig.subplots_adjust(hspace=0)
    fig.clf()
    fig.subplots_adjust(wspace=0)
    plot_bias(fig, 1., 'correlated normalisation (Peelle)', 
                axnum=(2, 1, 1), plot_taylor=False)
    plot_bias(fig, 0., 'uncorrelated normalisation', 
                 axnum=(2, 1, 2), plot_taylor=False)
    fig.show()
    fig.savefig('../pdf/uncorrelated-peelle.pdf')

def plot_figure9(tau=1.25, V=0.8, errsys=0.05, errsta=0.005, nsigma=2.2):
    plt.style.use('mnras')
    fig = plt.figure(9)
    fig.subplots_adjust(bottom=0.01)
    fig.clf()
    dnu = round(nsigma/np.sqrt(2) * errsta / tau, 3)
    dV = tau * dnu
    nu = V / tau
    nu1 = nu - dnu 
    nu2 = nu + dnu
    V1 = tau * nu1
    V2 = tau * nu2
    print((V2-V1)/errsta/1.414)
    print('{:.4f} ± {:.4f}'.format(tau, tau*errsys))
    print('{:.4f} ± {:.4f} -- {:.4f} ± {:.4f}'.format(nu1, errsta/tau, nu2, errsta/tau))
    print('{:.4f} ± {:.4f} (± {:.4f}) -- {:.4f} ± {:.4f} (± {:.4f})'.format(V1, errsta, V1 * errsys, V2, errsta, V2 * errsys))
    V = (V1 + V2) / 2 / (1 + (nsigma*errsys) ** 2)
    eV = errsta / np.sqrt(2)
    print(nsigma, errsys, V)
    ax = fig.add_subplot(111)
    ax.errorbar([0.4, 1.6], [V1, V2], yerr=[errsta, errsta], fmt='ko')    
    ax.text(0.4, V1 + 1.1*errsta, '${\\scriptstyle V}_1$', va='bottom', ha='center')
    ax.text(1.6, V2 - 1.1*errsta, '${\\scriptstyle V}_2$', va='top', ha='center')
    ax.fill_between([0.3, 1.7], [V-eV, V-eV], [V+eV, V+eV], color=(.6,.6,.6)) 
    ax.plot([0.3, 1.7], [V, V], 'k--')
    ax.text(1.75, V, '${\\scriptstyle V}$', va='center', ha='left')
    ax.set_xticks([])
    ax.set_yticks([0.79, .8, .81])
    ax.set_ylabel('visibility amplitude')
    ax.set_xlim(0.1, 1.9)
    fig.show()
    fig.savefig('../pdf/original-peelle.pdf')

# plot_figure2()
plot_figure9()
