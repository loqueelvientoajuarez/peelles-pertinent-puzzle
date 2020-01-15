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

def plot_figure9(tau=2, nu=0.5, sigma_sys=0.05, sigma_sta=0.006, 
        dnu=0.01, save=False):
    plt.style.use('mnras')
    fig = plt.figure(9)
    fig.clf()
    fig.subplots_adjust(bottom=0.01,hspace=0)
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    dtau_sys = tau * sigma_sys
    dnu_sta = sigma_sta * nu
    dnu = dnu * nu
    nu1, nu2 = nu - dnu, nu + dnu
    dV_sta = tau * dnu_sta
    V1, V2 = tau * nu1, tau * nu2
    dV1_sys, dV2_sys = V1 * sigma_sys, V2 * sigma_sys
    nsigma = abs(V2 - V1) / (np.sqrt(2) * dV_sta)
    mu = (V1 + V2) / 2 / (1 + (sigma_sys * nsigma) ** 2)
    dmu_sta = dV_sta / np.sqrt(2) 
    dmu_sys = mu * sigma_sys
    print('tau = {:.4f} ± {:.4f}'.format(tau, dtau_sys))
    print('nu =  {:.4f} ± {:.4f} -- {:.4f} ± {:.4f}'.format(nu1, dnu_sta, nu2, dnu_sta))
    print('V =   {:.4f} ± {:.4f} (± {:.4f}) -- {:.4f} ± {:.4f} (± {:.4f})'.format(V1, dV_sta, dV1_sys, V2, dV_sta, dV2_sys))
    print('mu =  {:.4f} ± {:.4f} ± {:.4f}'.format(mu, dmu_sta, dmu_sys))
    print('nsigma = {:.1f}'.format(nsigma))
    gray = (.7,.7,.7)
    lblue = gray
    # raw visibilities
    ax2.errorbar([0.4, 1.6], [nu1, nu2], yerr=[dnu_sta, dnu_sta], fmt='k.')
    ax2.text(0.4, nu1 - 2*dnu_sta, '$\\nu_1$', 
            va='top', ha='center')
    ax2.text(1.6, nu2 + 2*dnu_sta, '$\\nu_2$', 
            va='bottom', ha='center')
    # transfer function
    t = 1/tau
    t1, t2 = t/(1+sigma_sys), t/(1-sigma_sys)
    ax2.fill_between([0.3, 1.7], [t1, t1], [t2, t2], color=lblue)
    ax2.plot([0.3, 1.7], [t, t], 'k-')
    ax2.text(1.75, 1/tau, '$1/\\tau$', va='center', ha='left')
    ax2.set_xticks([])
    # ax.set_yticks([0.79, .8, .81])
    # ax.set_ylabel('visibility amplitude')
    # reduced visibilities 
    ax1.errorbar([0.4, 1.6], [V1, V2], yerr=[dV_sta, dV_sta], fmt='k.')    
    ax1.text(0.4, V1 + 1.5*dV_sta, '${\\scriptstyle V}_1$', 
            va='bottom', ha='center')
    ax1.text(1.6, V2 + 1.5*dV_sta, '${\\scriptstyle V}_2$', 
            va='bottom', ha='center')
    mu1, mu2 = mu - dmu_sta, mu + dmu_sta
    ax1.fill_between([0.3, 1.7], [mu1, mu1], [mu2, mu2], color=gray) 
    ax1.plot([0.3, 1.7], [mu, mu], 'k--')
    ax1.text(1.75, mu, '${\\scriptstyle V}$', va='center', ha='left')
    ax1.set_xlim(0.2, 1.95)
    ax1.set_ylim(0.97, 1.03)
    ax1.set_yticks([0.98, 1.00, 1.02])
    ax2.set_ylim(0.47, 0.53)
    ax1.set_ylabel('reduced visib.')
    ax2.set_ylabel('raw visibility')
    fig.show()
    if save:
        fig.savefig('../pdf/original-peelle.pdf')

# plot_figure2()
plot_figure9(save=True)
