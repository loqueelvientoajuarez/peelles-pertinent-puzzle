#! /usr/bin/env python3

import numpy as np
import numpy.linalg as la
from numpy.random import RandomState
from scipy.optimize import curve_fit
from matplotlib import pylab as plt
from joblib import Parallel, delayed
import os
import pickle
 
def constant(x, k):
    return np.full_like(x, k)
constant.p0 = [1.]
constant.desc = 'a'

def underresolved(x, a):
    return 1 - a * x ** 2
underresolved.p0 = [1.]
underresolved.desc = '1 - ax^2' 

def quadratic(x, a, b):
    return a - b * x ** 2
quadratic.p0 = [1., 1.]
quadratic.desc = 'a - bx^2'
quadratic.val = '1 - x^2'

def gauss(x, a, b):
    return a * np.exp(-(b*x) ** 2)
gauss.p0 = [1., 4.]
gauss.desc = 'a\\mathrm{e}^{-bx^2}'
gauss.val = '\\mathrm{e}^{-4x^2}'

def simulate_data(sigma_sys=0.02, sigma_sta=0.02, nsta=100, nsys=6,
    f=quadratic, p0=None, cov_model='fit', nsim=None, xmin=0.1, xmax=0.5):
    if p0 is None:
        p0 = f.p0
    if nsim is not None:
        n_jobs = -1
        kwarg = dict(sigma_sys=sigma_sys, sigma_sta=sigma_sta, nsta=nsta,
            nsys=nsys, f=f, p0=p0, cov_model=cov_model)
        call = delayed(simulate_data)(**kwarg)
        res = Parallel(n_jobs=n_jobs)(call for i in range(nsim))
        x, y, p, cov, chi2 = zip(*res)
        p = np.array(p).T
        cov = np.moveaxis(cov, 0, 2)
        chi2 = np.array(chi2)
        x = x[0]
        y = np.array(y).T
        return x, y, p, cov, chi2
    ndata = nsta * nsys
    # Data model is y = f(x, p₀)(1 + ε₂) + ε₁ with 
    #        statistical errors ε₁ ~ N(σ₁I) uncorrelated, and
    #        systematic errors  ε₂ ~ N(σ₂1) within a data group.
    x = np.linspace(xmin, xmax, 2*ndata).reshape((-1,nsta))[1::2].ravel()
    y0 = f(x, *p0)
    rs = RandomState() 
    err_sta = sigma_sta * rs.normal(size=ndata)
    ones = np.ones((nsys, nsta))
    err_sys = sigma_sys * (rs.normal(size=(nsys,1)) * ones).ravel()
    y = y0 * (1 + err_sys) + err_sta
    # initial guess for covariance matrix and correlation
    cov_sta = sigma_sta ** 2 * np.eye(ndata)
    cov = cov_sta + np.diag((y * sigma_sys)**2)
    u = np.array(np.floor(np.arange(ndata) / nsta), dtype=int)
    rho = np.array(u[:,None] == u[None,:], dtype=int) 
    # fit
    nfree = ndata - len(f.p0)
    popt = p0
    eps = [0]
    for i in range(10):
        popt_old = popt
        popt, pcov = curve_fit(f, x, y, p0=popt, sigma=cov)
        ym = f(x, *popt)
        eps = np.absolute((popt - popt_old) / popt)
        if i > 1 and all(eps < 1e-4):
            break
        if cov_model == 'none': 
            break
        elif i == 1 and cov_model in ['data', 'fit', 'uncorrelated fit', 'local fit']: 
            break
        elif cov_model == 'data': # naive correlation is modelled (cf. Peelle)
            ycov = y
        elif cov_model in ['fit', 'uncorrelated fit', 'corr. fit', 'correlated fit', 'recursive fit']: 
            ycov = ym
        elif cov_model in ['local fit', 'local corr. fit']: 
                               # set of correlated data
            ycov = np.empty_like(ym)
            for r in range(max(u) + 1):
                index = u == r 
                xi, yi, covi = x[index], y[index], cov[:,index][index,:]
                pi, unused = curve_fit(f, xi, yi, p0=popt, sigma=covi)
                ycov[index] = f(xi, *pi)
        else:
            raise RuntimeError('invalid covariance prescription:' + cov_model)
        cov_sys = sigma_sys ** 2 * rho * np.outer(ycov, ycov)
        cov = cov_sta + cov_sys
    res = ym - y
    chi2r = (res.T @ la.inv(cov) @ res) / nfree
    print(i, p0, popt, chi2r)
    return x, y, popt, pcov, chi2r

def print_model_summary(model, p, cov, chi, cov_model):
    print('{} using covariance model {}'.format(model.desc, cov_model))
    print('     mean    err    dev')
    for j, pname in enumerate('abcdef'[0:len(model.p0)]):
        pval = p[j]
        pmean = pval.mean()
        pdev = pval.std(ddof=len(model.p0))
        dp = np.sqrt(cov[j,j].mean())
        print(('{}: {:+.3f} ± {:.3f} ± {:.3f}').format(pname, pmean, pdev, dp))
    print('χ² = {:.3f}'.format(chi.mean()))
    print('')


def test_model(model=quadratic, 
        cov_models=['none', 'data', 'fit', 'recursive fit'], save=False):
    sigma_sys = 0.03
    sigma_sta = 0.02
    fmt = ['k-', 'k--', 'k-.', 'k:'] 
    plt.style.use('mnras')
    fig = plt.figure(1)
    fig.clf()
    nplots = 1
    xmin = 0.15
    xmax = 0.52
    nsta = 50
    nsys = 6
    kwarg = dict(f=model, sigma_sys=sigma_sys, sigma_sta=sigma_sta, 
            xmin=xmin, xmax=xmax, nsta=nsta, nsys=nsys)
    ax = fig.add_subplot(1, 1, 1)
    x0 = np.linspace(0, 1.0*xmax, 300)
    gray = (.3,.3,.3)
    ax.text(xmax, 1.07, '${\\scriptstyle V} = (1-x^2)(1 + \\eta_\\tau) + \\eta_\\nu$',
        fontsize=8, va='top', ha='right', color=gray)
    ax.text(xmax, 1.03, 'fitted with $\\mu = {}$'.format(model.desc),
        fontsize=8, va='top', ha='right', color=gray)
    ax.plot(x0, 1-x0**2, color=gray, lw=3)
    for k, (cm, f) in enumerate(zip(cov_models, fmt)):
        x, y, p, cov, chi = simulate_data(cov_model=cm, **kwarg) 
        yerr=np.full_like(x, sigma_sta)
        x0 = np.linspace(0, 1.01*xmax, 300)
        ym = model(x0, *p)
        if k == 0:
            gray=(.65,.65,.65)
            ax.errorbar(x, y, yerr=yerr, fmt='none', ecolor=gray,
                mec=gray, mfc=gray)
        ax.plot(x0, ym, f, label=cm)
    ax.set_xlabel('$x$')
    ax.set_ylabel('${\\scriptstyle V}$')
    ax.set_ylim(0.6, 1.08)
    ax.set_xlim(0, 1.01 * xmax)
    ax.legend(title='covariance determination method', ncol=2)
    fig.show()
    if save:
        fig.savefig('../pdf/fit-example.pdf')
    return fig

def test_covmodels(model=quadratic, nsim=100, sigma_sys=0.03, sigma_sta=0.02,
        save=False, recompute=False):
    nbin = min(40, max(10, nsim // 200))
    kwarg = dict(f=model, sigma_sys=sigma_sys, sigma_sta=sigma_sta, nsim=nsim)
    cov_models = ['none', 'data', 'fit', 'recursive fit']
    ncov = len(cov_models)
    npar = len(model.p0)
    plt.style.use('mnras-fullwidth')
    fig = plt.figure(2)
    fig.clf()
    fig.subplots_adjust(wspace=0, hspace=0, top=0.93)
    axes = [[fig.add_subplot(ncov, npar + 2, (npar + 2) * i + j + 1)
                for j in range(npar + 2)]
                    for i in range(ncov)]
    gray = (.4, .4, .4)
    kwargs = dict(ec=gray, fc=gray, density=True)
    models = { }
    filename = 'model={}_nsim={}_errsys{:.4f}_errsta={:.4f}.pickle'.format(
        model.__name__, nsim, sigma_sys, sigma_sta)
    if recompute or not os.path.exists(filename):
        for i, covm in enumerate(cov_models):
            models[covm] = simulate_data(cov_model=covm, **kwarg) 
        with open(filename, 'wb') as fout:
            pickle.dump(models, fout) 
    else:
        with open(filename, 'rb') as fin:
            models = pickle.load(fin)
    for i, covm in enumerate(cov_models):
        x, y, p, cov, chi = models[covm]
        print_model_summary(model, p, cov, chi, covm)
        # χ²
        c1, c2, cm, c4, c5 = np.percentile(chi, [.3,14,50,86,99.7])
        cdev = (c4 - c2) / 2
        bins = np.linspace(c1, c5, nbin)
        ax = axes[i][0]
        ax.hist(chi, bins=bins, **kwargs)
        ax.set_ylabel(covm)
        text = '${:+.3f} \\pm {:.3f}$'.format(cm, cdev)
        ax.text(0.02, 0.96, text, fontsize=8, va='top', ha='left', 
            transform=ax.transAxes)
        ax.set_yticks([])
        ax.vlines(1, 0, 0.8, linestyle='--',
                 transform=ax.get_xaxis_transform())
        if i == ncov - 1:
            ax.set_xlabel('$\\chi^2$')
        else:
            ax.set_xticklabels([])
        # η
        eta = model(x[:,None], *p[:,None,:]) - model(x, *model.p0)[:,None]
        eta = eta.mean(axis=0)
        e1, e2, em, e4, e5 = np.percentile(eta, [.3,14,50,86,99.7])
        edev = (e4 - e2) / 2
        bins = np.linspace(e1, e5, nbin)
        ax = axes[i][-1]
        ax.hist(eta, bins=bins, **kwargs)
        text = '${:+.3f} \\pm {:.3f}$'.format(em, edev)
        ax.text(0.02, 0.96, text, va='top', ha='left', fontsize=8, 
            transform=ax.transAxes)
        ax.set_yticks([])
        ax.vlines(0, 0, 0.8, linestyle='--',
             transform=ax.get_xaxis_transform())
        if i == ncov - 1:
            ax.set_xlabel('$\\mu - \\bar{\\scriptstyle V}$')
        else:
            ax.set_xticklabels([])
        for j, pname in enumerate('abcdef'[0:npar]):
            p1, p2, pm, p4, p5 = np.percentile(p[j], [.3,14,50,86,99.7])
            pdev = (p4 - p2) / 2
            bins = np.linspace(p1, p5, nbin) 
            dp = np.sqrt(np.median(cov[j,j]))
            ax = axes[i][j + 1]
            ax.hist(p[j], bins=bins, **kwargs) 
            text = '${:+.3f} \\pm {:.3f}\\ [{:.3f}]$'.format(pm, pdev, dp)
            ax.text(0.02, 0.96, text, va='top', ha='left', 
                transform=ax.transAxes, fontsize=8)
            ax.set_yticks([])
            if i == ncov - 1:
                ax.set_xlabel('${}$'.format(pname))
            else:
                ax.set_xticklabels([])
            ax.vlines(model.p0[j], 0, 0.8, linestyle='--',
                 transform=ax.get_xaxis_transform())
    for j in range(0, npar + 2):
        r = range(0 + 0 * (j == 0), ncov)
        xlim = np.array([axes[i][j].get_xlim() for i in r])
        ylim = np.array([axes[i][j].get_ylim() for i in r])
        xmin, xmax = xlim[:,0].min(), xlim[:,1].max()
        ymax = ylim[:,1].max() * 1.2
        for i in range(ncov):
            axes[i][j].set_xlim(xmin, xmax)
            axes[i][j].set_ylim(0, ymax)
    #for i in range(ncov):
    #    ylim = np.array([axes[i][j].get_ylim() for j in range(npar + 2)])
    #    ymin, ymax = ylim[:,0].min(), ylim[:,1].max()
    #     for j in range(npar + 2):
    #        axes[i][j].set_ylim(ymin, 1.15 * ymax)
    text=f'${{\\scriptstyle V}} = {model.val}(1 + \\eta_\\tau) + \\eta_\\nu\\mathrm{{\\ fitted\\ with\\ }} \\mu={model.desc}$'
    fig.suptitle(text)
    fig.text(0.005, 0.5, 'covariance determination method', 
        rotation=90, va='center', ha='left')
    fig.show()
    if save:
        if model.name != 'quadratic':
            filename = f'fit-quality-{model.name}.pdf'
        else:
            filename = '../pdf/fit-quality.pdf'
        fig.savefig(filename)
    return fig


if __name__ == "__main__":
    test_model(save=True ) # Fig 3
    test_covmodels(nsim=50_000, save=True) # Fig. 4
    test_covmodels(nsim=50_000, save=True, model=gauss) # check
