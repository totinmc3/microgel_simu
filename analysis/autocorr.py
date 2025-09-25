from espressomd import analyze
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt


def autocor(x):
    x = np.asarray(x)
    mean = x.mean()
    var = np.var(x)
    xp = x - mean
    corr = analyze.autocorrelation(xp) / var
    return corr


def fit_correlation_time(data, ts):
    data = np.asarray(data)
    data /= data[0]

    def fitfn(t, t_corr): return np.exp(-t / t_corr)
    popt, pcov = optimize.curve_fit(fitfn, ts, data)
    return popt[0]

if __name__ == "__main__":

    energy = np.genfromtxt('../test/energies.dat', delimiter="\t")
    e_total = energy[:,1]
    times = energy[:,0]
    times -= times[0]

    e_total_autocor = autocor(e_total)
    corr_time = fit_correlation_time(e_total_autocor[:100], times[:100])
    steps_per_uncorrelated_sample = int(np.ceil(3 * corr_time / 1))
    print(f'{steps_per_uncorrelated_sample=}')

    plt.figure(figsize=(10, 6))
    plt.plot(times[1:], e_total_autocor[1:], 'o', label='data', markersize=2)
    plt.plot(times[1:], np.exp(-times[1:] / corr_time), label='exponential fit')
    # plt.plot(2 * [steps_per_subsample * system.time_step],
    #         [min(e_total_autocor), 1], label='subsampling interval')
    plt.ylim(top=1.2, bottom=-0.15)
    plt.legend()
    plt.xscale('log')
    plt.xlabel('t')
    plt.ylabel('total energy autocorrelation')
    plt.show()
