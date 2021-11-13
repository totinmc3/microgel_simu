from espressomd import analyze
from scipy import optimize
import numpy as np


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

    energy = np.genfromtxt('../results/results_firstTrial/TotEner_warmup.dat', delimiter="\t")
    e_total = energy[10000:,1]
    times = energy[10000:,0]
    times -= times[0]

    e_total_autocor = autocor(e_total)
    corr_time = fit_correlation_time(e_total_autocor[:100], times[:100])
    steps_per_uncorrelated_sample = int(np.ceil(3 * corr_time / 1))
    print(steps_per_uncorrelated_sample)

