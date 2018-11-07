"""
Make a number of diagnostic plots of a given run.
"""

import matplotlib.pyplot as plt
import numpy as np

from py21cmmc.mcmc import analyse
import sys

try:
    model_name = "InstrumentalGridTestNoisePSFG" #sys.argv[1]
except IndexError:
    raise ValueError("Please give a model name as the only argument")

name = "data/"+model_name
figname = "plots/"+model_name+"_{}.png"

samples = analyse.get_samples(name)
data = np.load(name+'.npz')
noise = np.load(name+".noise.npz")


# Make trace plot
fig, ax = analyse.trace_plot(samples, include_lnl=True, start_iter=0, thin=1, colored=False, show_guess=True)
plt.savefig(figname.format("TracePlot"))
plt.clf()


# Make corner plot
fig = analyse.corner_plot(samples);
plt.savefig(figname.format("CornerPlot"))
plt.clf()


# Make data power spectrum plot
plt.figure(figsize=(12,8))
plt.imshow(np.log10(data['p_signal'].T), origin='lower', extent=(data["u_eta"][0][0], data["u_eta"][0][-1], data["u_eta"][1][0], data["u_eta"][1][-1]))
plt.xscale("log")
plt.yscale("log")
plt.ylabel("$\eta$", fontsize=12)
plt.xlabel("$\sqrt{u^2+v^2}$", fontsize=12)
plt.title("EoR Signal")
plt.colorbar()
plt.savefig(figname.format("DataPS"))
plt.clf()

# Make signal-to-noise power spectrum plot
plt.figure(figsize=(12,8))
noise_variance = np.array([np.diag(c) for c in noise['covariance']])
plt.imshow(np.log10(data['p_signal'].T/ noise_variance.T), origin='lower', extent=(data["u_eta"][0][0], data["u_eta"][0][-1], data["u_eta"][1][0], data["u_eta"][1][-1]))
plt.xscale("log")
plt.yscale("log")
plt.ylabel("$\eta$", fontsize=12)
plt.xlabel("$\sqrt{u^2+v^2}$", fontsize=12)
plt.title("Signal to Noise Power Spectrum")
plt.colorbar()
plt.savefig(figname.format("SignalNoisePS"))
plt.clf()

# Make noise power spectrum plot
plt.figure(figsize=(12,8))
plt.imshow(np.log10(noise_variance.T), origin='lower', extent=(data["u_eta"][0][0], data["u_eta"][0][-1], data["u_eta"][1][0], data["u_eta"][1][-1]))
plt.xscale("log")
plt.yscale("log")
plt.ylabel("$\eta$", fontsize=12)
plt.xlabel("$\sqrt{u^2+v^2}$", fontsize=12)
plt.title("Noise")
plt.colorbar()
plt.savefig(figname.format("NoisePS"))
plt.clf()

# Make covariance of noise power spectrum plot
plt.figure(figsize=(12,8))
plt.imshow(np.log10(noise['covariance'][0]), origin='lower', extent = (data["u_eta"][1][0], data["u_eta"][1][-1], data["u_eta"][1][0], data["u_eta"][1][-1]))
plt.xlabel("$\eta$", fontsize=12)
plt.ylabel("$\eta$", fontsize=12)
plt.title("Cov(Noise)")
plt.colorbar()
plt.savefig(figname.format("NoiseCov"))
plt.clf()
