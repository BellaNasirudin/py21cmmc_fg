"""
Make a number of diagnostic plots of a given run.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from py21cmmc.mcmc import analyse
import sys

try:
    model_name = sys.argv[1] # Don't change this, it allows you to pass a model name at the command line, eg "python plot.py InstrumentalGridTestNoise"
except IndexError:
    raise ValueError("Please give a model name as the only argument")

name = "data/"+model_name
figname = "plots/"+model_name+"_{}.png"

samples = analyse.get_samples(name)
data = np.load(name+'.npz')
try:
    noise = np.load(name+".noise.npz")
except:
    noise = None

blobs = samples.get_blobs()

ps_extent = (data['u'].min(), data['u'].max(), 1e6*data['eta'].min(), 1e6*data['eta'].max()) # times eta by 1e6 to get it into 1/MHz
if noise is not None:
    var = np.array([np.diag(c) for c in noise['covariance']])
    mn = noise['mean']

# Make trace plot
fig, ax = analyse.trace_plot(samples, include_lnl=True, start_iter=0, thin=1, colored=False, show_guess=True)
plt.savefig(figname.format("TracePlot"))
plt.clf()


# Make corner plot
fig = analyse.corner_plot(samples);
plt.savefig(figname.format("CornerPlot"))
plt.clf()


# Make 2D Power Spectrum Diagnosis
fig,ax = plt.subplots(2,4, figsize=(14,6), sharex=True, sharey=True,
                      subplot_kw={"xscale":"log", "yscale":'log'},
                     gridspec_kw={"hspace":0.05, "wspace":0.05})


im = ax[0,0].imshow(np.log10(data['p_signal'].T), origin='lower',
           extent=ps_extent)
plt.colorbar(im, ax=ax[0,0]);
ax[0,0].set_title("Mock 2D PS")

im = ax[0,1].imshow(np.log10(blobs['signal'][0,0].T), origin='lower',
           extent=ps_extent)
plt.colorbar(im, ax=ax[0,1]);
ax[0,1].set_title("Model 0,0 2D PS")

im = ax[0,2].imshow(np.log10(blobs['signal'][-1,-1].T), origin='lower',
           extent=ps_extent)
plt.colorbar(im, ax=ax[0,2]);
ax[0,2].set_title("Model -1,-1 2D PS")

im = ax[0,3].imshow(np.log10(mn.T), origin='lower',
           extent=ps_extent)
plt.colorbar(im, ax=ax[0,3]);
ax[0,3].set_title("Expected Noise+FG Power")

im = ax[1,3].imshow(np.log10(blobs['signal'][0,0].T - mn.T), origin='lower',
           extent=ps_extent)
plt.colorbar(im, ax=ax[1,3])
ax[1,3].set_title("EoR 0,0")


if noise is not None:
    im = ax[1,0].imshow(np.log10(np.sqrt(var).T), origin='lower', extent=ps_extent)
    plt.colorbar(im, ax=ax[1,0]);
    ax[1,0].set_title("$\sigma$")

    im = ax[1,1].imshow(blobs['sigma'][0,0].T, origin='lower', extent=ps_extent)
    plt.colorbar(im, ax=ax[1,1]);
    ax[1,1].set_title("#$\sigma$ 0,0")

    im = ax[1,2].imshow(blobs['sigma'][-1,-1].T, origin='lower', extent=ps_extent)
    plt.colorbar(im, ax=ax[1,2]);
    ax[1,2].set_title("#$\sigma$ -1,-1")

fig.suptitle("2D Power Spectrum Diagnosis", fontsize=16)

# Add common x,y labels
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel("Perpendicular Scale, $u$", labelpad=15, fontsize=15)
plt.ylabel("Line-of-Sight Scale, $\eta$", labelpad=15, fontsize=15)

fig.savefig(figname.format("2DPS"))
plt.clf()


# Make Baseline Layout / Weighting Diagnosis
fig, ax = plt.subplots(1,3, figsize=(12,4.5), squeeze=False)

im = ax[0,0].imshow(data['nbl_uvnu'][:,:,0].T, origin="lower")
plt.colorbar(im, ax = ax[0,0])
ax[0,0].set_title("# bl per UV(nu) cell")

im = ax[0,1].imshow(data['nbl_uv'].T, origin="lower")
plt.colorbar(im, ax = ax[0,1])
ax[0,1].set_title("Eff. # bl per UV(eta) cell")

ax[0,2].plot(data['u'], data['nbl_u'], label="Eff # bl per |u| cell")
ax[0,2].plot(data['u'], data['grid_weights'], label="Number of uv cells per u")
ax[0,2].set_xlabel("|u|")
ax[0,2].set_yscale('log')
ax[0,2].legend();

fig.suptitle("Baseline Layout / Weighting", fontsize=15)
plt.tight_layout()

fig.savefig(figname.format("Weighting"))
plt.clf()
