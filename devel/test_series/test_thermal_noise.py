from gridded_noise_nofg import core_eor, core_instr, likelihood, model_name
from py21cmmc.mcmc.mcmc import build_computation_chain
import pickle
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

nrealisations = 400
nthreads = 4

def _produce_mock(i):
    """Produces a mock power spectrum for purposes of getting numerical_covariances"""
    # Create an empty context with the given parameters.
    ctx = likelihood.LikelihoodComputationChain.createChainContext()

    # For each realisation, run every foreground core (not the signal!)
    for core in likelihood.foreground_cores:
        core.simulate_data(ctx)

    # And turn them into visibilities
    likelihood._instr_core.simulate_data(ctx)

    # The following is basically "compute_power", but saves the steps.
    visgrid = likelihood.grid_visibilities(ctx.get("visibilities"))

    # Transform frequency axis
    visgrid = likelihood.frequency_fft(visgrid, likelihood.frequencies, taper=likelihood.frequency_taper)

    # Get 2D power from gridded vis.
    power2d = likelihood.get_2d_power(visgrid)

    # Restrict power to eta modes above eta_min
    power2d = power2d[:, -len(likelihood.eta):]

    power3d = np.abs(visgrid)**2

    return power2d, power3d, visgrid


def numerical_variance():
    """
    Calculate the covariance of the foregrounds.

    Parameters
    ----------
    params: dict
        The parameters of this iteration. If empty, default parameters are used.

    nrealisations: int, optional
        Number of realisations to find the covariance.

    Output
    ------
    mean: (nperp, npar)-array
        The mean 2D power spectrum of the foregrounds.

    cov:
        The sparse block diagonal matrix of the covariance if nrealisation is not 1
        Else it is 0
    """

    if nrealisations < 2:
        raise ValueError("nrealisations must be more than one")

    pool = multiprocessing.Pool(nthreads)
    res = pool.map(_produce_mock, np.arange(nrealisations))

    power2d = np.array([r[0] for r in res])
    power3d = np.array([r[1] for r in res])
    visgrid = np.array([r[2] for r in res])

    mean_p2d = np.mean(power2d, axis=0)
    var_p2d = np.var(power2d, axis=0)
    var_p3d = np.var(power3d, axis=0)
    var_V3d = np.var(visgrid, axis=0)

    return mean_p2d, var_p2d, var_p3d, var_V3d


def make_the_plot(num_mean_p2d, num_var_p2d, num_var_V, num_var_p3d, anl_mean_p2d, anl_var_p2d, anl_var_V, anl_var_p3d):
    # Make a plot
    fig, ax = plt.subplots(
        4, 3,
        sharex=True, sharey=True,
        subplot_kw={"xscale": 'log', 'yscale': 'log'},
        figsize=(12, 12)
    )
    fig.suptitle("Thermal Noise Power")

    extent = (likelihood.u.min(), likelihood.u.max(), likelihood.eta.min(), likelihood.eta.max())

    im = ax[0, 0].imshow(num_mean_p2d.T, origin='lower', extent=extent)
    plt.colorbar(im, ax=ax[0, 0])
    ax[0, 0].set_title("Numerical mean")

    im = ax[0, 1].imshow(anl_mean_p2d.T, origin='lower', extent=extent)
    plt.colorbar(im, ax=ax[0, 1])
    ax[0, 1].set_title("Analytic Mean")

    im = ax[0, 2].imshow((num_mean_p2d / anl_mean_p2d).T, origin='lower', extent=extent)
    plt.colorbar(im, ax=ax[0, 2])
    ax[0, 2].set_title("Num/Anl Mean")

    im = ax[1, 0].imshow(num_var_p2d.T, origin='lower', extent=extent)
    plt.colorbar(im, ax=ax[1, 0])
    ax[1, 0].set_title("Numerical Var.")

    im = ax[1, 1].imshow(anl_var_p2d.T, origin='lower', extent=extent)
    plt.colorbar(im, ax=ax[1, 1])
    ax[1, 1].set_title("Analytic var.")

    im = ax[1, 2].imshow((num_var_p2d / anl_var_p2d).T, origin='lower', extent=extent)
    plt.colorbar(im, ax=ax[1, 2])
    ax[1, 2].set_title("Num./Anl. Var")

    im = ax[2, 0].imshow(num_var_V[0].T, origin='lower', extent=extent)
    plt.colorbar(im, ax=ax[2, 0])
    ax[2, 0].set_title("Numerical V Var.")

    # im = ax[2, 1].imshow(anl_var_V.T, origin='lower', extent=extent)
    # plt.colorbar(im, ax=ax[2, 1])
    # ax[2, 1].set_title("Analytic V var.")

    im = ax[2, 2].imshow((num_var_V[0].T / anl_var_V[0]), origin='lower', extent=extent)
    plt.colorbar(im, ax=ax[2, 2])
    ax[2, 2].set_title("Num./Anl. V Var")

    im = ax[3, 0].imshow(num_var_p3d[0].T, origin='lower', extent=extent)
    plt.colorbar(im, ax=ax[3, 0])
    ax[3, 0].set_title("Numerical P Var.")

    # im = ax[3, 1].imshow(anl_var_p3d.T, origin='lower', extent=extent)
    # plt.colorbar(im, ax=ax[3, 1])
    # ax[3, 1].set_title("Analytic P var.")

    im = ax[3, 2].imshow((num_var_p3d[0].T / anl_var_p3d[0]), origin='lower', extent=extent)
    plt.colorbar(im, ax=ax[3, 2])
    ax[3, 2].set_title("Num./Anl. P Var")

    # ADD SUPER AXIS LABELS
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Perpendicular Scale, $u$", labelpad=15, fontsize=15)
    plt.ylabel("Line-of-Sight Scale, $\eta$", labelpad=15, fontsize=15)

    return fig, ax


if __name__=="__main__":
    # Build the chain and run setup()
    build_computation_chain([core_eor, core_instr], likelihood)

    # Get numerical values
    num_mean_p2d, num_var_p2d, num_var_p3d, num_var_V = numerical_variance()

    # Get analytic values for p2d
    anl_mean_p2d, anl_var_p2d = likelihood.noise['mean'], likelihood.noise['covariance']
    anl_var_p2d = np.array([np.diag(c) for c in anl_var_p2d])
    anl_var_V = core_instr.thermal_variance_baseline / likelihood.nbl_uv
    anl_var_p3d = core_instr.thermal_variance_baseline**2 / likelihood.nbl_uv**2

    # Dump data in case plotting doesn't work
    with open("thermal_noise_data.pkl", 'wb') as f:
        pickle.dump(
            {"num_mean_p2d":num_mean_p2d, "num_var":num_var_p2d, "anl_mean":anl_mean_p2d, "anl_cov":anl_var_p2d,
             "num_var_p3d":num_var_p3d, "num_var_V":num_var_V, "anl_var_V":anl_var_V, "anl_var_p3d":anl_var_p3d}, f
        )

    # Make the plot
    fig, ax = make_the_plot(
        num_mean_p2d, num_var_p2d, num_var_V, num_var_p3d,
        anl_mean_p2d, anl_var_p2d, anl_var_V, anl_var_p3d
    )

    plt.savefig("thermal_noise_test.png")
