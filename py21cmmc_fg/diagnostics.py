"""
A module which contains functions providing diagnostic information/plots on chains, either pre- or post-MCMC.
"""
from py21cmmc.mcmc import build_computation_chain, CoreLightConeModule
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .likelihood import LikelihoodInstrumental2D
from .core import ForegroundsBase, CoreInstrumental
from powerbox.dft import fft, ifft


def imaging(chain=None, cores=None, lk=None, freq_ind=0):
    """
    Create a plot of the imaging capability of the current setup.

    Uses the loaded cores to create a simulated sky, then "observe" this with given baselines. Then uses an
    Instrumental2D likelihood to grid those baselines and transform back to the image plane. Every step of
    this process is output as a panel in a plot to be compared.

    Parameters
    ----------
    chain : :class:`~py21cmmc.mcmc.cosmoHammer.LikelihoodComputationChain.LikelihoodComputationChain` instance, optional
        A computation chain which contains loaded likelihoods and cores.
    cores : list of :class:`~py21cmmc.mcmc.core.CoreBase` instances, optional
        A list of cores defining the sky and instrument. Only required if `chain` is not given.
    lk : :class:`~likelihood.LikelihoodInstrumental2D` class, optional
        An instrumental likelihood, required if `chain` not given.
    freq_ind : int, optional
        The index of the frequency to actually show plots for (default is the first frequency channel).

    Returns
    -------
    fig :
        A matplotlib figure object.
    """
    if chain is None and (cores is None or lk is None):
        raise ValueError("Either chain or both cores and likelihood must be given.")

    # Create a likelihood computation chain.
    if chain is None:
        chain = build_computation_chain(cores, lk)
        chain.setup()
    else:
        lk = chain.getLikelihoodModules()[0]
        cores = chain.getCoreModules()

    if not isinstance(lk, LikelihoodInstrumental2D):
        raise ValueError("likelihood needs to be a Instrumental2D likelihood")

    if not hasattr(lk, "LikelihoodComputationChain"):
        chain.setup()

    # Call all core simulators.
    ctx = chain.core_simulated_context()

    visgrid = lk.grid_visibilities(ctx.get("visibilities"))

    # Do a direct FT there and back, rather than baselines.
    direct_vis, direct_u = fft(ctx.get("new_sky")[:, :, freq_ind], L=lk._instr_core.sky_size, a=0, b=2 * np.pi)
    direct_img, direct_l = ifft(direct_vis, Lk=(lk.uvgrid[1] - lk.uvgrid[0]) * len(lk.uvgrid), a=0, b=2 * np.pi)

    # Get the reconstructed image
    image_plane, image_grid = ifft(visgrid[:, :, freq_ind], Lk=(lk.uvgrid[1] - lk.uvgrid[0]) * len(lk.uvgrid), a=0, b=2 * np.pi)

    # Make a figure.
    if len(cores) == 2:
        fig, ax = plt.subplots(2, 4, figsize=(12, 6))
        mid_row = 0
    else:
        fig, ax = plt.subplots(3, max((4, len(cores))), figsize=(3*max((4, len(cores))), 9))
        mid_row = 1

    # Show original sky(s) (before Beam)
    i = 0
    for core in cores:
        if isinstance(core, ForegroundsBase):
            # TODO: frequency plotted here does not necessarily match the frequency in other plots.
            mp = ax[0, i].imshow(ctx.get("foregrounds")[i][:, :, freq_ind].T, origin='lower',
                                 extent=(-core.sky_size / 2, core.sky_size / 2) * 2)
            ax[0, i].set_title("Orig. %s FG" % core.__class__.__name__)
            cbar = plt.colorbar(mp, ax=ax[0, i])
            ax[0, i].set_xlabel("l")
            ax[0, i].set_ylabel("m")
            cbar.set_label("Brightness Temp. [Jy/sr]")
            i += 1

        # # TODO: add lightcone plot
        # if isinstance(core, CoreLightConeModule):
        #     mp = ax[0, i].imshow(ctx.get("foregrounds")[i][:, :, -freq_ind].T, origin='lower',
        #                          extent=(-core.sky_size / 2, core.sky_size / 2) * 2)
        #     ax[0, i].set_title("Original %s foregrounds" % core.__class__.__name__)
        #     cbar = plt.colorbar(mp, ax=ax[0, i])
        #     ax[0, i].set_xlabel("l")
        #     ax[0, i].set_ylabel("m")
        #     cbar.set_label("Brightness Temp. [K]")
        #     i += 1

    # Show tiled (if applicable) and attenuated sky
    mp = ax[mid_row, 1].imshow(
        ctx.get("new_sky")[:, :, freq_ind].T, origin='lower',
        extent=(-lk._instr_core.sky_size / 2, lk._instr_core.sky_size / 2) * 2
    )
    ax[mid_row, 1].set_title("Tiled+Beam FG")
    cbar = plt.colorbar(mp, ax=ax[mid_row, 1])
    ax[mid_row, 1].set_xlabel("l")
    ax[mid_row, 1].set_ylabel("m")
    cbar.set_label("Brightness Temp. [K]")

    # Show UV weights
    mp = ax[mid_row, 2].imshow(
        lk.nbl_uvnu[:, :, freq_ind].T, origin='lower',
        extent=(lk.uvgrid.min(), lk.uvgrid.max()) * 2
    )
    ax[mid_row, 2].set_title("UV weights")
    cbar = plt.colorbar(mp, ax=ax[mid_row, 2])
    ax[mid_row, 2].set_xlabel("u")
    ax[mid_row, 2].set_ylabel("v")
    cbar.set_label("Weight")

    # Show raw visibilities
    wvlength = 3e8 / ctx.get("frequencies")[freq_ind]
    mp = ax[mid_row, 3].scatter(ctx.get("baselines")[:, 0] / wvlength, ctx.get("baselines")[:, 1] / wvlength,
                                c=np.real(ctx.get("visibilities")[:, freq_ind]))
    ax[mid_row, 3].set_title("Raw Vis.")
    cbar = plt.colorbar(mp, ax=ax[mid_row, 3])
    ax[mid_row, 3].set_xlabel("u")
    ax[mid_row, 3].set_xlabel("v")
    cbar.set_label("Re[Vis] [Jy?]")

    # Show Gridded Visibilities
    mp = ax[mid_row+1, 3].imshow(
        np.real(visgrid[:, :, freq_ind].T), origin='lower',
        extent=(lk.uvgrid.min(), lk.uvgrid.max()) * 2
    )
    ax[mid_row+1, 3].set_title("Gridded Vis")
    cbar = plt.colorbar(mp, ax=ax[mid_row+1, 3])
    ax[mid_row+1, 3].set_xlabel("u")
    ax[mid_row+1, 3].set_ylabel("v")
    cbar.set_label("Jy")

    # Show directly-calculated UV plane
    mp = ax[mid_row+1, 2].imshow(
        np.real(direct_vis), origin='lower',
        extent=(direct_u[0].min(), direct_u[0].max()) * 2
    )
    ax[mid_row+1, 2].set_title("Direct Vis")
    cbar = plt.colorbar(mp, ax=ax[mid_row+1, 2])
    ax[mid_row+1, 2].set_xlabel("u")
    ax[mid_row+1, 2].set_ylabel("v")
    cbar.set_label("Jy")

    # Show final "image"
    mp = ax[mid_row+1, 1].imshow(np.abs(image_plane).T, origin='lower',
                         extent=(image_grid[0].min(), image_grid[0].max(),) * 2)
    ax[mid_row+1, 1].set_title("Recon. FG")
    cbar = plt.colorbar(mp, ax=ax[mid_row+1, 1])
    ax[mid_row+1, 1].set_xlabel("l")
    ax[mid_row+1, 1].set_ylabel("m")
    cbar.set_label("Flux Density. [Jy]")

    # Show direct reconstruction
    mp = ax[mid_row+1, 0].imshow(np.abs(direct_img).T, origin='lower',
                         extent=(direct_l[0].min(), direct_l[0].max(),) * 2)
    ax[mid_row+1, 0].set_title("Recon. direct FG")
    cbar = plt.colorbar(mp, ax=ax[mid_row+1, 0])
    ax[mid_row+1, 0].set_xlabel("l")
    ax[mid_row+1, 0].set_ylabel("m")
    cbar.set_label("Flux Density. [Jy]")

    plt.tight_layout()

    return fig
