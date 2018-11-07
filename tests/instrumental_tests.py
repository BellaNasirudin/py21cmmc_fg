"""
Some tests of the instrumental core/likelihood.
"""

from py21cmmc_fg.core import CoreInstrumental, ForegroundsBase
from py21cmmc_fg.likelihood import LikelihoodInstrumental2D
from py21cmmc.mcmc import build_computation_chain
import numpy as np

from powerbox.dft import fft, ifft

import matplotlib.pyplot as plt
import healpy as hp

RUNNING_AS_TEST = True

import logging
logger = logging.getLogger("21CMMC")
logger.setLevel(logging.DEBUG)

def test_imaging(fg_cls):
    """
    The idea of this test is to take a sky with a single source on it, sample onto baselines, then grid and
    reconstruct the image. It is a test of the all-round process of sampling/gridding.
    """

    core_ss = fg_cls(frequencies=np.array([149., 151.0]), nside_base=6)
    core_instr = CoreInstrumental(
        freq_min=150.0,
        freq_max=150.1,
        nfreq=2,
        antenna_posfile='baseline_grid',
        antenna_posfile_is_baselines=True,
        Tsys=0
    )

    lk = LikelihoodInstrumental2D(use_data=False)

    chain = build_computation_chain([core_ss, core_instr], lk)
    chain.setup()

    ctx = chain.core_simulated_context()

    ugrid, visgrid, weights = lk.grid_visibilities(ctx.get("visibilities"), ctx.get("baselines"),
                                                   ctx.get("frequencies"), lk.n_uv, lk.umax)

    image_plane, image_grid = ifft(visgrid[:, :, 0], Lk=(ugrid[1] - ugrid[0]) * len(ugrid), a=0, b=2 * np.pi)

    if not RUNNING_AS_TEST:
        fig, ax = plt.subplots(2,2, figsize=(12,10), squeeze=False)

        plt.axes(ax[0,0])
        hp.visufunc.mollview(ctx.get("foregrounds")[0][:,0], hold=True, rot=(0 , 90))

        #mp = ax[0, 0].imshow(ctx.get("foregrounds")[0], origin='lower',
        #                     extent=(0, core_ss.user_params.BOX_LEN)*2)
        ax[0, 0].set_title("Original foregrounds")
        #cbar = plt.colorbar(mp, ax=ax[0, 0])
        #ax[0, 0].set_xlabel("x [Mpc]")
        #ax[0, 0].set_ylabel("y [Mpc]")
        #cbar.set_label("Brightness Temp. [K]")

        #
        # mp = ax[1, 0].imshow(
        #     ctx.get("new_sky")[:, :, 0].T, origin='lower',
        #     extent=(core_instr.sky_coords.min(), core_instr.sky_coords.max(),)*2
        # )
        # ax[1, 0].set_title("Stitched/Coarsened Foregrounds")
        # cbar = plt.colorbar(mp, ax=ax[1, 0])
        # ax[1, 0].set_xlabel("l")
        # ax[1, 0].set_ylabel("m")
        # cbar.set_label("Brightness Temp. [K]")

        mp = ax[0, 1].imshow(np.abs(image_plane).T, origin='lower',
                             extent=(image_grid[0].min(), image_grid[0].max(),) * 2)
        ax[0, 1].set_title("Imaged foregrounds")
        cbar = plt.colorbar(mp, ax=ax[0, 1])
        ax[0, 1].set_xlabel("l")
        ax[0, 1].set_ylabel("m")
        cbar.set_label("Flux Density. [Jy]")

        l = np.sin(core_ss.angles[0])  # get |l| co-ordinate
        attenuation = core_instr.beam(l, core_ss.frequencies)
        beam_sky = ctx.get("foregrounds")[0] * attenuation
        plt.axes(ax[1,1])
        hp.visufunc.mollview(beam_sky[:, 0], hold=True, rot=(0, 90))

        # mp = ax[1, 1].imshow(ctx.get("new_sky")[:, :, 0].T * beam_sky[:, :, 0].T, origin='lower', extent=(
        # core_instr.sky_coords.min(), core_instr.sky_coords.max(), core_instr.sky_coords.min(),
        # core_instr.sky_coords.max()))
        # ax[1, 1].set_title("Stitched/Coarsened+Beam Foregrounds")
        # cbar = plt.colorbar(mp, ax=ax[1, 1])
        # ax[1, 1].set_xlabel("l")
        # ax[1, 1].set_ylabel("m")
        # cbar.set_label("Flux Density. [Jy]")

        plt.savefig("test_imaging_%s.png"%fg_cls.__name__)
        plt.clf()


def test_imaging_single_source():
    # Make a new foregrounds class that will create a sky that has a single source at zenith on it.
    class SingleSource(ForegroundsBase):
        def build_sky(self):
            """
            Create a grid of flux densities corresponding to a sample of point-sources drawn from a power-law source count
            model.
            """
            sky = np.zeros((self.npixels, len(self.frequencies)))
            sky[self.angles[0]<0.1] = 1.0

            return sky

    test_imaging(SingleSource)


def test_imaging_source_line():
    # Make a new foregrounds class that will create a sky that has a single source at zenith on it.
    class SourceLine(ForegroundsBase):
        def build_sky(self):
            """
            Create a grid of flux densities corresponding to a sample of point-sources drawn from a power-law source count
            model.
            """
            sky = np.zeros((self.npixels, len(self.frequencies)))
            sky[np.logical_and(self.angles[1]>-0.1, self.angles[1]<0.1)] = 1.0
            return sky

    test_imaging(SourceLine)


def test_imaging_source_ring():
    # Make a new foregrounds class that will create a sky that has a single source at zenith on it.
    class SourceRing(ForegroundsBase):
        def build_sky(self):
            """
            Create a grid of flux densities corresponding to a sample of point-sources drawn from a power-law source count
            model.
            """
            sky = []

            for i in range(len(self.frequencies)):
                inds = np.arange(-self.user_params.HII_DIM / 2, self.user_params.HII_DIM / 2)
                print(len(inds), self.user_params.HII_DIM)
                ind_mag = np.add.outer(inds**2, inds**2)

                thissky = np.zeros(ind_mag.shape)
                thissky[np.logical_and(ind_mag>150, ind_mag<200)] = 1
                sky.append(thissky)
#            sky[self.user_params.HII_DIM // 2] = 1.0
            return np.array(sky).T

    SourceRing.defaults['box_len'] = 10000
    SourceRing.defaults['sky_cells'] = 500

    test_imaging(SourceRing)

if __name__ == "__main__":
    RUNNING_AS_TEST = False
    test_imaging_single_source()
    test_imaging_source_line()
#    test_imaging_source_ring()
