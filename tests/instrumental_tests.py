"""
Some tests of the instrumental core/likelihood.
"""

from py21cmmc_fg.core import CoreInstrumental, ForegroundsBase
from py21cmmc_fg.likelihood import LikelihoodInstrumental2D
from py21cmmc.mcmc import build_computation_chain
import numpy as np

from powerbox.dft import fft, ifft

import matplotlib.pyplot as plt

RUNNING_AS_TEST = True

import logging
logger = logging.getLogger("21CMMC")
logger.setLevel(logging.DEBUG)


def test_imaging(core_ss):
    """
    The idea of this test is to take a sky with a single source on it, sample onto baselines, then grid and
    reconstruct the image. It is a test of the all-round process of sampling/gridding.
    """

    core_instr = CoreInstrumental(
        freq_min=150.0,
        freq_max=150.1,
        nfreq=2,
        n_cells=300,
        sky_extent=3,
        antenna_posfile='grid_centres',
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
        fig, ax = plt.subplots(2, 2, figsize=(12, 12))

        mp = ax[0, 0].imshow(ctx.get("foregrounds")[0][:, :, 0].T, origin='lower',
                             extent=(-core_ss.sky_size/2, core_ss.sky_size/2)*2)
        ax[0, 0].set_title("Original foregrounds")
        cbar = plt.colorbar(mp, ax=ax[0, 0])
        ax[0, 0].set_xlabel("x [Mpc]")
        ax[0, 0].set_ylabel("y [Mpc]")
        cbar.set_label("Brightness Temp. [K]")

        mp = ax[0, 1].imshow(
            ctx.get("new_sky")[:, :, 0].T, origin='lower',
            extent=(-core_instr.sky_size/2, core_instr.sky_size/2)*2
        )
        ax[0, 1].set_title("Tiled+Beam Foregrounds")
        cbar = plt.colorbar(mp, ax=ax[0, 1])
        ax[0, 1].set_xlabel("l")
        ax[0, 1].set_ylabel("m")
        cbar.set_label("Brightness Temp. [K]")

        mp = ax[1, 0].imshow(
            np.real(visgrid[:, :, 0].T), origin='lower',
            extent=(ugrid.min(), ugrid.max()) * 2
        )
        ax[1, 0].set_title("Gridded Vis")
        cbar = plt.colorbar(mp, ax=ax[1, 0])
        ax[1, 0].set_xlabel("u")
        ax[1, 0].set_ylabel("v")
        cbar.set_label("Jy")

        mp = ax[1, 1].imshow(np.abs(image_plane).T, origin='lower',
                             extent=(image_grid[0].min(), image_grid[0].max(),) * 2)
        ax[1, 1].set_title("Imaged foregrounds")
        cbar = plt.colorbar(mp, ax=ax[1, 1])
        ax[1, 1].set_xlabel("l")
        ax[1, 1].set_ylabel("m")
        cbar.set_label("Flux Density. [Jy]")

        plt.tight_layout()

        plt.savefig("test_imaging_%s.png"%core_ss.__class__.__name__)
        plt.clf()


def test_imaging_single_source():
    # Make a new foregrounds class that will create a sky that has a single source at zenith on it.
    class SingleSource(ForegroundsBase):
        def build_sky(self):
            """
            Create a grid of flux densities corresponding to a sample of point-sources drawn from a power-law source count
            model.
            """
            sky = np.zeros((self.n_cells, self.n_cells, len(self.frequencies)))
            sky[self.n_cells// 2, self.n_cells // 2] = 1.0
            return sky

    test_imaging(SingleSource())


def test_imaging_source_line():
    # Make a new foregrounds class that will create a sky that has a single source at zenith on it.
    class SourceLine(ForegroundsBase):
        def build_sky(self):
            """
            Create a grid of flux densities corresponding to a sample of point-sources drawn from a power-law source count
            model.
            """
            sky = np.zeros((self.n_cells, self.n_cells, len(self.frequencies)))
            sky[self.n_cells // 2] = 1.0
            return sky

    test_imaging(SourceLine())


def test_imaging_source_ring():
    # Make a new foregrounds class that will create a sky that has a single source at zenith on it.
    class SourceRing(ForegroundsBase):
        def build_sky(self):
            """
            Create a grid of flux densities corresponding to a sample of point-sources drawn from a power-law source count
            model.
            """
            sky = np.zeros((self.n_cells, self.n_cells, len(self.frequencies)))

            for i in range(len(self.frequencies)):
                inds = np.arange(-self.n_cells / 2, self.n_cells / 2)
                ind_mag = np.add.outer(inds**2, inds**2)

                thissky = np.zeros(ind_mag.shape)
                thissky[np.logical_and(ind_mag>150, ind_mag<200)] = 1
                sky[:,:,i] = thissky

            return sky

    test_imaging(SourceRing())


def test_imaging_gaussian():
    # Make a new foregrounds class that will create a sky that has a single source at zenith on it.
    class Gaussian(ForegroundsBase):
        def build_sky(self):
            """
            Create a grid of flux densities corresponding to a sample of point-sources drawn from a power-law source count
            model.
            """
            sky = np.zeros((self.n_cells, self.n_cells, len(self.frequencies)))

            for i in range(len(self.frequencies)):
                thissky = np.exp(-np.add.outer(self.sky_coords**2, self.sky_coords**2)/ (2 *0.1**2))
                sky[:,:,i] = thissky

            return sky

    test_imaging(Gaussian())

if __name__ == "__main__":
    RUNNING_AS_TEST = False
    test_imaging_single_source()
    test_imaging_source_line()
    test_imaging_source_ring()
    test_imaging_gaussian()
