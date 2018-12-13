"""
Some tests of the instrumental core/likelihood.
"""

from py21cmmc_fg.core import CoreInstrumental, ForegroundsBase
from py21cmmc_fg.likelihood import LikelihoodInstrumental2D
from py21cmmc_fg.diagnostics import imaging
import numpy as np

from powerbox.dft import fft, ifft
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNNING_AS_TEST = True

import logging

logger = logging.getLogger("21CMMC")
logger.setLevel(logging.DEBUG)


def test_imaging_grid(core_ss):
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

    fig = imaging(cores=[core_ss, core_instr], lk=lk)

    plt.savefig("test_imaging_%s.png" % core_ss.__class__.__name__)
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
            sky[self.n_cells // 2, self.n_cells // 2] = 1.0
            return sky

    test_imaging_grid(SingleSource())


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

    test_imaging_grid(SourceLine())


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
                ind_mag = np.add.outer(inds ** 2, inds ** 2)

                thissky = np.zeros(ind_mag.shape)
                thissky[np.logical_and(ind_mag > 150, ind_mag < 200)] = 1
                sky[:, :, i] = thissky

            return sky

    test_imaging_grid(SourceRing())


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
                thissky = np.exp(-np.add.outer(self.sky_coords ** 2, self.sky_coords ** 2) / (2 * 0.1 ** 2))
                sky[:, :, i] = thissky

            return sky

    test_imaging_grid(Gaussian())


def test_imaging_sin_tiled_not_coarsened():
    # Make a new foregrounds class that will create a sky that has a single source at zenith on it.
    class Sine(ForegroundsBase):
        def build_sky(self):
            """
            Create a grid of flux densities corresponding to a sample of point-sources drawn from a power-law source count
            model.
            """
            sky = np.zeros((self.n_cells, self.n_cells, len(self.frequencies)))

            l = np.arange(self.n_cells)
            L, M = np.meshgrid(l,l)

            for i in range(len(self.frequencies)):
                thissky = np.sin((L*2 + 3*M)*2*np.pi/self.n_cells + np.pi/2) + 1
                sky[:, :, i] = thissky

            return sky

    test_imaging_grid(Sine(sky_size=0.629565, n_cells=150))



if __name__ == "__main__":
    RUNNING_AS_TEST = False
    test_imaging_single_source()
    test_imaging_source_line()
    test_imaging_source_ring()
    test_imaging_gaussian()
    test_imaging_sin_tiled_not_coarsened()