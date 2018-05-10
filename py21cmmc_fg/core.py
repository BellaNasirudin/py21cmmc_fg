#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 14:13:39 2018

@author: bella

Foreground core for 21cmmc

"""
from scipy.integrate import quad
import numpy as np
from astropy import constants as const
from astropy.cosmology import Planck15 as cosmo  # TODO: this is not quite correct, if we pass cosmo parameters.
from astropy import units as un
from scipy.interpolate import RegularGridInterpolator
from powerbox.dft import fft
from os import path

class ForegroundCore:
    def __init__(self, S_min, S_max):
        """
        Setting up variables minimum and maximum flux
        """
        # print "Initializing the foreground core"

        self.S_min = S_min
        self.S_max = S_max

    def setup(self):
        print("Generating the foregrounds")

    def __call__(self, ctx):
        """
        Reading data and applying the foregrounds based on the variables
        """
        print("Getting the simulation data")

        EoR_lightcone = ctx.get("output").lightcone_box
        redshifts = ctx.get("output").redshifts_slices
        boxsize = ctx.get("output").box_len

        new_lightcone, frequencies, sky_size = self.add_foregrounds(EoR_lightcone, redshifts, boxsize)

        ctx.add("foreground_lightcone", new_lightcone)
        ctx.add("frequencies", frequencies)
        ctx.add("sky_size", sky_size)

    def add_foregrounds(self, EoR_lightcone, redshifts, boxsize):
        """
        Adding foregrounds in unit of Jy to the EoR signal (lightcone)
        
        Parameters
        ----------
        EoR_lightcone : The EoR signal lightcone outputted by 21cmFAST
        
        redshifts : The redshifts (instead of comoving distance) corresponding to each slice of EoR lightcone
        
        boxsize : The size of the EoR lightcone in Mpc
        
        S_min : The minimum flux in Jy for the point sources
        
        S_max : The maximum flux in Jy for the point sources
        
        Output
        ------
        
        sky : The EoR signal and foreground in unit of Jy/sr
        
        linFrequencies : The linearly-spaced frequencies corresponding to each slice of sky
        
        """
        print("Adding the foregrounds")

        # Number of 2D cells in sky array
        sky_cells = EoR_lightcone.shape[0]

        # Convert the boxsize from Mpc to radian
        # IF USING SMALL BOX, THIS IS WHERE WE DO STITCHING!!!
        sky_size = 2 * np.arctan(boxsize / (2 * (cosmo.comoving_transverse_distance([np.mean(redshifts)]).value)))
        
        print("Min and max Tb in mK",EoR_lightcone.min(),EoR_lightcone.max())
        
        # Change the units of brightness temperature from mK to Jy/sr
        EoR_lightcone = np.flip(EoR_lightcone * self.convert_factor_sources(), axis=2)
        
        print("Min and max Tb in Jy/sr",EoR_lightcone.min(),EoR_lightcone.max())

        # Convert redshifts to frequencies in Hz and generate linearly-spaced frequencies
        frequencies = 1420e6 / (1 + redshifts[::-1])

        # Interpolate linearly in frequency (POSSIBLY IN RADIAN AS WELL)
        linLightcone, linFrequencies = self.interpolate_freqs(EoR_lightcone, frequencies)

        # Generate the point sources foregrounds and
        foregrounds = np.repeat(self.add_points(self.S_min, self.S_max, sky_cells, sky_size),
                                np.shape(EoR_lightcone)[2], axis=2)

        self.add_diffuse()

        ## Add the foregrounds and the EoR signal
        sky = foregrounds + linLightcone
        
        print("Min and max sky in Jy/sr",sky.min(),sky.max())
        
        return sky, linFrequencies, sky_size

    def interpolate_freqs(self, data, frequencies, uv_range=100):
        """
        Interpolate the irregular frequencies so that they are linearly-spaced
        """

        linFrequencies = np.linspace(np.min(frequencies), np.max(frequencies), data.shape[2])

        ncells = np.shape(data)[0]
        # Create the xy data
        xy = np.linspace(-uv_range, uv_range, ncells)

        # generate the interpolation function
        func = RegularGridInterpolator([xy, xy, frequencies], data, bounds_error=False, fill_value=0)

        # Create a meshgrid to interpolate the points
        XY, YX, LINZREDS = np.meshgrid(xy, xy, linFrequencies)

        # Flatten the arrays so the can be put into pts array
        XY = XY.flatten()
        YX = YX.flatten()
        LINZREDS = LINZREDS.flatten()

        # Create the points to interpolate
        numpts = XY.size
        pts = np.zeros([numpts, 3])
        pts[:, 0], pts[:, 1], pts[:, 2] = XY, YX, LINZREDS

        # Interpolate the points
        interpData = func(pts)

        # Reshape the data 
        interpData = interpData.reshape(ncells, ncells, len(linFrequencies))

        return interpData, linFrequencies

    def add_points(self, S_min, S_max, sky_cells, sky_area):

        ## Create a function for source count distribution
        alpha = 4100
        beta = 1.59
        source_count = lambda x: alpha * x ** (-beta)

        ## Find the mean number of sources
        n_bar = quad(source_count, S_min, S_max)[0]

        ## Generate the number of sources following poisson distribution
        N_sources = np.random.poisson(n_bar)

        ## Generate the point sources in unit of Jy and position using uniform distribution
        fluxes = ((S_max ** (1 - beta) - S_min ** (1 - beta)) * np.random.uniform(size=N_sources) + S_min ** (
                    1 - beta)) ** (1 / (1 - beta))
        pos = np.rint(np.random.uniform(0, sky_cells - 1, size=(N_sources, 2))).astype(int)

        ## Create an empty array and fill it up by adding the point sources
        sky = np.zeros((sky_cells, sky_cells, 1))
        for ii in range(N_sources):
            sky[pos[ii, 0], pos[ii, 1]] += fluxes[ii]

        ## Divide by area of each sky cell; Jy/sr
        sky = sky / (sky_area / sky_cells)
        
        print("Min and max foregrounds flux in Jy/sr",sky.min(), sky.max())

        return sky

    def add_diffuse(self):
        print("Please input diffuse sources")

    def convert_factor_sources(self, nu=0):

        ## Can either do it with the beam or without the beam (frequency dependent)
        if (nu == 0):
            A_eff = 20 * un.m ** 2

            flux_density = (2 * 1e26 * const.k_B * 1e-3 * un.K / (A_eff * (1 * un.Hz) * (1 * un.s))).to(
                un.W / (un.Hz * un.m ** 2))

        else:
            flux_density = (2 * const.k_B * 1e-3 * un.K / (((const.c) / (nu.to(1 / un.s))) ** 2) * 1e26).to(
                un.W / (un.Hz * un.m ** 2))

        return flux_density.value


class CoreInstrumentalSampling:
    """
    Core MCMC class which converts 21cmFAST *lightcone* output into a mock observation, sampled at specific baselines.

    Assumes that a :class:`ForegroundCore` is also being used (and loaded before this).

    Parameters
    ----------
    antenna_posfile : str, {"mwa_phase2", "ska_low_v5"}
        Path to a file containing antenna positions. File must be in the format such that the second column is the
        x position (in metres), and third column the y position (in metres). Some files are built-in, and these can
        be accessed by using the options defined above.

    freq_min, freq_max : float
        min/max frequencies of the observation, in MHz.
    """

    def __init__(self, antenna_posfile, freq_min, freq_max, nfreq, tile_diameter, max_bl_length=150.0, **kwargs):
        super().__init__(**kwargs)
        self.antenna_posfile = antenna_posfile
        self.instrumental_frequencies = np.linspace(freq_min*1e6, freq_max*1e6, nfreq)
        self.tile_diameter = tile_diameter * un.m
        self.max_bl_length = max_bl_length

    def setup(self):
        # If antenna_posfile is a simple string, we'll try to find it in the data directory.
        if self.antenna_posfile == "grid_centres":
            self.baselines = None

        else:
            data_path = path.join(path.dirname(__file__), 'data', self.antenna_posfile+'.txt')
            if path.exists(data_path):
                ant_pos = np.genfromtxt(data_path, float)
            else:
                ant_pos = np.genfromtxt(self.antenna_posfile, float)


            # Find all the possible combination of tile displacement
            # baselines is a tuple of x and y displacements.
            self.baselines = self.get_baselines(ant_pos[:, 1], ant_pos[:, 2]) * un.m

            self.baselines = self.baselines[
                self.baselines[:, 0].value ** 2 + self.baselines[:, 1].value ** 2 <= self.max_bl_length ** 2]

    def __call__(self, ctx):
        new_lightcone = ctx.get("foreground_lightcone")
        frequencies = ctx.get("frequencies")
        sky_size = ctx.get("sky_size")

        vis = self.add_instrument(new_lightcone, frequencies, sky_size)

        ctx.add("visibilities", vis)
        ctx.add("baselines", self.baselines)
        ctx.add("frequencies", self.instrumental_frequencies)

    def add_instrument(self, lightcone, frequencies, sky_size):

        print("Adding instrument model")

        # Number of 2D cells in sky array
        sky_cells = np.shape(lightcone)[0]

        # Add the beam attenuation
        beam_sky = lightcone * self.beam(frequencies, sky_cells, sky_size)

        # Fourier transform image plane to UV plane.
        uvplane, uv = self.image_to_uv(beam_sky, sky_size)

        # This is probably bad, but set baselines
        if self.baselines is None:
            self.baselines = np.zeros((len(uv[0])**2, 2))
            U,V = np.meshgrid(uv[0], uv[1])
            self.baselines[:, 0] = U.flatten()*(const.c/frequencies.max())
            self.baselines[:, 1] = V.flatten()*(const.c/frequencies.max())

        # Fourier Transform over the (u,v) dimension and baselines sampling
        visibilities = self.add_baselines_sampling(uvplane, uv, frequencies)
        visibilities = self.interpolate_frequencies(visibilities, frequencies)

        # Add thermal noise
        visibilities = self.add_thermal_noise(visibilities)

        return visibilities

    def beam(self, frequencies, sky_cells, sky_size):

        print("Adding beam attenuation")
        ## First find the sigma of the beam
        epsilon = 0.42
        D = self.tile_diameter

        frequencies = frequencies * (un.s ** (-1))
        sigma = ((epsilon * const.c) / (frequencies * D)).to(un.dimensionless_unscaled)

        # Then create a meshgrid for the beam attenuation on sky array
        sky_coords_lm = np.sin(np.linspace(-sky_size / 2, sky_size / 2, sky_cells))

        L, M = np.meshgrid(sky_coords_lm, sky_coords_lm)

        beam = np.outer(np.exp(-L ** 2 + M ** 2), 1. / sigma ** 2).reshape((sky_cells, sky_cells, len(frequencies)))
        return beam

    def image_to_uv(self, sky, L):
        FT, uv_scale = fft(sky, [L, L], axes=(0, 1))
        return FT, uv_scale

    def add_baselines_sampling(self, uvplane, uv, frequencies):

        vis = np.zeros((len(self.baselines), len(frequencies)), dtype=np.complex128)

        frequencies = frequencies * un.s ** (-1)

        print("Sampling the data onto baselines")

        for i, ff in enumerate(frequencies):
            lamb = const.c / ff.to(1 / un.s)
            u = (self.baselines[:,0] / lamb).value
            v = (self.baselines[:,1] / lamb).value
            vis[:, i] = self.sample_on_baselines(uvplane[:, :, i], uv, u, v)

        return vis

    def sample_on_baselines(self, uvplane, uvgrid, u, v):
        real = np.real(uvplane)
        imag = np.imag(uvplane)

        f_real = RegularGridInterpolator([uvgrid[0], uvgrid[1]], real)
        f_imag = RegularGridInterpolator([uvgrid[0], uvgrid[1]], imag)

        arr = np.zeros((len(u), 2))
        arr[:, 0] = u
        arr[:, 1] = v

        FT_real = f_real(arr)
        FT_imag = f_imag(arr)

        return FT_real + FT_imag * 1j

    def interpolate_frequencies(self, visibilities, freq_grid):
        new_vis = np.zeros((visibilities.shape[0], len(self.instrumental_frequencies)), dtype=np.complex64)

        for i, vis in enumerate(visibilities):
            rl = RegularGridInterpolator((freq_grid,), np.real(vis))(self.instrumental_frequencies)
            im = RegularGridInterpolator((freq_grid,), np.imag(vis))(self.instrumental_frequencies)
            new_vis[i] = rl + im * 1j

        return new_vis

    def get_baselines(self, x, y):

        # ignore up.
        ind = np.tril_indices(len(x), k=-1)
        Xsep = np.add.outer(x, -x)[ind]
        Ysep = np.add.outer(y, -y)[ind]

        # Remove autocorrelations
        zeros = np.logical_and(Xsep.flatten()==0, Ysep.flatten()==0)

        return np.array([Xsep.flatten()[np.logical_not(zeros)], Ysep.flatten()[np.logical_not(zeros)]]).T

    def add_thermal_noise(self,  visibilities):
        return visibilities