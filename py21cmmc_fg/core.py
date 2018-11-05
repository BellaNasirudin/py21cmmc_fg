# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 14:13:39 2018

@author: bella

Foreground core for 21cmmc

"""
from scipy.integrate import quad
import numpy as np
from astropy import constants as const
from astropy import units as un
from scipy.interpolate import RegularGridInterpolator
from powerbox.dft import fft
from powerbox import LogNormalPowerBox
from os import path
import py21cmmc as p21
from astropy.cosmology import Planck15

from py21cmmc import LightCone, UserParams, CosmoParams
from py21cmmc.mcmc.core import CoreBase, CoreLightConeModule

import logging
logger = logging.getLogger("21CMMC")

import warnings

import time
from . import c_wrapper as cw
from cached_property import cached_property


class ForegroundsBase(CoreBase):
    """
    A base class which implements some higher-level functionality that all foreground core modules will require.

    All subclasses of :class:`CoreBase` receive the argument `store`, whose documentation is in the `21CMMC` docs.
    """
    # TODO: this needs much better docstrings!

    # The defaults dict specifies arguments that need to be set if no CoreLightConeModule is loaded.
    defaults = dict(
        box_len=150.0,
        sky_cells=100,
        cosmo=Planck15,
    )

    def __init__(self, *, frequencies=None, model_params={}, simulate_post_setup=True, **kwargs):
        """
        Initialise the ForegroundsBase object.

        Parameters
        ----------
        frequencies : array, optional
            The frequencies (in MHz) at which the foregrounds are defined. These need not be provided if a lightcone
            core is pre-loaded, as they will be read from that.
        model_params : dict
            A dictionary of model parameters for a specific foreground type.
        simulate_post_setup : bool, optional
            Whether the core should add any foregrounds after the initial setup call. This is useful if an analytic
            formulation of the foreground noise is known, and they don't need to be simulated for each iteration. In
            this case, it might be useful to set `simulate_post_setup` to True in order to produce a mock dataset,
            and then set it to False for analysis.

        Other Parameters
        ----------------
        All other parameters are passed to the :class:`py21cmmc.mcmc.CoreBase` class. These include `store`, which is a
        dictionary of options for storing data in the MCMC chain.
        """
        super().__init__(**kwargs)

        self.model_params = model_params
        self.simulate_post_setup = simulate_post_setup

        # This parameter controls whether simulations are actually produced in the __call__
        self._make_sims = True

        # These save the default values. These will be *overwritten* in setup() if a LightCone is loaded
        self.frequencies = frequencies
        self.user_params = UserParams(HII_DIM=self.defaults['sky_cells'], BOX_LEN=self.defaults['box_len'])
        self.cosmo_params = CosmoParams(hlittle=self.defaults['cosmo'].h, OMm=self.defaults['cosmo'].Om0,
                                        OMb=self.defaults['cosmo'].Ob0)

    @property
    def lightcone_module(self):
        for m in self.LikelihoodComputationChain.getCoreModules():
            if isinstance(m, p21.mcmc.core.CoreLightConeModule):
                return m
        raise AttributeError("this chain was not setup with a lightcone module (or setup() has not yet been called)")

    def setup(self):

        if hasattr(self, "lightcone_module"):
            self.user_params = self.lightcone_module.user_params
            self.cosmo_params = self.lightcone_module.cosmo_params
            self.frequencies = 1420.0e6 / (1 + self.lightcone_module.lightcone_slice_redshifts)
        elif self.frequencies is None:
            raise ValueError("If no lightcone core is supplied, frequencies must be supplied.")
        elif self.frequencies.max() < 1e6:
            self.frequencies *= 1e6  #Make it Hz

        self._make_sims = bool(self.simulate_post_setup)

        # Get the names of parameters
        self.parameter_names = getattr(self.LikelihoodComputationChain.params, "keys", [])

        # Figure out if any of the model parameters are being constrained in the MCMC
        self._updating = any([p in self.model_params for p in self.parameter_names])

    def simulate_data(self, ctx):
        # Update our parameters, if they are being constrained.
        self.model_params.update({k: v for k, v in ctx.getParams().items() if k in self.model_params})

        fg_lightcone = self.mock_lightcone(self.model_params)

        # Get the foregrounds list out of the context, defaulting to empty list.
        fg = ctx.get("foregrounds", [])
        fg.append(fg_lightcone)
        if len(fg) == 1:
            ctx.add("foregrounds", fg)

    def __call__(self, ctx):
        """
        This doesn't add anything to the context, rather it just updates the parameters of the class appropriately.
        All foreground models are determined by an appropriate likelihood, calling the `power_spectrum_expectation`
        and `power_spectrum_covariance` methods.
        """
        # Update our parameters, if they are being constrained.
        if self._updating:
            self.model_params.update({k:v for k,v in ctx.getParams().items() if k in self.model_params})

    def mock_lightcone(self, model_params, fg=None):
        """
        Create and return a mock LightCone object with foregrounds only.
        """
        if fg is None:
            fg = self.build_sky(**model_params)

        return LightCone(  # Create a mock lightcone that can be read by future likelihoods as if it were the real deal.
            redshift=self.lightcone_slice_redshifts.min(),
            user_params=self.user_params,
            cosmo_params=self.cosmo_params,
            astro_params=None, flag_options=None, brightness_temp=fg
        )

    @staticmethod
    def get_sky_size(boxsize, redshifts, cosmo):
        """Get the size of the simulated sky, across the zenith, in radians"""
        return 2 * np.arctan(boxsize / (2 * float(cosmo.comoving_transverse_distance([np.mean(redshifts)]).value)))

    @property
    def sky_size(self):
        """The size of the simulated sky, across the zenith, in radians"""
        return self.get_sky_size(self.user_params.BOX_LEN, self.lightcone_slice_redshifts, self.cosmo_params.cosmo)

    @property
    def sky_size_lm(self):
        """the size of the simulated sky, across the zenith, in lm"""
        return 2 * np.sin(self.get_sky_size(self.user_params.BOX_LEN, self.lightcone_slice_redshifts, self.cosmo_params.cosmo) / 2)

    @property
    def lightcone_slice_redshifts(self):
        """The cosmological redshift (of signal) associated with each frequency"""
        return 1420e6 / self.frequencies - 1

    def power_spectrum_expectation_analytic(self, f, u, sigma):
        """
        Calculate the expectation of the power spectrum of the foregrounds.

        Note that this expects that the beam is Gaussian, and is very brittle.

        Parameters
        ----------
        f : array
            Frequencies of observation.
        u : array
            Wavelength-normalised baselines at reference frequency.
        sigma :
            Gaussian beam width.

        Returns
        -------

        """
        pass


class CorePointSourceForegrounds(ForegroundsBase):
    """
    A Core module which simulates point-source foregrounds across a cuboid sky.
    """

    def __init__(self, *, S_min=1e-1, S_max=1e-1, alpha=4100., beta=1.59, gamma=0.8, f0=150e6, **kwargs):
        """
        The initializer for the point-source foreground core.

        Note that this class is sub-classed from :class:`ForegroundsBase`, and an extra keyword arguments will be passed
        to its initializer. See its docs for more information.

        Parameters
        ----------
        S_min : float, optional
            The minimum flux density of point sources in the simulation (Jy).
        S_max : float, optional
            The maximum flux density of point sources in the simulation, representing the 'peeling limit' (Jy)
        alpha : float, optional
            The normalisation coefficient of the source counts (sr^-1 Jy^-1).
        beta : float, optional
            The power-law index of source counts.
        gamma : float, optional
            The power-law index of the spectral energy distribution of sources (assumed universal).
        f0 : float, optional
            The reference frequency, at which the other parameters are defined, in Hz.

        Notes
        -----
        The box is populated with point-sources with number counts givem by

        .. math::\frac{dN}{dS} (\nu)= \alpha \left(\frac{S_{\nu}}{S_0}\right)^{-\beta} {\rm Jy}^{-1} {\rm sr}^{-1}.

        """
        super().__init__(model_params=dict(S_min=S_min, S_max=S_max, alpha=alpha, beta=beta, gamma=gamma, f0=f0),
                         **kwargs)

    def build_sky(self, S_min=1e-1, S_max=1.0, alpha=4100., beta=1.59, gamma=0.8, f0=150e6):
        """
        Create a grid of flux densities corresponding to a sample of point-sources drawn from a power-law source count
        model.
        """
        logger.info("Populating point sources... ")
        t1 = time.time()
        # Find the mean number of sources
        n_bar = quad(lambda x: alpha * x ** (-beta), S_min, S_max)[0] * self.sky_size ** 2  # Need to multiply by sky size in steradian!

        # Generate the number of sources following poisson distribution
        # Make sure it's not 0!
        # TODO: bella, I don't think it's statistically correct to force there to be sources. If the source count
        # TODO: is so low that you can get zero sources, a warning should be raised to alert the user, but it's still
        # TODO: statistically valid.
        n_sources = np.random.poisson(n_bar)

        if not n_sources:
            warnings.warn("There are no point-sources in the sky!")

        # Generate the point sources in unit of Jy and position using uniform distribution
        S_0 = ((S_max ** (1 - beta) - S_min ** (1 - beta)) * np.random.uniform(size=n_sources) + S_min ** (
                1 - beta)) ** (1 / (1 - beta))
        pos = np.rint(np.random.uniform(0, self.user_params.HII_DIM - 1, size=(n_sources, 2))).astype(int)

        # Grid the fluxes at nu = 150
        S_0 = np.histogram2d(pos[:, 0], pos[:, 1], bins=np.arange(0, self.user_params.HII_DIM + 1, 1), weights=S_0)
        S_0 = S_0[0]

        # Find the fluxes at different frequencies based on spectral index and divide by cell area
        sky = np.outer(S_0, (self.frequencies / f0) ** (-gamma)).reshape(
            (np.shape(S_0)[0], np.shape(S_0)[0], len(self.frequencies)))
        sky /= (self.sky_size / self.user_params.HII_DIM) ** 2

        # Change the unit from Jy/sr to mK
        sky /= CoreInstrumental.conversion_factor_K_to_Jy(self.frequencies)
        logger.info("\t... took %s sec." % (time.time() - t1))
        return sky


class CoreDiffuseForegrounds(ForegroundsBase):
    """
    A 21CMMC Core MCMC module which adds diffuse foregrounds to the base signal.
    """

    def __init__(self, *args, u0=10.0, eta=0.01, rho=-2.7, mean_temp=253e3, kappa=-2.55, **kwargs):
        super().__init__(*args,
                         model_params=dict(u0=u0, eta=eta, rho=rho, mean_temp=mean_temp, kappa=kappa),
                         **kwargs)

    @staticmethod
    def build_sky(frequencies, ncells, sky_size, u0=10.0, eta=0.01, rho=-2.7, mean_temp=253e3, kappa=-2.55):
        """
        This creates diffuse structure according to Eq. 55 from Trott+2016.

        Parameters
        ----------
        frequencies : array
            The frequencies at which to determine the diffuse emission.

        ncells : int
            Number of cells on a side for the 2 sky dimensions.

        sky_size : float
            Size (in radians) of the angular dimensions of the box.

        u0, eta, rho, mean_temp, kappa : float, optional
            Parameters of the diffuse sky model (see notes)

        Returns
        -------
        Tbins : (ncells, ncells, nfreq)- array
            Brightness temperature of the diffuse emission, in mK.


        Notes
        -----
        The box is populated with a lognormal field with power spectrum given by

        .. math:: \left(\frac{2k_B}{\lambda^2}\right)^2 \Omega (eta \bar{T}_B) \left(\frac{u}{u_0}\right)^{\rho} \left(\frac{\nu}{100 {\rm MHz}}\right)^{\kappa}
        """
        logger.info("Populating diffuse foregrounds...")
        t1 = time.time()

        # Calculate mean flux density per frequency. Remember to take the square root because the Power is squared.
        Tbar = np.sqrt((frequencies / 1e8) ** kappa * mean_temp ** 2)
        power_spectrum = lambda u: eta ** 2 * (u / u0) ** rho

        # Create a log normal distribution of fluctuations
        pb = LogNormalPowerBox(N=ncells, pk=power_spectrum, dim=2, boxlength=sky_size[0], a=0, b=2 * np.pi, seed=1234)

        density = pb.delta_x() + 1

        # Multiply the inherent fluctuations by the mean flux density.
        if np.std(density) > 0:
            Tbins = np.outer(density, Tbar).reshape((ncells, ncells, len(frequencies)))
        else:
            Tbins = np.zeros((ncells, ncells, len(frequencies)))
            for i in range(len(frequencies)):
                Tbins[:, :, i] = Tbar[i]

        logger.info(f"\t... took {time.time() - t1} sec.")
        return Tbins


class CoreInstrumental(CoreBase):
    """
    Core MCMC class which converts 21cmFAST *lightcone* output into a mock observation, sampled at specific baselines.

    Assumes that either a :class:`ForegroundBase` instance, or :class:`py21cmmc.mcmc.core.CoreLightConeModule` is also
    being used (and loaded before this).
    """

    def __init__(self, antenna_posfile, freq_min, freq_max, nfreq, tile_diameter=4.0, max_bl_length=None,
                 integration_time=120, Tsys=240, sky_size=1, sky_size_coord="rad", max_tile_n=50, n_cells=None,
                 effective_collecting_area=16.0,
                 *args, **kwargs):
        """
        Parameters
        ----------
        antenna_posfile : str, {"mwa_phase2", "ska_low_v5", "grid_centres"}
            Path to a file containing antenna positions. File must be in the format such that the second column is the
            x position (in metres), and third column the y position (in metres). Some files are built-in, and these can
            be accessed by using the options defined above. The option "grid_centres" is for debugging purposes, and
            dynamically produces baselines at the UV grid nodes. This can be used to approximate an exact DFT.

        freq_min, freq_max : float
            min/max frequencies of the observation, in MHz.

        nfreq : int
            Number of frequencies in the observation.

        tile_diameter : float, optional
            The physical diameter of the tiles, in metres.

        max_bl_length : float, optional
            The maximum length (in metres) of the baselines to be included in the analysis. By default, uses all
            baselines.

        integration_time : float,optional
            The length of the observation, in seconds.
            
        sky_size : float, optional
            The sky size to use. If an EoR simulation is used underneath, it will be tiled to ensure it has this
            final size. If a foreground core is used, its size will be set to be consistent with this value (i.e.
            this parameter is preferred).

        sky_size_coord : str, optional
            What co-ordinates the sky size is in. Options are "rad" for radians, "lm" for sin-projected co-ordinates,
            or "sigma" for beam-widths (the beam is considered Gaussian in lm coordinates).

        n_cells: int, optional
            The number of pixels per side of the sky grid. Simulations will be coarsened to match this number, where
            applicable. This is useful for reducing memory usage. Default is the same number of cells as the underlying
            simulation. If set to zero, no coarsening will be performed.

        effective_collecting_area : float, optional
            The effective collecting area of a tile (equal to the geometric area times the efficiency).

        Notes
        -----
        The choice of sky size can easily result in unphysical results (i.e. more than 180 degrees, or lm > 1). An
        exception will be raised if this is the case. However, even if not unphysical, it could lead to a very large
        number of tiled simulations. This is why the `max_tile_n` option is provided. If this is exceeded, an exception
        will be raised.
        """
        super().__init__(*args, **kwargs)

        self.antenna_posfile = antenna_posfile
        self.instrumental_frequencies = np.linspace(freq_min * 1e6, freq_max * 1e6, nfreq)
        self.tile_diameter = tile_diameter * un.m
        self.max_bl_length = max_bl_length
        self.integration_time = integration_time
        self.Tsys = Tsys

        self.effective_collecting_area = effective_collecting_area * un.m**2

        if self.effective_collecting_area > self.tile_diameter**2:
            warnings.warn("The effective collecting area (%s) is greater than the tile diameter squared!")

        # Sky size parameters.
        self._sky_size = sky_size
        self._sky_size_coord = sky_size_coord
        self._max_tile_n = max_tile_n

        self.n_cells = n_cells

        # Setup baseline lengths.
        if self.antenna_posfile == "grid_centres":
            self.baselines = None

        else:
            # If antenna_posfile is a simple string, we'll try to find it in the data directory.
            data_path = path.join(path.dirname(__file__), 'data', self.antenna_posfile + '.txt')

            if path.exists(data_path):
                ant_pos = np.genfromtxt(data_path, float)
            else:
                ant_pos = np.genfromtxt(self.antenna_posfile, float)

            # Find all the possible combination of tile displacement
            # baselines is a dim2 array of x and y displacements.
            self.baselines = self.get_baselines(ant_pos[:, 1], ant_pos[:, 2]) * un.m

            if self.max_bl_length:
                self.baselines = self.baselines[
                    self.baselines[:, 0].value ** 2 + self.baselines[:, 1].value ** 2 <= self.max_bl_length ** 2]

    def setup(self):

        if self.n_cells is None:
            self.n_cells = self._base_module.user_params.HII_DIM
        elif self.n_cells == 0:
            self.n_cells = self._n_stitched

    @property
    def lightcone_slice_redshifts(self):
        return self._base_module.lightcone_slice_redshifts

    @property
    def sky_size(self):
        "The sky size in lm co-ordinates"
        if self._sky_size_coord == "lm":
            size = self._sky_size
        elif self._sky_size_coord == "rad":
            size = 2 * np.sin(self._sky_size / 2)  # 2 factors account for centering of sky
        elif self._sky_size_coord == "sigma":
            size = self._sky_size * np.max(self.sigma(self.instrumental_frequencies))
        else:
            raise ValueError("sky_size_coord must be one of (lm, rad, sigma)")

        if np.sqrt(2) * size / 2 > 1:
            raise ValueError("sky size is unphysical (%s > 1)" % size)

        if size / self.eor_size_lm.min() > self._max_tile_n:
            raise ValueError("sky size is larger than max requested (must be tiled %s times, but max is %s)" % (
            size / self.eor_size_lm, self._max_tile_n))

        return size

    @property
    def eor_size_lm(self):
        """
        The size of the base 21cmFAST simulation in lm co-ordinates.
        """
        for m in self.LikelihoodComputationChain.getCoreModules():
            if isinstance(m, CoreLightConeModule) or isinstance(m, ForegroundsBase):
                return ForegroundsBase.get_sky_size(m.user_params.BOX_LEN, self.lightcone_slice_redshifts,
                                                    m.cosmo_params.cosmo)

        # If no lightcone module is found, raise an exception.
        raise AttributeError("No eor_size is applicable, as no LightCone or ForegroundsBase modules are loaded")

    @property
    def n_stitch(self):
        """The number of times the simulation needs to be stitched to get to *at least* sky_size"""
        return int(np.ceil(self.sky_size / ForegroundsBase.get_sky_size(self._base_module.user_params.BOX_LEN,
                                                                        self.lightcone_slice_redshifts,
                                                                        self._base_module.cosmo_params.cosmo)))

    @property
    def _base_module(self):
        "Basic foreground/lightcone module which this requires"
        if not hasattr(self, "LikelihoodComputationChain"):
            raise AttributeError("setup() must have been called for the _base_module to exist")

        for m in self.LikelihoodComputationChain.getCoreModules():
            if isinstance(m, ForegroundsBase) or isinstance(m, p21.mcmc.core.CoreLightConeModule):
                return m  # Doesn't really matter which one, we only want to access basic properties.

        # If nothing is returned, the correct modules must not have been loaded.
        raise ValueError("For CoreInstrumental, either a ForegroundBase or CoreLightConeModule must be loaded")

    @property
    def sim_frequencies(self):
        "The frequencies associated with slice redshifts, in Hz"
        return 1420e6 / (self.lightcone_slice_redshifts + 1)

    @property
    def sim_sky_size(self):
        """Size of the simulated sky (before stitching/coarsening) in radians"""
        return ForegroundsBase.get_sky_size(
            self._base_module.user_params.BOX_LEN,
            self.lightcone_slice_redshifts,
            self._base_module.cosmo_params.cosmo
        )

    def simulate_data(self, ctx):
        """
        Generate a set of realistic visibilities (i.e. the output we expect from an interferometer) and add it to the
        context. Also, add the linear frequencies of the observation to the context.
        """
        # Get the basic signal lightcone out of context
        lightcone = ctx.get("lightcone", [])

        # Make it a list
        if lightcone != []:
            lightcone = [lightcone]

        # Get any foreground lightcones
        foregrounds = ctx.get("foregrounds", [])

        # Get the total brightness
        total_brightness = 0
        for lc in lightcone + foregrounds:
            total_brightness += lc.brightness_temp

        total_brightness = self.stitch_and_coarsen(total_brightness)
        vis = self.add_instrument(total_brightness)

        ctx.add("visibilities", vis)
        ctx.add("baselines", self.baselines.value)
        ctx.add("frequencies", self.instrumental_frequencies)
        ctx.add("new_sky", total_brightness)

    def add_instrument(self, lightcone):
        # Find beam attenuation
        # n_cells is the number of cells per side in the sky of the stitched/coarsened array.
        # sky_size is in lm co-ordinates
        attenuation = self.beam(self.sim_frequencies)

        # Change the units of lightcone from mK to Jy
        beam_sky = lightcone * self.conversion_factor_K_to_Jy(self.sim_frequencies, self.beam_area(self.sim_frequencies)) * attenuation

        # Fourier transform image plane to UV plane.
        uvplane, uv = self.image_to_uv(beam_sky, self.sky_size)

        # This is probably bad, but set baselines if none are given, to coincide exactly with the uv grid at centre
        # frequency.
        if self.baselines is None:
            self.baselines = np.zeros((len(uv[0]) ** 2, 2))
            U, V = np.meshgrid(uv[0], uv[1])
            self.baselines[:, 0] = U.flatten() * (const.c / np.mean(self.sim_frequencies))
            self.baselines[:, 1] = V.flatten() * (const.c / np.mean(self.sim_frequencies))
            self.baselines = self.baselines * un.m

        # Fourier Transform over the (u,v) dimension and baselines sampling
        visibilities = self.sample_onto_baselines(uvplane, uv, self.baselines, self.sim_frequencies)

        visibilities = self.interpolate_frequencies(visibilities, self.sim_frequencies, self.instrumental_frequencies)

        # Add thermal noise using the mean beam area
        visibilities = self.add_thermal_noise(visibilities)

        return visibilities

    def sigma(self, frequencies):
        "The Gaussian beam width at each frequency"
        epsilon = 0.42  # scaling from airy disk to Gaussian
        return ((epsilon * const.c) / (frequencies / un.s * self.tile_diameter)).to(un.dimensionless_unscaled)

    @cached_property
    def sky_coords(self):
        """Grid-coordinates of the (stitched/coarsened) simulation in lm units"""
        return np.linspace(-self.sky_size / 2, self.sky_size / 2, self.n_cells+1)[:-1]

    @cached_property
    def cell_size(self):
        """Size (in lm) of a cell of the stitched/coarsened simulation"""
        return self.sky_size/self.n_cells

    def beam_area(self, frequencies):
        """
        The integrated beam area. Assumes a frequency-dependent Gaussian beam (in lm-space, as this class implements).

        Parameters
        ----------
        frequencies : array-like
            Frequencies at which to compute the area.

        Returns
        -------
        beam_area: (nfreq)-array
            The beam area of the sky, in lm.
        """
        sig = self.sigma(frequencies)
        return np.pi * sig**2

    def beam(self, frequencies):
        """
        Generate a frequency-dependent Gaussian beam attenuation across the sky per frequency.

        Parameters
        ----------
        ncells : int
            Number of cells in the sky grid.

        sky_size : float
            The extent of the sky in lm.

        Returns
        -------
        attenuation : (ncells, ncells, nfrequencies)-array
            The beam attenuation (maximum unity) over the sky.
        
        beam_area : (nfrequencies)-array
            The beam area of the sky (in sr).
        """

        # Create a meshgrid for the beam attenuation on sky array
        L, M = np.meshgrid(self.sky_coords, self.sky_coords)

        attenuation = np.exp(
            np.outer(-(L ** 2 + M ** 2), 1. /(2 * self.sigma(frequencies) ** 2)).reshape((self.n_cells, self.n_cells, len(frequencies))))

        return attenuation


    @staticmethod
    def conversion_factor_K_to_Jy(nu, omega=None):
        """

        Parameters
        ----------
        nu : float array, optional
            The frequencies of the observation (in Hz).
        
        omega: float or array
            The area of the observation or the beam (in sr).

        Returns
        -------
        conversion_factor : float or array
            The conversion factor(s) (per frequency) which convert temperature in Kelvin to flux density in Jy.

        """
        if omega is None:
            flux_density = (2 * const.k_B * 1e-3 * un.K / (((const.c) / (nu * (1 / un.s))) ** 2) * 1e26).to(
                un.W / (un.Hz * un.m ** 2))
        else:
            flux_density = (2 * const.k_B * 1e-3 * un.K / (((const.c) / (nu * (1 / un.s))) ** 2) * 1e26).to(
                un.W / (un.Hz * un.m ** 2)) / omega

        return flux_density.value

    @staticmethod
    def image_to_uv(sky, L):
        """
        Transform a box from image plan to UV plane.

        Parameters
        ----------
        sky : (ncells, ncells, nfreq)-array
            The frequency-dependent sky brightness (in arbitrary units)

        L : float
            The size of the box in radians.

        Returns
        -------
        uvsky : (ncells, ncells, nfreq)-array
            The UV-plane representation of the sky. Units are units of the sky times radians.

        uv_scale : list of two arrays.
            The u and v co-ordinates of the uvsky, respectively. Units are inverse of L.
        """
        logger.info("Converting to UV space...")
        t1 = time.time()
        ft, uv_scale = fft(sky, [L, L], axes=(0, 1), a = 0, b=2*np.pi)
        logger.info("... took %s sec." % (time.time() - t1))
        return ft, uv_scale

    @staticmethod
    def sample_onto_baselines(uvplane, uv, baselines, frequencies):
        """
        Sample a gridded UV sky onto a set of baselines.

        Sampling is done via linear interpolation over the regular grid.

        Parameters
        ----------
        uvplane : (ncells, ncells, nfreq)-array
            The gridded UV sky, in Jy.

        uv : list of two 1D arrays
            The u and v coordinates of the uvplane respectively.

        baselines : (N,2)-array
            Each row should be the (x,y) co-ordinates of a baseline, in metres.

        frequencies : 1D array
            The frequencies of the uvplane.

        Returns
        -------
        vis : complex (N, nfreq)-array
             The visibilities defined at each baseline.

        """
        vis = np.zeros((len(baselines), len(frequencies)), dtype=np.complex128)

        frequencies = frequencies / un.s

        logger.info("Sampling the data onto baselines...")
        t1 = time.time()

        for i, ff in enumerate(frequencies):
            lamb = const.c / ff.to(1 / un.s)
            arr = np.zeros(np.shape(baselines))
            arr[:, 0] = (baselines[:, 0] / lamb).value
            arr[:, 1] = (baselines[:, 1] / lamb).value

            real = np.real(uvplane[:, :, i])
            imag = np.imag(uvplane[:, :, i])

            f_real = RegularGridInterpolator([uv[0], uv[1]], real, bounds_error=False, fill_value=0)
            f_imag = RegularGridInterpolator([uv[0], uv[1]], imag, bounds_error=False, fill_value=0)

            FT_real = f_real(arr)
            FT_imag = f_imag(arr)

            vis[:, i] = FT_real + FT_imag * 1j

        logger.info("... took %s sec." % (time.time() - t1))
        return vis

    @staticmethod
    def interpolate_frequencies(visibilities, freq_grid, linear_freq):
        """
        Interpolate a set of visibilities from a non-linear grid of frequencies onto a linear grid. Interpolation
        is linear.

        Parameters
        ----------
        visibilities : complex (n_baselines, n_freq)-array
            The visibilities at each baseline and frequency.

        freq_grid : (nfreq)-array
            The grid of frequencies at which the visibilities are defined.

        linear_freq : (N,)-array
            The set of frequencies on which to interpolate the visibilities.

        Returns
        -------
        new_vis : complex (n_baselines, N)-array
            The interpolated visibilities.
        """
        logger.info("Interpolating frequencies...")
        t1 = time.time()

        # Make sure the input frequencies are ascending.
        if freq_grid[0] > freq_grid[-1]:
            freq_grid = freq_grid[::-1]
            visibilities = np.flip(visibilities, 1)

        # USING C Code reduces time by about 200%!!
        out = cw.interpolate_visibility_frequencies(visibilities, freq_grid, linear_freq)

        logger.info("... took %s sec."%(time.time() - t1))
        return out

    @staticmethod
    def get_baselines(x, y):
        """
        From a set of antenna positions, determine the non-autocorrelated baselines.

        Parameters
        ----------
        x, y : 1D arrays of the same length.
            The positions of the arrays (presumably in metres).

        Returns
        -------
        baselines : (n_baselines,2)-array
            Each row is the (x,y) co-ordinate of a baseline, in the same units as x,y.
        """
        # ignore up.
        ind = np.tril_indices(len(x), k=-1)
        Xsep = np.add.outer(x, -x)[ind]
        Ysep = np.add.outer(y, -y)[ind]

        # Remove autocorrelations
        zeros = np.logical_and(Xsep.flatten() == 0, Ysep.flatten() == 0)

        return np.array([Xsep.flatten()[np.logical_not(zeros)], Ysep.flatten()[np.logical_not(zeros)]]).T

    @property
    def thermal_variance_baseline(self):
        """
        The thermal variance of each baseline (assumed constant across baselines/times/frequencies.

        Equation comes from Trott 2016 (from Morales 2005)
        """
        df = self.instrumental_frequencies[1] - self.instrumental_frequencies[0]
        sigma = 2 * 1e26 * const.k_B.value * self.Tsys / self.effective_collecting_area / np.sqrt(df * self.integration_time)
        return (sigma**2).value

    def add_thermal_noise(self, visibilities):
        """
        Add thermal noise to each visibility.

        Parameters
        ----------
        visibilities : (n_baseline, n_freq)-array
            The visibilities at each baseline and frequency.

        frequencies : (n_freq)-array
            The frequencies of the observation.
        
        beam_area : float
            The area of the beam (in sr).

        delta_t : float, optional
            The integration time.

        Returns
        -------
        visibilities : array
            The visibilities at each baseline and frequency with the thermal noise from the sky.

        """
        logger.info("Adding thermal noise...")
        # TODO: should there be a root(2) here?
        rl_im = np.random.normal(0, 1, (2,)+visibilities.shape)

        return visibilities + np.sqrt(self.thermal_variance_baseline) * (rl_im[0, :] + rl_im[1, :] * 1j)

    @property
    def _n_stitched(self):
        """The number of cells in the stitched box (pre-coarsening)"""
        return int(self.sky_size * self._base_module.user_params.HII_DIM / self.sim_sky_size)

    def stitch_and_coarsen(self, lightcone):
        logger.info("Stitching and coarsening boxes...")
        t1 = time.time()

        new = cw.stitch_and_coarsen_sky(lightcone, self.sim_sky_size, self.sky_size, self.n_cells)

        logger.info("... took %s sec."%(time.time() - t1))
        return new