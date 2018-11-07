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
from py21cmmc.mcmc.core import CoreBase, CoreLightConeModule, NotSetupError

import logging

logger = logging.getLogger("21CMMC")

import warnings

import time
from . import c_wrapper as cw
from cached_property import cached_property
import healpy as hp


class ForegroundsBase(CoreBase):
    """
    A base class which implements some higher-level functionality that all foreground core modules will require.

    All subclasses of :class:`CoreBase` receive the argument `store`, whose documentation is in the `21CMMC` docs.
    """

    # TODO: this needs much better docstrings!

    def __init__(self, *, nside_base=10, frequencies=None, model_params={}, **kwargs):
        """
        Initialise the ForegroundsBase object.

        Parameters
        ----------
        frequencies : array, optional
            The frequencies (in MHz) at which the foregrounds are defined. These need not be provided if a lightcone
            core is pre-loaded, as they will be read from that.
        model_params : dict
            A dictionary of model parameters for a specific foreground type.

        Other Parameters
        ----------------
        All other parameters are passed to the :class:`py21cmmc.mcmc.CoreBase` class. These include `store`, which is a
        dictionary of options for storing data in the MCMC chain.
        """
        super().__init__(**kwargs)

        if nside_base >= 30:
            raise ValueError("nside_base must be less than 30!")

        self._nside = 2 ** nside_base
        self.model_params = model_params

        # These save the default values. These will be *overwritten* in setup() if a LightCone is loaded
        self.frequencies = frequencies * 1e6

    @cached_property
    def parameter_names(self):
        """The names of the parameters that are being constrained."""
        return getattr(self.LikelihoodComputationChain.params, "keys", [])

    @cached_property
    def _updating(self):
        """Whether any of the parameters being constrained are part of this model"""
        return any([p in self.model_params for p in self.parameter_names])

    @cached_property
    def npixels(self):
        return hp.nside2npix(self._nside)

    @cached_property
    def cell_area(self):
        return hp.nside2pixarea(self._nside)

    def simulate_data(self, ctx):
        # Update our parameters, if they are being constrained.
        self.model_params.update({k: v for k, v in ctx.getParams().items() if k in self.model_params})

        logger.info(f"Building sky for {self.__class__.__name__}...")
        t1 = time.time()

        sky = self.build_sky(**self.model_params)

        logger.info(f"... took {time.time()-t1} sec.")

        # Get the foregrounds list out of the context, defaulting to empty list.
        fg = ctx.get("foregrounds", [])
        fg.append(sky)
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
            self.model_params.update({k: v for k, v in ctx.getParams().items() if k in self.model_params})

    @property
    def redshifts(self):
        """The cosmological redshift (of signal) associated with each frequency"""
        return 1420e6 / self.frequencies - 1

    @property
    def angles(self):
        """
        The angles (radians) associated with each pixel of the foreground map.
        """
        return hp.pix2ang(self._nside, np.arange(self.npixels))


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
        Create a healpix sky corresponding to a sample of point-sources drawn from a power-law source count
        model.

        Parameters
        ----------
        See init function for model parameters.

        Returns
        -------
        sky : 1d-array
            The flux-density-per-steradian of each cell [Jy/sr]. The cells correspond to a healpix map of the sky, with
            :meth:`~npixels` pixels.
        """
        logger.info("Populating point sources... ")
        t1 = time.time()

        # Find the mean number of sources
        n_bar = quad(lambda x: alpha * x ** (-beta), S_min, S_max)[0] * 4 * np.pi  # over whole sky

        # Generate the number of sources following poisson distribution
        n_sources = np.random.poisson(n_bar)

        if not n_sources:
            logger.warn("There are no point-sources in the sky!")

        # Generate the point sources in unit of Jy and position using uniform distribution
        S_0 = ((S_max ** (1 - beta) - S_min ** (1 - beta)) * np.random.uniform(size=n_sources) + S_min ** (
                1 - beta)) ** (1 / (1 - beta))
        pos = np.rint(np.random.uniform(0, self.npixels - 1, size=n_sources)).astype(int)

        # Grid the fluxes at nu = f0
        S_0 = np.bincount(pos, weights=S_0, minlength=self.npixels)

        # Find the fluxes at different frequencies based on spectral index and divide by cell area
        sky = np.outer(S_0, (self.frequencies / f0) ** (-gamma)).reshape(
            (np.shape(S_0)[0], np.shape(S_0)[0], len(self.frequencies)))
        sky /= self.cell_area

        # Change the unit from Jy/sr to mK
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
        raise NotImplementedError("Diffuse foregrounds are wrong as they don't do healpix...")

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
    Core MCMC class which can convert both 21cmFAST *lightcones* and/or ForegroundBase outputs into a mock observation,
    sampled at specific baselines.

    Assumes that either a :class:`ForegroundBase` instance, or :class:`py21cmmc.mcmc.core.CoreLightConeModule` is also
    being used (and loaded before this).
    """

    def __init__(self, antenna_posfile, freq_min, freq_max, nfreq, tile_diameter=4.0, max_bl_length=None,
                 integration_time=120, Tsys=240, sky_extent=3, max_tile_n=50, n_cells=None,
                 effective_collecting_area=16.0, antenna_posfile_is_baselines=False,
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

        Tsys : float, optional
            The system temperature of the instrument, in K.

        sky_extent : float, optional
            Sets the minimum size of the underlying EoR simulation, in units of the (maximum) beam width, when
            calculating visibilities. While the foregrounds are always specified over the entire sky, the EoR simulation
            is in linear co-ordinates, and is often much smaller than a beam width. If so, it will be appropriately
            tiled until it matches this size. It is always assumed to be centred at zenith, and the tiling ensures
            that this `sky_extent` is reached in every line through zenith. See notes on tiling below.

        n_cells: int, optional
            The number of pixels per side of the sky grid. Simulations will be coarsened to match this number, where
            applicable. This is useful for reducing memory usage. Default is the same number of cells as the underlying
            simulation. If set to zero, no coarsening will be performed.

        max_tile_n : int, optional
            Maximum number of tiling to perform on any given axis. This is really a safety-catch parameter which
            raises an error if the number of tilings will be very large. In these cases, it is likely that any results
            will be very inaccurate in any case.

        effective_collecting_area : float, optional
            The effective collecting area of a tile (equal to the geometric area times the efficiency).

        antenna_posfile_is_baselines : bool, optional
            If True, specifies that the values read in the `antenna_posfile` are baselines (in m), not antenna positions.
        Notes
        -----
        The beam is assumed to be Gaussian in (l,m) co-ordinates, with a sharp truncation at |l|=1. If the `sky_extent`
        gives a sky that extends beyond |l|=1, it will be truncated at unity.

        There is a problem in converting the Euclidean-coordinates of the EoR simulation, if given, to angular/frequency
        co-ordinates. As any slice of the "lightcone" is assumed to be at the same redshift, it should trace a curved
        surface through physical space to approximate an observable surface. This is however not the case, as the
        surfaces chosen as slices are flat. One then has a choice: either interpret each 2D cell as constant-comoving-size,
        or constant-solid-angle. The former is true in the original simulation, but is a progressively worse approximation
        as the simulation is tiled and converted to angular co-ordinates (i.e. at angles close to the horizon, many
        simulations fit in a small angle, which distorts the actual physical scales -- in reality, those angles should
        exhibit the same structure as zenith). The latter is slightly incorrect in that it compresses what should be
        3D structure onto a 2D plane, but it does not grow catastophically bad as simulations are tiled. That is, the
        sizes of the structures remain approximately correct right down to the horizon. Thus we choose to use this
        method of tiling.
        """
        super().__init__(*args, **kwargs)

        # Instrument properties
        self.antenna_posfile = antenna_posfile
        self.instrumental_frequencies = np.linspace(freq_min * 1e6, freq_max * 1e6, nfreq)
        self.tile_diameter = tile_diameter * un.m
        self._max_bl_length = max_bl_length
        self.integration_time = integration_time
        self.Tsys = Tsys
        self.effective_collecting_area = effective_collecting_area * un.m ** 2

        if self.effective_collecting_area > self.tile_diameter ** 2:
            logger.warn("The effective collecting area (%s) is greater than the tile diameter squared!")

        # Sky size parameters.
        self.sky_extent = sky_extent

        if self.sky_extent <= 0:
            raise ValueError("sky_extent must be positive")

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
            if not antenna_posfile_is_baselines:
                self.baselines = self.get_baselines(ant_pos[:, -2], ant_pos[:, -1]) * un.m
            else:
                self.baselines = ant_pos * un.m

            if self._max_bl_length:
                self.baselines = self.baselines[
                    self.baselines[:, 0].value ** 2 + self.baselines[:, 1].value ** 2 <= self._max_bl_length ** 2]

    @cached_property
    def lightcone_core(self):
        "Lightcone core module"
        if not hasattr(self, "LikelihoodComputationChain"):
            raise NotSetupError

        for m in self.LikelihoodComputationChain.getCoreModules():
            if isinstance(m, p21.mcmc.core.CoreLightConeModule):
                return m  # Doesn't really matter which one, we only want to access basic properties.

        # If nothing is returned, we don'e have a lightcone
        raise AttributeError("no lightcone modules were loaded")

    @cached_property
    def foreground_cores(self):
        """List of foreground core modules"""
        if not hasattr(self, "LikelihoodComputationChain"):
            raise NotSetupError

        return [m for m in self.LikelihoodComputationChain.getCoreModules() if isinstance(m, ForegroundsBase)]

    @property
    def sky_size_required(self):
        "The sky-size in (l,m) co-ordinates required for the EoR lightcone"
        eor_frequencies = 1420e6 / (1 + self.lightcone_core.lightcone_slice_redshifts)
        size = self.sky_extent * np.max(self.sigma(eor_frequencies))

        size = min((1, size))

        if size / self.eor_size_lm.min() > self._max_tile_n:
            raise ValueError("sky size is larger than max requested (must be tiled %s times, but max is %s)" % (
                size / self.eor_size_lm, self._max_tile_n))

        return size

    def mpc_to_rad(self, redshift, cosmo):
        """
        Conversion factor between Mpc and radians for small angles.

        Parameters
        ----------
        cosmo : :class:`~astropy.cosmology.FLRW` instance
            A cosmology.

        Returns
        -------
        factor : float
            Conversion factor which gives the size in radians of a 1 Mpc object at given redshift.
        """
        return cosmo.comoving_transverse_distance(redshift).value

    @property
    def eor_size(self):
        """
        The minimum size of the base 21cmFAST simulation in lm co-ordinates.
        """
        return np.array(
            [self.lightcone_core.user_params.BOX_LEN * self.mpc_to_rad(z, self.lightcone_core.cosmo_params.cosmo)
             for z in self.lightcone_core.lightcone_slice_redshifts])

    def _simulate_data(self, ctx):

        """
        Generate a set of realistic visibilities (i.e. the output we expect from an interferometer) and add it to the
        context. Also, add the linear frequencies of the observation to the context.
        """
        # Get the basic signal lightcone out of context
        lightcone = ctx.get("lightcone")

        # Get visibilities from lightcone
        visibilities = 0
        if lightcone is not None:
            logger.info("Sampling EoR simulation onto baselines...")
            t1 = time.time()
            visibilities += self.get_eor_visibilities(lightcone.brightness_temp)
            logger.info(f"... finished in {time.time() - t1} sec")

        # Get any foreground lightcones
        foregrounds = ctx.get("foregrounds", [])

        for lc, cls in zip(foregrounds, self.foreground_cores):
            logger.info(f"Sampling {cls.__class__.__name__} foregrounds onto baselines...")
            print(len(self.baselines))

            t1 = time.time()
            visibilities += self.get_foreground_visibilities(lc, cls)
            logger.info(f"... finished in {time.time() - t1} sec.")

        return visibilities

    def simulate_data(self, ctx):
        visibilities = self._simulate_data(ctx)

        # When simulating, we add thermal noise!
        visibilities = self.add_thermal_noise(visibilities)

        ctx.add("visibilities", visibilities)
        ctx.add("baselines", self.baselines.value)
        ctx.add("frequencies", self.instrumental_frequencies)

    def __call__(self, ctx):
        visibilities = self._simulate_data(ctx)

        # Don't add thermal noise on every iteration... we can just use the expected value and variance.

        ctx.add("visibilities", visibilities)
        ctx.add("baselines", self.baselines.value)
        ctx.add("frequencies", self.instrumental_frequencies)

    def get_eor_visibilities(self, lightcone):
        frequencies = 1420.0 / (1 + self.lightcone_core.lightcone_slice_redshifts)

        vis = cw.get_tiled_visibilities(
            lightcone, frequencies, self.baselines, self.eor_size/self.lightcone_core.user_params.HII_DIM,
            self.sky_size_required, self.tile_diameter
        )

        vis = self.interpolate_frequencies(vis, frequencies, self.instrumental_frequencies)

        return vis

    def get_foreground_visibilities(self, lightcone, cls):
        # Find beam attenuation
        angles = cls.angles

        l = np.sin(angles[0])  # get |l| co-ordinate

        mask = l <= 1
        l = l[mask]

        attenuation = self.beam(l, cls.frequencies)

        # Attenuate the sky, and convert from Jy/sr to Jy (because we are doing the direct visibility calc)
        lightcone = lightcone[mask] * attenuation * cls.cell_area

        # Now get both l and m:
        m = l * np.cos(np.pi - angles[1][mask])
        l *= np.cos(angles[1][mask])

        vis = cw.get_direct_visibilities(cls.frequencies, self.baselines, lightcone, l, m)

        vis = self.interpolate_frequencies(vis, cls.frequencies, self.instrumental_frequencies)

        return vis

    def sigma(self, frequencies):
        "The Gaussian beam width at each frequency"
        epsilon = 0.42  # scaling from airy disk to Gaussian
        return ((epsilon * const.c) / (frequencies / un.s * self.tile_diameter)).to(un.dimensionless_unscaled)

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
        return np.pi * sig ** 2

    def beam(self, l, frequencies):
        """
        Generate a frequency-dependent Gaussian beam attenuation across the sky per frequency.

        Parameters
        ----------
        l : array
            The sky-coordinates
        frequencies : array
            Frequencies at which to evaluate the beam width.

        Returns
        -------
        attenuation : (len(l), len(frequencies))-array
            The beam attenuation (maximum unity) over the sky.
        """
        return np.exp(np.outer(-(l ** 2), 1. / (2 * self.sigma(frequencies) ** 2)))

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

    #
    # @staticmethod
    # def image_to_uv(sky, L):
    #     """
    #     Transform a box from image plan to UV plane.
    #
    #     Parameters
    #     ----------
    #     sky : (ncells, ncells, nfreq)-array
    #         The frequency-dependent sky brightness (in arbitrary units)
    #
    #     L : float
    #         The size of the box in radians.
    #
    #     Returns
    #     -------
    #     uvsky : (ncells, ncells, nfreq)-array
    #         The UV-plane representation of the sky. Units are units of the sky times radians.
    #
    #     uv_scale : list of two arrays.
    #         The u and v co-ordinates of the uvsky, respectively. Units are inverse of L.
    #     """
    #     logger.info("Converting to UV space...")
    #     t1 = time.time()
    #     ft, uv_scale = fft(sky, [L, L], axes=(0, 1), a = 0, b=2*np.pi)
    #     logger.info("... took %s sec." % (time.time() - t1))
    #     return ft, uv_scale
    #
    # @staticmethod
    # def sample_onto_baselines(uvplane, uv, baselines, frequencies):
    #     """
    #     Sample a gridded UV sky onto a set of baselines.
    #
    #     Sampling is done via linear interpolation over the regular grid.
    #
    #     Parameters
    #     ----------
    #     uvplane : (ncells, ncells, nfreq)-array
    #         The gridded UV sky, in Jy.
    #
    #     uv : list of two 1D arrays
    #         The u and v coordinates of the uvplane respectively.
    #
    #     baselines : (N,2)-array
    #         Each row should be the (x,y) co-ordinates of a baseline, in metres.
    #
    #     frequencies : 1D array
    #         The frequencies of the uvplane.
    #
    #     Returns
    #     -------
    #     vis : complex (N, nfreq)-array
    #          The visibilities defined at each baseline.
    #
    #     """
    #     vis = np.zeros((len(baselines), len(frequencies)), dtype=np.complex128)
    #
    #     frequencies = frequencies / un.s
    #
    #     logger.info("Sampling the data onto baselines...")
    #     t1 = time.time()
    #
    #     for i, ff in enumerate(frequencies):
    #         lamb = const.c / ff.to(1 / un.s)
    #         arr = np.zeros(np.shape(baselines))
    #         arr[:, 0] = (baselines[:, 0] / lamb).value
    #         arr[:, 1] = (baselines[:, 1] / lamb).value
    #
    #         real = np.real(uvplane[:, :, i])
    #         imag = np.imag(uvplane[:, :, i])
    #
    #         f_real = RegularGridInterpolator([uv[0], uv[1]], real, bounds_error=False, fill_value=0)
    #         f_imag = RegularGridInterpolator([uv[0], uv[1]], imag, bounds_error=False, fill_value=0)
    #
    #         FT_real = f_real(arr)
    #         FT_imag = f_imag(arr)
    #
    #         vis[:, i] = FT_real + FT_imag * 1j
    #
    #     logger.info("... took %s sec." % (time.time() - t1))
    #     return vis

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

        logger.info("... took %s sec." % (time.time() - t1))
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
        sigma = 2 * 1e26 * const.k_B.value * self.Tsys / self.effective_collecting_area / np.sqrt(
            df * self.integration_time)
        return (sigma ** 2).value

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
        rl_im = np.random.normal(0, 1, (2,) + visibilities.shape)

        return visibilities + np.sqrt(self.thermal_variance_baseline) * (rl_im[0, :] + rl_im[1, :] * 1j)

    # @property
    # def _n_stitched(self):
    #     """The number of cells in the stitched box (pre-coarsening)"""
    #     return int(self.sky_size * self._base_module.user_params.HII_DIM / self.sim_sky_size)
    #
    # def stitch_and_coarsen(self, lightcone):
    #     logger.info("Stitching and coarsening boxes...")
    #     t1 = time.time()
    #
    #     new = cw.stitch_and_coarsen_sky(lightcone, self.sim_sky_size, self.sky_size, self.n_cells)
    #
    #     logger.info("... took %s sec."%(time.time() - t1))
    #     return new
