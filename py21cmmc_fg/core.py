# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 14:13:39 2018

@author: bella

Foreground core for 21cmmc

"""
import logging
from os import path

import numpy as np
import py21cmmc as p21
from astropy import constants as const
from astropy import units as un
from powerbox import LogNormalPowerBox
from powerbox.dft import fft, fftfreq
from py21cmmc.mcmc.core import CoreBase, NotSetupError
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger("21CMMC")

import warnings

import time
from . import c_wrapper as cw
from cached_property import cached_property

# A little trick to create a profiling decorator if *not* running with kernprof
try:
    profile
except NameError:
    def profile(fnc):
        return fnc


class ForegroundsBase(CoreBase):
    """
    A base class which implements some higher-level functionality that all foreground core modules will require.

    All subclasses of :class:`CoreBase` receive the argument `store`, whose documentation is in the `21CMMC` docs.
    """

    def __init__(self, *, frequencies=None, sky_size=None, n_cells=None, model_params={}, **kwargs):
        """
        Initialise the ForegroundsBase object.

        Parameters
        ----------
        frequencies : array, optional
            The frequencies (in MHz) at which the foregrounds are defined. These need not be provided if an Instrumental
            Core is provided, as they will then be taken from that (unless they are explicitly set here).
        sky_size : float, optional
            The size of the sky, in (l,m) units. The sky will be assumed to stretch from -sky_size/2 to sky_size/2.
            Since the populated sky is square, this parameter can be no more than 2/sqrt(2). If not given, the class
            will attempt to set it from any loaded CoreInstrumental class, using its `sky_extent_required` parameter.
            If no CoreInstrumental class is loaded, this parameter must be set.
        n_cells : int, optional
            The number of regular (l,m) cells on a side of the sky grid. Like `sky_size` and `frequencies`, if unset
            and a `CoreInstrumental` class is loaded, will take its value from there.
        model_params : dict
            A dictionary of model parameters for a specific foreground type.

        Other Parameters
        ----------------
        All other parameters are passed to the :class:`py21cmmc.mcmc.CoreBase` class. These include `store`, which is a
        dictionary of options for storing data in the MCMC chain.
        """
        super().__init__(**kwargs)

        self.model_params = model_params
        self._sky_size = sky_size
        self._n_cells = n_cells

        # These save the default values. These will be *overwritten* in setup() if a LightCone is loaded
        self._frequencies = frequencies

    @cached_property
    def _instrumental_core(self):
        for m in self._cores:
            if isinstance(m, CoreInstrumental):
                return m

        raise AttributeError("No Instrumental Core is loaded")

    @cached_property
    def _updating(self):
        """Whether any MCMC parameters belong to the model parameters of this class"""
        return any([p in self.model_params for p in self.parameter_names])

    @property
    def frequencies(self):
        if self._frequencies is None:
            if hasattr(self, "_instrumental_core"):
                self._frequencies = self._instrumental_core.instrumental_frequencies / 1e6
            else:
                raise ValueError("As no instrumental core is loaded, frequencies need to be specified!")

        return self._frequencies * 1e6

    @property
    def sky_size(self):
        """
        The sky size, in (l,m) units.

        The sky will be assumed to stretch from -sky_size/2 to sky_size/2.
        """
        if self._sky_size is None:
            if hasattr(self, "_instrumental_core"):
                self._sky_size = self._instrumental_core.sky_size

            else:
                raise ValueError("As no instrumental core is loaded, sky_size needs to be specified.")

        return self._sky_size

    @property
    def n_cells(self):
        if self._n_cells is None:
            if hasattr(self, "_instrumental_core"):
                self._n_cells = self._instrumental_core.n_cells
            else:
                raise ValueError("As no instrumental core is loaded, n_cells needs to be specified.")
        return self._n_cells

    @property
    def cell_size(self):
        """Size, in (l,m), of a cell in one-dimension"""
        return self.sky_size / self.n_cells

    @property
    def cell_area(self):
        """(l,m) area of each sky cell"""
        return self.cell_size ** 2

    @property
    def sky_coords(self):
        """
        Co-ordinates of the left-edge of sky cells along a side.
        """
        return np.linspace(-self.sky_size / 2, self.sky_size / 2, self.n_cells)

    def convert_model_to_mock(self, ctx):
        """
        Simulate "observable" data (i.e. with noise included).

        Parameters
        ----------
        ctx : :class:`~CosmoHammer.ChainContext` object
            Much like a dictionary, but annoyingly different. Can contain the current parameters when passed in,
            and should be updated to contain the simulated data within this function.
        """
        # build_model_data is called before this method, so we do not need to
        # update parameters. We just build the sky:
        fg_lightcone = self.build_sky(**self.model_params)

        # Get the foregrounds list out of the context, defaulting to empty list.
        fg = ctx.get("foregrounds", [])
        fg.append(fg_lightcone)
        if len(fg) == 1:
            ctx.add("foregrounds", fg)

    def build_model_data(self, ctx):
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

    @profile
    def build_sky(self, S_min=1e-1, S_max=1.0, alpha=4100., beta=1.59, gamma=0.8, f0=150e6):
        """
        Create a grid of flux densities corresponding to a sample of point-sources drawn from a power-law source count
        model.

        Notes
        -----
        The sources are populated uniformly on an (l,m) grid. *This is not physical*. In reality, sources are more
        likely to be uniform an angular space, not (l,m) space. There are two reasons we do this: first, it corresponds
        to a simple analytic derivation of the statistics of such a sky, and (ii) it doesn't make too much of a difference
        as long as the beam (if there is one) is reasonably small.
        """
        logger.info("Populating point sources... ")

        # Find the mean number of sources
        n_bar = quad(lambda x: alpha * x ** (-beta), S_min, S_max)[
                    0] * self.sky_size ** 2  # Need to multiply by sky size in steradian

        # Generate the number of sources following poisson distribution
        n_sources = np.random.poisson(n_bar)

        if not n_sources:
            warnings.warn("There are no point-sources in the sky!")

        # Generate the point sources in unit of Jy and position using uniform distribution
        S_0 = ((S_max ** (1 - beta) - S_min ** (1 - beta)) * np.random.uniform(size=n_sources) + S_min ** (
                1 - beta)) ** (1 / (1 - beta))

        pos = np.rint(np.random.uniform(0, self.n_cells ** 2 - 1, size=n_sources)).astype(int)

        # Grid the fluxes at reference frequency, f0
        sky = np.bincount(pos, weights=S_0, minlength=self.n_cells ** 2)

        # Find the fluxes at different frequencies based on spectral index
        sky = np.outer(sky, (self.frequencies / f0) ** (-gamma)).reshape(
            (self.n_cells, self.n_cells, len(self.frequencies)))

        # Divide by cell area to get in Jy/sr (proportional to K)
        sky /= self.cell_area

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
        raise NotImplementedError("This is not working atm.")

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

    def __init__(self, *, antenna_posfile, freq_min, freq_max, nfreq, tile_diameter=4.0, max_bl_length=None,
                 integration_time=120, Tsys=240, effective_collecting_area=21.0,
                 sky_extent=3, n_cells=300, add_beam=True, padding_size = 3,
                 **kwargs):
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
            System temperature, K.

        effective_collecting_area : float, optional
            The effective collecting area of a tile (equal to the geometric area times the efficiency).

        sky_extent : float, optional
            Impose that any simulation (either EoR or foreground) have this size. The size is in units of the beam
            width, and defines the one-sided extent (i.e. if the beam width is sigma=0.2, and `sky_extent` is 3, then the
            sky will be required to extend to |l|=0.6).  Simulations are always assumed to be centred at zenith, and the
            tiling ensures that this `sky_extent` is reached in every line through zenith. See notes on tiling below.

        n_cells: int, optional
            The number of pixels per side of the sky grid, after any potential tiling to achieve `sky_size`. Simulations
            will be coarsened to match this number, where applicable. This is useful for reducing memory usage. Default
            is the same number of cells as the underlying simulation/foreground. If set to zero, no coarsening will be
            performed.


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
        super().__init__(**kwargs)

        self.antenna_posfile = antenna_posfile
        self.instrumental_frequencies = np.linspace(freq_min * 1e6, freq_max * 1e6, nfreq)
        self.tile_diameter = tile_diameter * un.m
        self.max_bl_length = max_bl_length
        self.integration_time = integration_time
        self.Tsys = Tsys
        self.add_beam = add_beam
        self.padding_size = padding_size

        self.effective_collecting_area = effective_collecting_area * un.m ** 2

        if self.effective_collecting_area > self.tile_diameter ** 2:
            warnings.warn("The effective collecting area (%s) is greater than the tile diameter squared!")

        # Sky size parameters.
        self._sky_extent = sky_extent

        self.n_cells = n_cells

        # Setup baseline lengths.
        if self.antenna_posfile == "grid_centres":
            self._baselines = None

        else:
            # If antenna_posfile is a simple string, we'll try to find it in the data directory.
            data_path = path.join(path.dirname(__file__), 'data', self.antenna_posfile + '.txt')

            if path.exists(data_path):
                ant_pos = np.genfromtxt(data_path, float)
            else:
                ant_pos = np.genfromtxt(self.antenna_posfile, float)

            # Find all the possible combination of tile displacement
            # baselines is a dim2 array of x and y displacements.
            self._baselines = self.get_baselines(ant_pos[:, 1], ant_pos[:, 2]) * un.m

            if self.max_bl_length:
                self._baselines = self._baselines[
                    self._baselines[:, 0].value ** 2 + self._baselines[:, 1].value ** 2 <= self.max_bl_length ** 2]

    @cached_property
    def lightcone_core(self):
        "Lightcone core module"
        for m in self._cores:
            if isinstance(m, p21.mcmc.core.CoreLightConeModule):
                return m  # Doesn't really matter which one, we only want to access basic properties.

        # If nothing is returned, we don't have a lightcone
        raise AttributeError("no lightcone modules were loaded")

    @cached_property
    def foreground_cores(self):
        """List of foreground core modules"""
        return [m for m in self._cores if isinstance(m, ForegroundsBase)]

    @property
    def sky_size(self):
        "The sky size in lm co-ordinates. This is the size *all the way across*"
        return 2 #* self._sky_extent * np.max(self.sigma(self.instrumental_frequencies))

    @cached_property
    def sky_coords(self):
        """Grid-coordinates of the (stitched/coarsened) simulation in lm units"""
        return np.linspace(-self.sky_size / 2, self.sky_size / 2, self.n_cells)

    @cached_property
    def cell_size(self):
        """Size (in lm) of a cell of the stitched/coarsened simulation"""
        return self.sky_size / self.n_cells

    @cached_property
    def cell_area(self):
        return self.cell_size ** 2

    def rad_to_cmpc(self, redshift, cosmo):
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


    def prepare_sky_lightcone(self, box):
        """
        Transform the raw brightness temperature of a simulated lightcone into the box structure required in this class.
        """
        frequencies = 1420.0e6 / (1 + self.lightcone_core.lightcone_slice_redshifts)

        if not np.all(frequencies == self.instrumental_frequencies):
            box = cw.interpolate_map_frequencies(box, frequencies, self.instrumental_frequencies)

        box_size = self.lightcone_core.user_params.BOX_LEN / self.rad_to_cmpc(
            np.mean(self.lightcone_core.lightcone_slice_redshifts), self.lightcone_core.cosmo_params.cosmo)

        # If the original simulation does not match the sky grid defined here, stitch and coarsen it
        if box_size != self.sky_size or len(box) != self.n_cells:
            box = self.tile_and_coarsen(box, box_size)

        # Convert to Jy/sr
        box *= self.mK_to_Jy_per_sr(self.instrumental_frequencies)

        return box

    @profile
    def prepare_sky_foreground(self, box, cls):
        frequencies = cls.frequencies

        if not np.all(frequencies == self.instrumental_frequencies):
            logger.info(f"Interpolating frequencies for {cls.__class__.__name__}...")
            box = cw.interpolate_map_frequencies(box, frequencies, self.instrumental_frequencies)

        # If the foregrounds don't match this sky grid, stitch and coarsen them.
        if cls.sky_size != self.sky_size or cls.n_cells != self.n_cells:
            logger.info(f"Tiling sky for {cls.__class__.__name__}...")
            box = self.tile_and_coarsen(box, cls.sky_size)

        return box

    def convert_model_to_mock(self, ctx):
        """
        Generate a set of realistic visibilities (i.e. the output we expect from an interferometer) and add it to the
        context. Also, add the linear frequencies of the observation to the context.
        """

        vis = ctx.get("visibilities")

        # Add thermal noise using the mean beam area
        vis = self.add_thermal_noise(vis)
        ctx.add("visibilities", vis)

    def build_model_data(self, ctx):
        """
        Generate a set of realistic visibilities (i.e. the output we expect from an interferometer).
        """
        # Get the basic signal lightcone out of context
        lightcone = ctx.get("lightcone")

        # Compute visibilities from EoR simulation
        box = 0
        if lightcone is not None:
            box += self.prepare_sky_lightcone(lightcone.brightness_temp)

        # Now get foreground visibilities and add them in
        foregrounds = ctx.get("foregrounds", [])

        # Get the total brightness
        for fg, cls in zip(foregrounds, self.foreground_cores):
            box += self.prepare_sky_foreground(fg, cls)

        vis = self.add_instrument(box)

        ctx.add("visibilities", vis)
        ctx.add("new_sky", box)

        # TODO: These aren't strictly necessary
        ctx.add("baselines", self.baselines)
        ctx.add("frequencies", self.instrumental_frequencies)
        ctx.add("baselines_type", self.antenna_posfile)

    @staticmethod
    def pad_with(vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 0)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value
        return vector

    def padding_image(self, image_cube, sky_size, big_sky_size):
        """
        Generate a spatial padding in image cube.

        Parameters
        ----------
        image_cube : (ncells, ncells, nfreq)-array
            The frequency-dependent sky brightness (in arbitrary units)

        sky_size : float
            The size of the box in radians.

        big_sky_size : float
            The size of the padded box in radians.

        Returns
        -------
        sky : (ncells, ncells, nfreq)-array
            The sky padded with zeros along the l,m plane.
        """
        sky = []
        N_pad = int((big_sky_size - sky_size)/2 * np.shape(image_cube)[0])
        for jj in range(np.shape(image_cube)[-1]):
            sky.append(np.pad(image_cube[:,:,jj], N_pad, self.pad_with))

        sky = np.array(sky).T

        return sky

    @profile
    def add_instrument(self, lightcone):
        # Find beam attenuation
        if self.add_beam is True:
            attenuation = self.beam(self.instrumental_frequencies)
            lightcone *= attenuation
        
        if self.padding_size is not None:
            lightcone = self.padding_image(lightcone, self.sky_size, self.padding_size * self.sky_size)
            uvplane, uv = self.image_to_uv(lightcone, self.padding_size * self.sky_size)
        else:
            # Fourier transform image plane to UV plane.
            uvplane, uv = self.image_to_uv(lightcone, self.sky_size)
        
        # Fourier Transform over the (u,v) dimension and baselines sampling
        if self.antenna_posfile != "grid_centres":
            visibilities = self.sample_onto_baselines(uvplane, uv, self.baselines, self.instrumental_frequencies)
        else:
            visibilities = uvplane
            self.baselines = uv[1]

        return visibilities

    @cached_property
    def uv_grid(self):
        """The grid of uv along one side"""
        return fftfreq(N=self.n_cells, d=self.cell_size, b=2 * np.pi)

    @cached_property
    def baselines(self):
        """The baselines of the array, in m"""
        #        if self._baselines is None:
        #            baselines = np.zeros((len(self.uv_grid) ** 2, 2))
        #            U, V = np.meshgrid(self.uv_grid, self.uv_grid)
        #            baselines[:, 0] = U.flatten() * (const.c / np.min(self.instrumental_frequencies))
        #            baselines[:, 1] = V.flatten() * (const.c / np.min(self.instrumental_frequencies))
        #            baselines = baselines * un.m
        #
        #            # Remove auto-correlations
        #            #baselines = baselines[baselines[:, 0] ** 2 + baselines[:, 1] ** 2 > 1.0 * un.m ** 2]
        #            return baselines
        #        else:
        if self._baselines is not None:
            return self._baselines.value
        else:
            return self._baselines

    def sigma(self, frequencies):
        "The Gaussian beam width at each frequency"
        epsilon = 0.42  # scaling from airy disk to Gaussian
        return ((epsilon * const.c) / (frequencies / un.s * self.tile_diameter)).to(un.dimensionless_unscaled).value

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
        L, M = np.meshgrid(np.sin(self.sky_coords), np.sin(self.sky_coords), indexing='ij')

        attenuation = np.exp(
            np.outer(-(L ** 2 + M ** 2), 1. / (2 * self.sigma(frequencies) ** 2)).reshape(
                (self.n_cells, self.n_cells, len(frequencies))))

        return attenuation

    @staticmethod
    def mK_to_Jy_per_sr(nu):
        """
        Conversion factor to convert a pixel of mK to Jy/sr.

        Taken from http://w.astro.berkeley.edu/~wright/school_2012.pdf

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
        wvlngth = const.c / (nu / un.s)

        intensity = 2 * const.k_B * 1e-3 * un.K / wvlngth ** 2

        flux_density = 1e26 * intensity.to(un.W / (un.Hz * un.m ** 2))

        return flux_density.value

    @profile
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
        ft, uv_scale = fft(np.fft.ifftshift(sky, axes=(0, 1)), L, axes=(0, 1), a=0, b=2 * np.pi)
        return ft, uv_scale

    @profile
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

        frequencies = frequencies / un.s
        vis = np.zeros((len(baselines), len(frequencies)), dtype=np.complex128)
        
        logger.info("Sampling the data onto baselines...")

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

        return vis

    @profile
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

    @profile
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
        if self.thermal_variance_baseline:
            logger.info("Adding thermal noise...")
            rl_im = np.random.normal(0, 1, (2,) + visibilities.shape)

            # NOTE: we divide the variance by two here, because the variance of the absolute value of the
            #       visibility should be equal to thermal_variance_baseline, which is true if the variance of both
            #       the real and imaginary components are divided by two.
            return visibilities + np.sqrt(self.thermal_variance_baseline / 2) * (rl_im[0, :] + rl_im[1, :] * 1j)
        else:
            return visibilities

    @profile
    def tile_and_coarsen(self, sim, sim_size):
        """"""
        logger.info("Tiling and coarsening boxes...")
        sim = cw.stitch_and_coarsen_sky(sim, sim_size, self.sky_size, self.n_cells)

        return sim

    @staticmethod
    def interpolate_frequencies(data, freqs, linFreqs, uv_range=100):

        if (freqs[0] > freqs[-1]):
            freqs = freqs[::-1]
            data = np.flip(data, 2)

        ncells = np.shape(data)[0]
        # Create the xy data
        xy = np.linspace(-uv_range / 2., uv_range / 2., ncells)

        # generate the interpolation function
        func = RegularGridInterpolator([xy, xy, freqs], data, bounds_error=False, fill_value=0)

        # Create a meshgrid to interpolate the points
        XY, YX, LINFREQS = np.meshgrid(xy, xy, linFreqs, indexing='ij')

        # Flatten the arrays so the can be put into pts array
        XY = XY.flatten()
        YX = YX.flatten()
        LINFREQS = LINFREQS.flatten()

        # Create the points to interpolate
        numpts = XY.size
        pts = np.zeros([numpts, 3])
        pts[:, 0], pts[:, 1], pts[:, 2] = XY, YX, LINFREQS

        # Interpolate the points
        interpData = func(pts)

        # Reshape the data
        interpData = interpData.reshape(ncells, ncells, len(linFreqs))

        return interpData
