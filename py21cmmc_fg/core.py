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
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
from powerbox.dft import fft
from powerbox import LogNormalPowerBox
from os import path
import py21cmmc as p21
from astropy.cosmology import Planck15

from py21cmmc import LightCone, UserParams, CosmoParams
from py21cmmc.mcmc.core import CoreBase

class ForegroundsBase(CoreBase):
    """
    A base class which implements some higher-level functionality that all foreground core modules will require.
    """
    # The defaults dict specifies arguments that need to be set if no CoreLightConeModule is loaded.
    defaults = dict(
        box_len = 150.0,
        sky_cells = 100,
        cosmo = Planck15,
    )

    def __init__(self, redshifts=None, sky_size = None, sky_cells = None, store=None, **kwargs):
        super().__init__(store)

        self.model_params = kwargs
        self.redshifts = redshifts

        self.user_params = UserParams(HII_DIM=self.defaults['sky_cells'], BOX_LEN=self.defaults['box_len'])
        self.cosmo_params = CosmoParams(hlittle=self.defaults['cosmo'].h, OMm=self.defaults['cosmo'].Om0,
                                        OMb=self.defaults['cosmo'].Ob0)
        
    def setup(self):
        for m in self.LikelihoodComputationChain.getCoreModules():
            if isinstance(m, p21.mcmc.core.CoreLightConeModule):
                self.lightcone_module = m

        if hasattr(self, "lightcone_module"):
            self.user_params = self.lightcone_module.user_params
            self.cosmo_params = self.lightcone_module.cosmo_params

            # This is not completely ideal, but because in this
            # case we depend on the underlying lightcone module,
            # we need to overwrite the redshifts on setup. Therefore we *run* the underlying lightcone just to
            # get its redshifts!
#            self.redshifts = self.default_ctx.get("lightcone").lightcone_redshifts

        elif self.redshifts is None:
            raise ValueError("If no lightcone core is supplied, redshifts must be supplied.")

    def __call__(self, ctx):
        """
        Apply foregrounds to an existing EoR lightcone.

        The net result of this function is to replace the EoR lightcone in the "output" variable of the context with
        a foreground-contaminated version with the same shape (but in Jy/sr rather than mK). It also adds the
        variables "frequencies" and "sky_size".
        """
        lightcone = ctx.get('lightcone', None)
        
        if lightcone is None: # Only do this is if there is no lightcone
            fg_lightcone = self.build_sky(self.frequencies, **self.model_params)
            self._add_to_ctx(ctx, fg_lightcone)
        else:
            pass

    def _add_to_ctx(self, ctx, new):

        if not ctx.contains("lightcone"):
            ctx.add("lightcone",self.mock_lightcone(self.frequencies, fg=new))
            ctx.add("redshifts", 1420e6 / self.frequencies -1 )

        else:
            # Add new bit to old lightcone
            ctx.get("lightcone").brightness_temp += new

    def mock_lightcone(self, frequencies, fg=None):
        """
        Create and return a mock LightCone object with foregrounds only.
        :return:
        """
        if fg is None:
            fg = self.build_sky(frequencies, **self.model_params)
  
        return LightCone( # Create a mock lightcone that can be read by future likelihoods as if it were the real deal.
                    redshift = self.redshifts.min(),
                    user_params = self.user_params,
                    cosmo_params = self.cosmo_params,
                    astro_params = None, flag_options = None, brightness_temp = fg
                )

    @staticmethod
    def get_sky_size(boxsize, redshifts, cosmo):
        return 2 * np.arctan(boxsize / (2 * float(cosmo.comoving_transverse_distance([np.mean(redshifts)]).value)))

    @property
    def sky_size(self):
        return self.get_sky_size(self.user_params.BOX_LEN, self.redshifts, self.cosmo_params.cosmo)

    @property
    def frequencies(self):
        "The frequencies associated with slice redshifts, in Hz"
        return 1420e6 / (self.redshifts + 1)


class CorePointSourceForegrounds(ForegroundsBase):
    """
    A 21CMMC Core MCMC module which adds point-source foregrounds to the base signal.
    """

    def __init__(self, *args, S_min=1e-1, S_max=1.0, alpha=4100., beta=1.59, gamma=0.8, f0=150e6, **kwargs):
        super().__init__(*args, S_min=S_min, S_max=S_max, alpha=alpha, beta=beta, gamma=gamma, f0=f0, **kwargs)

    @staticmethod
    def build_sky(frequencies, S_min=1e-1, S_max=1.0, alpha=4100., beta=1.59, gamma=0.8, f0=150e6, sky_size = 1.0, sky_cells = 600):
        """
        Create a grid of flux densities corresponding to a sample of point-sources drawn from a power-law source count
        model.

        Parameters
        ----------
        frequencies : array
            The frequencies at which to determine the diffuse emission.
        sky_cells : int
            Number of cells on a side for the 2 sky dimensions.
        sky_size : float
            Size (in radians) of the angular dimensions of the box.
        S_min : float
            The minimum flux (in Jy) of the foregrounds.
        S_max : float
            The maximum flux (in Jy) of the foregrounds.
        alpha : float
            The coefficient from Intema et al (2011). See notes below.
        beta : float
            The coefficient from Intema et al (2011). See notes below.

        Returns
        -------
        sky : array
            The sky filled with foregrounds (in mK).

        Notes
        -----
        The box is populated with point-sources foreground given by

        .. math::\frac{dN}{dS} (\nu)= \alpha \left(\frac{S_{\nu}}{S_0}\right)^{-\beta} {\rm Jy}^{-1} {\rm sr}^{-1}.

        """
        # Create a function for source count distribution
        source_count = lambda x: alpha * x ** (-beta)

        # Find the mean number of sources
        n_bar = quad(source_count, S_min, S_max)[0] * (sky_size)**2 # Need to multiply by sky size in steradian!

        # Generate the number of sources following poisson distribution
        # Make sure it's not 0!
        N_sources = 0
        while (N_sources==0):
            N_sources = np.random.poisson(n_bar)
        
        # Generate the point sources in unit of Jy and position using uniform distribution
        S_0 = ((S_max ** (1 - beta) - S_min ** (1 - beta)) * np.random.uniform(size=N_sources) + S_min ** (
                1 - beta)) ** (1 / (1 - beta))
        pos = np.rint(np.random.uniform(0, sky_cells - 1, size=(N_sources, 2))).astype(int)

        # Grid the fluxes at nu = 150
        S_0 = np.histogram2d(pos[:, 0], pos[:, 1], bins=np.arange(0, sky_cells + 1, 1), weights=S_0)
        S_0 = S_0[0]

        # Find the fluxes at different frequencies based on spectral index and divide by cell area
        sky = np.outer(S_0, (frequencies / f0) ** (-gamma)).reshape((np.shape(S_0)[0], np.shape(S_0)[0], len(frequencies)))
        sky /= (sky_size / sky_cells) ** 2

        # Change the unit from Jy/sr to mK
        sky /= CoreInstrumental.conversion_factor_K_to_Jy(frequencies)

        return sky


class CoreDiffuseForegrounds(ForegroundsBase):
    """
    A 21CMMC Core MCMC module which adds diffuse foregrounds to the base signal.
    """

    def __init__(self, u0=10.0, eta = 0.01, rho = -2.7, mean_temp=253e3, kappa=-2.55, *args,**kwargs):
        super().__init__(*args, u0=u0, eta = eta, rho = rho, mean_temp=mean_temp, kappa=kappa, **kwargs)

    @staticmethod
    def build_sky(frequencies, ncells, sky_size, u0=10.0, eta = 0.01, rho = -2.7, mean_temp=253e3, kappa=-2.55):
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

        # Calculate mean flux density per frequency. Remember to take the square root because the Power is squared.
        Tbar = np.sqrt((frequencies/1e8)**kappa * mean_temp**2)
        power_spectrum = lambda  u : eta**2 * (u/u0) ** rho

        # Create a log normal distribution of fluctuations
        pb = LogNormalPowerBox(N=ncells, pk=power_spectrum, dim=2, boxlength=sky_size[0], a=0, b=2 * np.pi, seed=1234)

        density = pb.delta_x() + 1

        # Multiply the inherent fluctuations by the mean flux density.
        if np.std(density)>0:
            Tbins = np.outer(density, Tbar).reshape((ncells, ncells, len(frequencies)))
        else:
            Tbins = np.zeros((ncells, ncells, len(frequencies)))
            for i in range(len(frequencies)):
                Tbins[:, :, i] = Tbar[i]

        return Tbins


class CoreInstrumental(CoreBase):
    """
    Core MCMC class which converts 21cmFAST *lightcone* output into a mock observation, sampled at specific baselines.

    Assumes that either a :class:`ForegroundBase` instance, or :class:`py21cmmc.mcmc.core.CoreLightConeModule` is also
    being used (and loaded before this).
    """
    def __init__(self, antenna_posfile, freq_min, freq_max, nfreq, tile_diameter=4.0, max_bl_length=300.0,
                 integration_time=1200, Tsys = 0, sky_size = 1.0, sky_cells = 600, store=None):
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

        nfreq : int
            Number of frequencies in the observation.

        tile_diameter : float, optional
            The physical diameter of the tiles, in metres.

        max_bl_length : float, optional
            The maximum length (in metres) of the baselines to be included in the analysis.

        integration_time : float,optional
            The length of the observation, in seconds.
            
        sky_size : float, optional
            The sky size in radian
        
        sky_cells: int, optional
            The number of pixels/grids in the sky
            
        """
        self.antenna_posfile = antenna_posfile
        self.instrumental_frequencies = np.linspace(freq_min*1e6, freq_max*1e6, nfreq)
        self.tile_diameter = tile_diameter * un.m
        self.max_bl_length = max_bl_length
        self.integration_time = integration_time
        self.sky_size = sky_size
        self.sky_cells = sky_cells
        self.Tsys = Tsys
        self.store = store or {}

    def setup(self):
        """
        Basically just read in the baselines from file that user gives.
        """

        # Setup dependencies
        for m in self.LikelihoodComputationChain.getCoreModules():
            if isinstance(m, ForegroundsBase) or isinstance(m, p21.mcmc.core.CoreLightConeModule):
                self._base_module = m # Doesn't really matter which one, we only want to access basic properties.

        if not hasattr(self, "_base_module"):
            raise ValueError("For CoreInstrumental, either a ForegroundBase or CoreLightConeModule must be loaded before it.")

        if self.antenna_posfile == "grid_centres":
            self.baselines = None

        else:
            # If antenna_posfile is a simple string, we'll try to find it in the data directory.
            data_path = path.join(path.dirname(__file__), 'data', self.antenna_posfile+'.txt')

            if path.exists(data_path):
                ant_pos = np.genfromtxt(data_path, float)
            else:
                ant_pos = np.genfromtxt(self.antenna_posfile, float)

            # Find all the possible combination of tile displacement
            # baselines is a dim2 array of x and y displacements.
            self.baselines = self.get_baselines(ant_pos[:, 1], ant_pos[:, 2]) * un.m

#            self.baselines = self.baselines[
#                self.baselines[:, 0].value ** 2 + self.baselines[:, 1].value ** 2 <= self.max_bl_length ** 2]
#            self.baselines = self.baselines[self.baselines[:, 0].value <= self.max_bl_length ]
#            self.baselines = self.baselines[self.baselines[:, 1].value <= self.max_bl_length ]

    @property
    def frequencies(self):
        "The frequencies associated with slice redshifts, in Hz"
        return 1420e6 / (self.redshifts + 1)

    def __call__(self, ctx):
        """
        Generate a set of realistic visibilities (i.e. the output we expect from an interferometer) and add it to the
        context. Also, add the linear frequencies of the observation to the context.
        """

        lightcone = ctx.get("lightcone")
        self.redshifts = ctx.get("redshifts")
        
        # Get the redshifts from lightcone if we don't already have them
        if self.redshifts is None:# if not hasattr(self, "redshifts"):
            self.redshifts = lightcone.lightcone_redshifts

            boxsize = self._base_module.user_params.BOX_LEN
            cosmo = self._base_module.cosmo_params.cosmo
            EoR_size = ForegroundsBase.get_sky_size(boxsize, self.redshifts, cosmo)
            
            # TODO: stitch stuff together and then coarsen the grid
            new_sky = self.stitch_boxes(lightcone.brightness_temp, EoR_size)
            new_sky = self.coarsen_sky(new_sky, n2=self.sky_cells)[0]
        else:
            new_sky = lightcone.brightness_temp            
       
        vis = self.add_instrument(new_sky, self.frequencies)
        
        ctx.add("visibilities", vis)
        ctx.add("baselines", self.baselines.value)
        ctx.add("frequencies", self.instrumental_frequencies)

    def add_instrument(self, lightcone, frequencies):
        # Number of 2D cells in sky array
        sky_cells = np.shape(lightcone)[0]
        
        # Find beam attenuation
        attenuation, beam_area = self.beam(frequencies, sky_cells, self.sky_size, self.tile_diameter)
        
        # Change the units of lightcone from mK to Jy
        beam_sky = lightcone * self.conversion_factor_K_to_Jy(frequencies, beam_area) * attenuation
                                                          
        # Fourier transform image plane to UV plane.
        uvplane, uv = self.image_to_uv(beam_sky, self.sky_size)
     
        # This is probably bad, but set baselines if none are given, to coincide exactly with the uv grid.
        if self.baselines is None:
            self.baselines = np.zeros((len(uv[0])**2, 2))
            U,V = np.meshgrid(uv[0], uv[1])
            self.baselines[:, 0] = U.flatten()*(const.c/frequencies.max())
            self.baselines[:, 1] = V.flatten()*(const.c/frequencies.max())

        # Fourier Transform over the (u,v) dimension and baselines sampling
        visibilities = self.sample_onto_baselines(uvplane, uv, self.baselines, frequencies)

        visibilities = self.interpolate_frequencies(visibilities, frequencies, self.instrumental_frequencies)

        # Just in case we forget, now the frequencies are all in terms of the instrumental frequencies.
        frequencies = self.instrumental_frequencies
        beam_area = self.beam(frequencies, sky_cells, self.sky_size, self.tile_diameter)
        
        # Add thermal noise using the mean beam area
        visibilities = self.add_thermal_noise(visibilities, frequencies, beam_area[1], self.integration_time,
                                              Tsys=self.Tsys)

        return visibilities

    @staticmethod
    def beam(frequencies, ncells, sky_size, D):
        """
        Generate a frequency-dependent Gaussian beam attenuation across the sky per frequency.

        Parameters
        ----------
        frequencies : array
            A set of frequencies (in Hz) at which to evaluate the beam.

        ncells : int
            Number of cells in the sky grid.

        sky_size : float
            The extent of the sky in radians.
        
        D : float
            The tile diameter (in m).

        Returns
        -------
        attenuation : (ncells, ncells, nfrequencies)-array
            The beam attenuation (maximum unity) over the sky.
        
        beam_area : (nfrequencies)-array
            The beam area of the sky (in sr).
        """
        # First find the sigma of the beam
        epsilon = 0.42

        frequencies = frequencies / un.s
        sigma = ((epsilon * const.c) / (frequencies * D)).to(un.dimensionless_unscaled)

        # Then create a meshgrid for the beam attenuation on sky array
        sky_coords_lm = np.sin(np.linspace(-sky_size / 2, sky_size / 2, ncells))
        L, M = np.meshgrid(sky_coords_lm, sky_coords_lm)
        
        attenuation = np.exp(np.outer(-(L ** 2 + M ** 2), 1. / sigma ** 2).reshape((ncells, ncells, len(frequencies))))
        beam_area = np.sum(attenuation, axis=(0,1)) * np.diff(sky_coords_lm )[0]**2

        return attenuation, beam_area

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
        ft, uv_scale = fft(sky, [L, L], axes=(0, 1))
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

        print("Sampling the data onto baselines")

        for i, ff in enumerate(frequencies):
            lamb = const.c / ff.to(1 / un.s)
            arr = np.zeros(np.shape(baselines))
            arr[:, 0] = (baselines[:,0] / lamb).value
            arr[:, 1] = (baselines[:,1] / lamb).value

            real = np.real(uvplane[:,:,i])
            imag = np.imag(uvplane[:,:,i])

            f_real = RegularGridInterpolator([uv[0], uv[1]], real, bounds_error=False, fill_value=0)
            f_imag = RegularGridInterpolator([uv[0], uv[1]], imag, bounds_error=False, fill_value=0)
            
            FT_real = f_real(arr)
            FT_imag = f_imag(arr)

            vis[:, i] = FT_real + FT_imag * 1j

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
        new_vis = np.zeros((visibilities.shape[0], len(linear_freq)), dtype=np.complex64)
        
        if(freq_grid[0]>freq_grid[-1]):
            freq_grid = freq_grid[::-1]
            visibilities = np.flip(visibilities,1)
            
        for i, vis in enumerate(visibilities):
            rl = RegularGridInterpolator((freq_grid,), np.real(vis))(linear_freq)
            im = RegularGridInterpolator((freq_grid,), np.imag(vis))(linear_freq)
            new_vis[i] = rl + im * 1j

        return new_vis

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
        zeros = np.logical_and(Xsep.flatten()==0, Ysep.flatten()==0)

        return np.array([Xsep.flatten()[np.logical_not(zeros)], Ysep.flatten()[np.logical_not(zeros)]]).T

    @staticmethod
    def add_thermal_noise(visibilities, frequencies, beam_area, delta_t = 1200, Tsys=0):
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
        sigma = Tsys * beam_area / (((const.c) / ((frequencies.max()-frequencies.min()) * (1 / un.s))) ** 2).value / np.sqrt((frequencies.max()-frequencies.min())*delta_t)

        rl_im = np.random.normal(0, 1, (2, np.shape(visibilities)[0],np.shape(visibilities)[1]))
        
        # TODO: check the units of sigma 
        visibilities += sigma * (rl_im[0, :] + rl_im[1, :] * 1j)
        
        return visibilities
    
    def stitch_boxes(self, lightcone, EoR_size):
        
        N_box = int((self.sky_size/EoR_size))
        
        new_sky = []
        
        for ff in range(np.shape(lightcone)[-1]):
            
            sky_temp = lightcone[:,:,ff].copy()
                
            for i in range(1,N_box):
                sky_temp = np.vstack((sky_temp, lightcone[:,:,ff]))
            
            sky = sky_temp.copy()
            
            for j in range(1,N_box):
                sky = np.hstack((sky, sky_temp))
            
            new_sky.append(sky)
        
        new_sky = np.array(new_sky).transpose(1,2,0)
        
        return new_sky
    
    def coarsen_sky(self, a, centres=None, n2=125):
        """
        Coarsen the square matrix a down to n2xn2.
        """
        n = np.shape(a)[0]
        if centres is None:
            dx = 1.
            centres = np.arange(n)
           
        else:
            dx = centres[1] - centres[0]
           
        db = float(dx)/(float(n2)/n)
       
        bvec = np.linspace(centres.min()- dx/2 + db/2, centres.max() + dx/2 - db/2, n2)
        
        new = []
        for ff in range(np.shape(a)[-1]):
            spl = RectBivariateSpline(centres, centres, a[:,:,ff], kx=1, ky=1)
           
            b = spl(bvec, bvec, grid=True)
            
            new.append(b)
        
        new = np.array(new).transpose(1,2,0)
        
        return new, bvec