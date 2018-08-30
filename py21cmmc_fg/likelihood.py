#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 16:54:22 2018

@author: bella
"""
from builtins import super

from powerbox.dft import fft
from powerbox.tools import angular_average_nd, get_power
import numpy as np
from astropy import constants as const
from astropy.cosmology import FlatLambdaCDM
from astropy import units as un

from py21cmmc.mcmc.likelihood import LikelihoodBase
from py21cmmc.mcmc.core import CoreLightConeModule
from py21cmmc.mcmc.cosmoHammer import ChainContext
from cosmoHammer.util import Params
from .core import CoreInstrumental, CoreDiffuseForegrounds, CorePointSourceForegrounds, ForegroundsBase
from scipy.sparse import block_diag
from sksparse.cholmod import cholesky

from numpy.linalg import slogdet, solve
from scipy.sparse import issparse
import h5py

class Likelihood2D(LikelihoodBase):
    required_cores = [CoreLightConeModule]

    def __init__(self, datafile=None, n_psbins=None, model_uncertainty=0.15,
                 error_on_model=True, nrealisations=200):
        """
        Initialize the likelihood.

        Parameters
        ----------
        datafile : str, optional
            The file from which to read the data. Alternatively, the file to which to write the data (see class
            docstring for how this works).
        n_psbins : int, optional
            The number of bins for the spherically averaged power spectrum. By default automatically
            calculated from the number of cells.
        model_uncertainty : float, optional
            The amount of uncertainty in the modelling, per power spectral bin (as fraction of the amplitude).
        error_on_model : bool, optional
            Whether the `model_uncertainty` is applied to the model, or the data.
        """
        super().__init__(datafile)

        # TODO: 21cmSense noise!

        self.n_psbins = n_psbins
        self.error_on_model = error_on_model
        self.model_uncertainty = model_uncertainty
        self.nrealisations = nrealisations

    def setup(self):
        super().setup()

        self.kperp_data, self.kpar_data, self.p_data = self.data['kperp'], self.data['kpar'], self.data['p']
        self.p_data += self.numerical_covariance(1)[0] ## adding foregrounds to our signal

        # Here define the variance of the foreground model, once for all.
        # Note: this will *not* work if foreground parameters are changing!
#        self.foreground_mean, self.foreground_variance = self.numerical_mean_and_variance(self.nrealisations)
        self.foreground_mean, self.foreground_covariance = self.numerical_covariance(self.nrealisations)

        # NOTE: we should actually use analytic_variance *and* analytic mean model, rather than numerical!!!


    @staticmethod
    def compute_power(lightcone, n_psbins=None):
        """
        Compute the 2D power spectrum
        Parameters
        ----------
        lightcone: 
        """
        p, kperp, kpar = get_power(lightcone.brightness_temp, boxlength=lightcone.lightcone_dimensions, res_ndim=2, bin_ave=False,
                        bins=n_psbins, get_variance=False)

        return p[:,int(len(kpar[0])/2):], kperp, kpar[0][int(len(kpar[0])/2):]

    def computeLikelihood(self, ctx, storage, variance=False):
        "Compute the likelihood"
        data = self.simulate(ctx)
        # add the power to the written data
        storage.update(**data)
        
        if variance is True:
        # TODO: figure out how to use 0.15*P_sig here.
            lnl = - 0.5 * np.sum((data['p']+self.foreground_mean - self.p_data) ** 2 / (self.foreground_variance + (0.15*data['p'])**2))
        else:
            lnl = self.lognormpdf(data['p'], self.foreground_covariance, len(self.kpar_data) )
        print("LIKELIHOOD IS ", lnl )

        return lnl
    
    def numerical_covariance(self,nrealisations=200):
        """
        Calculate the covariance of the foregrounds BEFORE the MCMC
    
        Parameters
        ----------
        
        nrealisations: int, optional
            Number of realisations to find the covariance.
        
        Output
        ------
        
        mean:(nperp, npar)-array
            The mean 2D power spectrum of the foregrounds.
            
        cov: 
            The sparse block diagonal matrix of the covariance if nrealisation is not 1
            Else it is 0
        """
        p = []
        mean = 0
        for core in self.foreground_cores:
            for ii in range(nrealisations):
                power = self.compute_power(core.mock_lightcone(), self.n_psbins)[0]
                p.append(power)
            
            mean += np.mean(p, axis=0)
            
        if(nrealisations>1):
            cov = block_diag([np.cov(x) for x in np.array(p).transpose((2,1,0))], format='csc')
        else:
            cov = 0

        return mean, cov

    def lognormpdf(self, model , cov, blocklen=None):
        """
        Calculate gaussian probability log-density of x, when x ~ N(mu,sigma), and cov is sparse.
    
        Code adapted from https://stackoverflow.com/a/16654259
        """
        chol_deco = cholesky(cov)

        nx = len(model.flatten())
        norm_coeff = nx * np.log(2 * np.pi) + chol_deco.logdet()
        
        err = ((self.p_data) - (model+self.foreground_mean)).T.flatten()

        numerator = chol_deco.solve_A(err).T.dot(err)
        
        return -0.5*(norm_coeff+numerator)            

    def numerical_mean_and_variance(self, nrealisations=200):
        p = []
        var = 0
        mean = 0
        for core in self.foreground_cores:
            for i in range(nrealisations):
                power = self.compute_power(core.mock_lightcone(), self.n_psbins)[0]
                p.append(power)
            mean += np.mean(p, axis=0)
            var += np.var(p, axis=0)

        return mean, var
    
    def analytic_variance(self, k):
        var = 0
        for m in self.LikelihoodComputationChain.getCoreModules():
            if isinstance(m, CorePointSourceForegrounds):
                params = m.model_params
                # <get analytic cov...>
                # var += ...
            elif isinstance(m, CoreDiffuseForegrounds):
                params = m.model_params
                # < get analytic cov...>
                # var += ....

        return var
    
    def simulate(self, ctx):
        p, kperp, kpar = self.compute_power(ctx.get('lightcone'), self.n_psbins)
        
        return dict(p=p, kperp=kperp, kpar=kpar)

    @property
    def foreground_cores(self):
        try:
            return [m for m in self.LikelihoodComputationChain.getCoreModules() if isinstance(m, ForegroundsBase)]
        except AttributeError:
            raise AttributeError("foreground_cores is not available unless emedded in a LikelihoodComputationChain, after setup")


class LikelihoodForeground2D(LikelihoodBase):
    def __init__(self, datafile, n_uv=None, n_psbins=50, umax = None, frequency_taper=np.hanning, **kwargs):
        """
        A likelihood for EoR physical parameters, based on a Gaussian 2D power spectrum.

        In this likelihood, any foregrounds are naturally suppressed by their imposed covariance, in 2D spectral space.
        Nevertheless, it is not required that the :class:`~core.CoreForeground` class be amongst the Core modules for
        this likelihood module to work. Without the foregrounds, the 2D modes are naturally weighted by the sample
        variance of the EoR signal itself.

        The likelihood *does* require the :class:`~core.CoreInstrumental` Core module however, as this class first
        re-grids the input visibilities from baselines onto a grid.

        Parameters
        ----------
        datafile : str
            A filename referring to a file which contains the observed data (or mock data) to be fit to. The file
            should be a compressed numpy binary (i.e. a npz file), and must contain at least the arrays "kpar", "kperp"
            and "p", which are the parallel/perpendicular modes (in 1/Mpc) and power spectrum (in Mpc^3) respectively.

        n_uv : int, optional
            The number of UV cells to grid the visibilities (per side). By default, tries to look at the 21cmFAST
            simulation and use the same number of grid cells as that.

        n_psbins : int, optional
            The number of kperp bins to use.

        umax : float, optional
            The extent of the UV grid. By default, uses the longest baseline at the highest frequency.

        taper : callable, optional
            A function which computes a taper function on an nfreq-array. Default is to have no taper. Callable should
            take single argument, N.
        """

        super().__init__(**kwargs)
        self.datafile = datafile
        self.n_uv = n_uv
        self.n_psbins = n_psbins
        self.umax = umax
        self.frequency_taper = frequency_taper

    def setup(self):
        """
        Read in observed data.

        Data should be in an npz file, and contain a "k" and "p" array. k should be in 1/Mpc, and p in Mpc**3.
        """
        data = np.load(self.datafile +".npz")

        # TODO: it may be better to read in visbilities, and transform them here an now. In any case, we'll end up with
        # the following variables after the transformation anyway, so we can work with them.
        self.baselines = data["baselines"]
        self.visibilities = data['visibilities']
        self.frequencies = data["frequencies"]

    def computeLikelihood(self, ctx):

        # On the first run-through, we *create* the data from visibilities
        if not hasattr(self, "power2d_data"):
            # First, check if the frequencies and baselines match up.
            if not np.all(self.frequencies==ctx.get("frequencies")):
                raise ValueError("The frequencies of observation and model must match.")
            if not np.all(self.baselines==ctx.get("baselines")):
                raise ValueError("The baselines of observation and model must match.")

            self.power2d_data = self.computePower(
                dict(
                    baselines = self.baselines,
                    visibilities = self.visibilities,
                    frequencies = self.frequencies,
                    output = ctx.get('output')
                )
            )[0]

        p, k, var = self.computePower(ctx)

        return -0.5 * np.sum((self.power2d_data - p) ** 2 / var)

    def computePower(self, ctx):
        """
        Compute the 2D power spectrum within the current context.

        Parameters
        ----------
        ctx : :class:`cosmoHammer.ChainContext.ChainContext` instance
            The context object, with objects within at least containing "baselines", "visibilities", "frequencies"
            and "output".

        Returns
        -------
        power2d : (nperp, npar)-array
            The 2D power spectrum.

        coords : list of 2 arrays
            The first is kperp, and the second is kpar.
        """
        # Read in data from ctx
        visibilities = ctx.get("visibilities")
        baselines = ctx.get('baselines')
        frequencies = ctx.get("frequencies")
        n_uv = self.n_uv or ctx.get("output").lightcone_box.shape[0]
        cosmo = ctx.get("output").cosmo

        # Compute 2D power.
        ugrid, visgrid, weights = self.grid(visibilities, baselines, frequencies, n_uv, self.umax)

        visgrid, eta = self.frequency_fft(visgrid, frequencies, taper=self.frequency_taper)

        # Ensure weights correspond to FT.
        weights = np.sum(weights*self.frequency_taper(len(frequencies)), axis=-1)

        power2d, coords, var = self.get_2d_power(visgrid, [ugrid, ugrid, eta], weights, frequencies.min(), frequencies.max(),
                                            bins=self.n_psbins, cosmo=cosmo)

        return power2d, coords, var

    def get_2d_power(self, fourier_vis, coords, weights, nu_min, nu_max, cosmo, bins=100):
        """
        Determine the 2D Power Spectrum of the observation.

        Parameters
        ----------

        fourier_vis : complex (ngrid, ngrid, neta)-array
            The gridded visibilities, fourier-transformed along the frequency axis. Units JyHz.

        coords: list of 3 1D arrays.
            The [u,v,eta] co-ordinates corresponding to the gridded fourier visibilities. u and v in 1/rad, and
            eta in 1/Hz.

        weights: (ngrid, ngrid, eta)-array
            The relative weight of each grid point. Conceptually, the number of baselines that have contributed
            to its value.

        nu_min, nu_max : (n_freq)-array
            The min and max frequency of observation, in Hz.

        bins : int, optional
            The number of radial bins, in which to average the u and v co-ordinates.

        Returns
        -------
        P : float (n_eta, bins)-array
            The cylindrical averaged (or 2D) Power Spectrum, with units Mpc**3.

        coords : list of 2 1D arrays
            The first value is the coordinates of k_perp (in 1/Mpc), and the second is k_par (in 1/Mpc).
        """
        # Change the units of coords to Mpc. No Need to do this if we don't write out the result.
        # z_mid = 1420e6 / (nu_min+nu_max)/2 - 1
        # coords[0] *= 2 * np.pi / cosmo.comoving_transverse_distance([z_mid])
        # coords[1] *= 2 * np.pi / cosmo.comoving_transverse_distance([z_mid])
        # coords[2] = 2 * np.pi * coords[2] * cosmo.H0.to(un.m / (un.Mpc * un.s)) * 1420e6 * un.Hz * cosmo.efunc(
        #     z_mid) / (const.c * (1 + z_mid) ** 2)

        # The 3D power spectrum
        power_3d = np.absolute(fourier_vis) ** 2

        # Generate the radial bins and coords
        # radial_bins = np.linspace(0, np.sqrt(2 * np.max(coords[0].value) ** 2), bins+1)
        # u, v = np.meshgrid(coords[0], coords[1])

        P, radial_bins, var = angular_average_nd(power_3d, coords, bins, n=2, weights=weights**2, bin_ave=False, get_variance=True)
        radial_bins = (radial_bins[1:] + radial_bins[:-1])/2
        # # Initialize the power and 2d weights.
        # P = np.zeros([fourier_vis.shape[-1], len(radial_bins)])
        # bins = np.zeros_like(P)
        #
        #
        # # Determine which radial bin each cell lies in.
        # bin_indx = np.digitize((u.value ** 2 + v.value ** 2), bins=radial_bins ** 2) - 1
        #
        # print(bin_indx.shape)
        # # Average within radial bins, weighting with weights.
        # for i in range(len(coords[2])):
        #     P[i, :] = np.bincount(bin_indx.flatten(), weights=(weights[:, :, i]*power_3d[:, :, i]).flatten())
        #     bins[i, :] = np.bincount(bin_indx.flatten(), weights=weights[:, :, i].flatten())
        # P[bins > 0] = P[bins > 0] / bins[bins > 0]

        # Convert the units of the power into Mpc**6. No need to do this if we don't write out the result, since it
        # has the same effect on power and its variance (thus cancels).
        # P /= ((CoreForegrounds.conversion_factor_K_to_Jy() * self.hz_to_mpc(nu_min, nu_max, cosmo) * self.sr_to_mpc2(z_mid, cosmo)) ** 2).value
        # P /= self.volume(z_mid, nu_min, nu_max, cosmo)

        return P, [radial_bins, coords[2]], var

    # def suppressedFg_1DPower(self, bins = 20):
    #
    #     annuli_bins = np.linspace(0, np.sqrt(self.k[0].max()**2+self.k[1].max()**2), bins)
    #
    #     k_par, k_perp = np.meshgrid(self.k[0], self.k[1])
    #
    #     k_indices = np.digitize(k_par**2+k_perp**2, bins = annuli_bins**2) -1
    #
    #     P_1D = np.zeros(len(annuli_bins))
    #     uncertainty_1D = np.zeros(len(annuli_bins))
    #
    #     P_1D[:] = [1 / np.sum(1 / self.uncertainty[k_indices == kk] ** 2) * np.sum((self.power[k_indices == kk] + self.uncertainty[k_indices == kk]) / self.uncertainty[k_indices == kk] ** 2) for kk in range(len(annuli_bins))]
    #     uncertainty_1D[:] = [np.sum([k_indices == kk]) / np.sum(1 / self.uncertainty[k_indices == kk]) for kk in range(len(annuli_bins))]
    #
    #     return P_1D, uncertainty_1D

    @staticmethod
    def grid(visibilities, baselines, frequencies, ngrid, umax=None):
        """
        Grid a set of visibilities from baselines onto a UV grid.

        Uses simple nearest-neighbour weighting to perform the gridding. This is fast, but not necessarily very
        accurate.

        Parameters
        ----------
        visibilities : complex (n_baselines, n_freq)-array
            The visibilities at each basline and frequency.

        baselines : (n_baselines, 2)-array
            The physical baselines of the array, in metres.

        frequencies : (n_freq)-array
            The frequencies of the observation

        ngrid : int
            The number of grid cells to form in the grid, per side. Note that the grid will extend to the longest
            baseline.

        umax : float, optional
            The extent of the UV grid. By default, uses the longest baseline at the highest frequency.

        Returns
        -------
        centres : (ngrid,)-array
            The co-ordinates of the grid cells, in UV.

        visgrid : (ngrid, ngrid, n_freq)-array
            The visibility grid, in Jy.

        weights : (ngrid, ngrid, n_freq)-array
            The weights of the visibility grid (i.e. how many baselines contributed to each).
        """
        if umax is None:
            umax = max([np.abs(b).max() for b in baselines]) * frequencies.max()/const.c

        ugrid = np.linspace(-umax, umax, ngrid+1) # +1 because these are bin edges.
        visgrid = np.zeros((ngrid, ngrid, len(frequencies)), dtype=np.complex128)
        weights = np.zeros((ngrid, ngrid, len(frequencies)))

        for j, f in enumerate(frequencies):
            # U,V values change with frequency.
            u = baselines[:, 0] * f / const.c
            v = baselines[:, 1] * f / const.c

            # TODO: doing three of the same histograms is probably unnecessary.
            weights[:, :, j] = np.histogram2d(u.value, v.value, bins=[ugrid, ugrid])[0]
            rl = np.histogram2d(u.value, v.value, bins=[ugrid, ugrid], weights=np.real(visibilities[:,j]))[0]
            im = np.histogram2d(u.value, v.value, bins=[ugrid, ugrid], weights=np.imag(visibilities[:,j]))[0]

            visgrid[:, :, j] = (rl + im * 1j) / weights[:, :, j]

        visgrid[np.isnan(visgrid)] = 0.0

        centres = (ugrid[1:] + ugrid[:-1])/2

        return centres, visgrid, weights

    @staticmethod
    def frequency_fft(vis, freq, taper=np.ones_like):
        """
        Fourier-transform a gridded visibility along the frequency axis.

        Parameters
        ----------
        vis : complex (ncells, ncells, nfreq)-array
            The gridded visibilities.

        freq : (nfreq)-array
            The linearly-spaced frequencies of the observation.

        taper : callable, optional
            A function which computes a taper function on an nfreq-array. Default is to have no taper. Callable should
            take single argument, N.

        Returns
        -------
        ft : (ncells, ncells, nfreq/2)-array
            The fourier-transformed signal, with negative eta removed.

        eta : (nfreq/2)-array
            The eta-coordinates, without negative values.
        """
        ft, eta =fft(vis*taper(len(freq)), (freq.max() - freq.min()), axes=(2,), a=0, b=2 * np.pi)
        ft = ft[:,:, (int(len(freq)/2)+1):]
        return ft, eta[0][(int(len(freq)/2)+1):]

    @staticmethod
    def hz_to_mpc(nu_min, nu_max, cosmo):
        """
        Convert a frequency range in Hz to a distance range in Mpc.
        """
        z_max = 1420e6/nu_min - 1
        z_min = 1420e6/nu_max - 1

        return (cosmo.comoving_distance(z_max) - cosmo.comoving_distance(z_min)) / (nu_max - nu_min)

    @staticmethod
    def sr_to_mpc2(z, cosmo):
        """
        Conversion factor from steradian to Mpc^2 at a given redshift.
        Parameters
        ----------
        z_mid

        Returns
        -------

        """
        return cosmo.comoving_distance(z) / (1 * un.sr)

    @staticmethod
    def volume(z_mid, nu_min, nu_max, A_eff=20):
        """
        Calculate the effective volume of an observation in Mpc**3, when co-ordinates are provided in Hz.

        Parameters
        ----------
        z_mid : float
            Mid-point redshift of the observation.

        nu_min, nu_max : float
            Min/Max frequency of observation, in Hz.

        A_eff : float
            Effective area of the telescope.

        Returns
        -------
        vol : float
            The volume.

        Notes
        -----
        How is this actually calculated? What assumptions are made?
        """
        # TODO: fix the notes in the docs above.

        diff_nu = nu_max - nu_min

        G_z = (cosmo.H0).to(un.m / (un.Mpc * un.s)) * 1420e6 * un.Hz * cosmo.efunc(z_mid) / (const.c * (1 + z_mid) ** 2)

        Vol = const.c ** 2 / (sigma * un.m ** 2 * nu_max * (1 / un.s) ** 2) * diff_nu * (
                    1 / un.s) * cosmo.comoving_distance([z_mid]) ** 2 / (G_z)

        return Vol.value

    def simulate_data(self, fg_core, instr_core, params):
        """
        Simulate datasets to which this very class instance should be compared, returning their mean and std.
        Writes the information to the datafile of this instance.

        Parameters
        ----------
        fg_core : :class:`~core.CoreForeground` instance

        instr_core : :class:`~core.CoreInstrumental` instance

        params : dict
            A dictionary of the same structure as that expected by the `run_mcmc` function from `py21cmmc`, holding
            the parameters that are going to be varied.

        niter : int, optional
            Number of iterations to aggregate.

        """
        core = Core21cmFastModule(
            parameter_names=params.keys(),
            box_dim=self._box_dim,
            flag_options=self._flag_options,
            astro_params=self._astro_params,
            cosmo_params=self._cosmo_params
        )

        core.setup()
        fg_core.setup()
        instr_core.setup()

        params = Params(*[(k, v[1]) for k, v in params.items()])
        ctx = ChainContext('derp', params)

        # Here is where the __call__ happens!
        core(ctx)
        fg_core(ctx)
        instr_core(ctx)

        np.savez(self.datafile, frequencies=ctx.get("frequencies"), baselines = ctx.get("baselines"),
                 visibilities=ctx.get("visibilities"))


# class LikelihoodForeground1D(LikelihoodForeground2D):
#     def computePower(self, ctx):
#         # Read in data from ctx
#         visibilities = ctx.get("visibilities")
#         baselines = ctx.get('baselines')
#         frequencies = ctx.get("frequencies")
#         n_uv = self.n_uv or ctx.get("output").lightcone_box.shape[0]
#
#         ugrid, visgrid, weights = self.grid(visibilities, baselines, frequencies, n_uv)
#
#         visgrid, eta = self.frequency_fft(visgrid, frequencies)
#
#         # TODO: this is probably wrong!
#         #weights = np.sum(weights, axis=-1)
#         power2d, coords  = self.get_1D_power(visgrid, [ugrid, ugrid, eta], weights, frequencies, bins=self.n_psbins )
#
#         # Find the 1D Power Spectrum of the visibility
#         #self.get_1D_power(visgrid, [ugrid, ugrid, eta[0]], weights, frequencies, bins=self.n_psbins)
#         return power2d, coords
#
#     def get_1D_power(self, visibility, coords, weights, linFrequencies, bins=100):
#
#         print("Finding the power spectrum")
#         ## Change the units of coords to Mpc
#         z_mid = (1420e6) / (np.mean(linFrequencies)) - 1
#         coords[0] = 2 * np.pi * coords[0] / cosmo.comoving_transverse_distance([z_mid])
#         coords[1] = 2 * np.pi * coords[1] / cosmo.comoving_transverse_distance([z_mid])
#         coords[2] = 2 * np.pi * coords[2] * (cosmo.H0).to(un.m / (un.Mpc * un.s)) * 1420e6 * un.Hz * cosmo.efunc(
#             z_mid) / (const.c * (1 + z_mid) ** 2)
#
#         # Change the unit of visibility
#         visibility = visibility / CoreForegrounds.conversion_factor_K_to_Jy() * self.hz_to_mpc(np.min(linFrequencies),
#                                                                                                np.max(
#                                                                                                    linFrequencies)) * self.sr_to_mpc2(
#             z_mid)
#
#         # Square the visibility
#         visibility_sq = np.abs(visibility) ** 2
#
#         # TODO: check if this is correct (reshaping might be in wrong order)
#         weights = np.repeat(weights, len(coords[2])).reshape((len(coords[0]), len(coords[1]), len(coords[2])))
#
#         PS_mK2Mpc6, k_Mpc = angular_average_nd(visibility_sq, coords, bins=bins, weights=weights)
#
#         PS_mK2Mpc3 = PS_mK2Mpc6 / self.volume(z_mid, np.min(linFrequencies), np.max(linFrequencies))
#
#         return PS_mK2Mpc3, k_Mpc
