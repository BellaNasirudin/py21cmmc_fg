#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 16:54:22 2018

@author: bella
"""

from powerbox.dft import fft
from powerbox.tools import angular_average_nd
import numpy as np
from astropy import constants as const
from astropy.cosmology import Planck15 as cosmo
from astropy import units as un

from py21cmmc.likelihood import LikelihoodBase, Core21cmFastModule
from cosmoHammer.ChainContext import ChainContext
from cosmoHammer.util import Params
from .core import CoreForegrounds


class LikelihoodForeground2D(LikelihoodBase):
    def __init__(self, datafile, n_uv=None, n_psbins=50, umax = None,**kwargs):
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
        """

        super().__init__(**kwargs)
        self.datafile = datafile
        self.n_uv = n_uv
        self.n_psbins = n_psbins
        self.umax = umax

    def setup(self):
        """
        Read in observed data.

        Data should be in an npz file, and contain a "k" and "p" array. k should be in 1/Mpc, and p in Mpc**3.
        """
        data = np.load(self.datafile +".npz")

        # TODO: it may be better to read in visbilities, and transform them here an now. In any case, we'll end up with
        # the following variables after the transformation anyway, so we can work with them.
        self.kpar = data["kpar"]
        self.kperp = data['kperp']
        self.power = data["p"]

        # This just monitors whether we've check the k dimensions.
        self._checked = False

    def computeLikelihood(self, ctx):

        p, k = self.computePower(ctx)

        # Make sure we have the correct kpar and kperp. We can remove this if the data itself is generated from
        # this class.
        if not self._checked:
            if len(k[0]) != len(self.kperp) or not np.isclose(k[0][0], self.kperp[0], 1e-4) or not np.isclose(k[0][-1], self.kperp[-1], 1e-4):
                raise ValueError("The kperp dimensions between data and model are incompatible.")
            if len(k[1]) != len(self.kpar) or not np.isclose(k[1][0], self.kpar[0], 1e-4) or not np.isclose(k[1][-1],
                                                                                                           self.kperp[-1],
                                                                                                           1e-4):
                raise ValueError("The kpar dimensions between data and model are incompatible.")

            self._checked = True

        # FIND CHI SQUARE OF PS!!!
        # TODO: this is a bit too simple. We need uncertainties!
        return -0.5 * np.sum((self.power - p) ** 2 / self.uncertainty ** 2)

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

        # Compute 2D power.
        ugrid, visgrid, weights = self.grid(visibilities, baselines, frequencies, n_uv, self.umax)
        visgrid, eta = self.frequency_fft(visgrid, frequencies)
        power2d, coords = self.get_2d_power(visgrid, [ugrid, ugrid, eta], weights, frequencies.min(), frequencies.max(), bins=self.n_psbins)

        return power2d, coords

    def get_2d_power(self, fourier_vis, coords, weights, nu_min, nu_max, bins=100):
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
        # Change the units of coords to Mpc
        z_mid = 1420e6 / (nu_min+nu_max)/2 - 1
        coords[0] *= 2 * np.pi / cosmo.comoving_transverse_distance([z_mid])
        coords[1] *= 2 * np.pi / cosmo.comoving_transverse_distance([z_mid])
        coords[2] = 2 * np.pi * coords[2] * cosmo.H0.to(un.m / (un.Mpc * un.s)) * 1420e6 * un.Hz * cosmo.efunc(
            z_mid) / (const.c * (1 + z_mid) ** 2)

        # The 3D power spectrum
        power_3d = np.absolute(fourier_vis) ** 2

        # Generate the radial bins and coords
        radial_bins = np.linspace(0, np.sqrt(2 * np.max(coords[0].value) ** 2), bins+1)
        u, v = np.meshgrid(coords[0], coords[1])

        # Initialize the power and 2d weights.
        P = np.zeros([fourier_vis.shape[-1], len(radial_bins)])
        bins = np.zeros_like(P)

        # Determine which radial bin each cell lies in.
        bin_indx = np.digitize((u.value ** 2 + v.value ** 2), bins=radial_bins ** 2) - 1

        print(bin_indx.shape)
        # Average within radial bins, weighting with weights.
        for i in range(len(coords[2])):
            P[i, :] = np.bincount(bin_indx.flatten(), weights=(weights[:, :, i]*power_3d[:, :, i]).flatten())
            bins[i, :] = np.bincount(bin_indx.flatten(), weights=weights[:, :, i].flatten())
        P[bins > 0] = P[bins > 0] / bins[bins > 0]

        # Convert the units of the power into Mpc**6
        P /= ((CoreForegrounds.conversion_factor_K_to_Jy() * self.hz_to_mpc(nu_min, nu_max) * self.sr_to_mpc2(z_mid)) ** 2).value
        P /= self.volume(z_mid, nu_min, nu_max)

        # TODO: In here we also need to calculate the VARIANCE of the power!!
        return P, [(radial_bins[1:]+radial_bins[:-1])/2, coords[2].value]

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
    def frequency_fft(vis, freq):
        """
        Fourier-transform a gridded visibility along the frequency axis.

        Parameters
        ----------
        vis : complex (ncells, ncells, nfreq)-array
            The gridded visibilities.

        freq : (nfreq)-array
            The linearly-spaced frequencies of the observation.

        Returns
        -------
        ft : (ncells, ncells, nfreq/2)-array
            The fourier-transformed signal, with negative eta removed.

        eta : (nfreq/2)-array
            The eta-coordinates, without negative values.
        """
        ft, eta =fft(vis, (freq.max() - freq.min()), axes=(2,), a=0, b=2 * np.pi)
        ft = ft[:,:, (int(len(freq)/2)+1):]
        return ft, eta[0][(int(len(freq)/2)+1):]

    @staticmethod
    def hz_to_mpc(nu_min, nu_max):
        """
        Convert a frequency range in Hz to a distance range in Mpc.
        """
        z_max = 1420e6/nu_min - 1
        z_min = 1420e6/nu_max - 1

        return (cosmo.comoving_distance(z_max) - cosmo.comoving_distance(z_min)) / (nu_max - nu_min)

    @staticmethod
    def sr_to_mpc2(z):
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

        Vol = const.c ** 2 / (A_eff * un.m ** 2 * nu_max * (1 / un.s) ** 2) * diff_nu * (
                    1 / un.s) * cosmo.comoving_distance([z_mid]) ** 2 / (G_z)

        return Vol.value

    def simulate_data(self, fg_core, instr_core, params, niter=20):
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

        p = [0] * niter
        for i in range(niter):
            # Here is where the __call__ happens!
            core(ctx)
            fg_core(ctx)
            instr_core(ctx)

            p[i], k = self.computePower(ctx)

        sigma = np.std(np.array(p), axis=0)
        p = np.mean(np.array(p), axis=0)

        np.savez(self.datafile, k=k, p=p, sigma=sigma)


class LikelihoodForeground1D(LikelihoodForeground2D):
    def computePower(self, ctx):
        # Read in data from ctx
        visibilities = ctx.get("visibilities")
        baselines = ctx.get('baselines')
        frequencies = ctx.get("frequencies")
        n_uv = self.n_uv or ctx.get("output").lightcone_box.shape[0]

        ugrid, visgrid, weights = self.grid(visibilities, baselines, frequencies, n_uv)

        visgrid, eta = self.frequency_fft(visgrid, frequencies)

        # TODO: this is probably wrong!
        #weights = np.sum(weights, axis=-1)
        power2d, coords  = self.get_1D_power(visgrid, [ugrid, ugrid, eta], weights, frequencies, bins=self.n_psbins )

        # Find the 1D Power Spectrum of the visibility
        #self.get_1D_power(visgrid, [ugrid, ugrid, eta[0]], weights, frequencies, bins=self.n_psbins)
        return power2d, coords

    def get_1D_power(self, visibility, coords, weights, linFrequencies, bins=100):

        print("Finding the power spectrum")
        ## Change the units of coords to Mpc
        z_mid = (1420e6) / (np.mean(linFrequencies)) - 1
        coords[0] = 2 * np.pi * coords[0] / cosmo.comoving_transverse_distance([z_mid])
        coords[1] = 2 * np.pi * coords[1] / cosmo.comoving_transverse_distance([z_mid])
        coords[2] = 2 * np.pi * coords[2] * (cosmo.H0).to(un.m / (un.Mpc * un.s)) * 1420e6 * un.Hz * cosmo.efunc(
            z_mid) / (const.c * (1 + z_mid) ** 2)

        # Change the unit of visibility
        visibility = visibility / CoreForegrounds.conversion_factor_K_to_Jy() * self.hz_to_mpc(np.min(linFrequencies),
                                                                                               np.max(
                                                                                                   linFrequencies)) * self.sr_to_mpc2(
            z_mid)

        # Square the visibility
        visibility_sq = np.abs(visibility) ** 2

        # TODO: check if this is correct (reshaping might be in wrong order)
        weights = np.repeat(weights, len(coords[2])).reshape((len(coords[0]), len(coords[1]), len(coords[2])))

        PS_mK2Mpc6, k_Mpc = angular_average_nd(visibility_sq, coords, bins=bins, weights=weights)

        PS_mK2Mpc3 = PS_mK2Mpc6 / self.volume(z_mid, np.min(linFrequencies), np.max(linFrequencies))

        return PS_mK2Mpc3, k_Mpc
