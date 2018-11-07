"""
Created on Fri Apr 20 16:54:22 2018

@author: bella
"""

from powerbox.dft import fft
from powerbox.tools import angular_average_nd, get_power
import numpy as np
from astropy import constants as const
from astropy import units as un

from py21cmmc.mcmc.likelihood import LikelihoodBaseFile, LikelihoodBase

from py21cmmc.mcmc.core import CoreLightConeModule
from .core import CoreInstrumental, CoreDiffuseForegrounds, CorePointSourceForegrounds, ForegroundsBase
from scipy.sparse import block_diag

from .util import lognormpdf

from scipy.integrate import quad
from scipy.special import erf

import logging
logger = logging.getLogger("21CMMC")


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
        self.foreground_data = self.numerical_covariance(1)[0]

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
        p, kperp, kpar = get_power(lightcone.brightness_temp * np.hanning(np.shape(lightcone.brightness_temp)[-1]), boxlength=lightcone.lightcone_dimensions, res_ndim=2, bin_ave=False,
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
        logger.debug("LIKELIHOOD IS ", lnl )

        return lnl
    
    def numerical_covariance(self, nrealisations=200):
        """
        Calculate the covariance of the foregrounds BEFORE the MCMC
    
        Parameters
        ----------
        
        nrealisations: int, optional
            Number of realisations to find the covariance.
        
        Output
        ------
        
        mean: (nperp, npar)-array
            The mean 2D power spectrum of the foregrounds.
            
        cov: 
            The sparse block diagonal matrix of the covariance if nrealisations > 1
            Else it is 0.
        """
        p = []
        mean = 0
        for core in self.foreground_cores:
            for ii in range(nrealisations):
                power = self.compute_power(core.mock_lightcone(), self.n_psbins)[0]
                p.append(power)
            
            mean += np.mean(p, axis=0)
            
        if(nrealisations>1):
            cov = [np.cov(x) for x in np.array(p).transpose((1,2,0))]
        else:
            cov = 0

        return mean, cov

    def get_signal_covariance(self, signal_power):
        """
        From a 2D signal (i.e. EoR) power spectrum, make a list of covariances in eta, with length u.

        Parameters
        ----------
        signal_power : (n_eta, n_u)-array
            The 2D power spectrum of the signal.

        Returns
        -------
        cov : list of arrays
            A length-u list of arrays of shape n_eta * n_eta.
        """
        cov = []
        for sig_eta in enumerate(signal_power.T):
            cov.append((0.15 * np.diag(sig_eta))**2)

        return cov

    def lognormpdf(self, model, cov):
        """
        Calculate gaussian probability log-density of x, when x ~ N(mu,sigma), and cov is sparse.
    
        Code adapted from https://stackoverflow.com/a/16654259
        
        Add the uncertainty of the model to the covariance and find the log-likelihood
        
        Parameters
        ----------
        
        model: (nperp, npar)-array
            The 2D power spectrum of the model signal.
        
        cov: (nperp * npar, nperp * npar)-array
            The sparse block diagonal matrix of the covariance
            
        Output
        ------
        
        returns the log-likelihood (float)
        """

        cov = block_diag(cov_new, format='csc')
        chol_deco = cholesky(cov)

        nx = len(model.flatten())
        norm_coeff = nx * np.log(2 * np.pi) + chol_deco.logdet()
        
        err = ((self.p_data + self.foreground_data) - (model + self.foreground_mean)).T.flatten()

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


class LikelihoodInstrumental2D(LikelihoodBaseFile):
    required_cores = [CoreInstrumental]

    def __init__(self, n_uv=None, n_ubins=30, umax = None, frequency_taper=np.blackman, nrealisations = 100,
                 model_uncertainty = 0.15, eta_min = 0, **kwargs):
        """
        A likelihood for EoR physical parameters, based on a Gaussian 2D power spectrum.

        In this likelihood, any foregrounds are naturally suppressed by their imposed covariance, in 2D spectral space.
        Nevertheless, it is not required that the :class:`~core.CoreForeground` class be amongst the Core modules for
        this likelihood module to work. Without the foregrounds, the 2D modes are naturally weighted by the sample
        variance of the EoR signal itself.

        The likelihood requires the :class:`~core.CoreInstrumental` Core module.

        Parameters
        ----------
        n_uv : int, optional
            The number of UV cells to grid the visibilities (per side). By default, uses the same number of UV cells
            as the Core (i.e. the same grid used to interpolate the simulation onto the baselines).

        n_ubins : int, optional
            The number of kperp (or u) bins to use when doing a cylindrical average of the power spectrum.

        umax : float, optional
            The extent of the UV grid. By default, uses the longest baseline at the highest frequency.

        frequency_taper : callable, optional
            A function which computes a taper function on an nfreq-array. Callable should
            take single argument, N.

        nrealisations : int, optional
            The number of realisations to use if calculating a *foreground* mean/covariance. Only applicable if
            a ForegroundBase instance is loaded as a core.

        model_uncertainty : float, optional
            Fractional uncertainty in the signal model power spectrum (this is modelling uncertainty of the code itself)

        eta_min : float, optional
            Minimum eta value to consider in the model. This will be applied at every u value.

        Other Parameters
        ----------------
        datafile : str
            A filename referring to a file which contains the observed data (or mock data) to be fit to. The file
            should be a compressed numpy binary (i.e. a npz file), and must contain at least the arrays "kpar", "kperp"
            and "p", which are the parallel/perpendicular modes (in 1/Mpc) and power spectrum (in Mpc^3) respectively.
        """

        super().__init__(**kwargs)

        self.n_uv = n_uv
        self.n_ubins = n_ubins
        self.umax = umax
        self.frequency_taper = frequency_taper
        self.nrealisations = nrealisations
        self.model_uncertainty = model_uncertainty
        self.eta_min = eta_min

        # Set the mean and covariance of foregrounds to zero by default
        self.foreground_mean, self.foreground_covariance = 0, 0

    def setup(self):
        """
        Read in observed data.

        Data should be in an npz file, and contain a "k" and "p" array. k should be in 1/Mpc, and p in Mpc**3.
        """
        # Get default value for n_uv. THIS HAS TO BE BEFORE THE SUPER() CALL!
        if self.n_uv is None:
            self.n_uv = self._instr_core.n_cells

        super().setup()

        # we can unpack data now because we know it's always a list of length 1.
        if self.data:
            self.data = self.data[0]
            self.baselines = self.data["baselines"]
            self.frequencies = self.data["frequencies"]
        if self.noise:
            self.noise = self.noise[0]

        self.parameter_names = getattr(self.LikelihoodComputationChain.params, "keys", [])

    def simulate(self, ctx):
        """
        Simulate datasets to which this very class instance should be compared.
        """
        baselines = ctx.get("baselines")
        frequencies = ctx.get("frequencies")
        visibilities = ctx.get("visibilities")
        p_signal, u_eta, weights = self.compute_power(visibilities, baselines, frequencies)

        return [dict(p_signal=p_signal, baselines=baselines, frequencies=frequencies, u_eta=u_eta, weights=weights)]

    def define_noise(self, ctx, model):
        """
        Define the properties of the noise... its mean and covariance.

        Note that in general this method should just calculate whatever noise properties are relevant, but in
        this case that is specifically the mean (of the noise) and its covariance.

        Note also that the outputs of this function are by default *saved* to a file within setup, so it can be
        checked later.

        It is *only* run on setup, not every iteration. So noise properties that are parameter-dependent must
        be performed elsewhere.

        Parameters
        ----------
        ctx : dict-like
            The Context object which is assumed to hold the core simulation context.

        model : list of dicts
            Exactly the output of :meth:`simulate`.

        Returns
        -------
        list of dicts
            In this case, a list with a single dict, which has the mean and covariance in it.

        """
        # Only save the mean/cov if we have foregrounds, and they don't update every iteration (otherwise, get them
        # every iter).
        if self.foreground_cores and not any([fg._updating for fg in self.foreground_cores]):
#            mean, covariance = self.numerical_covariance(
#                model[0]['baselines'], model[0]['frequencies'],
#                nrealisations=self.nrealisations
#            )
            print("DOING THIS ANALYTICAL THING")
            mean = self.numerical_covariance(
                model[0]['baselines'], model[0]['frequencies'],
                nrealisations=self.nrealisations
            )[0]
            print("UV IS ", model[0]["u_eta"])
            print("ETA IS ", model[0]["u_eta"][-1])
            print(np.median(model[0]["frequencies"]))
            covariance = self.analytical_covariance(model[0]["u_eta"][0], model[0]["u_eta"][-1], np.median(model[0]["frequencies"]), model[0]["frequencies"].max()-model[0]["frequencies"].min())
        else:
            # Only need thermal variance if we don't have foregrounds, otherwise it will be embedded in the
            # above foreground covariance

            weights = model[0]['weights']
            covariance = self.get_thermal_variance(weights)
            mean = 0

        return [{"mean":mean, "covariance":covariance}]

    def computeLikelihood(self, model):
        "Compute the likelihood"
        # remember that model is *exactly* the result of simulate(), which is a  *list* of dicts, so unpack
        model = model[0]

        sig_cov = self.get_signal_covariance(model['p_signal'])
        total_model = model['p_signal']

        # If we need to get the foreground covariance
        if self.foreground_cores and any([fg._updating for fg in self.foreground_cores]):
            mean, cov = self.numerical_covariance(model['baselines'], model['frequencies'], nrealisations=self.nrealisations)
            total_model += mean

        else:
            # Normal case (foreground parameters are not being updated, or there are no foregournds)
            total_model += self.noise["mean"]
            total_cov = [x+y for x,y in zip(self.noise['covariance'], sig_cov)]

        lnl = lognormpdf(self.data['p_signal'], total_model, total_cov)

        return lnl

    @property
    def _eor_core(self):
        for m in self.LikelihoodComputationChain.getCoreModules():
            if isinstance(m, CoreLightConeModule):
                return m
        return None


    @property
    def _instr_core(self):
        for m in self.LikelihoodComputationChain.getCoreModules():
            if isinstance(m, CoreInstrumental):
                return m

    @property
    def foreground_cores(self):
        try:
            return [m for m in self.LikelihoodComputationChain.getCoreModules() if isinstance(m, ForegroundsBase)]
        except AttributeError:
            raise AttributeError("foreground_cores is not available unless emedded in a LikelihoodComputationChain, after setup")

    def get_thermal_variance(self, weights):
        """
        Calculate the thermal noise variance on each 2D mode.

        Parameters
        ----------
        weights : (neta, n_u)-array
            The effective number of baselines averaged to create the power spectrum mode.

        Returns
        -------
        cov : list
            A list of arrays defining a block-diagonal covariance matrix, of which the thermal variance is really
            just the diagonal.
        """
        cov = []
        for weight_eta in weights:
            cov.append(np.diag(self._instr_core.thermal_variance_baseline / weight_eta))

        return cov

    def get_signal_covariance(self, signal_power):
        """
        From a 2D signal (i.e. EoR) power spectrum, make a list of covariances in eta, with length u.

        Parameters
        ----------
        signal_power : (n_eta, n_u)-array
            The 2D power spectrum of the signal.

        Returns
        -------
        cov : list of arrays
            A length-u list of arrays of shape n_eta * n_eta.
        """
        cov = []
        for sig_eta in signal_power:
            cov.append((self.model_uncertainty * np.diag(sig_eta))**2)

        return cov

    def numerical_covariance(self, baselines, frequencies, params={}, nrealisations=200, cov = None):
        """
        Calculate the covariance of the foregrounds.
    
        Parameters
        ----------
        params: dict
            The parameters of this iteration. If empty, default parameters are used.

        nrealisations: int, optional
            Number of realisations to find the covariance.
        
        Output
        ------
        mean: (nperp, npar)-array
            The mean 2D power spectrum of the foregrounds.
            
        cov: 
            The sparse block diagonal matrix of the covariance if nrealisation is not 1
            Else it is 0
        """
        p = []
        mean = 0

        for ii in range(nrealisations):
            # Create an empty context with the given parameters.
            ctx = self.LikelihoodComputationChain.createChainContext(params)

            # For each realisation, run every foreground core (not the signal!)
            for core in self.foreground_cores:
                core.simulate_data(ctx)

            # And turn them into visibilities
            self._instr_core.simulate_data(ctx)

            power, ks, pweights = self.compute_power(ctx.get("visibilities"), baselines, frequencies)
                
            p.append(power)
            
        mean += np.mean(p, axis=0)

        # Note, this covariance *already* has thermal noise built in.
        if nrealisations > 1 and cov is not None:
            cov = [np.cov(x) for x in np.array(p).transpose((1, 2, 0))]
        else:
            cov = 0
            
        return mean, cov
    
    @staticmethod
    def analytical_covariance(uv, eta, nu_mid, bwidth, S_min=1e-1, S_max=1.0, alpha=4100., beta=1.59, D = 4.0, gamma=0.8, f0=150e6):
        """
        from Cath's derivation: https://www.overleaf.com/3815868784cbgxmpzpphxm
        
        assumes:
            1. Gaussian beam
            2. Blackman-Harris taper
            3. S_max is 1 Jy
            
        Parameters
        ----------
        uv    : array
            The range of u,v's in Fourier space (sr^-1).
        eta   : array
            The range of eta in Fourier space (Hz^-1).
        nu_mid: float
            The central band frequency (Hz).
        bwidth: float
            The bandwidth (Hz).
        S_min : float, optional
            The minimum flux density of point sources in the simulation (Jy).
        S_max : float, optional
            The maximum flux density of point sources in the simulation, representing the 'peeling limit' (Jy)
        alpha : float, optional
            The normalisation coefficient of the source counts (sr^-1 Jy^-1).
        beta : float, optional
            The power-law index of source counts.
        D     : float, optional
            The physical diameter of the tiles, in metres.
        gamma : float, optional
            The power-law index of the spectral energy distribution of sources (assumed universal).
        f0    : float, optional
            The reference frequency, at which the other parameters are defined, in Hz.
        
        Returns
        -------
        cov   : (n_eta, n_eta)-array in a n_uv list
            The analytical covariance of the power spectrum over uv.
        
        """
        sigma = bwidth / 7
        
        cov = []
        
        for ii in range(len(uv)):
            
            x = lambda u: u * const.c.value / nu_mid
            
            #we only need the covariance of u with itself, so x1=x2
            C = 2 * sigma**2 * (2 * x(uv[ii])**2 ) / const.c.value**2 
            
            std_dev = const.c.value * eta / D            
            print(C, std_dev)            
            A = 1 / std_dev**2 + C
            
            # all combination of eta
            cov_uv = np.zeros((len(eta),len(eta)))
            
            for jj in range(len(eta)):
                
                avg_S2 = quad(lambda S: S**2 * alpha * (S**2 * (eta[jj] * f0) ** (gamma))** (-beta) * alpha / (3 + beta) * S **(3+ beta), S_min, S_max)[0]
                            
                B = 4 * sigma**2 * (x(uv[ii]) * (eta + eta[jj]) ) / const.c.value
                
                cov_eta = avg_S2 * np.sqrt(np.pi/(4 * A)) * np.exp(-2 * sigma**2 * (eta**2 + eta[jj]**2)) * np.exp(B**2 / (4 * A)) * ( erf((B + 2 * A)/(np.sqrt(2) * A)) - erf((B - 2 * A)/(np.sqrt(2) * A)))
                
                cov_uv[jj,:] = cov_eta
                cov_uv[:, jj] = cov_eta
                
            cov.append(cov_uv)
            
        return cov
        
    
    def compute_power(self, visibilities, baselines, frequencies):
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
        # Compute 2D power.
        ugrid, visgrid, weights = self.grid_visibilities(visibilities, baselines, frequencies, self.n_uv, self.umax)
 
        visgrid, eta = self.frequency_fft(visgrid, frequencies, taper=self.frequency_taper)
       
        # Ensure weights correspond to FT.
        weights = np.sum(weights * self.frequency_taper(len(frequencies)), axis=-1)
        
        power2d, coords, pweights = self.get_2d_power(visgrid, [ugrid, ugrid, eta], weights,
                                                       bins=self.n_ubins)

        u, eta = coords

        # Restrict power to eta modes above eta_min
        power2d = power2d[:, eta > self.eta_min]
        pweights = pweights[:, eta > self.eta_min]
        eta = eta[eta > self.eta_min]

        return power2d, [u, eta], pweights

    def get_2d_power(self, gridded_vis, coords, weights, bins=100):
        """
        Determine the 2D Power Spectrum of the observation.

        Parameters
        ----------

        gridded_vis : complex (ngrid, ngrid, neta)-array
            The gridded visibilities, fourier-transformed along the frequency axis. Units JyHz.

        coords: list of 3 1D arrays.
            The [u,v,eta] co-ordinates corresponding to the gridded fourier visibilities. u and v in 1/rad, and
            eta in 1/Hz.

        weights: (ngrid, ngrid, neta)-array
            The relative weight of each grid point. Conceptually, the number of baselines that have contributed
            to its value.

        bins : int, optional
            The number of radial bins, in which to average the u and v co-ordinates.

        Returns
        -------
        P : float (n_eta, bins)-array
            The cylindrical averaged (or 2D) Power Spectrum, with units Mpc**3.

        coords : list of 2 1D arrays
            The first value is the coordinates of k_perp (in 1/Mpc), and the second is k_par (in 1/Mpc).
        """
        # The 3D power spectrum
        power_3d = np.absolute(gridded_vis) ** 2
        
        weights[weights==0] = 1e-20
        
        P, radial_bins = angular_average_nd(power_3d, coords, bins, n=2, weights=weights**2, bin_ave=False, get_variance=False)

        # have to make a weight for every eta here..
        weights = np.tile(np.atleast_3d(weights), (1, 1, len(coords[-1])))
        weights = angular_average_nd(weights**2, coords, bins, n=2, bin_ave=False, get_variance=False, average=False)[0]

        radial_bins = (radial_bins[1:] + radial_bins[:-1])/2

        return P, [radial_bins, coords[2]], weights

    @staticmethod
    def grid_visibilities(visibilities, baselines, frequencies, ngrid, umax=None):
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
            umax = (max([np.abs(b).max() for b in baselines]) * frequencies.max()/const.c).value
        
        ugrid = np.linspace(-umax, umax, ngrid+1) # +1 because these are bin edges.
        visgrid = np.zeros((ngrid, ngrid, len(frequencies)), dtype=np.complex128)
        weights = np.zeros((ngrid, ngrid, len(frequencies)))
        
        centres = (ugrid[1:] + ugrid[:-1])/2
        # cgrid_u, cgrid_v = np.meshgrid(centres, centres)
        for j, f in enumerate(frequencies):
            # U,V values change with frequency.
            u = baselines[:, 0] * f / const.c
            v = baselines[:, 1] * f / const.c

            # Get number of baselines in each bin
            weights[:, :, j] = np.histogram2d(u.value, v.value, bins=[ugrid, ugrid])[0]

            # Histogram the baselines in each grid but interpolate to find the visibility at the centre
            # TODO: Bella, I don't think this is correct. You don't want to interpolate to the centre, you just
            # want to add the incoherent visibilities together.
            # visgrid[:, :, j] = griddata((u.value , v.value), np.real(visibilities[:,j]),(cgrid_u,cgrid_v),method="linear") + griddata((u.value , v.value), np.imag(visibilities[:,j]),(cgrid_u,cgrid_v),method="linear") *1j

            # So, instead, my version (SGM). One for real, one for complex.
            tmp_rl = np.histogram2d(u.value, v.value, bins=[ugrid, ugrid], weights=visibilities[:,j].real)[0]
            tmp_im = np.histogram2d(u.value, v.value, bins=[ugrid, ugrid], weights=visibilities[:,j].imag)[0]
            visgrid[:, :, j] = tmp_rl + 1j * tmp_im

        # Take the average visibility (divide by weights), being careful not to divide by zero.
        visgrid[weights!=0] /= weights[weights!=0]

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

