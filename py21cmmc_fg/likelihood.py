"""
Created on Fri Apr 20 16:54:22 2018

@author: bella
"""

from powerbox.dft import fft, fftfreq
from powerbox.tools import angular_average_nd, get_power
import numpy as np
from astropy import constants as const
from astropy import units as un

from py21cmmc.mcmc.likelihood import LikelihoodBaseFile, LikelihoodBase

from py21cmmc.mcmc.core import CoreLightConeModule, NotSetupError
from .core import CoreInstrumental, CoreDiffuseForegrounds, CorePointSourceForegrounds, ForegroundsBase
from scipy.sparse import block_diag

from .util import lognormpdf

from scipy.integrate import quad
from scipy.special import erf

import logging

logger = logging.getLogger("21CMMC")

from cached_property import cached_property


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
        p, kperp, kpar = get_power(lightcone.brightness_temp * np.hanning(np.shape(lightcone.brightness_temp)[-1]),
                                   boxlength=lightcone.lightcone_dimensions, res_ndim=2, bin_ave=False,
                                   bins=n_psbins, get_variance=False)

        return p[:, int(len(kpar[0]) / 2):], kperp, kpar[0][int(len(kpar[0]) / 2):]

    def computeLikelihood(self, ctx, storage, variance=False):
        "Compute the likelihood"
        data = self.simulate(ctx)
        # add the power to the written data
        storage.update(**data)

        if variance is True:
            # TODO: figure out how to use 0.15*P_sig here.
            lnl = - 0.5 * np.sum((data['p'] + self.foreground_mean - self.p_data) ** 2 / (
                        self.foreground_variance + (0.15 * data['p']) ** 2))
        else:
            lnl = self.lognormpdf(data['p'], self.foreground_covariance, len(self.kpar_data))
        logger.debug("LIKELIHOOD IS ", lnl)

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

        if (nrealisations > 1):
            cov = [np.cov(x) for x in np.array(p).transpose((1, 2, 0))]
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
            cov.append((0.15 * np.diag(sig_eta)) ** 2)

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

        return -0.5 * (norm_coeff + numerator)

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
            raise AttributeError(
                "foreground_cores is not available unless emedded in a LikelihoodComputationChain, after setup")


class LikelihoodInstrumental2D(LikelihoodBaseFile):
    required_cores = [CoreInstrumental]

    def __init__(self, n_uv=None, n_ubins=30, uv_max=None, u_min=None, u_max=None, frequency_taper=np.blackman,
                 nrealisations=100, model_uncertainty=0.15, eta_min=0, use_analytical_noise=True, **kwargs):
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

        uv_max : float, optional
            The extent of the UV grid. By default, uses the longest baseline at the highest frequency.

        u_min, u_max : float, optional
            The minimum and maximum of the grid of |u| = sqrt(u^2 + v^2). These define the *bin edges*. By default,
            they will be set as the min/max of the UV grid (along a side of the square).

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

        use_analytical_noise : bool, optional
            Whether to use analytical estimate of noise properties (eg. mean and covariance).

        Other Parameters
        ----------------
        datafile : str
            A filename referring to a file which contains the observed data (or mock data) to be fit to. The file
            should be a compressed numpy binary (i.e. a npz file), and must contain at least the arrays "kpar", "kperp"
            and "p", which are the parallel/perpendicular modes (in 1/Mpc) and power spectrum (in Mpc^3) respectively.
        """

        super().__init__(**kwargs)

        self._n_uv = n_uv
        self.n_ubins = n_ubins
        self._uv_max = uv_max
        self.frequency_taper = frequency_taper
        self.nrealisations = nrealisations
        self.model_uncertainty = model_uncertainty
        self.eta_min = eta_min
        self._u_min, self._u_max = u_min, u_max

        self.use_analytical_noise = use_analytical_noise

    def setup(self):
        super().setup()

        # we can unpack data now because we know it's always a list of length 1.
        if self.data:
            self.data = self.data[0]
        if self.noise:
            self.noise = self.noise[0]

    @cached_property
    def parameter_names(self):
        """Names of parameters that are being modified in the MCMC"""
        try:
            return getattr(self.LikelihoodComputationChain.params, "keys", [])
        except AttributeError:
            raise NotSetupError

    @cached_property
    def n_uv(self):
        """The number of cells on a side of the (square) UV grid"""
        if self._n_uv is None:
            return self._instr_core.n_cells
        else:
            return self._n_uv

    def simulate(self, ctx):
        """
        Simulate datasets to which this very class instance should be compared.
        """
        visibilities = ctx.get("visibilities")
        p_signal = self.compute_power(visibilities)

        # Remember that the results of "simulate" can be used in two places: (i) the computeLikelihood method, and (ii)
        # as data saved to file. In case of the latter, it is useful to save extra variables to the dictionary to be
        # looked at for diagnosis, even though they are not required in computeLikelihood().
        return [dict(p_signal=p_signal, baselines=self.baselines, frequencies=self.frequencies,
                     u=self.u, eta = self.eta, nbl_uv=self.nbl_uv, nbl_uvnu=self.nbl_uvnu,
                     nbl_u=self.nbl_u, grid_weights=self.grid_weights)]

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
            if not self.use_analytical_noise:
                mean, covariance = self.numerical_covariance(
                    self.baselines, self.frequencies,
                    nrealisations=self.nrealisations, cov = 1
                )
            else:
                logger.debug("DOING THIS ANALYTICAL THING")

                # Still getting mean numerically for now...
                mean = self.numerical_covariance(
                    self.baselines, self.frequencies,
                    nrealisations=self.nrealisations
                )[0]

                covariance = self.analytical_covariance(self.u, self.eta,
                                                        np.median(self.frequencies),
                                                        self.frequencies.max() - self.frequencies.min())

                thermal_covariance = self.get_thermal_covariance()
                covariance = [x + y for x, y in zip(covariance, thermal_covariance)]

        else:
            # Only need thermal variance if we don't have foregrounds, otherwise it will be embedded in the
            # above foreground covariance... BUT NOT IF THE FOREGROUND COVARIANCE IS ANALYTIC!!
            covariance = self.get_thermal_covariance()
            mean = np.repeat(self.noise_power_expectation, len(self.eta)).reshape((len(self.u), len(self.eta)))

        return [{"mean": mean, "covariance": covariance}]

    def computeLikelihood(self, model):
        "Compute the likelihood"
        # remember that model is *exactly* the result of simulate(), which is a  *list* of dicts, so unpack
        model = model[0]

        sig_cov = self.get_signal_covariance(model['p_signal'])
        total_model = model['p_signal']

        # If we need to get the foreground covariance
        if self.foreground_cores and any([fg._updating for fg in self.foreground_cores]):
            mean, cov = self.numerical_covariance(self.baselines, self.frequencies,
                                                  nrealisations=self.nrealisations)
            total_model += mean

        else:
            # Normal case (foreground parameters are not being updated, or there are no foregournds)
            total_model += self.noise["mean"]
            total_cov = [x + y for x, y in zip(self.noise['covariance'], sig_cov)]

        lnl = lognormpdf(self.data['p_signal'], total_model, total_cov)

        return lnl

    @cached_property
    def _lightcone_core(self):
        for m in self._cores:
            if isinstance(m, CoreLightConeModule):
                return m

        raise AttributeError("No lightcone core loaded")

    @property
    def _instr_core(self):
        for m in self._cores:
            if isinstance(m, CoreInstrumental):
                return m

    @property
    def foreground_cores(self):
        return [m for m in self._cores if isinstance(m, ForegroundsBase)]

    @cached_property
    def frequencies(self):
        return self._instr_core.instrumental_frequencies

    @cached_property
    def baselines(self):
        return self._instr_core.baselines

    def get_thermal_covariance(self):
        """
        Form the thermal variance per u into a full covariance matrix, in the same format as the other covariances.

        Returns
        -------
        cov : list
            A list of arrays defining a block-diagonal covariance matrix, of which the thermal variance is really
            just the diagonal.
        """
        cov = []
        for var in self.noise_power_variance:
            cov.append(np.diag(var * np.ones(len(self.eta))))

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
            cov.append((self.model_uncertainty * np.diag(sig_eta)) ** 2)

        return cov

    def numerical_covariance(self, params={}, nrealisations=200, cov=None):
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

            power = self.compute_power(ctx.get("visibilities"))

            p.append(power)

        mean += np.mean(p, axis=0)

        # Note, this covariance *already* has thermal noise built in.
        if nrealisations > 1 and cov is not None:
            cov = [np.cov(x) for x in np.array(p).transpose((1, 2, 0))]
        else:
            cov = 0

        return mean, cov

    @staticmethod
    def analytical_covariance(uv, eta, nu_mid, bwidth, S_min=1e-1, S_max=1.0, alpha=4100., beta=1.59, D=4.0, gamma=0.8,
                              f0=150e6):
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

            # we only need the covariance of u with itself, so x1=x2
            C = 2 * sigma ** 2 * (2 * x(uv[ii]) ** 2) / const.c.value ** 2

            std_dev = const.c.value * eta / D

            A = 1 / std_dev ** 2 + C

            # all combination of eta
            cov_uv = np.zeros((len(eta), len(eta)))

            for jj in range(len(eta)):
                avg_S2 = quad(lambda S: S ** 2 * alpha * (S ** 2 * (eta[jj] * f0) ** (gamma)) ** (-beta) * alpha / (
                            3 + beta) * S ** (3 + beta), S_min, S_max)[0]

                B = 4 * sigma ** 2 * (x(uv[ii]) * (eta + eta[jj])) / const.c.value

                cov_eta = avg_S2 * np.sqrt(np.pi / (4 * A)) * np.exp(
                    -2 * sigma ** 2 * (eta ** 2 + eta[jj] ** 2)) * np.exp(B ** 2 / (4 * A)) * (
                                      erf((B + 2 * A) / (np.sqrt(2) * A)) - erf((B - 2 * A) / (np.sqrt(2) * A)))

                cov_uv[jj, :] = cov_eta
                cov_uv[:, jj] = cov_eta

            cov.append(cov_uv)

        return cov

    def compute_power(self, visibilities):
        """
        Compute the 2D power spectrum within the current context.

        Parameters
        ----------
        visibilities : (nbl, nf)-complex-array
            The visibilities of each baseline at each frequency

        Returns
        -------
        power2d : (nperp, npar)-array
            The 2D power spectrum.

        coords : list of 2 arrays
            The first is kperp, and the second is kpar.
        """
        # Grid visibilities
        visgrid = self.grid_visibilities(visibilities)

        # Transform frequency axis
        visgrid = self.frequency_fft(visgrid, self.frequencies, taper=self.frequency_taper)

        # Get 2D power from gridded vis.
        power2d = self.get_2d_power(visgrid, [self.uvgrid, self.uvgrid, self.eta])

        # Restrict power to eta modes above eta_min
        power2d = power2d[:, -len(self.eta):]

        return power2d

    def get_2d_power(self, gridded_vis, coords):
        """
        Determine the 2D Power Spectrum of the observation.

        Parameters
        ----------

        gridded_vis : complex (ngrid, ngrid, neta)-array
            The gridded visibilities, fourier-transformed along the frequency axis. Units JyHz.

        coords: list of 3 1D arrays.
            The [u,v,eta] co-ordinates corresponding to the gridded fourier visibilities. u and v in 1/rad, and
            eta in 1/Hz.

        Returns
        -------
        P : float (n_eta, bins)-array
            The cylindrical averaged (or 2D) Power Spectrum, with units Mpc**3.
        """
        # The 3D power spectrum
        power_3d = np.absolute(gridded_vis) ** 2

        weights = self.nbl_uv.copy()
        weights[weights==0] = np.mean(weights[weights!=0]) / 1e-20

        P = angular_average_nd(
            field = power_3d,
            coords = coords,
            bins = self.u_edges, n=2,
            weights=weights,
            bin_ave=False,
            get_variance=False
        )[0]

        # have to make a weight for every eta here..
        # weights = np.tile(np.atleast_3d(weights), (1, 1, len(coords[-1])))
        # weights = angular_average_nd(weights ** 2, coords, bins, n=2, bin_ave=False, get_variance=False, average=False)[
        #     0]

        return P

    def grid_visibilities(self, visibilities):
        """
        Grid a set of visibilities from baselines onto a UV grid.

        Uses simple nearest-neighbour weighting to perform the gridding. This is fast, but not necessarily very
        accurate.

        Parameters
        ----------
        visibilities : complex (n_baselines, n_freq)-array
            The visibilities at each basline and frequency.

        Returns
        -------
        visgrid : (ngrid, ngrid, n_freq)-array
            The visibility grid, in Jy.
        """

        ugrid = np.linspace(-self.uv_max - np.diff((self.baselines[:, 0] * self.frequencies.min() / const.c).value)[0],
                            self.uv_max, self.n_uv + 1)  # +1 because these are bin edges.

        visgrid = np.zeros((self.n_uv, self.n_uv, len(self.frequencies)), dtype=np.complex128)

        for j, f in enumerate(self.frequencies):
            # U,V values change with frequency.
            u = self.baselines[:, 0] * f / const.c
            v = self.baselines[:, 1] * f / const.c

            # Histogram the baselines in each grid but interpolate to find the visibility at the centre
            # TODO: Bella, I don't think this is correct. You don't want to interpolate to the centre, you just
            # want to add the incoherent visibilities together.
            # visgrid[:, :, j] = griddata((u.value , v.value), np.real(visibilities[:,j]),(cgrid_u,cgrid_v),method="linear") + griddata((u.value , v.value), np.imag(visibilities[:,j]),(cgrid_u,cgrid_v),method="linear") *1j

            # So, instead, my version (SGM). One for real, one for complex.
            tmp_rl = np.histogram2d(u.value, v.value, bins=[ugrid, ugrid], weights=visibilities[:, j].real)[0]
            tmp_im = np.histogram2d(u.value, v.value, bins=[ugrid, ugrid], weights=visibilities[:, j].imag)[0]
            visgrid[:, :, j] = tmp_rl + 1j * tmp_im

        # Take the average visibility (divide by weights), being careful not to divide by zero.
        visgrid[self.nbl_uvnu != 0] /= self.nbl_uvnu[self.nbl_uvnu != 0]

        return visgrid

    @cached_property
    def uvgrid(self):
        """
        Centres of the uv grid along a side.
        """
        ugrid = np.linspace(-self.uv_max - np.diff((self.baselines[:, 0] * self.frequencies.min() / const.c).value)[0],
                            self.uv_max, self.n_uv + 1)  # +1 because these are bin edges.
        return (ugrid[1:] + ugrid[:-1]) / 2

    @cached_property
    def uv_max(self):
        if self._uv_max is None:
            return (max([np.abs(b).max() for b in self.baselines]) * self.frequencies.min() / const.c).value
        else:
            return self._uv_max

    @cached_property
    def u_min(self):
        """Minimum of |u| grid"""
        if self._u_min is None:
            return np.abs(self.uvgrid).min()
        else:
            return self._u_min

    @cached_property
    def u_max(self):
        """Maximum of |u| grid"""
        if self._u_max is None:
            return self.uv_max
        else:
            return self._u_max

    @cached_property
    def u_edges(self):
        """Edges of |u| bins"""
        return np.linspace(self.u_min, self.u_max, self.n_ubins + 1)

    @cached_property
    def u(self):
        """Centres of |u| bins"""
        return (self.u_edges[1:] + self.u_edges[:-1])/2

    @cached_property
    def nbl_uvnu(self):
        """The number of baselines in each u,v,nu cell"""

        ugrid = np.linspace(-self.uv_max - np.diff((self.baselines[:, 0] * self.frequencies.min() / const.c).value)[0],
                            self.uv_max, self.n_uv + 1)  # +1 because these are bin edges.
        weights = np.zeros((self.n_uv, self.n_uv, len(self.frequencies)))

        for j, f in enumerate(self.frequencies):
            # U,V values change with frequency.
            u = self.baselines[:, 0] * f / const.c
            v = self.baselines[:, 1] * f / const.c

            # Get number of baselines in each bin
            weights[:, :, j] = np.histogram2d(u.value, v.value, bins=[ugrid, ugrid])[0]

        return weights

    @cached_property
    def nbl_uv(self):
        """
        Effective number of baselines in a uv eta cell.

        See devel/noise_power_derivation for details.
        """
        dnu = self.frequencies[1] - self.frequencies[0]

        sm = dnu ** 2 * self.frequency_taper(len(self.frequencies)) ** 2 / self.nbl_uvnu

        # Some of the cells may have zero baselines, and since they have no variance at all, we set them to zero.
        sm[np.isinf(sm)] = 0

        return 1 / np.sum(sm, axis=-1)

    @cached_property
    def nbl_u(self):
        """
        Effective number of baselines in a |u| annulus.
        """
        return angular_average_nd(
            field=self.nbl_uv,
            coords=[self.uvgrid, self.uvgrid],
            bins=self.u_edges, n=2, bin_ave=False,
            average=False)[0]

    @cached_property
    def eta(self):
        "Grid of positive frequency fourier-modes"
        dnu = self.frequencies[1] - self.frequencies[0]
        eta = fftfreq(len(self.frequencies), d=dnu, b=2 * np.pi)
        return eta[eta > self.eta_min]

    @cached_property
    def grid_weights(self):
        """The number of uv cells that go into a single u annulus (unlrelated to baseline weights)"""
        return angular_average_nd(
            field=np.ones((len(self.uvgrid),) * 2),
            coords=[self.uvgrid, self.uvgrid],
            bins=self.u_edges, n=2, bin_ave=False,
            average=False)[0]

    @cached_property
    def noise_power_expectation(self):
        """The expectation of the power spectrum of thermal noise (same shape as u)"""
        return self._instr_core.thermal_variance_baseline * self.grid_weights / self.nbl_u

    @cached_property
    def noise_power_variance(self):
        """Variance of the noise power spectrum per u bin"""
        return 2 * self._instr_core.thermal_variance_baseline ** 2 * self.grid_weights / self.nbl_u ** 2

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
        ft = fft(vis * taper(len(freq)), (freq.max() - freq.min()), axes=(2,), a=0, b=2 * np.pi)[0]

        ft = ft[:, :, (int(len(freq) / 2) + 1):]
        return ft

    @staticmethod
    def hz_to_mpc(nu_min, nu_max, cosmo):
        """
        Convert a frequency range in Hz to a distance range in Mpc.
        """
        z_max = 1420e6 / nu_min - 1
        z_min = 1420e6 / nu_max - 1

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
