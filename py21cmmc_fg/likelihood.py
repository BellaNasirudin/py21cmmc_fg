"""
Created on Fri Apr 20 16:54:22 2018

@author: bella
"""

import logging
import multiprocessing
from functools import partial

import numpy as np
from astropy import constants as const
from astropy import units as un
from cached_property import cached_property
from powerbox.dft import fft, fftfreq
from powerbox.tools import angular_average_nd
from py21cmmc.mcmc.core import CoreLightConeModule
from py21cmmc.mcmc.likelihood import LikelihoodBaseFile
from scipy.integrate import quad
from scipy.special import erf
from scipy import signal
from .core import CoreInstrumental, ForegroundsBase
from .util import lognormpdf

logger = logging.getLogger("21CMMC")


class LikelihoodInstrumental2D(LikelihoodBaseFile):
    required_cores = [CoreInstrumental]

    def __init__(self, n_uv=None, n_ubins=30, uv_max=None, u_min=None, u_max=None, frequency_taper=np.blackman, n_obs=1,
                 nrealisations=100, nthreads=1, model_uncertainty=0.15, eta_min=0, use_analytical_noise=False, ps_dim=2,
                 **kwargs):
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

        nthreads : int, optional
            Number of processes to use if generating realisations for numerical covariance.

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
        self._nthreads = nthreads
        self.ps_dim = ps_dim
        self.n_obs = n_obs

        self.use_analytical_noise = use_analytical_noise
        
        self.kernel_weights = None # set this as None so we only do this once

    def setup(self):
        super().setup()

        # we can unpack data now because we know it's always a list of length 1.
        if self.data:
            self.data = self.data[0]
        if self.noise:
            self.noise = self.noise[0]

        #Store only the p_signal
        self.data = {"p_signal":self.data["p_signal"]}

    @cached_property
    def n_uv(self):
        """The number of cells on a side of the (square) UV grid"""
        if self._n_uv is None:
            return self._instr_core.n_cells
        else:
            return self._n_uv

    def reduce_data(self, ctx):
        """
        Simulate datasets to which this very class instance should be compared.
        """
        self.baselines_type = ctx.get("baselines_type")
        visibilities = ctx.get("visibilities")
        p_signal = self.compute_power(visibilities)

        # Remember that the results of "simulate" can be used in two places: (i) the computeLikelihood method, and (ii)
        # as data saved to file. In case of the latter, it is useful to save extra variables to the dictionary to be
        # looked at for diagnosis, even though they are not required in computeLikelihood().
        return [dict(p_signal=p_signal, baselines=self.baselines, frequencies=self.frequencies,
                     u=self.u, eta=self.eta)]
                     #, nbl_uv=self.nbl_uv, nbl_uvnu=self.nbl_uvnu, nbl_u=self.nbl_u, grid_weights=self.grid_weights)]

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
                    nrealisations=self.nrealisations, nthreads=self._nthreads
                )
            else:
                # Still getting mean numerically for now...
                mean = self.numerical_covariance(nrealisations=self.nrealisations, nthreads=self._nthreads)[0]

                covariance = self.analytical_covariance(self.u, self.eta,
                                                        np.median(self.frequencies),
                                                        self.frequencies.max() - self.frequencies.min())

                thermal_covariance = self.get_thermal_covariance()
                covariance = [x + y for x, y in zip(covariance, thermal_covariance)]

        else:
            # Only need thermal variance if we don't have foregrounds, otherwise it will be embedded in the
            # above foreground covariance... BUT NOT IF THE FOREGROUND COVARIANCE IS ANALYTIC!!
            #                covariance = self.get_thermal_covariance()
            #                mean = np.repeat(self.noise_power_expectation, len(self.eta)).reshape((len(self.u), len(self.eta)))
            mean = 0
            covariance = 0

        return [{"mean": mean, "covariance": covariance}]

    def computeLikelihood(self, model):
        "Compute the likelihood"
        # remember that model is *exactly* the result of reduce_data(), which is a  *list* of dicts, so unpack
        model = model[0]
        
        lnl = 0
        
        for ii in range(self.n_obs):
            total_model = model['p_signal'][ii]
            
            if self.ps_dim == 2:
                sig_cov = self.get_signal_covariance(model['p_signal'][ii]) # TODO: CHANGE THIS TO COSMIC VARIANCE
    
                if self.foreground_cores:
                    # Normal case (foreground parameters are not being updated, or there are no foregrounds)
                    total_model += self.noise["mean"][ii]
                    total_cov = self.noise['covariance'][ii]#[x + y for x, y in zip(self.noise['covariance'], sig_cov)]
                else:
                    total_cov = sig_cov
                
                lnl += lognormpdf(self.data['p_signal'][ii], total_model, total_cov)
            else:
                lnl += -0.5 * np.sum(
                    (self.data['p_signal'][ii] - total_model) ** 2 / (self.model_uncertainty * model['p_signal'][ii]) ** 2)
        
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
        if self.ps_dim != 1:
            cov = []
            for sig_eta in signal_power:
                cov.append((self.model_uncertainty * np.diag(sig_eta)) ** 2)

            return cov
        else:
            return 0

    def numerical_covariance(self, params={}, nrealisations=200, nthreads=1):
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

        if nrealisations < 2:
            raise ValueError("nrealisations must be more than one")

        # We use a hack where we define an external function which *passed*
        # this object just so that we can do multiprocessing on it.
        fnc = partial(_produce_mock, self, params)

        pool = multiprocessing.Pool(nthreads)
        
        power = pool.map(fnc, np.arange(nrealisations))
        
        # Note, this covariance *already* has thermal noise built in.
        cov = []
        mean = []
        
        for ii in range(self.n_obs):
            mean.append(np.mean(np.array(power)[:,ii,:,:], axis=0))

            if self.ps_dim == 2:
                cov.append([np.cov(x) for x in np.array(power)[:,ii,:,:].transpose((1, 2, 0))])
            else:
                cov = np.var(np.array(power)[:,ii,:,:], axis=0)
            
        pool.close()
        pool.join()

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
        # Grid visibilities only if we're not using "grid_centres"

        if self.baselines_type != "grid_centres":
            visgrid = self.grid_visibilities(visibilities)
        else:
            visgrid = visibilities
            
        # Transform frequency axis
        visgrid = self.frequency_fft(visgrid, self.frequencies, self.ps_dim, taper=signal.blackmanharris, n_obs = self.n_obs)#self.frequency_taper)

        # Get 2D power from gridded vis.
        power2d = self.get_power(visgrid, ps_dim=self.ps_dim)

        return power2d

    def get_power(self, gridded_vis, ps_dim=2):
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
        PS : float (n_obs, n_eta, bins)-list
            The cylindrical averaged (or 2D) Power Spectrum, with units JyHz**2.
        """
        logger.info("Calculating the power spectrum")
        PS = []
        for vis in gridded_vis:
            # The 3D power spectrum
            power_3d = np.absolute(vis) ** 2
    
            if ps_dim == 2:
                P = angular_average_nd(
                    field=power_3d,
                    coords=[self.uvgrid, self.uvgrid, self.eta],
                    bins=self.u_edges, n=ps_dim,
                    weights=np.sum(self.kernel_weights, axis=2),  # weights,
                    bin_ave=False,
                )[0][:, int(len(self.frequencies) / (2 * self.n_obs)):]  # return the positive part
    
            elif ps_dim == 1:
    
                P = angular_average_nd(
                    field=power_3d,
                    coords=[self.uvgrid, self.uvgrid, self.eta],
                    bins=self.u_edges,
                    weights=self.kernel_weights,
                    bin_ave=False,
                )[0]
                
            P[np.isnan(P)] = 0
            PS.append(P)
        
        self.eta = self.eta[self.eta > self.eta_min]

        return PS

    def fourierBeam(self, centres, u_bl, v_bl, frequency, min_attenuation = 1e-5, N = 20):
        """
        Find the Fourier Transform of the Gaussian beam
        
        Parameter
        ---------
        centres : (ngrid)-array
            The centres of the grid.
        
        u_bl : (n_baselines)-array
            The list of baselines in m.
            
        v_bl : (n_baselines)-array
            The list of baselines in m.
            
        frequency: float
            The frequency in Hz.
        """
    
        a = 1/ (2 * self._instr_core.sigma(frequency)**2)
        
        indx_u = np.digitize(u_bl, centres)
        indx_v = np.digitize(v_bl, centres)
        
        beam = []
        
        for jj in range(len(u_bl)):
            x, y = np.meshgrid(centres[indx_u[jj]-int(N/2):indx_u[jj]+int(N/2)], centres[indx_v[jj]-int(N/2):indx_v[jj]+int(N/2)])
            B = (np.exp(- ((x - u_bl[jj])**2 + (y - v_bl[jj])**2 )/ a)).T
            B[B<min_attenuation] = 0
            beam.append(B)
        
        indx_u+= -int(N/2)
        indx_v+= -int(N/2)
        
        indx_u[indx_u<0] = 0
        indx_v[indx_v<0] = 0         
                   
        return beam, indx_u, indx_v

    def grid_visibilities(self, visibilities, N = 50):
        """
        Grid a set of visibilities from baselines onto a UV grid.

        Uses Fourier (Gaussian) beam weighting to perform the gridding.
        
        Parameters
        ----------
        visibilities : complex (n_baselines, n_freq)-array
            The visibilities at each basline and frequency.

        Returns
        -------
        visgrid : (ngrid, ngrid, n_freq)-array
            The visibility grid, in Jy.
        """
        logger.info("Gridding the visibilities")
        
        ugrid = np.linspace(-self.uv_max, self.uv_max, self.n_uv +1 )  # +1 because these are bin edges.
        
        centres = (ugrid[1:] + ugrid[:-1]) / 2
        
        visgrid = np.zeros((self.n_uv, self.n_uv, len(self.frequencies)), dtype=np.complex128)
        
        if self.kernel_weights is None:
            weights = np.zeros((self.n_uv, self.n_uv, len(self.frequencies)))
        
        for jj, freq in enumerate(self.frequencies):
            u_bl = (self.baselines[:,0] * freq / const.c).value
            v_bl = (self.baselines[:,1] * freq / const.c).value
            
            beam, indx_u, indx_v = self.fourierBeam(centres, u_bl, v_bl, freq, N=N)
            
            for kk in range(len(indx_u)):
                visgrid[indx_u[kk]:indx_u[kk]+np.shape(beam[kk])[0], indx_v[kk]:indx_v[kk]+np.shape(beam[kk])[1], jj] += beam[kk] * visibilities[kk,jj]
           
                if self.kernel_weights is None:
                    weights[indx_u[kk]:indx_u[kk]+np.shape(beam[kk])[0], indx_v[kk]:indx_v[kk]+np.shape(beam[kk])[1], jj] += beam[kk] / np.sum(beam[kk])

        if self.kernel_weights is None:
            self.kernel_weights = weights
            
        visgrid[self.kernel_weights!=0] /= self.kernel_weights[self.kernel_weights!=0]
        
        return visgrid

    @cached_property
    def uvgrid(self):
        """
        Centres of the uv grid along a side.
        """
        if self.baselines_type != "grid_centres":
            ugrid = np.linspace(-self.uv_max, self.uv_max, self.n_uv + 1)  # +1 because these are bin edges.
            return (ugrid[1:] + ugrid[:-1]) / 2
        else:
            # return the uv
            return self.baselines

    @cached_property
    def uv_max(self):
        if self._uv_max is None:
            if self.baselines_type != "grid_centres":
                return (max([np.abs(b).max() for b in self.baselines]) * self.frequencies.min() / const.c).value
            else:
                # return the uv
                return self.baselines.max()
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
        """Edges of |u| bins where |u| = sqrt(u**2+v**2)"""
        return np.linspace(self.u_min, self.u_max, self.n_ubins + 1)

    @cached_property
    def u(self):
        """Centres of |u| bins"""
        return (self.u_edges[1:] + self.u_edges[:-1]) / 2

    @cached_property
    def nbl_uvnu(self):
        """The number of baselines in each u,v,nu cell"""

        if self.baselines_type != "grid_centres":
            ugrid = np.linspace(-self.uv_max, self.uv_max, self.n_uv + 1)  # +1 because these are bin edges.
            weights = np.zeros((self.n_uv, self.n_uv, len(self.frequencies)))

            for j, f in enumerate(self.frequencies):
                # U,V values change with frequency.
                u = self.baselines[:, 0] * f / const.c
                v = self.baselines[:, 1] * f / const.c

                # Get number of baselines in each bin
                weights[:, :, j] = np.histogram2d(u.value, v.value, bins=[ugrid, ugrid])[0]
        else:
            weights = np.ones((self.n_uv, self.n_uv, len(self.frequencies)))

        return weights

    @cached_property
    def nbl_uv(self):
        """
        Effective number of baselines in a uv eta cell.

        See devel/noise_power_derivation for details.
        """
        dnu = self.frequencies[1] - self.frequencies[0]

        sm = dnu ** 2 / self.nbl_uvnu

        # Some of the cells may have zero baselines, and since they have no variance at all, we set them to zero.
        sm[np.isinf(sm)] = 0

        nbl_uv = 1 / np.sum(sm, axis=-1)
        nbl_uv[np.isinf(nbl_uv)] = 0

        return nbl_uv

    @cached_property
    def nbl_u(self):
        """
        Effective number of baselines in a |u| annulus.
        """
        if self.ps_dim == 2:
            return angular_average_nd(
                field=self.nbl_uv,
                coords=[self.uvgrid, self.uvgrid],
                bins=self.u_edges, n=2,
                bin_ave=False,
                average=False
            )[0]
        else:
            return None

    @cached_property
    def eta(self):
        "Grid of positive frequency fourier-modes"
        dnu = self.frequencies[1] - self.frequencies[0]
        eta = fftfreq(int(len(self.frequencies) / self.n_obs), d=dnu, b=2 * np.pi)
        return eta

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
        return self._instr_core.thermal_variance_baseline ** 2 * self.grid_weights / self.nbl_u ** 2

    @staticmethod
    def frequency_fft(vis, freq, dim, taper=np.ones_like, n_obs =1):
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
        
        n_obs : int, optional
            Number of observations used to separate the visibilities into different bandwidths.

        Returns
        -------
        ft : (ncells, ncells, nfreq/2)-array
            The fourier-transformed signal, with negative eta removed.

        eta : (nfreq/2)-array
            The eta-coordinates, without negative values.
        """
        ft = []
        W = (freq.max() - freq.min()) / n_obs
        L = int(len(freq) / n_obs)
        
        for ii in range(n_obs):
            ft.append(fft(np.fft.ifftshift(vis[:,:,ii*L:(ii+1)*L] * taper(L), axes=(2,)), W, axes=(2,), a=0, b=2 * np.pi)[0])
        
        ft = np.array(ft)
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


def _produce_mock(self, params, i):
    """Produces a mock power spectrum for purposes of getting numerical_covariances"""
    # Create an empty context with the given parameters.
    np.random.seed(i)
    ctx = self.chain.createChainContext(params)

    # For each realisation, run every foreground core (not the signal!)
    for core in self.foreground_cores:
        core.simulate_mock(ctx)

    # And turn them into visibilities
    self._instr_core.simulate_mock(ctx)

    # And compute the power
    power = self.compute_power(ctx.get("visibilities"))

    return power