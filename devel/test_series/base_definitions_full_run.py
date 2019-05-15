#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 13:10:08 2019

@author: bella
"""

import logging
import os
import sys

import numpy as np
from powerbox.dft import fft
from powerbox.tools import angular_average_nd
from py21cmmc.mcmc import CoreLightConeModule, run_mcmc as _run_mcmc

from py21cmmc_fg.core import CoreInstrumental
from py21cmmc_fg.likelihood import LikelihoodInstrumental2D

DEBUG = int(os.environ.get("DEBUG", 0))

if DEBUG > 2 or DEBUG < 0:
    raise ValueError("DEBUG should be 0,1,2")

# Use -c at the end to continue sampling instead of starting again.
CONTINUE = "-c" in sys.argv

logger = logging.getLogger("21CMMC")
logger.setLevel([logging.DEBUG, logging.INFO, logging.WARNING][-DEBUG])

if DEBUG:
    logger.debug("Running in DEBUG=%s mode." % DEBUG)

# ============== SET THESE VARIABLES.


# ----- These should be kept the same between all tests. -------
freq_min = 150.0
freq_max = 180.0
n_obs = 3
u_min = 10
z_step_factor = 1.04
sky_size = 4.5  # in sigma
max_tile_n = 50
integration_time = 3600000  # 1000 hours of observation time
tile_diameter = 4.0
max_bl_length = 250 if DEBUG else 300

# MCMC OPTIONS
params = dict(  # Parameter dict as described above.
    HII_EFF_FACTOR=[20.0, 10.0, 250.0, 3.0],
    ION_Tvir_MIN=[3.0, 1, 100, 0.1],
    L_X = [40.5, 38, 42, 0.1],
    NU_X_THRESH =[500, 100, 1500, 50],
)

# ----- Options that differ between DEBUG levels --------
HII_DIM = [250, 125, 80][DEBUG]
DIM = 3 * HII_DIM
BOX_LEN = 3 * HII_DIM

# Instrument Options
nfreq = 50 * n_obs  if DEBUG else 100 * n_obs
n_cells = 300  if DEBUG else 1000

# Likelihood options
if DEBUG == 2:
    n_ubins = 30
else:
    n_ubins = 60

# ============== END OF USER-SETTABLE STUFF

z_min = 1420. / freq_max - 1
z_max = 1420. / freq_min - 1

def _store_lightcone(ctx):
    """A storage function for lightcone slices"""
    return ctx.get("lightcone").brightness_temp[0]


def _store_2dps(ctx):
    print("STORING 2DPS!!")
    lc = ctx.get('lightcone')
    p, k = fft(lc.brightness_temp, L=lc.lightcone_dimensions)
    p = np.abs(p) ** 2

    p = angular_average_nd(p, coords=k, n=2, bin_ave=False, bins=21)[0]

    return p

core_eor = CoreLightConeModule(
    redshift=z_min,  # Lower redshift of the lightcone
    max_redshift=z_max,  # Approximate maximum redshift of the lightcone (will be exceeded).
    user_params=dict(
        HII_DIM=HII_DIM,
        BOX_LEN=BOX_LEN,
        DIM=DIM
    ),
    z_step_factor=z_step_factor,  # How large the steps between evaluated redshifts are (log).
    regenerate=False,
    keep_data_in_memory=DEBUG,
    store={
        "lc_slices": _store_lightcone,
        "2DPS": _store_2dps
    },
    change_seed_every_iter=False,
    initial_conditions_seed=42
)


class CustomCoreInstrument(CoreInstrumental):
    def __init__(self, freq_min=freq_min, freq_max=freq_max, nfreq=nfreq,
                 sky_size=sky_size, n_cells=n_cells, tile_diameter=tile_diameter,
                 integration_time=integration_time,max_bl_length = max_bl_length,
                 **kwargs):
        super().__init__(freq_max=freq_max, freq_min=freq_min,
                         nfreq=nfreq, tile_diameter=tile_diameter, integration_time=integration_time,
                         sky_extent=sky_size, n_cells=n_cells, max_bl_length = max_bl_length,
                         **kwargs)


class CustomLikelihood(LikelihoodInstrumental2D):
    def __init__(self, n_ubins=n_ubins, uv_max=None, nrealisations=[1000, 100, 2][DEBUG],
                 **kwargs):
        super().__init__(n_ubins=n_ubins, uv_max=uv_max, u_min= u_min, n_obs = n_obs,
                         simulate=True, nthreads=[10, 3, 1][DEBUG], nrealisations=nrealisations, ps_dim=2,
                         **kwargs)
    def store(self, model, storage):
        """Store stuff"""
        storage['signal'] = model[0]['p_signal'] + self.noise['mean']
        
#        # Remember that the variance is actually the variance plus the model uncertainty
#        sig_cov = self.get_cosmic_variance(model[0]['p_signal'])
#        
#        # Add a "number of sigma" entry only if cov is not zero
#        if (self.noise['covariance'] == 0) or (self.noise['covariance'] is None):
#            var = 0
#        else:
#            var = []
#            for ii in range(len(sig_cov)):
#                print(np.shape(self.noise['covariance'][ii]), np.shape(sig_cov[ii]))
#                var.extend(([np.diag(p) + np.diag(s) for p,s in zip(self.noise['covariance'][ii], sig_cov[ii])]))
#                        
#        storage['sigma'] = (self.data['p_signal'] - model[0]['p_signal'])/np.sqrt(np.array(var))
        
def run_mcmc(*args, model_name, params=params, **kwargs):
    return _run_mcmc(
        *args,
        datadir='data',  # Directory for all outputs
        model_name=model_name,  # Filename of main chain output
        params=params,
        walkersRatio=[10, 3, 2][DEBUG],  # The number of walkers will be walkersRatio*nparams
        burninIterations=0,  # Number of iterations to save as burnin. Recommended to leave as zero.
        sampleIterations=[100, 50, 2][DEBUG],  # Number of iterations to sample, per walker.
        threadCount=[10, 3, 1][DEBUG],  # Number of processes to use in MCMC (best as a factor of walkersRatio)
        continue_sampling=CONTINUE,  # Whether to contine sampling from previous run *up to* sampleIterations.
        **kwargs
    )
