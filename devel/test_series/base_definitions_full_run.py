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
    
if DEBUG > 3 or DEBUG < 0:
    raise ValueError("DEBUG should be 0,1,2")

if DEBUG==3:
    import tracemalloc
    tracemalloc.start()
    snapshot = tracemalloc.take_snapshot()

    def trace_print():
        global snapshot
        snapshot2 = tracemalloc.take_snapshot()
        snapshot2 = snapshot2.filter_traces((
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
            tracemalloc.Filter(False, tracemalloc.__file__)
        ))

        if snapshot is not None:
            print("================================== Begin Trace:")
            top_stats = snapshot2.compare_to(snapshot, 'lineno', cumulative=True)
            for stat in top_stats[:10]:
                print(stat)
        snapshot = snapshot2

    class CoreLightConeModule(CoreLightConeModule):
        def build_model_data(self, ctx):
            trace_print()
            super().build_model_data(ctx)
            
# Use -c at the end to continue sampling instead of starting again.
CONTINUE = "-c" in sys.argv

logger = logging.getLogger("21CMMC")
logger.setLevel([logging.DEBUG, logging.INFO, logging.WARNING][-DEBUG])

if DEBUG and DEBUG<3:
    logger.debug("Running in DEBUG=%s mode." % DEBUG)
elif DEBUG==3:
    logger.debug("Running in Memory Debug mode")
    

# ============== SET THESE VARIABLES.

# ----- These should be kept the same between all tests. -------
freq_min = 150.0
bandwidth = 10.0
u_min = 10
z_step_factor = 1.04
sky_size = 4.5  # in sigma
max_tile_n = 50
integration_time = 3600000  # 1000 hours of observation time
max_bl_length = 150 if DEBUG else 300

# MCMC OPTIONS
params = dict(  # Parameter dict as described above.
    HII_EFF_FACTOR=[20.0, 10.0, 250.0, 3.0],
    ION_Tvir_MIN=[3.0, 1, 100, 1.0],
    L_X = [40.5, 38, 42, 2.0],
    NU_X_THRESH =[500, 100, 1500, 50],
)

if DEBUG:
    del params['L_X']
    del params['NU_X_THRESH']

astro_params = {k:v[0] for k,v in params.items()}

# ----- Options that differ between DEBUG levels --------
tile_diameter = 4.0 if DEBUG<3 else 12.0 # 
n_obs = 3 if DEBUG < 2 else 1

freq_max = freq_min + n_obs * bandwidth
HII_DIM = [250, 125, 80, 80][DEBUG]
DIM = 3 * HII_DIM
BOX_LEN = 3 * HII_DIM

# Instrument Options
nfreq = 50 * n_obs  if DEBUG else 100 * n_obs
n_cells = 300  if DEBUG else 800

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
    return 0

if DEBUG==1:
    store={
        "lc_slices": _store_lightcone,
        "2DPS": _store_2dps
    }
else:
    store=None
        
core_eor = CoreLightConeModule(
    redshift=z_min,  # Lower redshift of the lightcone
    max_redshift=z_max,  # Approximate maximum redshift of the lightcone (will be exceeded).
    user_params=dict(
        HII_DIM=HII_DIM,
        BOX_LEN=BOX_LEN,
        DIM=DIM
    ),
    astro_params=astro_params,
    z_step_factor=z_step_factor,  # How large the steps between evaluated redshifts are (log).
    regenerate=False,
    keep_data_in_memory=DEBUG,
    store=store,
    change_seed_every_iter=False,
    initial_conditions_seed=42
)


class CustomCoreInstrument(CoreInstrumental):
    def __init__(self, freq_min=freq_min, freq_max=freq_max, nfreq=nfreq,
                 sky_size=sky_size, n_cells=n_cells, tile_diameter=tile_diameter,
                 integration_time=integration_time,max_bl_length = max_bl_length,
                 **kwargs):
        super().__init__(freq_max=freq_max, freq_min=freq_min, n_obs = n_obs,
                         nfreq=nfreq, tile_diameter=tile_diameter, integration_time=integration_time,
                         sky_extent=sky_size, n_cells=n_cells, max_bl_length = max_bl_length,
                         **kwargs)


class CustomLikelihood(LikelihoodInstrumental2D):
    def __init__(self, n_ubins=n_ubins, uv_max=None, nrealisations=[700, 100, 2, 2][DEBUG],
                 simulate=bool(DEBUG),
                 **kwargs):
        super().__init__(n_ubins=n_ubins, uv_max=uv_max, u_min= u_min, n_obs = n_obs,
                         simulate=simulate, nthreads=[7, 3, 1, 1][DEBUG], nrealisations=nrealisations, ps_dim=2,
                         **kwargs)
    def store(self, model, storage):
        """Store stuff"""
        storage['signal'] = model[0]['p_signal'] #+ self.noise['mean']
            
def run_mcmc(*args, model_name, params=params, **kwargs):
    return _run_mcmc(
        *args,
        datadir='data',  # Directory for all outputs
        model_name=model_name,  # Filename of main chain output
        params=params,
        walkersRatio=[6, 3, 2, 2][DEBUG],  # The number of walkers will be walkersRatio*nparams
        burninIterations=0,  # Number of iterations to save as burnin. Recommended to leave as zero.
        sampleIterations=[20, 50, 2, 1][DEBUG],  # Number of iterations to sample, per walker.
        threadCount=[3, 3, 1, 4][DEBUG],  # Number of processes to use in MCMC (best as a factor of walkersRatio)
        continue_sampling=CONTINUE,  # Whether to contine sampling from previous run *up to* sampleIterations.
        **kwargs
    )
