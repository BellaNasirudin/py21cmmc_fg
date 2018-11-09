from py21cmmc.mcmc import CoreLightConeModule, run_mcmc as _run_mcmc
from py21cmmc_fg.core import CoreInstrumental, CorePointSourceForegrounds #, ForegroundsBase
from py21cmmc_fg.likelihood import LikelihoodInstrumental2D
import numpy as np
import os

DEBUG = bool(os.environ.get("DEBUG", False))

import logging
logger = logging.getLogger("21CMMC")

if DEBUG:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.WARNING)

if DEBUG:
    logger.debug("Running in DEBUG mode.")

# ============== SET THESE VARIABLES.
# These should be kept the same between all tests.
freq_min = 150.0
freq_max = 160.0

if DEBUG:
    HII_DIM = 80
else:
    HII_DIM = 250

DIM = 3 * HII_DIM
BOX_LEN = 3 * HII_DIM

z_step_factor = 1.04

# Instrument Options
if DEBUG:
    nfreq = 32
else:
    nfreq = 64

max_bl_length = 300.
sky_size = 3.0 # in sigma
max_tile_n = 50

if DEBUG:
    n_cells = 300
else:
    n_cells = 500

# Likelihood options
if DEBUG:
    n_ubins = 21
else:
    n_ubins = 40

taper = np.blackman


# MCMC OPTIONS
params=dict(  # Parameter dict as described above.
            HII_EFF_FACTOR=[30.0, 10.0, 50.0, 3.0],
            ION_Tvir_MIN=[4.7, 2, 8, 0.1],
        )
# ============== END OF USER-SETTABLE STUFF

z_min = 1420./freq_max - 1
z_max = 1420./freq_min - 1

core_eor = CoreLightConeModule(
    redshift = z_min,              # Lower redshift of the lightcone
    max_redshift = z_max,          # Approximate maximum redshift of the lightcone (will be exceeded).
    user_params = dict(
        HII_DIM = HII_DIM,
        BOX_LEN = BOX_LEN,
        DIM=DIM
    ),
    z_step_factor=z_step_factor,          # How large the steps between evaluated redshifts are (log).
    regenerate=False,
    keep_data_in_memory=DEBUG
)


class CustomCoreInstrument(CoreInstrumental):
    def __init__(self, freq_min=freq_min, freq_max = freq_max, nfreq=nfreq, max_bl_length=max_bl_length,
                 sky_size=sky_size, n_cells = n_cells,
                 **kwargs):
        super().__init__(freq_max=freq_max, freq_min=freq_min, max_bl_length=max_bl_length,
                         nfreq=nfreq, tile_diameter=4.0, integration_time=1200,
                         sky_extent=sky_size, n_cells=n_cells,
                         **kwargs)


class CustomLikelihood(LikelihoodInstrumental2D):
    def __init__(self, n_ubins=21, umax=290, frequency_taper=taper, **kwargs):
        super().__init__(n_ubins=n_ubins, umax=umax, frequency_taper=frequency_taper,
                         simulate=True,
                         **kwargs)

    def store(self, model, storage):
        """Store stuff"""
        storage['signal'] = model[0]['p_signal']

        # Add a "number of sigma" entry
        var = np.array([np.diag(p) for p in self.noise['covariance']])
        storage['sigma'] = (self.data['p_signal'] - self.noise['mean'] - model[0]['p_signal'])/np.sqrt(var)


def run_mcmc(*args, model_name, params=params, **kwargs):
    return _run_mcmc(
        *args,
        datadir='data',  # Directory for all outputs
        model_name=model_name,  # Filename of main chain output
        params=params,
        **kwargs
    )
