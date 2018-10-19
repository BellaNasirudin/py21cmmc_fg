from py21cmmc.mcmc import CoreLightConeModule, run_mcmc as _run_mcmc
from py21cmmc_fg.core import CoreInstrumental #, ForegroundsBase
from py21cmmc_fg.likelihood import LikelihoodInstrumental2D

# ============== SET THESE VARIABLES.
# These should be kept the same between all tests.
freq_min = 150.0
freq_max = 160.0

HII_DIM = 500
DIM = 3*HII_DIM
BOX_LEN = 2 * HII_DIM

z_step_factor = 1.04

# Instrument Options
nfreq = 64
max_bl_length = 300.
sky_size = 3.0 # in sigma
max_tile_n = 50,
n_cells = 300,

# Likelihood options
n_ubins = 21
umax = 290
taper = np.blackman


# MCMC OPTIONS
params=dict(  # Parameter dict as described above.
            HII_EFF_FACTOR=[30.0, 10.0, 50.0, 3.0],
            ION_Tvir_MIN=[4.7, 2, 8, 0.1],
        ),
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
    regenerate=False
)


class CustomCoreInstrument(CoreInstrumental):
    def __init__(self, freq_min=freq_min, freq_max = freq_max, nfreq=nfreq, max_bl_length=max_bl_length,
                 sky_size=sky_size, max_tile_n = max_tile_n, n_cells = n_cells,
                 **kwargs):
        super().__init__(freq_max=freq_max, freq_min=freq_min, max_bl_length=max_bl_length,
                         nfreq=nfreq, tile_diameter=4.0, integration_time=1200,
                         sky_size_coord='sigma', sky_size=sky_size, max_tile_n=max_tile_n, n_cells=n_cells,
                         **kwargs)


class CustomLikelihood(LikelihoodInstrumental2D):
    def __init__(self, n_ubins=21, umax=290, frequency_taper=taper, **kwargs):
        super().__init__(n_uv = None, n_ubins=n_ubins, umax=umax, frequency_taper=frequency_taper,
                         simulate=True,
                         **kwargs)


def run_mcmc(*args, model_name, params=params, **kwargs):
    return _run_mcmc(
        *args,
        datadir='data',  # Directory for all outputs
        model_name=model_name,  # Filename of main chain output
        params=params
        **kwargs
    )
