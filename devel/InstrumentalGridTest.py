import numpy as np

from py21cmmc.mcmc import CoreLightConeModule
from py21cmmc.mcmc import run_mcmc

from py21cmmc_fg.core import  CorePointSourceForegrounds, CoreInstrumental #, ForegroundsBase
from py21cmmc_fg.likelihood import LikelihoodInstrumental2D


model_name = "InstrumentalGridTest"
frequencies = np.linspace(150, 160.0, 200)

core_eor = CoreLightConeModule( # All core modules are prefixed by Core* and end with *Module
    redshift = 7.875,              # Lower redshift of the lightcone
    max_redshift = 8.5,          # Approximate maximum redshift of the lightcone (will be exceeded).
    user_params = dict(       
        HII_DIM = 150,         
        BOX_LEN = 600.0,
        SEED_NUM = 1
    ),
    z_step_factor=1.04,          # How large the steps between evaluated redshifts are (log).
    regenerate=False          
)
    
fg_core = CorePointSourceForegrounds(redshifts = 1420. / frequencies - 1)

core_instr = CoreInstrumental(
    antenna_posfile = 'grid_centres', # use a special grid of *baselines*.
    freq_min = 150.0, # MHz 
    freq_max = 160.0, # MHz
    nfreq = 64, 
    tile_diameter=4.0, 
    max_bl_length=300.0,
    integration_time=1200, 
    Tsys = 0, 
    sky_size = 3, 
    sky_size_coord="sigma", 
    max_tile_n=50,
    n_cells = 300,
    store = {},
)

# Now the likelihood...
likelihood = LikelihoodInstrumental2D(
    n_uv = None, # use underlying n_cells 
    n_ubins=21, 
    umax = 290, 
    frequency_taper=np.blackman, 
    datafile=['data/instrumental_grid_data.npz'],
    simulate = True
)

chain = run_mcmc(
    [core_eor, core_instr], likelihood,
    datadir='data',          # Directory for all outputs
    model_name=model_name,   # Filename of main chain output
    params=dict(             # Parameter dict as described above.
        HII_EFF_FACTOR = [30.0, 10.0, 50.0, 3.0],
        ION_Tvir_MIN = [4.7, 2, 8, 0.1],
    ), 
    walkersRatio=12,         # The number of walkers will be walkersRatio*nparams
    burninIterations=0,      # Number of iterations to save as burnin. Recommended to leave as zero.
    sampleIterations=50,    # Number of iterations to sample, per walker.
    threadCount=6,           # Number of processes to use in MCMC (best as a factor of walkersRatio)
    continue_sampling=False  # Whether to contine sampling from previous run *up to* sampleIterations.
)

