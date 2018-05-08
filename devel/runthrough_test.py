from py21cmmc_fg.core import ForegroundCore, CoreInstrumentalSampling
from py21cmmc_fg.likelihood import ForegroundLikelihood
from py21cmmc.mcmc import run_mcmc
import os

# ====== Manually set parameters for the run =================================
parameters = {"HII_EFF_FACTOR": ['alpha', 30.0, 10.0, 50.0, 3.0]}

storage_options = {
    "DATADIR": os.path.expanduser("~/Documents/MCMCData"),
    "KEEP_ALL_DATA": False,
    "KEEP_GLOBAL_DATA": False,
}

box_dim = {
    "HII_DIM": 30,
    "BOX_LEN": 100.0
}

flag_options = {
    'redshifts': 7.0
}

fg_core = ForegroundCore(S_min=1e-1, S_max=1.0)
instr_core = CoreInstrumentalSampling(
    antenna_posfile="grid_centres",
    freq_min=150.0, freq_max=160.0, nfreq=35,
    tile_diameter=4.0,
    max_bl_length=150.0
)
# ============================================================================

filename_of_data = os.path.join(storage_options['DATADIR'], "data.txt")
box_dim['DIREC'] = storage_options['DATADIR']

try:
    os.mkdir(box_dim['DIREC'])
except:
    pass

lk_fg = ForegroundLikelihood(filename_of_data, box_dim=box_dim, flag_options=flag_options, n_psbins=20)

if not os.path.exists(filename_of_data):
    lk_fg.simulate_data(fg_core, instr_core, parameters, niter=3)

run_mcmc(
    redshift=flag_options['redshifts'],
    parameters=parameters,
    storage_options = storage_options,
    box_dim=box_dim,
    flag_options=flag_options,
    extra_core_modules=[
        fg_core,
        instr_core
    ],
    likelihood_modules=[lk_fg],
    walkersRatio=4,
    burninIterations=1,
    sampleIterations=4,
    threadCount=1
)
