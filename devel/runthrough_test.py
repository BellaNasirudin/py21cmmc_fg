# import sys
# sys.path.insert(0, '../')
from py21cmmc_fg.core import CoreForegrounds, CoreInstrumental
from py21cmmc_fg.likelihood import LikelihoodForeground2D
from py21cmmc.mcmc import run_mcmc
import os

# ====== Manually set parameters for the run =================================
parameters = {"HII_EFF_FACTOR": ['alpha', 30.0, 10.0, 50.0, 3.0]}

datadir = os.path.expanduser("~/Documents/py21cmmc_fg_runs")
model_name = "test"

box_dim = {
    "HII_DIM": 30,
    "BOX_LEN": 50.0
}

flag_options = {
    'redshifts': 7.0
}

fg_core = CoreForegrounds(
    pt_source_params=dict(
        S_min=1e-1,
        S_max=1.0
    ),
    diffuse_params=dict(
        u0=10.0,
        eta = 0.01,
        rho = -2.7,
        mean_temp=253e3,
        kappa=-2.55
    ),
    add_point_sources=True,
    add_diffuse=True
)

instr_core = CoreInstrumental(
    antenna_posfile="grid_centres",
    freq_min=150.0, freq_max=160.0, nfreq=35,
    tile_diameter=4.0,
    max_bl_length=150.0
)
# ============================================================================

filename_of_data = os.path.join(datadir, "data.txt")
box_dim['DIREC'] = datadir

try:
    os.mkdir(datadir)
except:
    pass

lk_fg = LikelihoodForeground2D(filename_of_data, box_dim=box_dim, flag_options=flag_options, n_psbins=20)

if not os.path.exists(filename_of_data):
    lk_fg.simulate_data(fg_core, instr_core, parameters)

run_mcmc(
    redshift=flag_options['redshifts'],
    parameters=parameters,
    datadir=datadir,
    model_name=model_name,
    box_dim=box_dim,
    flag_options=flag_options,
    extra_core_modules=[
        fg_core,
        instr_core
    ],
    likelihood_modules=[lk_fg],
    walkersRatio=2,
    burninIterations=1,
    sampleIterations=1,
    threadCount=1
)
