import numpy as np
from py21cmmc_fg.likelihood import LikelihoodInstrumental2D
from py21cmmc_fg.core import CorePointSourceForegrounds, CoreInstrumental

from py21cmmc.mcmc.mcmc import run_mcmc
from py21cmmc.mcmc.core import CoreLightConeModule

model_name = "runthrough_test"
frequencies = np.linspace(150, 160.0, 200)

lc_core = CoreLightConeModule(
    redshift=6.0,
    max_redshift=9.0,
    user_params=dict(
        HII_DIM=125,
        BOX_LEN=500.0
    ),
    regenerate=False
)

fg_core = CorePointSourceForegrounds(redshifts = 1420. / frequencies - 1)

instr_core = CoreInstrumental("mwa_phase2", 150, 160, 200)


likelihood = LikelihoodInstrumental2D(
    datafile = "data/runthrough_test",
)
chain = run_mcmc(
    [lc_core, instr_core, fg_core], likelihood,
    datadir='data',
    model_name=model_name,
    params = dict(
        HII_EFF_FACTOR=[30.0, 10., 50.0, 3.0],
        ION_Tvir_MIN = [4.7, 2, 8, 0.1]
    ),
    walkersRatio = 3,
    burninIterations=0,
    sampleIterations=2,
    threadCount=3,
    continue_sampling=False
)
