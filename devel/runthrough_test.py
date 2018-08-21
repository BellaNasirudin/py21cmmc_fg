from py21cmmc_fg.likelihood import Likelihood2D
from py21cmmc_fg.core import CorePointSourceForegrounds

from py21cmmc.mcmc.mcmc import run_mcmc
from py21cmmc.mcmc.core import CoreLightConeModule

model_name = "runthrough_test"

lc_core = CoreLightConeModule(
    redshift=7.0,
    max_redshift=8.0,
    user_params=dict(
        HII_DIM=50,
        BOX_LEN=200.0
    ),
    regenerate=False
)

fg_core = CorePointSourceForegrounds()

likelihood = Likelihood2D(
    datafile = "data/runthrough_test",
)

chain = run_mcmc(
    [lc_core, fg_core], likelihood,
    datadir='data',
    model_name=model_name,
    params = dict(
        HII_EFF_FACTOR=[30.0, 10., 50.0, 3.0],
        ION_Tvir_MIN = [4.7, 2, 8, 0.1]
    ),
    walkersRatio = 2,
    burninIterations=0,
    sampleIterations=2,
    threadCount=6,
    continue_sampling=False
)
