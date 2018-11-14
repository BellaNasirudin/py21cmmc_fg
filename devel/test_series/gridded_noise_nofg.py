"""
The second test in a series of tests to prove that this code works.

Here are the tests:

1. Gridded baselines, no thermal noise, no foregrounds
2. ** Gridded baselines, thermal noise, no foregrounds
3. MWA baselines, thermal noise, no foregrounds
4. Gridded baselines, thermal noise, point-source foregrounds
5. MWA baselines, thermal noise, point-source foregrounds

"""
from base_definitions import CustomCoreInstrument, CustomLikelihood, core_eor, run_mcmc, DEBUG

model_name = "InstrumentalGridTestNoise"

core_instr = CustomCoreInstrument(
    antenna_posfile = 'grid_centres', # use a special grid of *baselines*.
    Tsys=240,
)

likelihood = CustomLikelihood(
    datafile=[f'data/{model_name}.npz'],
    noisefile=[f'data/{model_name}.noise.npz'],
)

chain = run_mcmc(
    [core_eor, core_instr], likelihood,
    model_name=model_name,             # Filename of main chain output
)

