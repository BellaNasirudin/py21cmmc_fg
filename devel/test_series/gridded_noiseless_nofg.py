"""
The first test in a series of tests to prove that this code works.

Here are the tests:

1. ** Gridded baselines, no thermal noise, no foregrounds
2. Gridded baselines, thermal noise, no foregrounds
3. MWA baselines, thermal noise, no foregrounds
4. Gridded baselines, thermal noise, point-source foregrounds
5. MWA baselines, thermal noise, point-source foregrounds

"""


from .base_definitions import CustomCoreInstrument, CustomLikelihood, core_eor, run_mcmc

model_name = "InstrumentalGridTest"

core_instr = CustomCoreInstrument(
    antenna_posfile = 'grid_centres', # use a special grid of *baselines*.
    Tsys = 0,
)

likelihood = CustomLikelihood(
    datafile=[f'data/{model_name}.npz']
)


chain = run_mcmc(
    [core_eor, core_instr], likelihood,
    model_name=model_name,   # Filename of main chain output
    walkersRatio=16,         # The number of walkers will be walkersRatio*nparams
    burninIterations=0,      # Number of iterations to save as burnin. Recommended to leave as zero.
    sampleIterations=25,    # Number of iterations to sample, per walker.
    threadCount=6,           # Number of processes to use in MCMC (best as a factor of walkersRatio)
    continue_sampling=False  # Whether to contine sampling from previous run *up to* sampleIterations.
)

