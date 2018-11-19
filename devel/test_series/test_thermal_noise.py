from gridded_noise_nofg import core_eor, core_instr, likelihood, model_name
from py21cmmc.mcmc.mcmc import build_computation_chain
import pickle
import numpy as np

nrealisations = 100

# Build the chain and run setup()
chain = build_computation_chain([core_eor, core_instr], likelihood)

num_mean, num_cov = likelihood.numerical_covariance(
    nrealisations=nrealisations, cov =1 # really unsure about reason for this.
)

num_var = np.array([np.diag(c) for c in num_cov])

anl_mean, anl_cov = likelihood.noise['mean'], likelihood.noise['covariance']

anl_var = np.array([np.diag(c) for c in anl_cov])

with open("thermal_noise_data.pkl", 'wb') as f:
    pickle.dump({"num_mean":num_mean, "num_var":num_var, "anl_mean":anl_mean, "anl_cov":anl_var}, f)

