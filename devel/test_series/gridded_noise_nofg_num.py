"""
The second test (b)in a series of tests to prove that this code works.

Here are the tests:

1. Gridded baselines, no thermal noise, no foregrounds
2. ** Gridded baselines, thermal noise, no foregrounds
2b. ** Gridded baselines, thermal noise, no foregrounds (numerically calculated noise).
3. MWA baselines, thermal noise, no foregrounds
4. Gridded baselines, thermal noise, point-source foregrounds
5. MWA baselines, thermal noise, point-source foregrounds

"""
from base_definitions import CustomCoreInstrument, CustomLikelihood, core_eor, run_mcmc, DEBUG
from py21cmmc_fg.core import ForegroundsBase
import numpy as np

model_name = "InstrumentalGridTestNoiseNumerical"

class NoFG(ForegroundsBase):
    def build_sky(self):
        return np.zeros((self.n_cells, self.n_cells, len(self.frequencies)))

nofg = NoFG() #This is used *only* to trigger the numerical covariance. There should be a better way of doing this.


core_instr = CustomCoreInstrument(
    antenna_posfile = 'grid_centres', # use a special grid of *baselines*.
    Tsys=240,
)

likelihood = CustomLikelihood(
    datafile=[f'data/{model_name}.npz'],
    noisefile=[f'data/{model_name}.noise.npz'],
)

if __name__ == "__main__":
    chain = run_mcmc(
        [core_eor, nofg, core_instr], likelihood,
        model_name=model_name,             # Filename of main chain output
    )

