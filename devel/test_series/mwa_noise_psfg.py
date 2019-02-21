#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 15:59:34 2019

@author: bella
"""

"""
The fifth test in a series of tests to prove that this code works.

Here are the tests:

1. Gridded baselines, no thermal noise, no foregrounds
2. Gridded baselines, thermal noise, no foregrounds
3. MWA baselines, thermal noise, no foregrounds
4. Gridded baselines, thermal noise, point-source foregrounds
5. **MWA baselines, thermal noise, point-source foregrounds

"""

from base_definitions import CustomCoreInstrument, CustomLikelihood, core_eor, run_mcmc, DEBUG
from py21cmmc_fg.core import CorePointSourceForegrounds

model_name = "MWATestNoisePSFG-SKA"

core_instr = CustomCoreInstrument(
    antenna_posfile = 'mwa_phase2', # use a special grid of *baselines*.
    Tsys = 240, effective_collecting_area = 300.0
)

# Add foregrounds core but set S_min=S_max so essentially no foregrounds so that we can add noise numerically
core_fg = CorePointSourceForegrounds(S_min=1e-4, S_max=1, gamma=0)

likelihood = CustomLikelihood(
    datafile=[f'data/{model_name}.npz'],
    noisefile=[f'data/{model_name}.noise.npz'],
    use_analytical_noise=False
)

if __name__ == "__main__":
    chain = run_mcmc(
        [core_eor, core_fg, core_instr], likelihood,
        model_name=model_name,             # Filename of main chain output
    )

