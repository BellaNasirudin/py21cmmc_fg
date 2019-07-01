#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 17:06:33 2019

@author: bella
"""
from base_definitions_full_run import CustomCoreInstrument, CustomLikelihood, core_eor, run_mcmc, DEBUG
from py21cmmc_fg.core import CorePointSourceForegrounds

model_name = "MWATestNoisePSFG-SKA-4params-800cells-3obs"

core_instr = CustomCoreInstrument(
    antenna_posfile = 'mwa_phase2', # use a special grid of *baselines*.
    Tsys = 240, effective_collecting_area = 300.0
)

# Add foregrounds core but set S_min=S_max so essentially no foregrounds so that we can add noise numerically
core_fg = CorePointSourceForegrounds(S_min=50e-6, S_max=50e-3, gamma=0)

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
