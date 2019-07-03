#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: steven
"""
from base_definitions_full_run import CustomCoreInstrument, CustomLikelihood, core_eor, run_mcmc, DEBUG

model_name = "memory_testing"

core_instr = CustomCoreInstrument(
    antenna_posfile = 'mwa_phase2_reduced',
    Tsys = 240,
    effective_collecting_area = 300.0
)

print(core_instr.antenna_posfile)

likelihood = CustomLikelihood(
    datafile=[f'data/{model_name}.npz'],
    noisefile=[f'data/{model_name}.noise.npz'],
    use_analytical_noise=False,
    simulate=True
)

if __name__ == "__main__":
    chain = run_mcmc(
        [core_eor, core_instr], likelihood,
        model_name=model_name,             # Filename of main chain output
    )
