#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 11:16:04 2018

@author: bella
"""

from py21cmmc.mcmc import analyse
import numpy as np
import matplotlib.pyplot as plt
import h5py

model_name = "runthrough_test"
samples = analyse.get_samples("data/%s" %model_name)

niter = samples.size
nwalkers, nparams = samples.shape

print(niter, nwalkers, nparams)

print(samples.accepted, np.mean(samples.accepted/niter))

analyse.trace_plot(samples, include_lnl=True, start_iter=0, thin=1, colored=False, show_guess=True)

print(samples.param_guess)

analyse.corner_plot(samples)

