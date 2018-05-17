#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 13:13:30 2018

@author: bella
"""
import sys
sys.path.insert(1,'../')
from py21cmmc_fg.core import CoreForegrounds, CoreInstrumental
from py21cmmc_fg.likelihood import LikelihoodForeground2D

import numpy as np
from cosmoHammer.ChainContext import ChainContext

import matplotlib.pyplot as plt

fg_core = CoreForegrounds(
    pt_source_params=dict(
        S_min=1e-1,
        S_max=1.0
    ),
    diffuse_params=dict(
        u0=10.0,
        eta = 0.01,
        rho = -2.7,
        mean_temp=253e3,
        kappa=-2.55
    ),
    add_point_sources=True,
    add_diffuse=False,
    redshifts = 1420./np.linspace(150, 160, 30) - 1,
    boxsize=300.0,
    sky_cells = 150
)

instr_core = CoreInstrumental(
    antenna_posfile="grid_centres",
    freq_min=150.0, freq_max=160.0, nfreq=30,
    tile_diameter=4.0,
    max_bl_length=150.0,
    Tsys=0
)

lk = LikelihoodForeground2D(datafile=None, n_psbins=50)

fg_core.setup()
instr_core.setup()

ctx = ChainContext('derp', {"a":1})

fg_core(ctx)
instr_core(ctx)

p, k = lk.computePower(ctx)

plt.imshow(np.log10(p), origin='lower', aspect='auto', extent=(k[0][0], k[0][-1], k[1][0], k[1][-1]))
plt.xscale('log')
plt.yscale('log')
plt.colorbar()

plt.imshow(ctx.get("output").lightcone_box[:,:,0])
plt.colorbar()

p = [0]*30
for i in range(30):
    fg_core(ctx)
    instr_core(ctx)
    p[i] = lk.computePower(ctx)[0].flatten()
    
mean = np.mean(p,axis=0)
cov = np.cov(p)

plt.imshow(np.log10(cov.T))