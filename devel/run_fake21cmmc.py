#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 15:01:35 2018

@author: bella
"""
import sys
import numpy as np

sys.path.insert(0, '../../EoR_21cmFAST_model/Codes')
from read_21cmFAST import  load_binary_data

from Cores_py21cmmc import ForegroundCore
from Likelihood_py21cmmc import ForegroundLikelihood

class ctx:
    
    def __init__(self):
        lightcone_dir = '../../../../../../../data/phd/Runs/delta_T_v3_no_halos__zstart006.00000_zend011.41539_FLIPBOXES0_300_1500Mpc_lighttravel'
        self.lightcone = load_binary_data(lightcone_dir, 300)
        
        self.boxsize = 1500
        
        redshift_dir = '../../../../../../../data/phd/Runs/zlistInterp_1500Mpc300.txt'
        redshifts = np.genfromtxt(redshift_dir, delimiter=',')[:300+1]
#        redshifts = (np.diff(redshifts)/2+redshifts[:-1])
        
        self.redshifts = redshifts
        
        
    def get(self,str):

        if(str=="lightcone"):
            return self.lightcone
        
        if(str=="boxsize"):
            return self.boxsize
        
        if(str=="redshifts"):
            return self.redshifts
        
        if(str=="foreground_lightcone"):
            return self.foreground_lightcone
        
        if (str=="frequencies"):
            return self.frequencies
        
        if (str=="observed_power"):
            return self.observed_power

        if (str=="sky_size"):
            return self.sky_size
            
    def add(self,str,data):
        
        if(str=="foreground_lightcone"):
            self.foreground_lightcone = data
        
        if (str=="frequencies"):
            self.frequencies = data
            
        if (str=="observed_power"):
            self.observed_power = data

        if (str=="sky_size"):
            self.sky_size = data

stuff = ctx()
fg = ForegroundCore(0.5,1)
fg(stuff, 0.5,1)

likelihood = ForegroundLikelihood(stuff)
