#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 14:13:39 2018

@author: bella

Foreground core for 21cmmc

"""
from scipy.integrate import quad
import numpy as np
from astropy import constants as const
from astropy.cosmology import Planck15 as cosmo
from astropy import units as un
from scipy.interpolate import RegularGridInterpolator
#from run_fake21cmmc import ctx

class ForegroundCore:
    
    def __init__(self, S_min, S_max):
        """
        Setting up variables minimum and maximum flux
        """
        #print "Initializing the foreground core"
        
        self.S_min = S_min
        self.S_max = S_max
        
    def setup():
        print("Generating the foregrounds")
    
    def __call__(self, ctx):
        """
        Reading data and applying the foregrounds based on the variables
        """
        print ("Getting the simulation data")
        
        EoR_lightcone = ctx.get("output")
        
        redshifts = ctx.get("redshifts")
        
        boxsize = ctx.get("box_len")
       
        new_lightcone, frequencies, sky_size = self.add_foregrounds(EoR_lightcone, redshifts, boxsize)       
        
        ctx.add("foreground_lightcone", new_lightcone)
        ctx.add("frequencies", frequencies)
        ctx.add("sky_size", sky_size)
        
        
    def add_foregrounds(self, EoR_lightcone, redshifts, boxsize):
        """
        Adding foregrounds in unit of Jy to the EoR signal (lightcone)
        
        Parameters
        ----------
        EoR_lightcone : The EoR signal lightcone outputted by 21cmFAST
        
        redshifts : The redshifts (instead of comoving distance) corresponding to each slice of EoR lightcone
        
        boxsize : The size of the EoR lightcone in Mpc
        
        S_min : The minimum flux in Jy for the point sources
        
        S_max : The maximum flux in Jy for the point sources
        
        Output
        ------
        
        sky : The EoR signal and foreground in unit of Jy/sr
        
        linFrequencies : The linearly-spaced frequencies corresponding to each slice of sky
        
        """
        print ("Adding the foregrounds")        
        ## Number of 2D cells in sky array
        sky_cells = np.shape(EoR_lightcone)[0]
        
        ## Convert the boxsize from Mpc to radian
        ## IF USING SMALL BOX, THIS IS WHERE WE DO STITCHING!!!
        sky_size = 2*np.arctan(boxsize/(2*(cosmo.comoving_transverse_distance([np.mean(redshifts)]).value)))
        
        ## Change the units of brightness temperature from mK to Jy/sr
        EoR_lightcone = np.flip(EoR_lightcone*self.convert_factor_sources(),axis=2)
        
        ## Convert redshifts to frequencies in Hz and generate linearly-spaced frequencies
        frequencies = 1420e6/(1+np.flipud(np.diff(redshifts)/2+redshifts[:-1]))
        
        ### Interpolate linearly in frequency (POSSIBLY IN RADIAN AS WELL)
        linLightcone, linFrequencies = self.interpolate_freqs(EoR_lightcone, frequencies)
        
        ## Generate the point sources foregrounds and 
        foregrounds = np.repeat(self.add_points(self.S_min, self.S_max, sky_cells, sky_size), np.shape(EoR_lightcone)[2], axis=2)
        
        self.add_diffuse()
        
        ## Add the foregrounds and the EoR signal
        sky = foregrounds + linLightcone
        
        return sky, linFrequencies, sky_size
    
    def interpolate_freqs(self,data,frequencies,uv_range=100):
        """
        Interpolate the irregular frequencies so that they are linearly-spaced
        """
        
        linFrequencies = np.linspace(np.min(frequencies), np.max(frequencies), np.shape(data)[2])
        
        ncells = np.shape(data)[0]
        #Create the xy data
        xy = np.linspace(-uv_range,uv_range,ncells)
    
        # generate the interpolation function
        func = RegularGridInterpolator([xy,xy,frequencies],data, bounds_error=False, fill_value=0)
    
        #Create a meshgrid to interpolate the points
        XY,YX,LINZREDS = np.meshgrid(xy,xy,linFrequencies)
    
        #Flatten the arrays so the can be put into pts array
        XY=XY.flatten()
        YX=YX.flatten()
        LINZREDS = LINZREDS.flatten()
    
        #Create the points to interpolate
        numpts = XY.size
        pts = np.zeros([numpts,3])
        pts[:,0],pts[:,1],pts[:,2]=XY,YX,LINZREDS
    
        #Interpolate the points
        interpData = func(pts)
    
        # Reshape the data 
        interpData=interpData.reshape(ncells,ncells,len(linFrequencies))
    
        return interpData, linFrequencies
    
    def add_points(self, S_min, S_max, sky_cells, sky_area):
        
        ## Create a function for source count distribution
        alpha = 4100
        beta = 1.59
        source_count = lambda x : alpha*x**(-beta)
        
        ## Find the mean number of sources
        n_bar = quad(source_count, S_min, S_max) [0]
        
        ## Generate the number of sources following poisson distribution
        N_sources = np.random.poisson(n_bar)
        
        ## Generate the point sources in unit of Jy and position using uniform distribution
        fluxes = ((S_max**(1-beta)-S_min**(1-beta))*np.random.uniform(size=N_sources)+S_min**(1-beta))**(1/(1-beta))
        pos = np.rint(np.random.uniform(0,sky_cells-1,size=(N_sources,2))).astype(int)
        
        ## Create an empty array and fill it up by adding the point sources
        sky = np.zeros((sky_cells,sky_cells,1))        
        for ii in range(N_sources):
            sky[pos[ii,0],pos[ii,1]] += fluxes[ii]
        
        ## Divide by area of each sky cell; Jy/sr
        sky = sky/(sky_area/sky_cells)
        
        return sky
    
    def add_diffuse(self):
        print("Please input diffuse sources")
    
    def convert_factor_sources(self, nu=0):
        
        ## Can either do it with the beam or without the beam (frequency dependent)
        if (nu==0):
            A_eff = 20 * un.m**2
            
            flux_density = (2*1e26*const.k_B*1e-3*un.K/(A_eff*(1*un.Hz)*(1*un.s))).to(un.W/(un.Hz*un.m**2))
            
        else:           
            flux_density =  (2*const.k_B*1e-3*un.K/(((const.c)/(nu.to(1/un.s)))**2)*1e26).to(un.W/(un.Hz*un.m**2))
        
        return flux_density.value
        
        