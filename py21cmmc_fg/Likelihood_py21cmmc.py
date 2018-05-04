#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 16:54:22 2018

@author: bella
"""

#import LikelihoodBase
from powerbox.dft import fft
from powerbox.tools import angular_average_nd
import numpy as np
from astropy import constants as const
from astropy.cosmology import Planck15 as cosmo
from astropy import units as un
import itertools as it
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import chisquare
from py21cmmc.likelihood import LikelihoodBase, Core21cmFastModule
from cosmoHammer.ChainContext import ChainContext
    

class ForegroundLikelihood(LikelihoodBase):
    
    def __init__(self, datafile, **kwargs): ##not supposed to be here, delete when combined with 21cmmc
        
        super().__init__(**kwargs)
        self.datafile = datafile
#        PS_mK2Mpc3, chi_sq = self.compute_likelihood(ctx)
        
#        ctx.add("observed_power", PS_mK2Mpc3)
#        print PS_mK2Mpc3
    
        
    def setup(self):
        print("read in data")
        data = np.genfromtxt(self.datafile)

        self.k = data[:,0]
        self.power = data[:,1]
        self.uncertainty = data[:,2]

    def computeLikelihood(self, ctx):
        
        PS_mK2Mpc3, k_Mpc = self.computePower(ctx)

        # ctx.add("power_spectrum", PS_mK2Mpc3)
        ## FIND CHI SQUARE OF PS!!!
        #this is a bit too simple. Firstly, you need to make sure that the k values line up. secondly, you need uncertainties.
        return -0.5 * (self.power - PS_mK2Mpc3)**2 / self.uncertainty**2
        
    def computePower(self, ctx):
        ## Read in data from ctx
        new_lightcone = ctx.get("foreground_lightcone")
        frequencies = ctx.get("frequencies")
        boxsize = ctx.get("boxsize")
        sky_size = ctx.get("sky_size")
        
        # print "Computing likelihood"
        
        ## CONSIDER MOVING INTERPOLATING LINERALY SPACED FREQUENCY HERE INSTEAD
        
        visibility, coords, weights = self.add_instrument(new_lightcone,frequencies, sky_size)
        
        ## Find the 1D Power Spectrum of the visibility
        return self.power_spectrum(visibility, coords, weights, frequencies, bins = 50)
        
        
    def add_instrument(self, lightcone, frequencies, sky_size):
        
        print ("Adding instrument model")
        ## Number of 2D cells in sky array
        sky_cells = np.shape(lightcone)[0]
        
        ## Convert the boxsize from Mpc to radian
        z_mid = (1420e6)/(np.mean(frequencies))-1 
        
        ## Add the beam attenuation
        beam_sky = lightcone*self.add_beam(frequencies, sky_cells, sky_size)
        
        ## Fourier Transform over the (u,v) dimension and baselines sampling
        visibility_2D, uv_scales, weights = self.add_baselines_sampling(beam_sky, frequencies, sky_size)
        
        ## Fourier Transform over the eta dimension
        full_visibility, eta_scales = fft(visibility_2D, np.max(frequencies)-np.min(frequencies), axes=(2,))
        
        return full_visibility, [uv_scales[0], uv_scales[1], eta_scales], weights
        
    
    def add_beam(self, frequencies, sky_cells, sky_size):
        
        print ("Adding beam attenuation")
        ## First find the sigma of the beam
        epsilon = 0.42
        D = 4 * un.m
        frequencies = (frequencies)*(un.s**(-1))
        sigma = ((epsilon*const.c)/(frequencies*D)).to(un.dimensionless_unscaled)
        
        ## Then create a meshgrid for the beam attenuation on sky array
        range_rad = np.linspace(-sky_size/2,sky_size/2,sky_cells)
        degree = (range_rad*un.rad).to(un.deg)
        
        lm_range = np.sin(degree)

        l, m = np.meshgrid(lm_range, np.flipud(lm_range))

        ## Create an empty beam array and fill it up with beam attenuation over frequency
        beam = np.zeros((sky_cells, sky_cells, len(frequencies)))
        
        for ff in range(len(frequencies)):
            beam[:,:,ff] = np.exp(-(l**2 + m**2)/(sigma[ff]**2))
        print (beam[:,:,-1])
        return beam
    
    def add_baselines_sampling(self, beam_sky, frequencies, sky_size, new_cells = 1200, max_uv = 300):
        
        print ("Sampling the Fourier space with baselines")
        ## Read the tiles position
        MWA_array = np.genfromtxt("../Data/hex_pos.txt", float)
        
        ## Find all the possible combination of tile displacement
        x_displacement, y_displacement = self.get_baselines(MWA_array[:,1], MWA_array[:,2])*un.m
        
        ## 2D Fourier Transform
        FT, uv_scale = fft(beam_sky, [sky_size, sky_size], axes =(0,1))
        
        vis_fft2 = np.zeros((new_cells,new_cells,len(frequencies)), dtype=np.complex128)
        weights = np.zeros((new_cells,new_cells,len(frequencies)))
        
        frequencies = (frequencies)*(un.s**(-1))
        
        print ("Regridding the data in Fourier space")

        for ff in range(len(frequencies)):
            lamb = const.c/frequencies[ff].to(1/un.s)
            mwa_u = (x_displacement/lamb).value
            mwa_v = (y_displacement/lamb).value
            
            vis_fft2[:,:,ff], coords, weights[:,:,ff] = self.sampling_regGrid(FT[:,:,ff], mwa_u, mwa_v, uv_scale, cutoff = max_uv, new_cells = new_cells)
        
        return vis_fft2, coords, weights
    
    def sampling_regGrid(self, FT, mwa_u, mwa_v, uv_values, cutoff=0, new_cells=0):
        real = np.real(FT)
        imag = np.imag(FT)
        
        x_cells = np.shape(FT)[1]
        y_cells = np.shape(FT)[0]
    
        mwa_u_new = mwa_u[(np.abs(mwa_u)<=cutoff)&(np.abs(mwa_v)<=cutoff)]
        mwa_v_new = mwa_v[(np.abs(mwa_u)<=cutoff)&(np.abs(mwa_v)<=cutoff)]
        
        if(x_cells%2==0):
            mwa_u = mwa_u_new[(mwa_u_new>=(-cutoff-np.diff(uv_values)[0,0]))]
            mwa_v = mwa_v_new[(mwa_u_new>=(-cutoff-np.diff(uv_values)[0,0]))]
            x_range = np.linspace((-cutoff-np.diff(uv_values)[0,0]),cutoff,x_cells)
            
        if(y_cells%2==0):
            mwa_u = mwa_u_new[(mwa_v_new>=(-cutoff-np.diff(uv_values)[0,0]))]
            mwa_v = mwa_v_new[(mwa_v_new>=(-cutoff-np.diff(uv_values)[0,0]))]
            y_range = np.linspace((-cutoff-np.diff(uv_values)[0,0]),cutoff,y_cells)
            
        mwa_u = mwa_u_new
        mwa_v = mwa_v_new
    
        ##########################################################################
        
        f_real = RegularGridInterpolator([x_range,y_range], real)
        f_imag = RegularGridInterpolator([x_range, y_range], imag)
    
        MWA_Arr = np.zeros((len(mwa_u),2))
        MWA_Arr[:,0] = mwa_u
        MWA_Arr[:,1] = mwa_v
    
        FT_real = f_real(MWA_Arr)
        FT_imag = f_imag(MWA_Arr)
        
        x_range = np.linspace(np.min(x_range), np.max(x_range), new_cells)
        y_range = np.linspace(np.max(y_range), np.min(y_range), new_cells)
            
        vis_fft2 = np.zeros((len(x_range),len(y_range)), dtype=np.complex128)
        weights = np.zeros((len(x_range),len(y_range)))
    
        pos_u = np.digitize(mwa_u, bins = x_range)-1
        pos_v = np.digitize(mwa_v,y_range[::-1])-1
    
        for i in range(len(mwa_u)):    
            if ((pos_u[i]<new_cells)&(pos_v[i]<new_cells)):
                vis_fft2[pos_u[i],pos_v[i]] += FT_real[i]+1j*FT_imag[i]
                weights[pos_u[i],pos_v[i]] +=1
    
        vis_fft2[weights>0] = vis_fft2[weights>0]/weights[weights>0]
    
        coords = [x_range,y_range]
        
        return vis_fft2, coords, weights    
        
    def get_baselines(self, x,y):
        
        xx = (x,x)
        yy = (y,y)
        len_xy = len(x)
        
        X = np.zeros(len_xy*(len_xy-1))
        i = 0
        ii = 0
        
        for t in it.product(*xx):
            if (ii%(len_xy+1)!=0):
                X[i] = t[0]-t[1]
                i +=1
            ii+= 1
        
        Y = np.zeros(len_xy*(len_xy-1))
        j = 0
        jj = 0
        
        for t in it.product(*yy):
            if (jj%(len_xy+1)!=0):
                Y[j] = t[0]-t[1]
                j +=1
            jj +=1
            
        return X, Y
    
    def power_spectrum(self, visibility, coords, weights, linFrequencies, bins = 100):
        
        print ("Finding the power spectrum")
        ## Change the units of coords to Mpc
        z_mid = (1420e6)/(np.mean(linFrequencies))-1 
        coords[0] = 2*np.pi*coords[0]/cosmo.comoving_transverse_distance([z_mid])
        coords[1] = 2*np.pi*coords[1]/cosmo.comoving_transverse_distance([z_mid])
        coords[2] = 2*np.pi*coords[2]*(cosmo.H0).to(un.m/(un.Mpc * un.s))/1420e6*un.Hz*cosmo.efunc(z_mid)/(const.c*(1+z_mid)**2)
        
        ## Change the unit of visibility
        visibility = visibility/self.convert_factor_sources()*self.convert_factor_HztoMpc(np.min(linFrequencies),np.max(linFrequencies))*self.convert_factor_SrtoMpc2( z_mid)
        
        ## Square the visibility
        visibility_sq = np.abs(visibility)**2
        
        PS_mK2Mpc6, k_Mpc, c_Mpc = angular_average_nd(visibility_sq, coords, bins = bins, n=3)
        
        PS_mK2Mpc3 = PS_mK2Mpc6/self.volume(z_mid, np.min(linFrequencies), np.max(linFrequencies))
        
        return PS_mK2Mpc3, k_Mpc
    
    def convert_factor_HztoMpc(self, nu_min, nu_max):
        
        z_max = (1420e6)/(nu_min)-1 
        z_min = (1420e6)/(nu_max)-1 
        
        Mpc_Hz = (cosmo.comoving_distance([z_max])-cosmo.comoving_distance([z_min]))/(nu_max - nu_min)
        
        return Mpc_Hz
    
    def convert_factor_SrtoMpc2(self, z_mid):
        
        Mpc2_sr = cosmo.comoving_distance([z_mid])/(1*un.sr)
        
        return Mpc2_sr
    
    def volume(self, z_mid, nu_min, nu_max, A_eff=20):
        
        diff_nu = nu_max - nu_min
        print (diff_nu)

        G_z = (cosmo.H0).to(un.m/(un.Mpc * un.s))/1420e6*un.Hz*cosmo.efunc(z_mid)/(const.c*(1+z_mid)**2)
        
        Vol = const.c**2/(A_eff*un.m**2*nu_max*(1/un.s)**2)*diff_nu*(1/un.s)*cosmo.comoving_distance([z_mid])**2/(G_z)
        print (Vol)
        return Vol.value
    
    # def chi_square(self,power_spectrum, deg_of_freedom = 3):
    #     print("Do this the formula way")
    #     chi_sq, p_value = chisquare(power_spectrum, ddof=deg_of_freedom)
        
    #     return chi_sq, p_value
    
    def convert_factor_sources(self, nu=0):
        
        ## Can either do it with the beam or without the beam (frequency dependent)
        if (nu==0):
            A_eff = 20 * un.m**2
            
            flux_density = (2*1e26*const.k_B*1e-3*un.K/(A_eff*(1*un.Hz)*(1*un.s))).to(un.W/(un.Hz*un.m**2))
            
        else:           
            flux_density =  (2*const.k_B*1e-3*un.K/(((const.c)/(nu.to(1/un.s)))**2)*1e26).to(un.W/(un.Hz*un.m**2))
        
        return flux_density.value

    def simulate_data(self, Smin, Smax, params, niter=20):
        core = Core21cmFastModule(parameters=params, box_dim = self._box_dim, flag_options = self._flag_options, astro_params = self._astro_params, cosmo_params = self._cosmo_params)
        fg_core = ForegroundCore(Smin, Smax)

        ctx = ChainContext('derp',{"HII_EFF_FACTOR":30.0})
        

        p = [0]*niter
        k = [0]*niter
        for i in range(niter):
            core(ctx)
            fg_core(ctx)
            p[i], k[i] = self.computePower(ctx)

        sigma = np.std(np.array(p), axis=-1)
        p = np.mean(np.array(p), axis=-1)

        np.savetxt(self.datafile, [k,p,sigma])


if __name__ == "__main__":

    from Cores_py21cmmc import ForegroundCore
    from py21cmmc.mcmc import run_mcmc
    import os

    filename_of_data = "../../data.txt"
    Smin  = 1e-1
    Smax = 1.0
    if not os.path.exists(filename_of_data):
        lk = ForegroundLikelihood(filename_of_data)
        lk.simulate_data(Smin, Smax, {"HII_EFF_FACTOR":['alpha', 30.0, 10.0, 50.0, 3.0]}, niter=20)


    run_mcmc(
        redshift = 7.0,
        parameters = {"HII_EFF_FACTOR": ['alpha', 30.0, 10.0, 50.0, 3.0]},
        storage_options = {
            "DATADIR": "../../MCMCData",
            "KEEP_ALL_DATA":False,
            "KEEP_GLOBAL_DATA":False,
        },
        box_dim = {
            "HII_DIM": 30,
            "BOX_LEN": 50.0
        },
        extra_core_modules = [ForegroundCore( Smin=Smin, Smax=Smax)],
        likelihood_modules = [ForegroundLikelihood(filename_of_data)],
        walkersRatio = 4,
        burninIterations = 1,
        sampleIterations = 10,
        threadCount = 1
    )




