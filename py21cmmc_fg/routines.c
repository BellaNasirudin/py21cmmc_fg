/*
    Some routines used for speeding up certain functions which are tooooo sloooooow....
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#define DEBUG

void interpolate_visibility_frequencies(int n_bl, int nf_in, int nf_out, complex double *visibilities, double *freq_in,
                                        double *freq_out, complex double *out){
    int i,j, k;
    double d1,d2;
    j = 0;

    for(i=0;i<nf_out;i++){
        while(freq_out[i]>freq_in[j]){
            j++;
        }

        d1 = freq_out[i] - freq_in[j-1];
        d2 = freq_in[j] - freq_out[i];

        for(k=0;k<n_bl;k++){
            out[i * n_bl + k] = (d2 * visibilities[k * nf_in + (j-1)] + d1 * visibilities[k * nf_in + j])/(d1 + d2);
        }
    }
}

#define INDB(x,y,z) ((z) + ((y)*nf) + ((x) * nf*n_out))
#define INDS(x,y,z) ((z) + ((y)*nf) + ((x) * nf*n_sim))


void stitch_and_coarsen_sky(int n_sim, int nf, int n_out, double small_sky_size, double big_sky_size,
                            double *small_sky, double *big_sky){
    /*
        We put all of this in *one* function to save memory and time. In one function, it doesn't have to actually
        *stitch* boxes and therefore use too much memory, it can re-use the same initial box.

        NOTE: at this point, we work directly in linear space, assuming that we can tile each cell and its length
        adds linearly. This is not technically true, due to wide-field effects (the sizes are in radians). Will
        have to fix this later.

        This function takes a box of a given size, and effectively linearly tiles the box in the first two dimensions,
        then places down *larger* cells on top of it, and calculates the average of the hi-res cells within each low-res
        cell. A cell is deemed to be fully in a cell if its *LEFT EDGE* is in the larger cell, and fully out of the cell
        otherwise.

        This could be improved in many ways:
            1) Use centres instead of left edge,
            2) Average cells by the amount that they're actually in the larger cell.
            3) Acknowledge that there are curved-sky effects.

    */

    int j_x, i_x, j_y, i_y, i_f;
    double dlo, dhi;
    int xstart, xend, ystart, yend;

#ifdef DEBUG
    // Check input parameters
    printf("n_sim: %d, nf: %d, n_out: %d, small_sky_size: %g, big_sky_size: %g\n", n_sim, nf, n_out, small_sky_size, big_sky_size);
#endif

    // Get the number of cells within a lo-res cell
    dlo = big_sky_size/n_out;
    dhi = small_sky_size/n_sim;

    j_x = 0;
    j_y = 0;

   for(i_x=0;i_x<n_out;i_x++){ // Go through lo-res box x
        xstart = (int) ((i_x*dlo)/dhi);
        xend = (int) (((i_x+1)*dlo)/dhi) + 1;


        for(i_y=0;i_y<n_out;i_y++){ // Go through lo-res box y
            ystart = (int) ((i_y*dlo)/dhi);
            yend = (int) (((i_y+1)*dlo)/dhi) + 1;

            // Add all bits in this cell
            for(j_x=xstart;j_x<xend;j_x++){
                for(j_y=ystart;j_y<yend;j_y++){
                    for(i_f=0; i_f<nf; i_f++){
                        big_sky[INDB(i_x, i_y, i_f)] += small_sky[INDS(j_x%n_sim, j_y%n_sim, i_f)];
                    }
                }
            }

            // Divide by how many cells there were
            big_sky[INDB(i_x, i_y, i_f)] /= ((xend - xstart) * (yend - ystart));
        }
    }
}

double attenuation(double l, double wavelength, double tile_diameter){
    double sigma = 0.42 * wavelength/tile_diameter;
    return exp(-l*l/(2*sigma*sigma));
}

void get_tiled_visibilities(int n_sim, int nf, int n_bl, double dtheta, double l_extent,
                            double tile_diameter, int n_new,
                            double *sim, double *x_bl, double *y_bl, double *wavelengths, complex double *visibilities,
                            double *new_image){
    /*
        We put all of this in *one* function to save memory and time. In one function, it doesn't have to actually
        *stitch* boxes and therefore use too much memory, it can re-use the same initial box.

        NOTE: the sky_size is in *radians*, and is the size of the simulation assuming it had been wrapped in an
              observing sphere. However, for the purposes of this function, we *equate* angles with (l,m).
              This is a necessary evil, because the simulation is euclidean. If we try to correct the co-ordinate
              transform, it will result in un-physical pushing of the corners of the simulation. Assuming angles
              and (l,m) are interchangeable is much simpler, and is correct in the most important regions.

              To fix this properly, one must modify the *simulation* box itself when creating the lightcone.

              The sky_size_required is in (l,m), and is the minimum size in any direction that is
              required to tile the original simulation in order to consider the result "converged".
              To avoid distorting line-of-sight modes, we assume each redshift slice of the simulation has the same
              *angular* width (i.e. the angle-size relation doesn't change over the redshift range).

        This function takes a box of a given size, and effectively linearly tiles the box in the first two dimensions,

        Ultimately, since the simulation is periodic in the transverse plane, the choice of where "zenith" is does
        not matter, and we naturally choose it to be the centre of the cell at index (0,0).

    */

    // Size of the cells in radians.
    double absl = 0, l=0, m=0;
    double u,v, beam;
    int i_f, i_bl, ind, ix=n_sim/2,iy=n_sim/2;

    int i_up=n_new/2, j_up=n_new/2, i_down=n_new/2, j_down=n_new/2;

#ifdef DEBUG
    // Check input parameters
    printf("n_sim: %d, nf: %d, n_bl: %d, dtheta: %g, l_extent: %g, n_new=%d\n", n_sim, nf, n_bl, dtheta, l_extent, n_new);
#endif

    complex double PPI = -2 * 3.141592653589 * I;


    while(absl<=l_extent){
        while (absl<=l_extent){

             for(i_bl=0;i_bl<n_bl;i_bl++){
                 for(i_f=0;i_f<nf;i_f++){
                    ind = i_f + i_bl * nf;

                    beam = attenuation(absl, wavelengths[i_f], tile_diameter);


                    u = x_bl[i_bl] / wavelengths[i_f];
                    v = y_bl[i_bl] / wavelengths[i_f];

                    // Top-right quadrant

                    visibilities[ind] += beam * cexp(PPI*(u*l + v*m)) * sim[INDS(ix, iy, i_f)] ;
/*
                    if(ind==1) {
                        printf("beam=%lf uv=(%lf,%lf) lm=(%lf,%lf) arg=%lf arg2=%lf rl,im=(%lf,%lf)\n", beam,
                                u, v, l, m, u*l+v*m, cimag(PPI*(u*l+v*m)), creal(cexp(PPI*(u*l+v*m))),
                                cimag(cexp(PPI*(u*l+v*m))));
                        printf("sim(s)=%lf %lf %lf %lf (ix=%d iy=%d i_f=%d ind=(%d,%d,%d,%d))\n", sim[INDS(ix, iy, i_f)], sim[INDS(n_sim-ix-1, iy, i_f)],
                                                         sim[INDS(ix, n_sim-iy-1, i_f)],  sim[INDS(n_sim-ix-1, n_sim-iy-1, i_f)],
                                                         ix,iy,i_f, INDS(ix, iy, i_f),INDS(n_sim-ix-1, iy, i_f),INDS(ix, n_sim-iy-1, i_f),INDS(n_sim-ix-1, n_sim-iy-1, i_f)
                        );
                    }
*/

                    // Top-left quadrant (unless l is zero, avoids double counting the centre/central lines)
                    if (l>0) visibilities[ind] += beam * cexp(PPI*(-u*l + v*m)) * sim[INDS(n_sim-ix-1, iy, i_f)];

                    if (m>0) {
                        // Bottom-right quadrant
                        visibilities[ind] += beam * cexp(PPI * (u*l - v*m)) * sim[INDS(ix, n_sim-iy-1, i_f)];

                        // Bottom-left quadrant
                        if(l>0) visibilities[ind] += beam * cexp(-PPI * (u*l + v*m)) * sim[INDS(n_sim-ix-1, n_sim-iy-1, i_f)];
                    }
                 }
             }

             //printf("%d %d %d\n", i_up, j_up, n_new);
             new_image[j_up + i_up*n_new] = beam * sim[INDS(ix, iy, i_f)];
             new_image[j_up + i_down*n_new] = beam * sim[INDS(n_sim-ix-1, iy, i_f)];
             new_image[j_down + i_up*n_new] = beam * sim[INDS(ix, n_sim-iy-1, i_f)];
             new_image[j_down + i_down*n_new] = beam * sim[INDS(n_sim-ix-1, n_sim-iy-1, i_f)];

             j_up++;
             j_down--;

             iy++;
             iy = iy%n_sim;
             m += dtheta;
             absl = sqrt(l*l + m*m);

             if(absl>l_extent) printf("BIGGER: %g %g %g %g\n", l, m, absl, visibilities[1]);

        }

         i_up++;
         i_down--;

         j_up = n_new/2;
         j_down = n_new/2;

         ix++;
         ix = ix%n_sim;
         l += dtheta;
         m = 0;
         iy=n_sim/2;
         absl = sqrt(l*l + m*m);

    }
}


void getvis(int nf, int nbl, int nsource,  double *wavelengths, double *x_bl, double *y_bl,
            double *source_flux, double *l, double *m, double complex *vis){
    /*
        Direct visibility calculation over multiple frequencies/wavelengths

        Parameters
        ----------
        nf :
            number of frequencies/wavelengths
        nbl :
            number of baselines
        nsource :
            number of sources
        wavelengths : length=nf
            Wavelengths (in m) observed.
        x_bl, y_bl : length=nbl
            Baseline lengths (in m).
        source_flux : length=nsource*nf
            Source flux density (Jy) for each source at each frequency
        l, m : length=nsource
            Positions (in l,m coordinates) of the sources on the sky.

        Returns
        -------
        Returns nothing, but *fills* `vis`.

        vis : length=nbl*nf
            The resultant visibilities, after direct calculation.

    */
    int i,j,k;
    double u,v;

    complex double PPI = -2*3.1415653589*I;


    for(k=0;k<nbl;k++){
        for(i=0;i<nf;i++){
            u = x_bl[k]/wavelengths[i];
            v = y_bl[k]/wavelengths[i];

            for(j=0;j<nsource;j++){
                vis[k*nf + i] += source_flux[j*nf + i] * cexp(PPI*(u*l[j] + v*m[j]));
            }
        }
    }
}
