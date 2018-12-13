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

void interpolate_map_frequencies(int n_map, int nf_in, int nf_out, double *map, double *freq_in,
                                 double *freq_out, int nmap_is_all, double *out){
    /*
        Linearly interpolate each cell in a map onto a given set of frequencies.

        Assumes freq_in and freq_out are monotonically increasing (not necessarily regular). Also cannot deal
        with the case where freq_out is oiutside of freq_in.

        Parameters
        ----------
        n_map :
            The number of pixels in the map if nmap_is_all>0, otherwise the number of pixels on a side of a square.
        nf_in :
            Number of frequencies in the input map
        nf_out :
            Number of frequencies in the output map
        map : length=n_map*nf_in or n_map*n_map*nf_in
            The input map.
        freq_in : length=nf_in
            Frequencies of the input (arbitrary units)
        freq_out : length=nf_out
            Frequencies of the output (same units as freq_in)
        nmap_is_all :
            Wether n_map is all of the pixels, or just the length of a  side.
        out : length=n_map*nf_out or n_map*n_map*nf_out
            The output, interpolated, map
    */

    int i,j, k;
    double d1,d2;

    int N;

    if(nmap_is_all>0){
        N = n_map;
    }else{
        N = n_map*n_map;
    }

    j = 0;

    for(i=0;i<nf_out;i++){
        while(freq_out[i]>freq_in[j]){
            j++;
        }

        d1 = freq_out[i] - freq_in[j-1];
        d2 = freq_in[j] - freq_out[i];

        for(k=0;k<N;k++){
            out[i * N + k] = (d2 * map[k * nf_in + (j-1)] + d1 * map[k * nf_in + j])/(d1 + d2);
        }
    }
}


#define INDB(x,y,z) (z + (y*nf) + (x * nf*n_out))
#define INDS(x,y,z) (z + (y*nf) + (x * nf*n_sim))


void stitch_and_coarsen_sky(int n_sim, int nf, int n_out, double small_sky_size, double big_sky_size,
                            double *small_sky, double *big_sky){
    /*
        We put all of this in *one* function to save memory and time. In one function, it doesn't have to actually
        *stitch* boxes and therefore use too much memory, it can re-use the same initial box.

        NOTE: the original and final simulations are both assumed to be in (l,m) space, which means they can be tiled
              linearly. To be useful, the simulation must be periodic in (l,m).

        This function takes a box of a given size, and effectively linearly tiles the box in the first two dimensions,
        then places down *larger* cells on top of it, and calculates the average of the hi-res cells within each low-res
        cell. A cell is deemed to be fully in a (broader) cell if its *LEFT EDGE* is in the larger cell, and fully out
        of the cell otherwise. The final returned value is the *average* of the input (i.e. the sum divided by the
        number of cells deemed to be in it -- this is better than averaging by area).

        The box is tiled starting from the left edge, which should be without loss of generality for a periodic box (but
        should be kept in mind when comparing input to output).

        This function could be improved by averaging cells by the amount that they're actually in the larger cell.
    */

    int j_x, i_x, j_y, i_y, i_f;
    double dlo, dhi;
    int xstart, xend, ystart, yend;

#ifdef DEBUG
    // Check input parameters
    printf("IN C CODE: n_sim: %d, nf: %d, n_out: %d, small_sky_size: %g, big_sky_size: %g\n", n_sim, nf, n_out, small_sky_size, big_sky_size);
#endif

    // Get the number of cells within a lo-res cell
    dlo = big_sky_size/n_out;
    dhi = small_sky_size/n_sim;

    for(i_x=0;i_x<n_out;i_x++){ // Go through lo-res box x
        xstart = (int) ((i_x*dlo)/dhi);
        xend = (int) (((i_x+1)*dlo)/dhi) + 1;


        for(i_y=0;i_y<n_out;i_y++){ // Go through lo-res box y
            ystart = (int) ((i_y*dlo)/dhi);
            yend = (int) (((i_y+1)*dlo)/dhi) + 1;

            // Add all bits in this cell
            for(i_f=0; i_f<nf; i_f++){
                for(j_x=xstart;j_x<xend;j_x++){
                    for(j_y=ystart;j_y<yend;j_y++){
                        big_sky[INDB(i_x, i_y, i_f)] += small_sky[INDS(j_x%n_sim, j_y%n_sim, i_f)];
                    }
                }
            // Divide by how many cells there were
            big_sky[INDB(i_x, i_y, i_f)] /= ((xend - xstart) * (yend - ystart));
            }

        }
    }
}

