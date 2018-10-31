/*
    Some routines used for speeding up certain functions which are tooooo sloooooow....
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

void interpolate_visibility_frequencies(int n_bl, int nf_in, int nf_out, complex double *visibilities, double *freq_in,
                                        double *freq_out, complex double *out){
    int i,j, k;
    double d1,d2;
    j = 0;

    for(i=0;i<nf_out;i++){
       // printf("Out Freq: %g\n", freq_out[i]);
        // Go through input frequencies until we pass current out frequency (assumes both are strictly increasing).
        while(freq_out[i]>freq_in[j]){
            j++;
        }

        //printf("\t Surrounding in freqs: %g %g\n", freq_in[j-1], freq_in[j]);

        d1 = freq_out[i] - freq_in[j-1];
        d2 = freq_in[j] - freq_out[i];

        //printf("\t d1, d1: %g, %g\n", d1, d2);

        for(k=0;k<n_bl;k++){
            out[i * n_bl + k] = (d2 * visibilities[k * nf_in + (j-1)] + d1 * visibilities[k * nf_in + j])/(d1 + d2);
        }

    }
}