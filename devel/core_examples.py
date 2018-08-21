"""
A few examples of how to call the core functions (and provides a way to check they're kind of working...
"""

import numpy as np
from py21cmmc_fg import core
from py21cmmc.mcmc.core import CoreLightConeModule
from py21cmmc.mcmc.mcmc import run_mcmc, build_computation_chain
from py21cmmc.mcmc.cosmoHammer import ChainContext

import matplotlib.pyplot as plt

PLOT = True


def mkplot(lightcone):
    if PLOT:
        plt.imshow(lightcone[:, :, 0])
        plt.colorbar()
        plt.savefig("figure.png")

    else:
        print(lightcone)


def point_sources_only():
    # To change the size parameters for every kind of foreground, modify
    # the class variable:
    core.ForegroundsBase.defaults['box_len'] = 10000.0
    core.ForegroundsBase.defaults['sky_cells'] = 350

    # Make a core, with no 21cmFAST signal.
    ptsource_core = core.CorePointSourceForegrounds(redshifts=np.linspace(7, 8, 25))
    
    lightcone = ptsource_core.build_sky(ptsource_core.frequencies, ptsource_core.sky_cells, ptsource_core.sky_size)
    mkplot(lightcone)


def point_sources_with_21cmfast():
    # No need to pass redshifts, because it will be gotten from 21cmFAST lightcone.
    ptsource_core = core.CorePointSourceForegrounds()

    signal_core = CoreLightConeModule(
        redshift=7.0,  # Minimum redshift of the lightcone
        max_redshift=10.0,  # Maximum redshift will be *at least* this big.
        user_params={"HII_DIM": 50, "BOX_LEN": 100}
    )

    # Build a chain to combine the cores. NOTE: signal_core must be *before* ptsource_core
    chain = build_computation_chain([signal_core, ptsource_core], [])
    ctx = chain.core_context()  # this is a helper method that just runs the cores.

    mkplot(ctx)


def point_sources_and_diffuse():
    # To change the size parameters for every kind of foreground, modify
    # the class variable:
    core.ForegroundsBase.defaults['box_len'] = 100.0
    core.ForegroundsBase.defaults['sky_cells'] = 80

    # Make a core, with no 21cmFAST signal.
    ptsource_core = core.CorePointSourceForegrounds(redshifts=np.linspace(7, 8, 25))
    diffuse_core = core.CoreDiffuseForegrounds(redshifts=np.linspace(7, 8, 25))

    # Create an empty context, required to call the core.
    ctx = ChainContext(parent=None, params={})

    # Call the core.
    ptsource_core(ctx)
    diffuse_core(ctx)

    mkplot(ctx)


def ptsource_and_instrumental():
    frequencies = np.linspace(150, 160.0, 30)

    ptsource_core = core.CorePointSourceForegrounds(redshifts=1420. / frequencies - 1)

    instrumental_core = core.CoreInstrumental(
        antenna_posfile='grid_centres',
        freq_min=frequencies.min(),
        freq_max=frequencies.max(),
        nfreq=len(frequencies)
    )

    # Build a chain to combine the cores. NOTE: instrumental_core must be *after* ptsource_core
    chain = build_computation_chain([ptsource_core, instrumental_core], [])
    ctx = chain.core_context()  # this is a helper method that just runs the cores.

    mkplot(ctx)

## Change this to whichever one you want to run
if __name__ == "__main__":
    point_sources_only()
