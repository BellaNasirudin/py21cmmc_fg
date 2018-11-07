"""
Tests of the c functions via the c_wrapper
"""

from py21cmmc_fg import c_wrapper as cw

RUNNING_AS_TEST = True

import numpy as np
from powerbox.dft import fft, ifft, ifftshift


def test_tiled_visibility_gaussian_untiled():
    dtheta = 0.01
    size = 1.0
    l = np.arange(-size/2, size/2, dtheta)
    sim = np.exp(-np.add.outer(l**2, l**2))

    _test_tiled_visibility(sim, dtheta, l, size, name="gaussian_untiled")


def test_tiled_visibility_sin_untiled():
    dtheta = 0.01
    size= 1.0
    l = np.arange(-size/2, size/2, dtheta)
    sim = np.sin(np.add.outer(2 * 2 * np.pi * l, 0 * l))  # create a sin-wave pattern in simulation

    _test_tiled_visibility(sim, dtheta, l, size, name='sine_untiled')


def test_tiled_visibility_sin_tiled():
    dtheta = 0.01
    size = 0.2
    l = np.arange(-size/2, size/2, dtheta)
    sim = np.sin(np.add.outer(2 * 2 * np.pi * l/size, 0 * l))  # create a sin-wave pattern in simulation

    _test_tiled_visibility(sim, dtheta, l, size, name='sine_tiled')



def test_direct_visibility_gaussian():
    dtheta = 0.01
    size = 1.0
    l = np.arange(-size/2, size/2, dtheta)
    sim = np.exp(-np.add.outer(l**2, l**2))

    _test_direct_visibility(sim, dtheta, l, size, name="gaussian")


def test_direct_visibility_sine():
    dtheta = 0.01
    size= 1.0
    l = np.arange(-size/2, size/2, dtheta)
    sim = np.sin(np.add.outer(2 * 2 * np.pi * l, 0 * l))  # create a sin-wave pattern in simulation

    _test_direct_visibility(sim, dtheta, l, size, name='sine')


def _test_tiled_visibility(sim, dtheta, l, size, name):
    print(l[len(l)//2])
    D = 10.0
    sigma = 0.42 * 2 / D

    attenuation = np.exp(-np.add.outer(l ** 2, l ** 2) / (2 * sigma ** 2))

    sim_uv, uv = fft(sim * attenuation, L=size, a=0, b=2 * np.pi)

    # Use an FFT grid of baselines for testing
    X, Y = np.meshgrid(uv[0], uv[1])

    # Convert to metres, and flatten
    X = 2 * X.flatten()
    Y = 2 * Y.flatten()

    baselines = np.array([X, Y]).T

    sim = np.atleast_3d(sim)  # Give it a frequency axis
    frequencies = np.array([150.0e6])

    # Use no actual tiling, and make the beam really small so we know we've converged.
    vis, new_image = cw.get_tiled_visibilities(sim * dtheta**2, frequencies, baselines, dtheta=dtheta, l_extent=0.5,
                                               tile_diameter=D)

    vis = vis.reshape((len(l), len(l))).T

    # Now do the FT back
    recon = ifft(vis, L=size, a=0, b=2 * np.pi)[0]

    if not RUNNING_AS_TEST:
        fig, ax = plt.subplots(2, 4, figsize=(13, 6))
        cmap = ax[0, 0].imshow(sim[:, :, 0].T, origin='lower')
        ax[0, 0].set_title("Original")
        plt.colorbar(cmap, ax=ax[0,0])

        cmap = ax[0, 1].imshow((sim[:, :, 0] * attenuation).T, origin='lower')
        ax[0, 1].set_title("Original w/ beam")
        plt.colorbar(cmap, ax=ax[0, 1])

        cmap = ax[0, 2].imshow(np.real(sim_uv.T), origin='lower')
        ax[0, 2].set_title("Re[FFT]")
        plt.colorbar(cmap, ax=ax[0, 2])

        cmap = ax[0, 3].imshow(np.imag(sim_uv.T), origin='lower')
        ax[0, 3].set_title("Im[FFT]")
        plt.colorbar(cmap, ax=ax[0, 3])

        cmap = ax[1, 0].imshow(new_image.T / dtheta**2, origin='lower')
        ax[1, 0].set_title("new image")
        plt.colorbar(cmap, ax=ax[1, 0])

        # TODO: really unsure why I have to use ifftshift here...
        cmap = ax[1, 1].imshow(ifftshift(np.real(recon.T)), origin='lower')
        ax[1, 1].set_title("reconstructed")
        plt.colorbar(cmap, ax=ax[1,1])

        cmap = ax[1, 2].imshow(np.real(vis).T, origin='lower')
        ax[1, 2].set_title("real Direct UV")
        plt.colorbar(cmap, ax=ax[1,2])

        cmap = ax[1, 3].imshow(np.imag(vis).T, origin='lower')
        ax[1, 3].set_title("imag Direct UV")
        plt.colorbar(cmap, ax=ax[1,3])

        plt.savefig("test_tiled_visibility_%s.png"%name)


def _test_direct_visibility(sim, dtheta, l, size, name):
    print(l[len(l)//2])

    sim_uv, uv = fft(sim, L=size, a=0, b=2 * np.pi)

    # Use an FFT grid of baselines for testing
    X, Y = np.meshgrid(uv[0], uv[1])

    # Convert to metres, and flatten
    X = 2 * X.flatten()
    Y = 2 * Y.flatten()
    baselines = np.array([X, Y]).T

    sim = np.atleast_3d(sim)  # Give it a frequency axis
    frequencies = np.array([150.0e6])

    L, M = np.meshgrid(l,l, indexing='ij')
    # Use no actual tiling, and make the beam really small so we know we've converged.
    vis = cw.get_direct_visibilities(frequencies, baselines, sim.flatten() * dtheta**2, L.flatten(), M.flatten())

    vis = vis.reshape((len(l), len(l))).T

    # Now do the FT back
    recon = ifft(vis, L=size, a=0, b=2 * np.pi)[0]

    if not RUNNING_AS_TEST:
        fig, ax = plt.subplots(2, 3, figsize=(13, 6))
        cmap = ax[0, 0].imshow(sim[:, :, 0].T, origin='lower')
        ax[0, 0].set_title("Original")
        plt.colorbar(cmap, ax=ax[0,0])

        cmap = ax[0, 1].imshow(np.real(sim_uv.T), origin='lower')
        ax[0, 1].set_title("Re[FFT]")
        plt.colorbar(cmap, ax=ax[0, 1])

        cmap = ax[0, 2].imshow(np.imag(sim_uv.T), origin='lower')
        ax[0, 2].set_title("Im[FFT]")
        plt.colorbar(cmap, ax=ax[0, 2])

        # TODO: really unsure why I have to use ifftshift here...
        cmap = ax[1, 0].imshow(ifftshift(np.real(recon.T)), origin='lower')
        ax[1, 0].set_title("reconstructed")
        plt.colorbar(cmap, ax=ax[1,0])

        cmap = ax[1, 1].imshow(np.real(vis).T, origin='lower')
        ax[1, 1].set_title("real Direct UV")
        plt.colorbar(cmap, ax=ax[1,1])

        cmap = ax[1, 2].imshow(np.imag(vis).T, origin='lower')
        ax[1, 2].set_title("imag Direct UV")
        plt.colorbar(cmap, ax=ax[1,2])

        plt.savefig("test_direct_visibility_%s.png"%name)


if __name__ == "__main__":
    RUNNING_AS_TEST = False

    import matplotlib.pyplot as plt

    test_tiled_visibility_gaussian_untiled()
    test_tiled_visibility_sin_untiled()
    test_tiled_visibility_sin_tiled()

    test_direct_visibility_gaussian()
    test_direct_visibility_sine()