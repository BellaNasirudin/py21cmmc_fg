import os
import ctypes
import glob
import numpy as np

# Build the extension function (this should be negligible performance-wise)
fl = glob.glob(os.path.join(os.path.dirname(__file__), "c_routines*"))[0]


def interpolate_visibility_frequencies(visibilities, freq_in, freq_out):
    cfunc = ctypes.CDLL(fl).interpolate_visibility_frequencies
    cfunc.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        np.ctypeslib.ndpointer("complex128", flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer("complex128", flags="C_CONTIGUOUS"),
    ]

    if freq_out.min() < freq_in.min() or freq_out.max() > freq_in.max():
        raise ValueError(
            f"c interpolation routine cannot deal with out of bounds frequencies! Input: ({freq_in.min()},{freq_in.max()}). Output ({freq_out.min()},{freq_out.max()})")

    n_bl, nf_in = visibilities.shape
    nf_out = len(freq_out)

    visibilities = visibilities.flatten()
    vis_out = np.zeros((n_bl, nf_out), dtype=np.complex128).flatten()

    cfunc(n_bl, nf_in, nf_out, np.ascontiguousarray(visibilities),
          np.ascontiguousarray(freq_in), np.ascontiguousarray(freq_out), np.ascontiguousarray(vis_out))

    return vis_out.reshape((nf_out, n_bl)).T


def stitch_and_coarsen_sky(sky, small_sky_size, big_sky_size, nsky_out):
    cfunc = ctypes.CDLL(fl).stitch_and_coarsen_sky
    cfunc.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_double,
        np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")
    ]

    n_sim = sky.shape[0]
    if sky.shape[1] != n_sim:
        raise ValueError("The sky must have the same number of cells in the first two dimensions!")

    nf = sky.shape[-1]

    out = np.zeros(nsky_out * nsky_out * nf)
    cfunc(n_sim, nf, nsky_out, small_sky_size, big_sky_size,
          np.ascontiguousarray(sky.flatten()), np.ascontiguousarray(out))

    return out.reshape((nsky_out, nsky_out, nf))


def get_tiled_visibilities(sim, frequencies, baselines, dtheta, l_extent, tile_diameter):
    cfunc = ctypes.CDLL(fl).get_tiled_visibilities
    cfunc.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int,
        np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.complex128, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ]

    n_sim = sim.shape[0]

    if sim.shape[1] != n_sim:
        raise ValueError("Simulation should have the same shape for the first two dimensions")

    nf = len(frequencies)
    n_bl = len(baselines)

    wavelengths = 3e8 / frequencies

    visibilities = np.zeros(n_bl * nf, dtype=np.complex128)

    n_new = n_sim * int(2 * 1.2 * l_extent/(dtheta*n_sim))
    print(n_new)
    new_image = np.zeros(n_new*n_new)

    cfunc(n_sim, nf, n_bl, dtheta, l_extent, tile_diameter,
          n_new,
          np.ascontiguousarray(sim.flatten()),
          np.ascontiguousarray(baselines[:, 0]),
          np.ascontiguousarray(baselines[:, 1]),
          np.ascontiguousarray(wavelengths),
          np.ascontiguousarray(visibilities),
          np.ascontiguousarray(new_image)
          )

    return visibilities.reshape((n_bl, nf)), new_image.reshape((n_new, n_new))


def get_direct_visibilities(f, baselines, source_flux, source_pos_l, source_pos_m):
    cfunc = ctypes.CDLL(fl).getvis
    cfunc.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, np.ctypeslib.ndpointer(np.float64),
        np.ctypeslib.ndpointer(np.float64), np.ctypeslib.ndpointer(np.float64), np.ctypeslib.ndpointer(np.float64),
        np.ctypeslib.ndpointer(np.float64), np.ctypeslib.ndpointer(np.float64), np.ctypeslib.ndpointer(np.complex128)
    ]

    assert len(f.shape)==1
    assert len(baselines.shape)==2
    assert len(source_pos_l.shape)==1
    assert len(source_pos_m.shape)==1

    assert source_pos_l.shape == source_pos_l.shape
    assert len(source_flux) == len(source_pos_l)

    nbl = len(baselines)

    vis = np.zeros(len(f) * nbl, dtype=np.complex128)

    wavelengths = 3e8/f

    cfunc(len(f), nbl, len(source_flux), wavelengths, np.ascontiguousarray(baselines[:, 0]),
          np.ascontiguousarray(baselines[:, 1]),
          np.ascontiguousarray(source_flux.flatten()),
          np.ascontiguousarray(source_pos_l),
          np.ascontiguousarray(source_pos_m),
          vis
          )

    return vis.reshape((nbl, len(f)))
