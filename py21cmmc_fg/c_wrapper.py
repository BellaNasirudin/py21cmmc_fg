import os
import ctypes
import glob
import numpy as np
# Build the extension function (this should be negligible performance-wise)
fl = glob.glob(os.path.join(os.path.dirname(__file__), "c_routines.cpython*"))[0]


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
        raise ValueError(f"c interpolation routine cannot deal with out of bounds frequencies! Input: ({freq_in.min()},{freq_in.max()}). Output ({freq_out.min()},{freq_out.max()})")

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

    out = np.zeros(nsky_out*nsky_out*nf)
    cfunc(n_sim, nf, nsky_out, small_sky_size, big_sky_size,
          np.ascontiguousarray(sky.flatten()), np.ascontiguousarray(out))

    return out.reshape((nsky_out, nsky_out, nf))
