from numpy.linalg import slogdet, solve, LinAlgError
import numpy as np
import warnings


def logdet_block_matrix(S):
    sm = 0
    for s in S:
        if not np.all(s==0): # if all s is zero, then there's no signal here at all... move on.
            try:
                sm += slogdet(s)[1]
            except LinAlgError as e:
                # If the log-det can't be found for this block, just ignore it and move on.
                # TODO: this is probably a bad idea!
                warnings.warn("log-determinant not working: %s"%s)

    return sm


def solve_block_matrix(S, x):
    bits = []
    inds = []
    for i, (s,xx) in enumerate(zip(S, x)):
        if not np.all(s == 0):  # if all s is zero, then there's no signal here at all... move on.
            try:
                sol = solve(s, xx)
                bits += [sol]
                inds += [i]
            except LinAlgError:
                # Sometimes, the covariance might be all zeros, or singular.
                # Then we just ignore those values of u (or kperp) and keep going.
                # TODO: this might not be a great idea.

                warnings.warn("solve didn't work for index %s" % i)

    bits = np.array(bits)
    return bits, inds


def lognormpdf(x, mu, cov):
    """
    Calculate gaussian probability log-density of x, when x ~ N(mu,cov), and cov is block diagonal.

    Code adapted from https://stackoverflow.com/a/16654259
    """
    nx = len(x)
    norm_coeff = nx * np.log(2 * np.pi) + logdet_block_matrix(cov)

    err = x - mu
    sol, inds = solve_block_matrix(cov, err)

    # The "inds" here means we only use the u co-ordinates that had a solution to the covariance.
    numerator = 0
    for i, (s, e) in enumerate(zip(sol, err[inds])):
        numerator += s.dot(e)
    
    return -0.5 * numerator#(norm_coeff + numerator)