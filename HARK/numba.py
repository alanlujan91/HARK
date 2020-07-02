"""
General purpose  / miscellaneous functions in numba.  Includes utility functions (and their
derivatives).
"""

from __future__ import division  # Import Python 3.x division function
from __future__ import print_function

import numpy as np
from numba import njit, vectorize, float64


# ==============================================================================
# ============== Define utility functions        ===============================
# ==============================================================================


def CRRAutility(gam, target="parallel"):
    """
    Evaluates constant relative risk aversion (CRRA) utility of consumption c
    given risk aversion parameter gam.

    Parameters
    ----------
    c : float
        Consumption value
    gam : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Utility

    Tests
    -----
    Test a value which should pass:
    >>> c, gamma = 1.0, 2.0    # Set two values at once with Python syntax
    >>> utility(c=c, gam=gamma)
    -1.0
    """

    @vectorize([float64(float64)], target=target)
    def logU(c):
        return np.log(c)

    @vectorize([float64(float64)], target=target)
    def CRRAU(c):
        return c ** (1.0 - gam) / (1.0 - gam)

    if gam == 1:
        return logU
    else:
        return CRRAU


def CRRAutilityP(gam, target="parallel"):
    """
    Evaluates constant relative risk aversion (CRRA) marginal utility of consumption
    c given risk aversion parameter gam.

    Parameters
    ----------
    c : float
        Consumption value
    gam : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Marginal utility
    """

    @vectorize([float64(float64)], target=target)
    def uP(c):
        return c ** -gam


def CRRAutilityPP(gam, target="parallel"):
    """
    Evaluates constant relative risk aversion (CRRA) marginal marginal utility of
    consumption c given risk aversion parameter gam.

    Parameters
    ----------
    c : float
        Consumption value
    gam : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Marginal marginal utility
    """

    @vectorize([float64(float64)], target=target)
    def uPP(c):
        return -gam * c ** (-gam - 1.0)


def CRRAutilityPPP(gam, target="parallel"):
    """
    Evaluates constant relative risk aversion (CRRA) marginal marginal marginal
    utility of consumption c given risk aversion parameter gam.

    Parameters
    ----------
    c : float
        Consumption value
    gam : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Marginal marginal marginal utility
    """

    @vectorize([float64(float64)], target=target)
    def uPPP(c):
        return (gam + 1.0) * gam * c ** (-gam - 2.0)


def CRRAutilityPPPP(gam, target="parallel"):
    """
    Evaluates constant relative risk aversion (CRRA) marginal marginal marginal
    marginal utility of consumption c given risk aversion parameter gam.

    Parameters
    ----------
    c : float
        Consumption value
    gam : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Marginal marginal marginal marginal utility
    """

    @vectorize([float64(float64)], target=target)
    def uPPPP(c):
        return -(gam + 2.0) * (gam + 1.0) * gam * c ** (-gam - 3.0)


def CRRAutility_inv(gam, target="parallel"):
    """
    Evaluates the inverse of the CRRA utility function (with risk aversion para-
    meter gam) at a given utility level u.

    Parameters
    ----------
    u : float
        Utility value
    gam : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Consumption corresponding to given utility value
    """

    @vectorize([float64(float64)], target=target)
    def invLogU(u):
        return np.exp(u)

    @vectorize([float64(float64)], target=target)
    def invU(u):
        return ((1.0 - gam) * u) ** (1 / (1.0 - gam))

    if gam == 1:
        return invLogU
    else:
        return invU


def CRRAutilityP_inv(gam, target="parallel"):
    """
    Evaluates the inverse of the CRRA marginal utility function (with risk aversion
    parameter gam) at a given marginal utility level uP.

    Parameters
    ----------
    uP : float
        Marginal utility value
    gam : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Consumption corresponding to given marginal utility value.
    """

    @vectorize([float64(float64)], target=target)
    def invUP(uP):
        return uP ** (-1.0 / gam)


def CRRAutility_invP(gam, target="parallel"):
    """
    Evaluates the derivative of the inverse of the CRRA utility function (with
    risk aversion parameter gam) at a given utility level u.

    Parameters
    ----------
    u : float
        Utility value
    gam : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Marginal consumption corresponding to given utility value
    """

    @vectorize([float64(float64)], target=target)
    def invPLogU(u):
        return np.exp(u)

    @vectorize([float64(float64)], target=target)
    def invPU(u):
        return ((1.0 - gam) * u) ** (gam / (1.0 - gam))

    if gam == 1:
        return invPLogU
    else:
        return invPU


def CRRAutilityP_invP(gam, target="parallel"):
    """
    Evaluates the derivative of the inverse of the CRRA marginal utility function
    (with risk aversion parameter gam) at a given marginal utility level uP.

    Parameters
    ----------
    uP : float
        Marginal utility value
    gam : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Marginal consumption corresponding to given marginal utility value
    """

    @vectorize([float64(float64)], target=target)
    def invPUP(uP):
        return (-1.0 / gam) * uP ** (-1.0 / gam - 1.0)


def CARAutility(alpha, target="parallel"):
    """
    Evaluates constant absolute risk aversion (CARA) utility of consumption c
    given risk aversion parameter alpha.

    Parameters
    ----------
    c: float
        Consumption value
    alpha: float
        Risk aversion

    Returns
    -------
    (unnamed): float
        Utility
    """

    @vectorize([float64(float64)], target=target)
    def u(c):
        return 1 - np.exp(-alpha * c) / alpha


def CARAutilityP(alpha, target="parallel"):
    """
    Evaluates constant absolute risk aversion (CARA) marginal utility of
    consumption c given risk aversion parameter alpha.

    Parameters
    ----------
    c: float
        Consumption value
    alpha: float
        Risk aversion

    Returns
    -------
    (unnamed): float
        Marginal utility
    """

    @vectorize([float64(float64)], target=target)
    def uP(c):
        return np.exp(-alpha * c)


def CARAutilityPP(alpha, target="parallel"):
    """
    Evaluates constant absolute risk aversion (CARA) marginal marginal utility
    of consumption c given risk aversion parameter alpha.

    Parameters
    ----------
    c: float
        Consumption value
    alpha: float
        Risk aversion

    Returns
    -------
    (unnamed): float
        Marginal marginal utility
    """

    @vectorize([float64(float64)], target=target)
    def uPP(c):
        return -alpha * np.exp(-alpha * c)


def CARAutilityPPP(alpha, target="parallel"):
    """
    Evaluates constant absolute risk aversion (CARA) marginal marginal marginal
    utility of consumption c given risk aversion parameter alpha.

    Parameters
    ----------
    c: float
        Consumption value
    alpha: float
        Risk aversion

    Returns
    -------
    (unnamed): float
        Marginal marginal marginal utility
    """

    @vectorize([float64(float64)], target=target)
    def uPPP(c):
        return alpha ** 2.0 * np.exp(-alpha * c)


def CARAutility_inv(alpha, target="parallel"):
    """
    Evaluates inverse of constant absolute risk aversion (CARA) utility function
    at utility level u given risk aversion parameter alpha.

    Parameters
    ----------
    u: float
        Utility value
    alpha: float
        Risk aversion

    Returns
    -------
    (unnamed): float
        Consumption value corresponding to u
    """

    @vectorize([float64(float64)], target=target)
    def invU(u):
        return -1.0 / alpha * np.log(alpha * (1 - u))


def CARAutilityP_inv(alpha, target="parallel"):
    """
    Evaluates the inverse of constant absolute risk aversion (CARA) marginal
    utility function at marginal utility uP given risk aversion parameter alpha.

    Parameters
    ----------
    u: float
        Utility value
    alpha: float
        Risk aversion

    Returns
    -------
    (unnamed): float
        Consumption value corresponding to uP
    """

    @vectorize([float64(float64)], target=target)
    def invUP(u):
        return -1.0 / alpha * np.log(u)


def CARAutility_invP(alpha, target="parallel"):
    """
    Evaluates the derivative of inverse of constant absolute risk aversion (CARA)
    utility function at utility level u given risk aversion parameter alpha.

    Parameters
    ----------
    u: float
        Utility value
    alpha: float
        Risk aversion

    Returns
    -------
    (unnamed): float
        Marginal onsumption value corresponding to u
    """

    @vectorize([float64(float64)], target=target)
    def invPU(u):
        return 1.0 / (alpha * (1.0 - u))


@njit(cache=True)
def splrep(x, y):
    """

    Parameters
    ----------
    x
    y

    Returns
    -------

    """
    size = x.size

    xdiff = np.diff(x)
    ydiff = np.diff(y)
    slope = ydiff / xdiff

    d = np.empty(size)
    d[0] = 0
    d[1:-1] = 6 * slope[1:] - 6 * slope[:-1]
    d[-1] = 0

    Ad = np.empty(size)
    Ad[1:-1] = 2 * (xdiff[:-1] + xdiff[1:])

    A = np.diag(Ad) + np.diag(xdiff, -1) + np.diag(xdiff, 1)
    A[0, :3] = [1, -2, 1]
    A[-1, -3:] = [1, -2, 1]

    return np.linalg.solve(A, d)


# interpolate and extrapolate vectors
# extrapolation using end polynomials
@njit(cache=True)
def splevec(x0, x, y, z):
    """
    Parameters
    ----------
    x0: internal point to be evaluated, can be vector
    x: vector of basis points where function is defined
    y: vector of functional values for each point in x
    z: spline coefficients calculated by splrep
    """

    # find index
    index = np.searchsorted(x, x0)
    nx = x.size

    index[index == 0] = 1
    index[index == nx] = nx - 1

    xi1, xi0 = x[index], x[index - 1]
    yi1, yi0 = y[index], y[index - 1]
    zi1, zi0 = z[index], z[index - 1]
    hi1 = xi1 - xi0

    # calculate cubic
    f0 = (
        zi0 / (6 * hi1) * (xi1 - x0) ** 3
        + zi1 / (6 * hi1) * (x0 - xi0) ** 3
        + (yi1 / hi1 - zi1 * hi1 / 6) * (x0 - xi0)
        + (yi0 / hi1 - zi0 * hi1 / 6) * (xi1 - x0)
    )

    return f0


# interpolate and extrapolate scalars
# extrapolation using end polynomials
@njit
def spleval(x0, x, y, z):
    index = np.searchsorted(x, x0)
    nx = x.size

    index = 1 if index == 0 else index
    index = nx - 1 if index == nx else index

    xi1, xi0 = x[index], x[index - 1]
    yi1, yi0 = y[index], y[index - 1]
    zi1, zi0 = z[index], z[index - 1]
    hi1 = xi1 - xi0

    # calculate cubic
    f0 = (
        zi0 / (6 * hi1) * (xi1 - x0) ** 3
        + zi1 / (6 * hi1) * (x0 - xi0) ** 3
        + (yi1 / hi1 - zi1 * hi1 / 6) * (x0 - xi0)
        + (yi0 / hi1 - zi0 * hi1 / 6) * (xi1 - x0)
    )

    return f0


@njit
def spldervec(x0, x, y, z):
    # find index
    index = np.searchsorted(x, x0)
    nx = x.size

    index[index == 0] = 1
    index[index == nx] = nx - 1

    xi1, xi0 = x[index], x[index - 1]
    yi1, yi0 = y[index], y[index - 1]
    zi1, zi0 = z[index], z[index - 1]
    hi1 = xi1 - xi0

    # calculate cubic
    df0 = (
        -zi0 / (2 * hi1) * (xi1 - x0) ** 2
        + zi1 / (2 * hi1) * (x0 - xi0) ** 2
        + (yi1 / hi1 - zi1 * hi1 / 6)
        - (yi0 / hi1 - zi0 * hi1 / 6)
    )

    return df0


@njit
def splder(x0, x, y, z):
    # find index
    index = np.searchsorted(x, x0)
    nx = x.size

    index = 1 if index == 0 else index
    index = nx - 1 if index == nx else index

    xi1, xi0 = x[index], x[index - 1]
    yi1, yi0 = y[index], y[index - 1]
    zi1, zi0 = z[index], z[index - 1]
    hi1 = xi1 - xi0

    # calculate cubic
    df0 = (
        -zi0 / (2 * hi1) * (xi1 - x0) ** 2
        + zi1 / (2 * hi1) * (x0 - xi0) ** 2
        + (yi1 / hi1 - zi1 * hi1 / 6)
        - (yi0 / hi1 - zi0 * hi1 / 6)
    )

    return df0


@njit
def splkd(x, y, z):
    size = x.size

    xdiff = np.diff(x)
    ydiff = np.diff(y)
    slope = ydiff / xdiff

    df = np.empty(size)

    df[:-1] = -xdiff / 3 * z[:-1] - xdiff / 6 * z[1:] + slope

    df[-1] = xdiff[-1] / 3 * z[-1] + xdiff[-1] / 6 * z[-2] + slope[-1]

    return df
