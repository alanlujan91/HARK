import math
from itertools import product

import numpy as np
import scipy.stats as stats
from scipy.special import erf, erfc
from xarray import DataArray


class Distribution:
    """
    Base class for all probability distributions.

    Parameters
    ----------
    seed : int
        Seed for random number generator.
    """

    def __init__(self, seed=0):
        self.RNG = np.random.RandomState(seed)
        self.seed = seed

    def reset(self):
        """
        Reset the random number generator of this distribution.

        Parameters
        ----------
        """
        self.RNG = np.random.RandomState(self.seed)


class IndexDistribution(Distribution):
    """
    This class provides a way to define a distribution that
    is conditional on an index.

    The current implementation combines a defined distribution
    class (such as Bernoulli, LogNormal, etc.) with information
    about the conditions on the parameters of the distribution.

    For example, an IndexDistribution can be defined as
    a Bernoulli distribution whose parameter p is a function of
    a different inpute parameter.

    Parameters
    ----------

    engine : Distribution class
        A Distribution subclass.

    conditional: dict
        Information about the conditional variation
        on the input parameters of the engine distribution.
        Keys should match the arguments to the engine class
        constructor.

    seed : int
        Seed for random number generator.
    """

    conditional = None
    engine = None

    def __init__(self, engine, conditional, RNG=None, seed=0):

        if RNG is None:
            # Set up the RNG
            super().__init__(seed)
        else:
            # If an RNG is received, use it in whatever state it is in.
            self.RNG = RNG
            # The seed will still be set, even if it is not used for the RNG,
            # for whenever self.reset() is called.
            # Note that self.reset() will stop using the RNG that was passed
            # and create a new one.
            self.seed = seed

        self.conditional = conditional
        self.engine = engine

        self.dstns = []

        # Test one item to determine case handling
        item0 = list(self.conditional.values())[0]

        if type(item0) is list:
            # Create and store all the conditional distributions
            for y in range(len(item0)):
                cond = {key: val[y] for (key, val) in self.conditional.items()}
                self.dstns.append(
                    self.engine(seed=self.RNG.randint(0, 2**31 - 1), **cond)
                )

        elif type(item0) is float:

            self.dstns = [
                self.engine(seed=self.RNG.randint(0, 2**31 - 1), **conditional)
            ]

        else:
            raise (
                Exception(
                    f"IndexDistribution: Unhandled case for __getitem__ access. item0: {item0}; conditional: {self.conditional}"
                )
            )

    def __getitem__(self, y):

        return self.dstns[y]

    def approx(self, N, **kwds):
        """
        Approximation of the distribution.

        Parameters
        ----------
        N : init
            Number of discrete points to approximate
            continuous distribution into.

        kwds: dict
            Other keyword arguments passed to engine
            distribution approx() method.

        Returns:
        ------------
        dists : [DiscreteDistribution]
            A list of DiscreteDistributions that are the
            approximation of engine distribution under each condition.

            TODO: It would be better if there were a conditional discrete
            distribution representation. But that integrates with the
            solution code. This implementation will return the list of
            distributions representations expected by the solution code.
        """

        # test one item to determine case handling
        item0 = list(self.conditional.values())[0]

        if type(item0) is float:
            # degenerate case. Treat the parameterization as constant.
            return self.dstns[0].approx(N, **kwds)

        if type(item0) is list:
            return TimeVaryingDiscreteDistribution(
                [self[i].approx(N, **kwds) for i, _ in enumerate(item0)]
            )

    def draw(self, condition):
        """
        Generate arrays of draws.
        The input is an array containing the conditions.
        The output is an array of the same length (axis 1 dimension)
        as the conditions containing random draws of the conditional
        distribution.

        Parameters
        ----------
        condition : np.array
            The input conditions to the distribution.

        Returns:
        ------------
        draws : np.array
        """
        # for now, assume that all the conditionals
        # are of the same type.
        # this matches the HARK 'time-varying' model architecture.

        # test one item to determine case handling
        item0 = list(self.conditional.values())[0]

        if type(item0) is float:
            # degenerate case. Treat the parameterization as constant.
            N = condition.size

            return self.engine(
                seed=self.RNG.randint(0, 2**31 - 1), **self.conditional
            ).draw(N)

        if type(item0) is list:
            # conditions are indices into list
            # somewhat convoluted sampling strategy retained
            # for test backwards compatibility

            draws = np.zeros(condition.size)

            for c in np.unique(condition):
                these = c == condition
                N = np.sum(these)

                cond = {key: val[c] for (key, val) in self.conditional.items()}
                draws[these] = self[c].draw(N)

            return draws


class TimeVaryingDiscreteDistribution(Distribution):
    """
    This class provides a way to define a discrete distribution that
    is conditional on an index.

    Wraps a list of discrete distributions.

    Parameters
    ----------

    distributions : [DiscreteDistribution]
        A list of discrete distributions

    seed : int
        Seed for random number generator.
    """

    distributions = []

    def __init__(self, distributions, seed=0):
        # Set up the RNG
        super().__init__(seed)

        self.distributions = distributions

    def __getitem__(self, y):
        return self.distributions[y]

    def draw(self, condition):
        """
        Generate arrays of draws.
        The input is an array containing the conditions.
        The output is an array of the same length (axis 1 dimension)
        as the conditions containing random draws of the conditional
        distribution.

        Parameters
        ----------
        condition : np.array
            The input conditions to the distribution.

        Returns:
        ------------
        draws : np.array
        """
        # for now, assume that all the conditionals
        # are of the same type.
        # this matches the HARK 'time-varying' model architecture.

        # conditions are indices into list
        # somewhat convoluted sampling strategy retained
        # for test backwards compatibility

        draws = np.zeros(condition.size)

        for c in np.unique(condition):
            these = c == condition
            N = np.sum(these)

            draws[these] = self.distributions[c].draw(N)

        return draws


### CONTINUOUS DISTRIBUTIONS


class Lognormal(Distribution):
    """
    A Lognormal distribution

    Parameters
    ----------
    mu : float or [float]
        One or more means of underlying normal distribution.
        Number of elements T in mu determines number of rows of output.
    sigma : float or [float]
        One or more standard deviations of underlying normal distribution.
        Number of elements T in sigma determines number of rows of output.
    seed : int
        Seed for random number generator.
    """

    mu = None
    sigma = None

    def __init__(self, mu=0.0, sigma=1.0, seed=0):
        self.mu = np.array(mu)
        self.sigma = np.array(sigma)
        # Set up the RNG
        super().__init__(seed)

        if self.mu.size != self.sigma.size:
            raise Exception(
                "mu and sigma must be of same size, are %s, %s"
                % ((self.mu.size), (self.sigma.size))
            )

    def draw(self, N):
        """
        Generate arrays of lognormal draws. The sigma parameter can be a number
        or list-like.  If a number, output is a length N array of draws from the
        lognormal distribution with standard deviation sigma. If a list, output is
        a length T list whose t-th entry is a length N array of draws from the
        lognormal with standard deviation sigma[t].

        Parameters
        ----------
        N : int
            Number of draws in each row.

        Returns:
        ------------
        draws : np.array or [np.array]
            T-length list of arrays of mean one lognormal draws each of size N, or
            a single array of size N (if sigma is a scalar).
        """

        draws = []
        for j in range(self.mu.size):
            draws.append(
                self.RNG.lognormal(
                    mean=self.mu.item(j), sigma=self.sigma.item(j), size=N
                )
            )
        # TODO: change return type to np.array?
        return draws[0] if len(draws) == 1 else draws

    def approx(self, N, tail_N=0, tail_bound=None, tail_order=np.e):
        """
        Construct a discrete approximation to a lognormal distribution with underlying
        normal distribution N(mu,sigma).  Makes an equiprobable distribution by
        default, but user can optionally request augmented tails with exponentially
        sized point masses.  This can improve solution accuracy in some models.

        Parameters
        ----------
        N: int
            Number of discrete points in the "main part" of the approximation.
        tail_N: int
            Number of points in each "tail part" of the approximation; 0 = no tail.
        tail_bound: [float]
            CDF boundaries of the tails vs main portion; tail_bound[0] is the lower
            tail bound, tail_bound[1] is the upper tail bound.  Inoperative when
            tail_N = 0.  Can make "one tailed" approximations with 0.0 or 1.0.
        tail_order: float
            Factor by which consecutive point masses in a "tail part" differ in
            probability.  Should be >= 1 for sensible spacing.

        Returns
        -------
        d : DiscreteDistribution
            Probability associated with each point in array of discrete
            points for discrete probability mass function.
        """
        tail_bound = tail_bound if tail_bound is not None else [0.02, 0.98]
        # Find the CDF boundaries of each segment
        if self.sigma > 0.0:
            if tail_N > 0:
                lo_cut = tail_bound[0]
                hi_cut = tail_bound[1]
            else:
                lo_cut = 0.0
                hi_cut = 1.0
            inner_size = hi_cut - lo_cut
            inner_CDF_vals = [
                lo_cut + x * N ** (-1.0) * inner_size for x in range(1, N)
            ]
            if inner_size < 1.0:
                scale = 1.0 / tail_order
                mag = (1.0 - scale**tail_N) / (1.0 - scale)
            lower_CDF_vals = [0.0]
            if lo_cut > 0.0:
                for x in range(tail_N - 1, -1, -1):
                    lower_CDF_vals.append(
                        lower_CDF_vals[-1] + lo_cut * scale**x / mag
                    )
            upper_CDF_vals = [hi_cut]
            if hi_cut < 1.0:
                for x in range(tail_N):
                    upper_CDF_vals.append(
                        upper_CDF_vals[-1] + (1.0 - hi_cut) * scale**x / mag
                    )
            CDF_vals = lower_CDF_vals + inner_CDF_vals + upper_CDF_vals
            temp_cutoffs = list(
                stats.lognorm.ppf(
                    CDF_vals[1:-1], s=self.sigma, loc=0, scale=np.exp(self.mu)
                )
            )
            cutoffs = [0] + temp_cutoffs + [np.inf]
            CDF_vals = np.array(CDF_vals)

            K = CDF_vals.size - 1  # number of points in approximation
            prob = CDF_vals[1 : (K + 1)] - CDF_vals[0:K]
            data = np.zeros(K)
            for i in range(K):
                zBot = cutoffs[i]
                zTop = cutoffs[i + 1]
                # Manual check to avoid the RuntimeWarning generated by "divide by zero"
                # with np.log(zBot).
                if zBot == 0:
                    tempBot = np.inf
                else:
                    tempBot = (self.mu + self.sigma**2 - np.log(zBot)) / (
                        np.sqrt(2) * self.sigma
                    )
                tempTop = (self.mu + self.sigma**2 - np.log(zTop)) / (
                    np.sqrt(2) * self.sigma
                )
                if tempBot <= 4:
                    data[i] = (
                        -0.5
                        * np.exp(self.mu + (self.sigma**2) * 0.5)
                        * (erf(tempTop) - erf(tempBot))
                        / prob[i]
                    )
                else:
                    data[i] = (
                        -0.5
                        * np.exp(self.mu + (self.sigma**2) * 0.5)
                        * (erfc(tempBot) - erfc(tempTop))
                        / prob[i]
                    )

        else:
            prob = np.ones(N) / N
            data = np.exp(self.mu) * np.ones(N)
        return DiscreteDistribution(
            prob, data, seed=self.RNG.randint(0, 2**31 - 1, dtype="int32")
        )

    @classmethod
    def from_mean_std(cls, mean, std, seed=0):
        """
        Construct a LogNormal distribution from its
        mean and standard deviation.

        This is unlike the normal constructor for the distribution,
        which takes the mu and sigma for the normal distribution
        that is the logarithm of the Log Normal distribution.

        Parameters
        ----------
        mean : float or [float]
            One or more means.  Number of elements T in mu determines number
            of rows of output.
        std : float or [float]
            One or more standard deviations. Number of elements T in sigma
            determines number of rows of output.
        seed : int
            Seed for random number generator.

        Returns
        ---------
        LogNormal

        """
        mean_squared = mean**2
        variance = std**2
        mu = np.log(mean / (np.sqrt(1.0 + variance / mean_squared)))
        sigma = np.sqrt(np.log(1.0 + variance / mean_squared))

        return cls(mu=mu, sigma=sigma, seed=seed)


class MeanOneLogNormal(Lognormal):
    def __init__(self, sigma=1.0, seed=0):
        mu = -0.5 * sigma**2
        super().__init__(mu=mu, sigma=sigma, seed=seed)


class Normal(Distribution):
    """
    A Normal distribution.

    Parameters
    ----------
    mu : float or [float]
        One or more means.  Number of elements T in mu determines number
        of rows of output.
    sigma : float or [float]
        One or more standard deviations. Number of elements T in sigma
        determines number of rows of output.
    seed : int
        Seed for random number generator.
    """

    mu = None
    sigma = None

    def __init__(self, mu=0.0, sigma=1.0, seed=0):
        self.mu = np.array(mu)
        self.sigma = np.array(sigma)
        super().__init__(seed)

    def draw(self, N):
        """
        Generate arrays of normal draws.  The mu and sigma inputs can be numbers or
        list-likes.  If a number, output is a length N array of draws from the normal
        distribution with mean mu and standard deviation sigma. If a list, output is
        a length T list whose t-th entry is a length N array with draws from the
        normal distribution with mean mu[t] and standard deviation sigma[t].

        Parameters
        ----------
        N : int
            Number of draws in each row.

        Returns
        -------
        draws : np.array or [np.array]
            T-length list of arrays of normal draws each of size N, or a single array
            of size N (if sigma is a scalar).
        """
        draws = []
        for t in range(self.sigma.size):
            draws.append(self.sigma.item(t) * self.RNG.randn(N) + self.mu.item(t))

        return draws

    def approx(self, N):
        """
        Returns a discrete approximation of this distribution.
        """
        x, w = np.polynomial.hermite.hermgauss(N)
        # normalize w
        prob = w * np.pi**-0.5
        # correct x
        data = math.sqrt(2.0) * self.sigma * x + self.mu
        return DiscreteDistribution(
            prob, data, seed=self.RNG.randint(0, 2**31 - 1, dtype="int32")
        )

    def approx_equiprobable(self, N):

        CDF = np.linspace(0, 1, N + 1)
        lims = stats.norm.ppf(CDF)
        pdf = stats.norm.pdf(lims)

        # Find conditional means using Mills's ratio
        prob = np.diff(CDF)
        data = self.mu - np.diff(pdf) / prob * self.sigma

        return DiscreteDistribution(
            prob, data, seed=self.RNG.randint(0, 2**31 - 1, dtype="int32")
        )


class MVNormal(Distribution):
    """
    A Multivariate Normal distribution.

    Parameters
    ----------
    mu : numpy array
        Mean vector.
    Sigma : 2-d numpy array. Each dimension must have length equal to that of
            mu.
        Variance-covariance matrix.
    seed : int
        Seed for random number generator.
    """

    mu = None
    Sigma = None

    def __init__(self, mu=np.array([1, 1]), Sigma=np.array([[1, 0], [0, 1]]), seed=0):
        self.mu = mu
        self.Sigma = Sigma
        self.M = len(self.mu)
        super().__init__(seed)

    def draw(self, N):
        """
        Generate an array of multivariate normal draws.

        Parameters
        ----------
        N : int
            Number of multivariate draws.

        Returns
        -------
        draws : np.array
            Array of dimensions N x M containing the random draws, where M is
            the dimension of the multivariate normal and N is the number of
            draws. Each row represents a draw.
        """
        draws = self.RNG.multivariate_normal(self.mu, self.Sigma, N)

        return draws

    def approx(self, N, equiprobable=False):
        """
        Returns a discrete approximation of this distribution.

        The discretization will have N**M points, where M is the dimension of
        the multivariate normal.

        It uses the fact that:
            - Being positive definite, Sigma can be factorized as Sigma = QVQ',
              with V diagonal. So, letting A=Q*sqrt(V), Sigma = A*A'.
            - If Z is an N-dimensional multivariate standard normal, then
              A*Z ~ N(0,A*A' = Sigma).

        The idea therefore is to construct an equiprobable grid for a standard
        normal and multiply it by matrix A.
        """

        # Start by computing matrix A.
        v, Q = np.linalg.eig(self.Sigma)
        sqrtV = np.diag(np.sqrt(v))
        A = np.matmul(Q, sqrtV)

        # Now find a discretization for a univariate standard normal.
        if equiprobable:
            z_approx = Normal().approx_equiprobable(N)
        else:
            z_approx = Normal().approx(N)

        # Now create the multivariate grid and prob
        Z = np.array(list(product(*[z_approx.data.flatten()] * self.M)))
        prob = np.prod(np.array(list(product(*[z_approx.prob] * self.M))), axis=1)

        # Apply mean and standard deviation to the Z grid
        data = self.mu[None, ...] + np.matmul(Z, A.T)

        # Construct and return discrete distribution
        return DiscreteDistribution(
            prob, data.T, seed=self.RNG.randint(0, 2**31 - 1, dtype="int32")
        )


class Weibull(Distribution):
    """
    A Weibull distribution.

    Parameters
    ----------
    scale : float or [float]
        One or more scales.  Number of elements T in scale
        determines number of
        rows of output.
    shape : float or [float]
        One or more shape parameters. Number of elements T in scale
        determines number of rows of output.
    seed : int
        Seed for random number generator.
    """

    scale = None
    shape = None

    def __init__(self, scale=1.0, shape=1.0, seed=0):
        self.scale = np.array(scale)
        self.shape = np.array(shape)
        # Set up the RNG
        super().__init__(seed)

    def draw(self, N):
        """
        Generate arrays of Weibull draws.  The scale and shape inputs can be
        numbers or list-likes.  If a number, output is a length N array of draws from
        the Weibull distribution with the given scale and shape. If a list, output
        is a length T list whose t-th entry is a length N array with draws from the
        Weibull distribution with scale scale[t] and shape shape[t].

        Note: When shape=1, the Weibull distribution is simply the exponential dist.

        Mean: scale*Gamma(1 + 1/shape)

        Parameters
        ----------
        N : int
            Number of draws in each row.

        Returns:
        ------------
        draws : np.array or [np.array]
            T-length list of arrays of Weibull draws each of size N, or a single
            array of size N (if sigma is a scalar).
        """
        draws = []
        for j in range(self.scale.size):
            draws.append(
                self.scale.item(j)
                * (-np.log(1.0 - self.RNG.rand(N))) ** (1.0 / self.shape.item(j))
            )
        return draws[0] if len(draws) == 1 else draws


class Uniform(Distribution):
    """
    A Uniform distribution.

    Parameters
    ----------
    bot : float or [float]
        One or more bottom values.
        Number of elements T in mu determines number
        of rows of output.
    top : float or [float]
        One or more top values.
        Number of elements T in top determines number of
        rows of output.
    seed : int
        Seed for random number generator.
    """

    bot = None
    top = None

    def __init__(self, bot=0.0, top=1.0, seed=0):
        self.bot = np.array(bot)
        self.top = np.array(top)
        # Set up the RNG
        self.RNG = np.random.RandomState(seed)

    def draw(self, N):
        """
        Generate arrays of uniform draws.  The bot and top inputs can be numbers or
        list-likes.  If a number, output is a length N array of draws from the
        uniform distribution on [bot,top]. If a list, output is a length T list
        whose t-th entry is a length N array with draws from the uniform distribution
        on [bot[t],top[t]].

        Parameters
        ----------
        N : int
            Number of draws in each row.

        Returns
        -------
        draws : np.array or [np.array]
            T-length list of arrays of uniform draws each of size N, or a single
            array of size N (if sigma is a scalar).
        """
        draws = []
        for j in range(self.bot.size):
            draws.append(
                self.bot.item(j)
                + (self.top.item(j) - self.bot.item(j)) * self.RNG.rand(N)
            )
        return draws[0] if len(draws) == 1 else draws

    def approx(self, N):
        """
        Makes a discrete approximation to this uniform distribution.

        Parameters
        ----------
        N : int
            The number of points in the discrete approximation

        Returns
        -------
        d : DiscreteDistribution
            Probability associated with each point in array of discrete
            points for discrete probability mass function.
        """
        prob = np.ones(N) / float(N)
        center = (self.top + self.bot) / 2.0
        width = (self.top - self.bot) / 2.0
        data = center + width * np.linspace(-(N - 1.0) / 2.0, (N - 1.0) / 2.0, N) / (
            N / 2.0
        )
        return DiscreteDistribution(
            prob, data, seed=self.RNG.randint(0, 2**31 - 1, dtype="int32")
        )


### DISCRETE DISTRIBUTIONS


class Bernoulli(Distribution):
    """
    A Bernoulli distribution.

    Parameters
    ----------
    p : float or [float]
        Probability or probabilities of the event occurring (True).

    seed : int
        Seed for random number generator.
    """

    p = None

    def __init__(self, p=0.5, seed=0):
        self.p = np.array(p)
        # Set up the RNG
        super().__init__(seed)

    def draw(self, N):
        """
        Generates arrays of booleans drawn from a simple Bernoulli distribution.
        The input p can be a float or a list-like of floats; its length T determines
        the number of entries in the output.  The t-th entry of the output is an
        array of N booleans which are True with probability p[t] and False otherwise.

        Arguments
        ---------
        N : int
            Number of draws in each row.

        Returns
        -------
        draws : np.array or [np.array]
            T-length list of arrays of Bernoulli draws each of size N, or a single
        array of size N (if sigma is a scalar).
        """
        draws = []
        for j in range(self.p.size):
            draws.append(self.RNG.uniform(size=N) < self.p.item(j))
        return draws[0] if len(draws) == 1 else draws


class DiscreteDistribution(Distribution):
    """
    A representation of a discrete probability distribution.

    Parameters
    ----------
    prob : np.array
        An array of floats representing a probability mass function.
    data : np.array
        Discrete point values for each probability mass.
        For multivariate distributions, the last dimension of data must index
        "nature" or the random realization. For instance, if data.shape == (2,6,4),
        the random variable has 4 possible realizations and each of them has shape (2,6).
    seed : int
        Seed for random number generator.
    """

    prob = None
    data = None

    def __init__(self, prob, data, seed=0):

        self.prob = prob

        if len(data.shape) < 2:
            self.data = data[None, ...]
        else:
            self.data = data

        # Set up the RNG
        super().__init__(seed)

        # Check that prob and data have compatible dimensions.
        same_dims = len(prob) == data.shape[-1]
        if not same_dims:
            raise ValueError(
                "Provided prob and data arrays have incompatible dimensions. "
                + "The length of the prob must be equal to that of data's last dimension."
            )

    def dim(self):
        """
        Last dimension of self.data indexes "nature."
        """
        return self.data.shape[:-1]

    def draw_events(self, n):
        """
        Draws N 'events' from the distribution PMF.
        These events are indices into data.
        """
        # Generate a cumulative distribution
        base_draws = self.RNG.uniform(size=n)
        cum_dist = np.cumsum(self.prob)

        # Convert the basic uniform draws into discrete draws
        indices = cum_dist.searchsorted(base_draws)

        return indices

    def draw(self, N, data=None, exact_match=False):
        """
        Simulates N draws from a discrete distribution with probabilities P and outcomes data.

        Parameters
        ----------
        N : int
            Number of draws to simulate.
        data : None, int, or np.array
            If None, then use this distribution's data for point values.
            If an int, then the index of data for the point values.
            If an np.array, use the array for the point values.
        exact_match : boolean
            Whether the draws should "exactly" match the discrete distribution (as
            closely as possible given finite draws).  When True, returned draws are
            a random permutation of the N-length list that best fits the discrete
            distribution.  When False (default), each draw is independent from the
            others and the result could deviate from the input.

        Returns
        -------
        draws : np.array
            An array of draws from the discrete distribution; each element is a value in data.
        """
        if data is None:
            data = self.data
        elif isinstance(data, int):
            data = self.data[data]

        if exact_match:
            events = np.arange(self.prob.size)  # just a list of integers
            cutoffs = np.round(np.cumsum(self.prob) * N).astype(
                int
            )  # cutoff points between discrete outcomes
            top = 0

            # Make a list of event indices that closely matches the discrete distribution
            event_list = []
            for j in range(events.size):
                bot = top
                top = cutoffs[j]
                event_list += (top - bot) * [events[j]]

            # Randomly permute the event indices
            indices = self.RNG.permutation(event_list)

        # Draw event indices randomly from the discrete distribution
        else:
            indices = self.draw_events(N)

        # Create and fill in the output array of draws based on the output of event indices
        draws = data[..., indices]

        # TODO: some models expect univariate draws to just be a 1d vector. Fix those models.
        if len(draws.shape) == 2 and draws.shape[0] == 1:
            draws = draws.flatten()

        return draws

    def expected_value(self, func=None, *args):
        """
        Expected value of a function, given an array of configurations of its
        inputs along with a DiscreteDistribution object that specifies the
        probability of each configuration.

        Parameters
        ----------
        func : function
            The function to be evaluated.
            This function should take the full array of distribution values
            and return either arrays of arbitrary shape or scalars.
            It may also take other arguments *args.
            This function differs from the standalone `calc_expectation`
            method in that it uses numpy's vectorization and broadcasting
            rules to avoid costly iteration.
            Note: If you need to use a function that acts on single outcomes
            of the distribution, consier `distribution.calc_expectation`.
        *args :
            Other inputs for func, representing the non-stochastic arguments.
            The the expectation is computed at f(dstn, *args).

        Returns
        -------
        f_exp : np.array or scalar
            The expectation of the function at the queried values.
            Scalar if only one value.
        """

        if func is None:
            # if no function is provided, it's much faster to go straight
            # to dot product instead of calling the dummy function.
            f_query = self.data
        else:
            # if a function is provided, we need to add one more dimension,
            # the nature dimension, to any inputs that are n-dim arrays.
            # This allows numpy to easily broadcast the function's output.
            # For more information on broadcasting, see:
            # https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules
            args = [
                arg[..., np.newaxis] if isinstance(arg, np.ndarray) else arg
                for arg in args
            ]

            f_query = func(self.data, *args)

        f_exp = np.dot(f_query, self.prob)

        return f_exp

    def dist_of_func(self, func=lambda x: x, xarray=False, *args, **kwargs):
        """
        Finds the distribution of a random variable Y that is a function
        of discrete random variable data, Y=f(data).

        Parameters
        ----------
        func : function
            The function to be evaluated.
            This function should take the full array of distribution values.
            It may also take other arguments *args.
        *args :
            Additional non-stochastic arguments for func,
            The function is computed as f(dstn, *args).

        Returns
        -------
        f_dstn : DiscreteDistribution or DiscreteDistributionXRA
            The distribution of func(dstn).
        """
        # we need to add one more dimension,
        # the nature dimension, to any inputs that are n-dim arrays.
        # This allows numpy to easily broadcast the function's output.
        args = [
            arg[..., np.newaxis] if isinstance(arg, np.ndarray) else arg for arg in args
        ]
        f_query = func(self.data, *args)

        if xarray:
            f_dstn = DiscreteDistributionXRA(
                list(self.prob), f_query, seed=self.seed, **kwargs
            )
        else:
            f_dstn = DiscreteDistribution(list(self.prob), f_query, seed=self.seed)

        return f_dstn


class DiscreteDistributionXRA(DiscreteDistribution):
    """
    A representation of a discrete probability distribution
    stored in an underlying `xarray.DataArray` array.

    Parameters
    ----------
    prob : np.array
        An array of floats representing a probability mass function.
    data : np.array
        Discrete point values for each probability mass.
        For multivariate distributions, the last dimension of data must index
        "nature" or the random realization. For instance, if data.shape == (2,6,4),
        the random variable has 4 possible realizations and each of them has shape (2,6).
    seed : int
        Seed for random number generator.
    coords : dict
        Coordinate values/names for each dimension of the underlying array.
    dims : tuple or list
        Dimension names for each dimension of the underlying array.
    name : str
        Name of the distribution.
    attrs: dict
        Attributes for the distribution.
    """

    def __init__(
        self,
        prob,
        data,
        seed=0,
        coords=None,
        dims=None,
        name="DiscreteDistributionXRA",
        attrs=None,
    ):

        if data.ndim < 2:
            data = data[np.newaxis, ...]

        if attrs is None:
            attrs = {}

        attrs["prob"] = np.asarray(prob)
        attrs["seed"] = seed
        attrs["RNG"] = np.random.RandomState(seed)

        self._xarray = DataArray(
            data=data,
            coords=coords,
            dims=dims,
            name=name,
            attrs=attrs,
        )

    @property
    def xarray(self):
        """
        Returns the underlying xarray.DataArray object.
        """
        return self._xarray

    @property
    def values(self):
        """
        Returns the distribution's data as a numpy.ndarray.
        """
        return self._xarray.values

    @property
    def prob(self):
        """
        Returns the distribution's probability mass function.
        """
        return self._xarray.prob

    @property
    def RNG(self):
        """
        Returns the distribution's random number generator.
        """
        return self._xarray.RNG

    @property
    def data(self):
        """
        The distribution's data as an array. The underlying
        array type (e.g. dask, sparse, pint) is preserved.
        """
        return self._xarray.data

    @property
    def coords(self):
        """
        The distribution's coordinates.
        """
        return self._xarray.coords

    @property
    def dims(self):
        """
        The distribution's dimensions.
        """
        return self._xarray.dims

    @property
    def name(self):
        """
        The distribution's name.
        """
        return self._xarray.name

    @property
    def attrs(self):
        """
        The distribution's attributes.
        """
        return self._xarray.attrs

    def expected_value(self, func=None, *args, labels=False):
        """
        Expectation of a function, given an array of configurations of its inputs
        along with a DiscreteDistributionXRA object that specifies the probability
        of each configuration.

        Parameters
        ----------
        func : function
            The function to be evaluated.
            This function should take the full array of distribution values
            and return either arrays of arbitrary shape or scalars.
            It may also take other arguments *args.
            This function differs from the standalone `calc_expectation`
            method in that it uses numpy's vectorization and broadcasting
            rules to avoid costly iteration.
            Note: If you need to use a function that acts on single outcomes
            of the distribution, consier `distribution.calc_expectation`.
        *args :
            Other inputs for func, representing the non-stochastic arguments.
            The the expectation is computed at f(dstn, *args).
        labels : bool
            If True, the function should use labeled indexing instead of integer
            indexing using the distribution's underlying rv coordinates. For example,
            if `dims = ('rv', 'x')` and `coords = {'rv': ['a', 'b'], }`, then
            the function can be `lambda x: x["a"] + x["b"]`.

        Returns
        -------
        f_exp : np.array or scalar
            The expectation of the function at the queried values.
            Scalar if only one value.
        """

        def func_wrapper(x, *args):
            """
            Wrapper function for `func` that handles labeled indexing.
            """
            dim_0 = self.dims[0]
            idx = self.coords[dim_0].values

            wrapped = dict(zip(idx, x))

            return func(wrapped, *args)

        if labels:
            which_func = func_wrapper
        else:
            which_func = func

        return super().expected_value(which_func, *args)


def approx_lognormal_gauss_hermite(N, mu=0.0, sigma=1.0, seed=0):
    d = Normal(mu, sigma).approx(N)
    return DiscreteDistribution(d.prob, np.exp(d.data), seed=seed)


def calc_normal_style_pars_from_lognormal_pars(avg_lognormal, std_lognormal):
    varLognormal = std_lognormal**2
    varNormal = math.log(1 + varLognormal / avg_lognormal**2)
    avgNormal = math.log(avg_lognormal) - varNormal * 0.5
    std_normal = math.sqrt(varNormal)
    return avgNormal, std_normal


def calc_lognormal_style_pars_from_normal_pars(mu_normal, std_normal):
    varNormal = std_normal**2
    avg_lognormal = math.exp(mu_normal + varNormal * 0.5)
    varLognormal = (math.exp(varNormal) - 1) * avg_lognormal**2
    std_lognormal = math.sqrt(varLognormal)
    return avg_lognormal, std_lognormal


def approx_beta(N, a=1.0, b=1.0):
    """
    Calculate a discrete approximation to the beta distribution.  May be quite
    slow, as it uses a rudimentary numeric integration method to generate the
    discrete approximation.

    Parameters
    ----------
    N : int
        Size of discrete space vector to be returned.
    a : float
        First shape parameter (sometimes called alpha).
    b : float
        Second shape parameter (sometimes called beta).

    Returns
    -------
    d : DiscreteDistribution
        Probability associated with each point in array of discrete
        points for discrete probability mass function.
    """
    P = 1000
    vals = np.reshape(stats.beta.ppf(np.linspace(0.0, 1.0, N * P), a, b), (N, P))
    data = np.mean(vals, axis=1)
    prob = np.ones(N) / float(N)
    return DiscreteDistribution(prob, data)


def make_markov_approx_to_normal(x_grid, mu, sigma, K=351, bound=3.5):
    """
    Creates an approximation to a normal distribution with mean mu and standard
    deviation sigma, returning a stochastic vector called p_vec, corresponding
    to values in x_grid.  If a RV is distributed x~N(mu,sigma), then the expectation
    of a continuous function f() is E[f(x)] = numpy.dot(p_vec,f(x_grid)).

    Parameters
    ----------
    x_grid: numpy.array
        A sorted 1D array of floats representing discrete values that a normally
        distributed RV could take on.
    mu: float
        Mean of the normal distribution to be approximated.
    sigma: float
        Standard deviation of the normal distribution to be approximated.
    K: int
        Number of points in the normal distribution to sample.
    bound: float
        Truncation bound of the normal distribution, as +/- bound*sigma.

    Returns
    -------
    p_vec: numpy.array
        A stochastic vector with probability weights for each x in x_grid.
    """
    x_n = x_grid.size  # Number of points in the outcome grid
    lower_bound = -bound  # Lower bound of normal draws to consider, in SD
    upper_bound = bound  # Upper bound of normal draws to consider, in SD
    raw_sample = np.linspace(
        lower_bound, upper_bound, K
    )  # Evenly spaced draws between bounds
    f_weights = stats.norm.pdf(raw_sample)  # Relative probability of each draw
    sample = mu + sigma * raw_sample  # Adjusted bounds, given mean and stdev
    w_vec = np.zeros(x_n)  # A vector of outcome weights

    # Find the relative position of each of the draws
    sample_pos = np.searchsorted(x_grid, sample)
    sample_pos[sample_pos < 1] = 1
    sample_pos[sample_pos > x_n - 1] = x_n - 1

    # Make arrays of the x_grid point directly above and below each draw
    bot = x_grid[sample_pos - 1]
    top = x_grid[sample_pos]
    alpha = (sample - bot) / (top - bot)

    # Keep the weights (alpha) in bounds
    alpha_clipped = np.clip(alpha, 0.0, 1.0)

    # Loop through each x_grid point and add up the probability that each nearby
    # draw contributes to it (accounting for distance)
    for j in range(1, x_n):
        c = sample_pos == j
        w_vec[j - 1] = w_vec[j - 1] + np.dot(f_weights[c], 1.0 - alpha_clipped[c])
        w_vec[j] = w_vec[j] + np.dot(f_weights[c], alpha_clipped[c])

    # Reweight the probabilities so they sum to 1
    W = np.sum(w_vec)
    p_vec = w_vec / W

    # Check for obvious errors, and return p_vec
    assert (
        (np.all(p_vec >= 0.0))
        and (np.all(p_vec <= 1.0))
        and (np.isclose(np.sum(p_vec), 1.0))
    )
    return p_vec


def make_markov_approx_to_normal_by_monte_carlo(x_grid, mu, sigma, N_draws=10000):
    """
    Creates an approximation to a normal distribution with mean mu and standard
    deviation sigma, by Monte Carlo.
    Returns a stochastic vector called p_vec, corresponding
    to values in x_grid.  If a RV is distributed x~N(mu,sigma), then the expectation
    of a continuous function f() is E[f(x)] = numpy.dot(p_vec,f(x_grid)).

    Parameters
    ----------
    x_grid: numpy.array
        A sorted 1D array of floats representing discrete values that a normally
        distributed RV could take on.
    mu: float
        Mean of the normal distribution to be approximated.
    sigma: float
        Standard deviation of the normal distribution to be approximated.
    N_draws: int
        Number of draws to use in Monte Carlo.

    Returns
    -------
    p_vec: numpy.array
        A stochastic vector with probability weights for each x in x_grid.
    """

    # Take random draws from the desired normal distribution
    random_draws = np.random.normal(loc=mu, scale=sigma, size=N_draws)

    # Compute the distance between the draws and points in x_grid
    distance = np.abs(x_grid[:, np.newaxis] - random_draws[np.newaxis, :])

    # Find the indices of the points in x_grid that are closest to the draws
    distance_minimizing_index = np.argmin(distance, axis=0)

    # For each point in x_grid, the approximate probability of that point is the number
    # of Monte Carlo draws that are closest to that point
    p_vec = np.zeros_like(x_grid)
    for p_index, p in enumerate(p_vec):
        p_vec[p_index] = np.sum(distance_minimizing_index == p_index) / N_draws

    # Check for obvious errors, and return p_vec
    assert (
        (np.all(p_vec >= 0.0))
        and (np.all(p_vec <= 1.0))
        and (np.isclose(np.sum(p_vec)), 1.0)
    )
    return p_vec


def make_tauchen_ar1(N, sigma=1.0, ar_1=0.9, bound=3.0):
    """
    Function to return a discretized version of an AR1 process.
    See http://www.fperri.net/TEACHING/macrotheory08/numerical.pdf for details

    Parameters
    ----------
    N: int
        Size of discretized grid
    sigma: float
        Standard deviation of the error term
    ar_1: float
        AR1 coefficient
    bound: float
        The highest (lowest) grid point will be bound (-bound) multiplied by the unconditional
        standard deviation of the process

    Returns
    -------
    y: np.array
        Grid points on which the discretized process takes values
    trans_matrix: np.array
        Markov transition array for the discretized process
    """
    yN = bound * sigma / ((1 - ar_1**2) ** 0.5)
    y = np.linspace(-yN, yN, N)
    d = y[1] - y[0]
    trans_matrix = np.ones((N, N))
    for j in range(N):
        for k_1 in range(N - 2):
            k = k_1 + 1
            trans_matrix[j, k] = stats.norm.cdf(
                (y[k] + d / 2.0 - ar_1 * y[j]) / sigma
            ) - stats.norm.cdf((y[k] - d / 2.0 - ar_1 * y[j]) / sigma)
        trans_matrix[j, 0] = stats.norm.cdf((y[0] + d / 2.0 - ar_1 * y[j]) / sigma)
        trans_matrix[j, N - 1] = 1.0 - stats.norm.cdf(
            (y[N - 1] - d / 2.0 - ar_1 * y[j]) / sigma
        )

    return y, trans_matrix


# ================================================================================
# ==================== Functions for manipulating discrete distributions =========
# ================================================================================


def add_discrete_outcome_constant_mean(distribution, x, p, sort=False):
    """
    Adds a discrete outcome of x with probability p to an existing distribution,
    holding constant the relative probabilities of other outcomes and overall mean.

    Parameters
    ----------
    distribution : DiscreteDistribution
        A one-dimensional DiscreteDistribution.
    x : float
        The new value to be added to the distribution.
    p : float
        The probability of the discrete outcome x occuring.
    sort: bool
        Whether or not to sort data before returning it

    Returns
    -------
    d : DiscreteDistribution
        Probability associated with each point in array of discrete
        points for discrete probability mass function.
    """

    if type(distribution) != TimeVaryingDiscreteDistribution:
        data = np.append(x, distribution.data * (1 - p * x) / (1 - p))
        prob = np.append(p, distribution.prob * (1 - p))

        if sort:
            indices = np.argsort(data)
            data = data[indices]
            prob = prob[indices]

        return DiscreteDistribution(prob, data)
    elif type(distribution) == TimeVaryingDiscreteDistribution:
        # apply recursively on all the internal distributions
        return TimeVaryingDiscreteDistribution(
            [
                add_discrete_outcome_constant_mean(d, x, p)
                for d in distribution.distributions
            ],
            seed=distribution.seed,
        )


def add_discrete_outcome(distribution, x, p, sort=False):
    """
    Adds a discrete outcome of x with probability p to an existing distribution,
    holding constant the relative probabilities of other outcomes.

    Parameters
    ----------
    distribution : DiscreteDistribution
        One-dimensional distribution to which the outcome is to be added.
    x : float
        The new value to be added to the distribution.
    p : float
        The probability of the discrete outcome x occuring.

    Returns
    -------
    d : DiscreteDistribution
        Probability associated with each point in array of discrete
        points for discrete probability mass function.
    """

    data = np.append(x, distribution.data)
    prob = np.append(p, distribution.prob * (1 - p))

    if sort:
        indices = np.argsort(data)
        data = data[indices]
        prob = prob[indices]

    return DiscreteDistribution(prob, data)


def combine_indep_dstns(*distributions, seed=0, xarray=False, **kwargs):
    """
    Given n independent vector-valued discrete distributions, construct their joint discrete distribution.
    Can take multivariate discrete distributions as inputs.

    Parameters
    ----------
    distributions : DiscreteDistribution
        Arbitrary number of discrete distributionss to combine. Their realizations must be
        vector-valued (for each D in distributions, it must be the case that len(D.dim())==1).

    Returns
    -------
    A DiscreteDistribution representing the joint distribution of the given
    random variables.
    """
    # Get information on the distributions
    dist_lengths = ()
    dist_dims = ()
    for dist in distributions:

        if len(dist.dim()) > 1:
            raise NotImplementedError(
                "We currently only support combining vector-valued distributions."
            )

        dist_dims += (dist.dim(),)
        dist_lengths += (len(dist.prob),)

    number_of_distributions = len(distributions)

    # We need the combinations of indices of realizations in all
    # distributions
    inds = np.meshgrid(
        *[np.array(range(l), dtype=int) for l in dist_lengths], indexing="ij"
    )
    inds = [x.flatten() for x in inds]

    data_out = []
    P_temp = []
    for i, ind_vec in enumerate(inds):
        data_out += [distributions[i].data[..., ind_vec]]
        P_temp += [distributions[i].prob[ind_vec]]

    data_out = np.concatenate(data_out, axis=0)
    P_temp = np.stack(P_temp, axis=0)
    P_out = np.prod(P_temp, axis=0)

    assert np.isclose(np.sum(P_out), 1), "Probabilities do not sum to 1!"

    if xarray:
        which_dist = DiscreteDistributionXRA
    else:
        which_dist = DiscreteDistribution
    return which_dist(P_out, data_out, seed=seed, **kwargs)


def calc_expectation(dstn, func=lambda x: x, *args):
    """
    Expectation of a function, given an array of configurations of its inputs
    along with a DiscreteDistribution object that specifies the probability
    of each configuration.

    Parameters
    ----------
    dstn : DiscreteDistribution
        The distribution over which the function is to be evaluated.
    func : function
        The function to be evaluated.
        This function should take an array of shape dstn.dim() and return
        either arrays of arbitrary shape or scalars.
        It may also take other arguments *args.
    *args :
        Other inputs for func, representing the non-stochastic arguments.
        The the expectation is computed at f(dstn, *args).

    Returns
    -------
    f_exp : np.array or scalar
        The expectation of the function at the queried values.
        Scalar if only one value.
    """

    f_query = [func(dstn.data[..., i], *args) for i in range(len(dstn.prob))]

    f_query = np.stack(f_query, axis=-1)

    # From the numpy.dot documentation:
    # If both a and b are 1-D arrays, it is inner product of vectors (without complex conjugation).
    # If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b.
    # Thus, if func returns scalars, f_exp will be a scalar and if it returns arrays f_exp
    # will be an array of the same shape.
    f_exp = np.dot(f_query, dstn.prob)

    return f_exp


def distr_of_function(dstn, func=lambda x: x, *args):
    """
    Finds the distribution of a random variable Y that is a function
    of discrete random variable data, Y=f(data).

    Parameters
    ----------
    dstn : DiscreteDistribution
        The distribution over which the function is to be evaluated.
    func : function
        The function to be evaluated.
        This function should take an array of shape dstn.dim().
        It may also take other arguments *args.
    *args :
        Additional non-stochastic arguments for func,
        The function is computed at f(dstn, *args).

    Returns
    -------
    f_dstn : DiscreteDistribution
        The distribution of func(dstn).
    """
    # Apply function to every event realization
    f_query = [func(dstn.data[..., i], *args) for i in range(len(dstn.prob))]

    # Stack results along their last (new) axis
    f_query = np.stack(f_query, axis=-1)

    f_dstn = DiscreteDistribution(dstn.prob, f_query)

    return f_dstn


class MarkovProcess(Distribution):
    """
    A representation of a discrete Markov process.

    Parameters
    ----------
    transition_matrix : np.array
        An array of floats representing a probability mass for
        each state transition.
    seed : int
        Seed for random number generator.

    """

    transition_matrix = None

    def __init__(self, transition_matrix, seed=0):
        """
        Initialize a discrete distribution.

        """
        self.transition_matrix = transition_matrix

        # Set up the RNG
        super().__init__(seed)

    def draw(self, state):
        """
        Draw new states fromt the transition matrix.

        Parameters
        ----------
        state : int or nd.array
            The state or states (1-D array) from which to draw new states.

        Returns
        -------
        new_state : int or nd.array
            New states.
        """

        def sample(s):
            return self.RNG.choice(
                self.transition_matrix.shape[1], p=self.transition_matrix[s, :]
            )

        array_sample = np.frompyfunc(sample, 1, 1)

        return array_sample(state)


def ExpectedValue(func=None, dist=None, args=(), labels=False):
    """
    Expectation of a function, given an array of configurations of its inputs
    along with a DiscreteDistribution(dataRA) object that specifies the probability
    of each configuration.

    Parameters
    ----------
    func : function
        The function to be evaluated.
        This function should take the full array of distribution values
        and return either arrays of arbitrary shape or scalars.
        It may also take other arguments *args.
        This function differs from the standalone `calc_expectation`
        method in that it uses numpy's vectorization and broadcasting
        rules to avoid costly iteration.
        Note: If you need to use a function that acts on single outcomes
        of the distribution, consier `distribution.calc_expectation`.
    dist : DiscreteDistribution or DiscreteDistributionXRA
        The distribution over which the function is to be evaluated.
    args : tuple
        Other inputs for func, representing the non-stochastic arguments.
        The the expectation is computed at f(dstn, *args).
    labels : bool
        If True, the function should use labeled indexing instead of integer
        indexing using the distribution's underlying rv coordinates. For example,
        if `dims = ('rv', 'x')` and `coords = {'rv': ['a', 'b'], }`, then
        the function can be `lambda x: x["a"] + x["b"]`.

    Returns
    -------
    f_exp : np.array or scalar
        The expectation of the function at the queried values.
        Scalar if only one value.
    """

    if not isinstance(args, tuple):
        args = (args,)

    if isinstance(dist, DiscreteDistributionXRA):
        return dist.expected_value(func, *args, labels=labels)
    elif isinstance(dist, DiscreteDistribution):
        return dist.expected_value(func, *args)
