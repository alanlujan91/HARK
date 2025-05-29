"""
MarkovChain class for HARK - Convenient tools for discrete Markov processes.

This module provides a high-level interface for working with discrete Markov chains,
including vectorized conditional expectations, stationary distributions, and simulation.
"""

import numpy as np
import scipy.linalg
from typing import Union, Optional, Callable
from HARK.distributions.base import MarkovProcess
from HARK.distributions.discrete import (
    DiscreteDistribution,
    DiscreteDistributionLabeled,
)


class MarkovChain(MarkovProcess):
    """
    A comprehensive representation of a discrete Markov process with rich functionality
    for economic modeling.

    This class extends MarkovProcess to provide vectorized operations for conditional
    expectations, stationary distributions, multi-step transitions, and other common
    operations needed when working with Markov processes in economic models.

    Parameters
    ----------
    transition_matrix : np.array
        An N×N array of transition probabilities where entry (i,j) gives the
        probability of transitioning from state i to state j.
    state_values : np.array, optional
        Array of length N containing the values associated with each state.
        If None, states are assumed to be integers 0, 1, ..., N-1.
    state_labels : list or np.array, optional
        Labels for each state (for integration with labeled distributions).
    seed : int, optional
        Seed for random number generator. Default is 0.

    Examples
    --------
    >>> import numpy as np
    >>> from HARK.distributions import make_tauchen_ar1, MarkovChain
    >>>
    >>> # Create AR(1) process: y_t = 0.9 * y_{t-1} + epsilon_t
    >>> y_grid, P = make_tauchen_ar1(N=7, ar_1=0.9, sigma=0.1)
    >>> mc = MarkovChain(P, y_grid)
    >>>
    >>> # Calculate E[y_{t+1} | y_t] for all current states
    >>> conditional_means = mc.expected(lambda y: y)
    >>>
    >>> # Should approximately equal 0.9 * y_grid for AR(1)
    >>> print(np.allclose(conditional_means, 0.9 * y_grid, atol=1e-3))
    >>>
    >>> # Get stationary distribution
    >>> stationary = mc.stationary_dist()
    >>>
    >>> # Simulate a path
    >>> path = mc.simulate(T=100, initial_state=3)
    """

    def __init__(
        self,
        transition_matrix: np.ndarray,
        state_values: Optional[np.ndarray] = None,
        state_labels: Optional[Union[list, np.ndarray]] = None,
        seed: int = 0,
    ):
        # Initialize parent MarkovProcess
        super().__init__(transition_matrix, seed)

        # Validate transition matrix
        if (
            transition_matrix.ndim != 2
            or transition_matrix.shape[0] != transition_matrix.shape[1]
        ):
            raise ValueError("Transition matrix must be square (N×N)")

        if not np.allclose(transition_matrix.sum(axis=1), 1.0):
            raise ValueError("Each row of transition matrix must sum to 1.0")

        if np.any(transition_matrix < 0):
            raise ValueError("Transition probabilities must be non-negative")

        self.n_states = transition_matrix.shape[0]

        # Set state values
        if state_values is None:
            self.state_values = np.arange(self.n_states)
        else:
            self.state_values = np.array(state_values)
            if len(self.state_values) != self.n_states:
                raise ValueError("Length of state_values must equal number of states")

        # Set state labels
        self.state_labels = state_labels

        # Create conditional distributions for each state
        self._conditional_dists = [
            DiscreteDistribution(
                pmv=self.transition_matrix[i],
                atoms=self.state_values.reshape(
                    1, -1
                ),  # Fix: ensure 2D shape for atoms
                seed=self._rng.integers(0, 2**31 - 1),
            )
            for i in range(self.n_states)
        ]

        # Cache for stationary distribution
        self._stationary_cache = None

    def expected(
        self,
        func: Callable[[np.ndarray], Union[float, np.ndarray]],
        current: Optional[Union[int, float, str]] = None,
    ) -> Union[float, np.ndarray]:
        """
        Calculate conditional expectations E[f(z_{t+1}) | z_t] for all states or a specific state.

        This is the core method that provides vectorized conditional expectation
        calculations, addressing the main limitation of current HARK Markov tools.

        Parameters
        ----------
        func : callable
            Function to apply to next period states. Should accept an array of
            state values and return either a scalar or array of the same length.
        current : int, float, str, or None, optional
            If provided, return expectation conditional on this specific current state.
            Can be state index (int), state value (float), or state label (str).
            If None, return expectations for all current states.

        Returns
        -------
        float or np.array
            If current is None: array of length N with E[f(z_{t+1}) | z_t = s_i]
            for each state s_i.
            If current is specified: scalar E[f(z_{t+1}) | z_t = current].

        Examples
        --------
        >>> # Calculate E[z_{t+1}^2 | z_t] for all states
        >>> squared_expectations = mc.expected(lambda z: z**2)
        >>>
        >>> # Calculate E[z_{t+1} | z_t = 0.5] for specific current state
        >>> conditional_mean = mc.expected(lambda z: z, current=0.5)
        """
        # Evaluate function at all possible next states
        try:
            f_vals = func(self.state_values)
        except Exception as e:
            raise ValueError(f"Function evaluation failed: {e}")

        # Ensure f_vals is an array and has correct shape
        f_vals = np.asarray(f_vals)
        if f_vals.ndim == 0:
            f_vals = np.full(self.n_states, f_vals)
        elif f_vals.ndim == 1 and len(f_vals) != self.n_states:
            raise ValueError(
                f"Function must return scalar or array of length {self.n_states}"
            )
        elif f_vals.ndim == 2:
            # Handle vector-valued functions: f_vals should be (n_states, n_outputs)
            if f_vals.shape[1] != self.n_states:
                # If it's (n_outputs, n_states), transpose it
                if f_vals.shape[0] == self.n_states:
                    f_vals = f_vals.T
                else:
                    raise ValueError(
                        f"Vector function output shape {f_vals.shape} incompatible with {self.n_states} states"
                    )
        elif f_vals.ndim > 2:
            raise ValueError("Function output cannot have more than 2 dimensions")

        # Calculate conditional expectations using matrix multiplication
        # E[f(z_{t+1}) | z_t = i] = sum_j P[i,j] * f(z_j)
        if f_vals.ndim == 1:
            conditional_expectations = self.transition_matrix @ f_vals
        else:
            # For vector-valued functions: P @ f_vals where f_vals is (n_outputs, n_states)
            conditional_expectations = f_vals @ self.transition_matrix.T

        if current is None:
            return conditional_expectations
        else:
            # Find index of current state
            current_idx = self._state_to_index(current)
            return conditional_expectations[current_idx]

    def _state_to_index(self, state: Union[int, float, str]) -> int:
        """Convert state value, label, or index to array index."""
        if isinstance(state, (int, np.integer)) and 0 <= state < self.n_states:
            # Assume it's already an index
            return int(state)
        elif self.state_labels is not None and state in self.state_labels:
            # Look up by label
            return list(self.state_labels).index(state)
        else:
            # Look up by value
            indices = np.where(np.isclose(self.state_values, state))[0]
            if len(indices) == 0:
                raise ValueError(f"State {state} not found")
            elif len(indices) > 1:
                raise ValueError(f"State {state} matches multiple indices")
            return indices[0]

    def stationary_dist(
        self, method: str = "eigenvector", tol: float = 1e-12, max_iter: int = 10000
    ) -> np.ndarray:
        """
        Calculate the stationary distribution of the Markov chain.

        For an ergodic Markov chain, this finds the unique distribution π such that
        π = π P (i.e., the left eigenvector with eigenvalue 1).

        Parameters
        ----------
        method : str, optional
            Method to use: 'eigenvector' (default) or 'iteration'.
        tol : float, optional
            Tolerance for convergence (iteration method only).
        max_iter : int, optional
            Maximum iterations (iteration method only).

        Returns
        -------
        np.array
            Array of length N containing the stationary probabilities.

        Examples
        --------
        >>> stationary = mc.stationary_dist()
        >>> # Verify it's actually stationary
        >>> np.allclose(stationary @ mc.transition_matrix, stationary)
        """
        if self._stationary_cache is not None:
            return self._stationary_cache.copy()

        if method == "eigenvector":
            # Find left eigenvector with eigenvalue 1
            # For P.T @ π = π, we solve (P.T - I) @ π = 0
            eigenvals, eigenvecs = scipy.linalg.eig(self.transition_matrix.T)

            # Find eigenvalue closest to 1
            idx = np.argmin(np.abs(eigenvals - 1.0))
            stationary = np.real(eigenvecs[:, idx])

            # Normalize to probability vector
            stationary = np.abs(stationary)  # Handle numerical errors in sign
            stationary = stationary / stationary.sum()

        elif method == "iteration":
            # Power iteration: π_{n+1} = π_n P
            pi = np.ones(self.n_states) / self.n_states  # Start uniform

            for i in range(max_iter):
                pi_new = pi @ self.transition_matrix
                if np.max(np.abs(pi_new - pi)) < tol:
                    break
                pi = pi_new
            else:
                raise RuntimeError(
                    f"Stationary distribution did not converge in {max_iter} iterations"
                )

            stationary = pi_new
        else:
            raise ValueError("Method must be 'eigenvector' or 'iteration'")

        # Cache result
        self._stationary_cache = stationary.copy()
        return stationary

    def simulate(
        self,
        T: int,
        initial_state: Union[int, float, str] = 0,
        return_indices: bool = False,
    ) -> np.ndarray:
        """
        Simulate a path of the Markov chain.

        Parameters
        ----------
        T : int
            Length of simulation.
        initial_state : int, float, or str, optional
            Starting state (index, value, or label). Default is 0.
        return_indices : bool, optional
            If True, return state indices. If False, return state values.

        Returns
        -------
        np.array
            Array of length T containing the simulated path.

        Examples
        --------
        >>> # Simulate 100 periods starting from state 0
        >>> path = mc.simulate(T=100, initial_state=0)
        >>>
        >>> # Get path as indices instead of values
        >>> index_path = mc.simulate(T=100, return_indices=True)
        """
        # Convert initial state to index
        current_idx = self._state_to_index(initial_state)

        # Simulate path
        path_indices = np.zeros(T, dtype=int)
        path_indices[0] = current_idx

        for t in range(1, T):
            current_idx = self.draw(current_idx)
            path_indices[t] = current_idx

        if return_indices:
            return path_indices
        else:
            return self.state_values[path_indices]

    def simulate_many(
        self,
        T: int,
        n_sims: int,
        initial_state: Union[int, float, str] = 0,
        return_indices: bool = False,
    ) -> np.ndarray:
        """
        Simulate multiple independent paths of the Markov chain.

        Parameters
        ----------
        T : int
            Length of each simulation.
        n_sims : int
            Number of independent simulations.
        initial_state : int, float, or str, optional
            Starting state for all simulations.
        return_indices : bool, optional
            If True, return state indices. If False, return state values.

        Returns
        -------
        np.array
            Array of shape (n_sims, T) containing the simulated paths.
        """
        paths = np.zeros((n_sims, T))
        for i in range(n_sims):
            paths[i] = self.simulate(T, initial_state, return_indices)
        return paths

    def n_step_transition(
        self,
        n: int,
        initial_dist: Optional[np.ndarray] = None,
        initial_state: Optional[Union[int, float, str]] = None,
    ) -> np.ndarray:
        """
        Calculate the distribution after n steps.

        If initial_dist is provided, computes μ P^n where μ is the initial distribution.
        If initial_state is provided, computes the n-step transition probabilities
        from that state.

        Parameters
        ----------
        n : int
            Number of steps.
        initial_dist : np.array, optional
            Initial distribution (array of length N summing to 1).
        initial_state : int, float, or str, optional
            Initial state (alternative to initial_dist).

        Returns
        -------
        np.array
            Distribution after n steps.

        Examples
        --------
        >>> # Distribution after 5 steps starting from uniform
        >>> uniform = np.ones(mc.n_states) / mc.n_states
        >>> dist_5 = mc.n_step_transition(5, initial_dist=uniform)
        >>>
        >>> # Probability distribution 10 steps from state 0
        >>> probs_10 = mc.n_step_transition(10, initial_state=0)
        """
        if initial_dist is not None and initial_state is not None:
            raise ValueError("Specify either initial_dist or initial_state, not both")

        if initial_dist is not None:
            dist = np.array(initial_dist)
            if not np.allclose(dist.sum(), 1.0):
                raise ValueError("initial_dist must sum to 1.0")
        elif initial_state is not None:
            dist = np.zeros(self.n_states)
            dist[self._state_to_index(initial_state)] = 1.0
        else:
            raise ValueError("Must specify either initial_dist or initial_state")

        # Calculate P^n by repeated multiplication
        # For large n, could use matrix exponentiation, but this is clearer
        for _ in range(n):
            dist = dist @ self.transition_matrix

        return dist

    def conditional_dist(
        self, current_state: Union[int, float, str]
    ) -> DiscreteDistribution:
        """
        Get the conditional distribution of next period's state.

        Parameters
        ----------
        current_state : int, float, or str
            Current state (index, value, or label).

        Returns
        -------
        DiscreteDistribution
            Distribution of z_{t+1} | z_t = current_state.

        Examples
        --------
        >>> # Get distribution of next state given current state 0
        >>> next_dist = mc.conditional_dist(0)
        >>> expected_next = next_dist.expected()
        """
        idx = self._state_to_index(current_state)
        return self._conditional_dists[idx]

    def dist_of_func(
        self, func: Callable[[np.ndarray], Union[float, np.ndarray]]
    ) -> list:
        """
        Get the distribution of a function of next period's state for each current state.

        This is analogous to distr_of_function but for Markov processes.

        Parameters
        ----------
        func : callable
            Function to apply to state values.

        Returns
        -------
        list
            List of DiscreteDistribution objects, one for each current state,
            representing the distribution of f(z_{t+1}) | z_t = s_i.

        Examples
        --------
        >>> # Distribution of z_{t+1}^2 conditional on each current state
        >>> squared_dists = mc.dist_of_func(lambda z: z**2)
        >>> # Get expectation of z^2 given current state 0
        >>> expected_z_squared_given_0 = squared_dists[0].expected()
        """
        f_vals = func(self.state_values)
        f_vals = np.asarray(f_vals)

        result = []
        for i in range(self.n_states):
            result.append(
                DiscreteDistribution(
                    pmv=self.transition_matrix[i],
                    atoms=f_vals.reshape(1, -1),  # Fix: ensure 2D shape for atoms
                    seed=self._rng.integers(0, 2**31 - 1),
                )
            )

        return result

    def ergodic_mean(
        self, func: Callable[[np.ndarray], Union[float, np.ndarray]]
    ) -> float:
        """
        Calculate the ergodic mean E_π[f(z)] where π is the stationary distribution.

        Parameters
        ----------
        func : callable
            Function to apply to state values.

        Returns
        -------
        float
            Long-run average value of f(z).

        Examples
        --------
        >>> # Long-run average of the state values
        >>> ergodic_mean_state = mc.ergodic_mean(lambda z: z)
        >>>
        >>> # Long-run variance
        >>> mean_val = mc.ergodic_mean(lambda z: z)
        >>> ergodic_var = mc.ergodic_mean(lambda z: (z - mean_val)**2)
        """
        stationary = self.stationary_dist()
        f_vals = func(self.state_values)
        return stationary @ f_vals

    def autocorrelation(self, lags: Union[int, list] = 1) -> Union[float, np.ndarray]:
        """
        Calculate autocorrelation of the state process.

        Parameters
        ----------
        lags : int or list, optional
            Lag(s) for autocorrelation. Default is 1.

        Returns
        -------
        float or np.array
            Autocorrelation at specified lag(s).

        Examples
        --------
        >>> # First-order autocorrelation
        >>> rho_1 = mc.autocorrelation(1)
        >>>
        >>> # Autocorrelations at lags 1, 2, 3, 4, 5
        >>> rhos = mc.autocorrelation([1, 2, 3, 4, 5])
        """
        if isinstance(lags, int):
            lags = [lags]

        # Calculate stationary mean and variance
        stationary = self.stationary_dist()
        mean_z = stationary @ self.state_values
        var_z = stationary @ (self.state_values - mean_z) ** 2

        autocorrs = []
        for lag in lags:
            # E[z_t * z_{t+lag}] = π' @ diag(z) @ P^lag @ z
            z_diag = np.diag(self.state_values)
            P_lag = np.linalg.matrix_power(self.transition_matrix, lag)
            cross_moment = stationary @ z_diag @ P_lag @ self.state_values

            # Correlation = (E[z_t * z_{t+lag}] - μ^2) / σ^2
            autocorr = (cross_moment - mean_z**2) / var_z
            autocorrs.append(autocorr)

        return autocorrs[0] if len(autocorrs) == 1 else np.array(autocorrs)

    def __repr__(self) -> str:
        return (
            f"MarkovChain(n_states={self.n_states}, "
            f"state_range=[{self.state_values.min():.3f}, {self.state_values.max():.3f}])"
        )


class LabeledMarkovChain(MarkovChain):
    """
    A MarkovChain with labeled states for enhanced usability.

    This subclass integrates with HARK's labeled distribution framework to allow
    referencing states by meaningful names rather than indices.

    Parameters
    ----------
    transition_matrix : np.array
        Transition probability matrix.
    state_values : np.array
        Numeric values for each state.
    state_labels : list
        Labels for each state.
    seed : int, optional
        Random seed.

    Examples
    --------
    >>> labels = ['recession', 'normal', 'expansion']
    >>> values = [-0.02, 0.03, 0.08]  # GDP growth rates
    >>> P = np.array([[0.7, 0.3, 0.0],
    ...               [0.1, 0.8, 0.1],
    ...               [0.0, 0.4, 0.6]])
    >>> lmc = LabeledMarkovChain(P, values, labels)
    >>>
    >>> # Reference states by label
    >>> recession_mean = lmc.expected(lambda g: g, current='recession')
    """

    def __init__(
        self,
        transition_matrix: np.ndarray,
        state_values: np.ndarray,
        state_labels: list,
        seed: int = 0,
    ):
        if len(state_labels) != len(state_values):
            raise ValueError("Length of state_labels must equal length of state_values")

        super().__init__(transition_matrix, state_values, state_labels, seed)

        # Create labeled conditional distributions
        self._labeled_conditional_dists = []
        for i in range(self.n_states):
            # Create a simple labeled distribution by manually constructing it
            # to avoid compatibility issues with DiscreteDistributionLabeled.from_unlabeled
            labeled_dist = DiscreteDistributionLabeled(
                pmv=self.transition_matrix[i],
                atoms=self.state_values.reshape(1, -1),  # Ensure 2D shape
                name=f"Next state | current = {state_labels[i]}",
                var_names=["next_state"],  # Single variable for next state
                seed=self._rng.integers(0, 2**31 - 1),
            )
            self._labeled_conditional_dists.append(labeled_dist)

    def conditional_dist_labeled(
        self, current_state: Union[int, str]
    ) -> DiscreteDistributionLabeled:
        """Get labeled conditional distribution."""
        idx = self._state_to_index(current_state)
        return self._labeled_conditional_dists[idx]


def create_markov_chain_from_tauchen(
    N: int = 7, sigma: float = 1.0, ar_1: float = 0.9, bound: float = 3.0, **kwargs
) -> MarkovChain:
    """
    Convenience function to create a MarkovChain from Tauchen discretization of AR(1).

    Parameters
    ----------
    N : int
        Number of discrete states.
    sigma : float
        Standard deviation of innovations.
    ar_1 : float
        AR(1) coefficient.
    bound : float
        Number of standard deviations for grid bounds.
    **kwargs
        Additional arguments passed to MarkovChain constructor.

    Returns
    -------
    MarkovChain
        Discretized AR(1) process as a MarkovChain.

    Examples
    --------
    >>> # Standard AR(1): y_t = 0.95 * y_{t-1} + ε_t
    >>> mc = create_markov_chain_from_tauchen(N=9, ar_1=0.95, sigma=0.1)
    >>>
    >>> # Verify AR(1) property: E[y_{t+1} | y_t] ≈ 0.95 * y_t
    >>> conditional_means = mc.expected(lambda y: y)
    >>> theoretical_means = 0.95 * mc.state_values
    >>> print(np.allclose(conditional_means, theoretical_means, atol=1e-3))
    """
    from HARK.distributions.utils import make_tauchen_ar1

    y_grid, P = make_tauchen_ar1(N, sigma, ar_1, bound)
    return MarkovChain(P, y_grid, **kwargs)
