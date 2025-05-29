---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# MarkovChain Tools in HARK

This notebook demonstrates the new `MarkovChain` functionality in HARK that provides convenient, vectorized operations for discrete Markov processes. This addresses a key limitation in HARK's toolkit by bringing Markov processes to the same convenience level as i.i.d. distributions.

## The Problem: Limited Markov Process Tools

Previously, HARK had excellent tools for i.i.d. distributions (`distr_of_func`, `expect`) but limited support for processes with memory. When working with AR(1) processes or other Markovian dynamics, users had to manually iterate over state-conditioned distributions:

```python
# Old cumbersome approach for E[f(z_{t+1}) | z_t]
# expectations = []
# for i, current_state in enumerate(states):
#     conditional_dist = DiscreteDistribution(P[i], next_states)
#     expectations.append(conditional_dist.expected(func))
```

## The Solution: MarkovChain Class

The new `MarkovChain` class provides vectorized operations that make working with Markov processes as intuitive as working with i.i.d. distributions.

```python
import numpy as np
import matplotlib.pyplot as plt
from HARK.distributions import (
    MarkovChain,
    LabeledMarkovChain,
    create_markov_chain_from_tauchen,
    make_tauchen_ar1,
)

# Set random seed for reproducibility
np.random.seed(42)
```

## Basic MarkovChain Usage

Let's start with a simple 3-state Markov chain representing economic regimes:

```python
# Define transition matrix and state values
P = np.array(
    [
        [0.7, 0.3, 0.0],  # Recession transitions
        [0.1, 0.8, 0.1],  # Normal transitions
        [0.0, 0.4, 0.6],  # Expansion transitions
    ]
)

state_values = np.array([-0.02, 0.03, 0.08])  # GDP growth rates
state_labels = ["Recession", "Normal", "Expansion"]

# Create MarkovChain
mc = MarkovChain(P, state_values)
print(f"Created MarkovChain with {mc.n_states} states")
print(f"State values: {mc.state_values}")
```

## Vectorized Conditional Expectations

The key feature is computing E[f(z_{t+1}) | z_t] for all states at once:

```python
# Calculate E[z_{t+1} | z_t] for all current states
conditional_means = mc.expected(lambda z: z)
print("Conditional means E[z_{t+1} | z_t]:")
for i, (label, mean) in enumerate(zip(state_labels, conditional_means)):
    print(f"  {label}: {mean:.4f}")

# Calculate E[z_{t+1}^2 | z_t] for all current states
conditional_second_moments = mc.expected(lambda z: z**2)
print("\nConditional second moments E[z_{t+1}^2 | z_t]:")
for i, (label, moment) in enumerate(zip(state_labels, conditional_second_moments)):
    print(f"  {label}: {moment:.6f}")

# Calculate expectation for a specific current state
specific_expectation = mc.expected(lambda z: z, current=0)  # Given recession
print(f"\nE[z_{{t+1}} | z_t = Recession] = {specific_expectation:.4f}")
```

## Comparison: Old vs New Approach

Let's demonstrate the improvement in code clarity and performance:

```python
import time


# Old manual approach
def old_approach(transition_matrix, state_vals, func):
    """Manual calculation of conditional expectations"""
    n_states = len(state_vals)
    result = np.zeros(n_states)
    for i in range(n_states):
        for j in range(n_states):
            result[i] += transition_matrix[i, j] * func(state_vals[j])
    return result


# New vectorized approach
def new_approach(markov_chain, func):
    """Vectorized calculation using MarkovChain"""
    return markov_chain.expected(func)


# Function to test
test_func = lambda z: z**2 + 0.1 * z

# Timing comparison
start = time.time()
old_result = old_approach(P, state_values, test_func)
old_time = time.time() - start

start = time.time()
new_result = new_approach(mc, test_func)
new_time = time.time() - start

print("Performance Comparison:")
print(f"Old approach time: {old_time:.6f} seconds")
print(f"New approach time: {new_time:.6f} seconds")
print(f"Speedup: {old_time / new_time:.2f}x")
print(f"Results identical: {np.allclose(old_result, new_result)}")
```

## AR(1) Process Integration

One of the most common use cases is working with AR(1) processes discretized via Tauchen's method:

```python
# Create AR(1) process: y_t = ρ * y_{t-1} + ε_t
rho = 0.9
sigma = 0.1
N = 7

# Method 1: Manual creation
y_grid, P_ar1 = make_tauchen_ar1(N=N, ar_1=rho, sigma=sigma)
mc_ar1_manual = MarkovChain(P_ar1, y_grid)

# Method 2: Convenience function
mc_ar1 = create_markov_chain_from_tauchen(N=N, ar_1=rho, sigma=sigma)

print("AR(1) Markov Chain Properties:")
print(f"States: {mc_ar1.state_values}")
print(f"Range: [{mc_ar1.state_values.min():.3f}, {mc_ar1.state_values.max():.3f}]")

# Verify AR(1) property: E[y_{t+1} | y_t] ≈ ρ * y_t
conditional_means = mc_ar1.expected(lambda y: y)
theoretical_means = rho * mc_ar1.state_values

print("\nAR(1) Property Verification:")
print("E[y_{t+1} | y_t] vs ρ * y_t:")
for i in range(N):
    print(
        f"  State {i}: {conditional_means[i]:.4f} vs {theoretical_means[i]:.4f} "
        f"(error: {abs(conditional_means[i] - theoretical_means[i]):.4f})"
    )

max_error = np.max(np.abs(conditional_means - theoretical_means))
print(f"Maximum approximation error: {max_error:.4f}")
```

## Stationary Distribution Analysis

```python
# Calculate and analyze stationary distribution
stationary = mc_ar1.stationary_dist()

print("Stationary Distribution Analysis:")
print(f"Stationary probabilities: {stationary}")
print(f"Sum of probabilities: {stationary.sum():.10f}")

# Verify stationarity: π = π * P
verification = stationary @ mc_ar1.transition_matrix
print(f"Stationarity check (π * P = π): {np.allclose(stationary, verification)}")
print(f"Maximum deviation: {np.max(np.abs(stationary - verification)):.2e}")

# Calculate ergodic moments
ergodic_mean = mc_ar1.ergodic_mean(lambda y: y)
ergodic_variance = mc_ar1.ergodic_mean(lambda y: (y - ergodic_mean) ** 2)

print(f"\nErgodic Properties:")
print(f"Long-run mean: {ergodic_mean:.4f}")
print(f"Long-run variance: {ergodic_variance:.4f}")

# Compare with theoretical AR(1) variance
theoretical_variance = sigma**2 / (1 - rho**2)
print(f"Theoretical AR(1) variance: {theoretical_variance:.4f}")
print(f"Approximation error: {abs(ergodic_variance - theoretical_variance):.4f}")
```

## Simulation and Dynamics

```python
# Simulate paths
T = 1000
initial_state = N // 2  # Start from middle state

# Single path simulation
path = mc_ar1.simulate(T=T, initial_state=initial_state)

# Multiple path simulation
n_sims = 100
paths = mc_ar1.simulate_many(T=100, n_sims=n_sims, initial_state=initial_state)

print("Simulation Results:")
print(f"Single path: {T} periods, starts at {mc_ar1.state_values[initial_state]:.3f}")
print(f"Path mean: {path.mean():.4f}, Path std: {path.std():.4f}")
print(f"Multiple paths: {n_sims} simulations of {paths.shape[1]} periods each")
print(f"Cross-simulation mean: {paths.mean():.4f}")

# Plot simulation results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Single path
ax1.plot(path[:200], alpha=0.8, linewidth=1)
ax1.axhline(ergodic_mean, color="red", linestyle="--", alpha=0.7, label="Ergodic Mean")
ax1.set_title("AR(1) Process Simulation (Single Path, First 200 Periods)")
ax1.set_xlabel("Time")
ax1.set_ylabel("State Value")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Multiple paths (first 50 periods)
for i in range(min(10, n_sims)):
    ax2.plot(paths[i, :50], alpha=0.3, linewidth=0.8)
ax2.axhline(ergodic_mean, color="red", linestyle="--", alpha=0.7, label="Ergodic Mean")
ax2.set_title("AR(1) Process Simulation (Multiple Paths, First 50 Periods)")
ax2.set_xlabel("Time")
ax2.set_ylabel("State Value")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Autocorrelation Analysis

```python
# Calculate autocorrelations
lags = [1, 2, 3, 4, 5, 10]
autocorrs = mc_ar1.autocorrelation(lags)

print("Autocorrelation Analysis:")
print("Lag | Empirical | Theoretical")
print("----+----------+------------")
for lag, emp_autocorr in zip(lags, autocorrs):
    theoretical_autocorr = rho**lag
    print(f" {lag:2d} |  {emp_autocorr:.4f}  |   {theoretical_autocorr:.4f}")

# Plot autocorrelation function
lags_extended = list(range(1, 21))
autocorrs_extended = mc_ar1.autocorrelation(lags_extended)
theoretical_extended = [rho**lag for lag in lags_extended]

plt.figure(figsize=(10, 6))
plt.plot(
    lags_extended, autocorrs_extended, "bo-", label="Discretized Chain", markersize=6
)
plt.plot(
    lags_extended, theoretical_extended, "r--", label="Theoretical AR(1)", linewidth=2
)
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.title(f"Autocorrelation Function (ρ = {rho})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Multi-step Transition Dynamics

```python
# Analyze convergence to stationarity
steps = [1, 2, 5, 10, 20, 50]
initial_state_idx = 0  # Start from lowest state

print("Convergence to Stationarity:")
print("Step | Distribution after n steps")
print("-----+", "-" * 50)

for n in steps:
    dist_n = mc_ar1.n_step_transition(n, initial_state=initial_state_idx)
    # Print first 3 and last 3 probabilities
    print(
        f" {n:2d}  | [{dist_n[0]:.3f}, {dist_n[1]:.3f}, {dist_n[2]:.3f}, ..., "
        f"{dist_n[-3]:.3f}, {dist_n[-2]:.3f}, {dist_n[-1]:.3f}]"
    )

print(
    f"Stat | [{stationary[0]:.3f}, {stationary[1]:.3f}, {stationary[2]:.3f}, ..., "
    f"{stationary[-3]:.3f}, {stationary[-2]:.3f}, {stationary[-1]:.3f}]"
)

# Plot convergence
fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(steps)))

for i, (n, color) in enumerate(zip(steps, colors)):
    dist_n = mc_ar1.n_step_transition(n, initial_state=initial_state_idx)
    ax.plot(
        mc_ar1.state_values,
        dist_n,
        "o-",
        color=color,
        label=f"After {n} steps",
        markersize=5,
        alpha=0.8,
    )

ax.plot(
    mc_ar1.state_values,
    stationary,
    "k--",
    linewidth=3,
    label="Stationary Distribution",
    alpha=0.9,
)

ax.set_xlabel("State Value")
ax.set_ylabel("Probability")
ax.set_title(
    f"Convergence to Stationarity (Starting from state {mc_ar1.state_values[initial_state_idx]:.3f})"
)
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

## Labeled MarkovChain

For better interpretability, you can use labeled states:

```python
# Create labeled Markov chain for economic regimes
regime_P = np.array(
    [
        [0.8, 0.15, 0.05],  # Recession persistence
        [0.1, 0.8, 0.1],  # Normal transitions
        [0.05, 0.25, 0.7],  # Expansion persistence
    ]
)

regime_values = np.array([-0.03, 0.02, 0.06])  # GDP growth rates
regime_labels = ["Recession", "Normal", "Expansion"]

lmc = LabeledMarkovChain(regime_P, regime_values, regime_labels)

print("Labeled MarkovChain Example:")
print(f"States: {lmc.state_labels}")
print(f"Values: {lmc.state_values}")

# Use labels for calculations
recession_expectation = lmc.expected(lambda g: g, current="Recession")
normal_expectation = lmc.expected(lambda g: g, current="Normal")
expansion_expectation = lmc.expected(lambda g: g, current="Expansion")

print("\nConditional Expectations by Regime:")
print(f"E[growth | Recession]: {recession_expectation:.4f}")
print(f"E[growth | Normal]:    {normal_expectation:.4f}")
print(f"E[growth | Expansion]: {expansion_expectation:.4f}")

# Calculate regime persistence
print("\nRegime Persistence (staying in same state):")
for i, label in enumerate(regime_labels):
    persistence = regime_P[i, i]
    expected_duration = 1 / (1 - persistence)
    print(f"{label}: {persistence:.1%} (avg duration: {expected_duration:.1f} periods)")
```

## Advanced Features

```python
# Distribution of functions
print("Distribution of Functions:")

# Get conditional distributions for each current state
squared_dists = mc_ar1.dist_of_func(lambda y: y**2)

print("E[y_{t+1}^2 | y_t] for each current state:")
for i, dist in enumerate(squared_dists):
    expectation = dist.expected()
    print(
        f"  State {i} (y={mc_ar1.state_values[i]:.3f}): E[y^2] = {expectation[0]:.4f}"
    )

# Individual conditional distributions
print("\nConditional Distribution Analysis:")
middle_state = N // 2
cond_dist = mc_ar1.conditional_dist(middle_state)

print(f"Given y_t = {mc_ar1.state_values[middle_state]:.3f}:")
print(f"  Probabilities: {cond_dist.pmv}")
print(f"  Next states:   {cond_dist.atoms[0]}")  # atoms is 2D for discrete dist
print(f"  Expected next: {cond_dist.expected()[0]:.4f}")

# Vector-valued functions
print("\nVector-Valued Functions:")
# Function that returns both the value and its square
both_moments = mc_ar1.expected(lambda y: np.column_stack([y, y**2]))
print("E[y_{t+1}, y_{t+1}^2 | y_t] for each state:")
print("State | E[y]    | E[y^2]")
print("------+--------+--------")
for i in range(N):
    print(f" {i:2d}   | {both_moments[0, i]:.4f} | {both_moments[1, i]:.4f}")
```


## Performance Benchmark


```python
# Comprehensive performance test
def benchmark_approaches(n_states=50, n_functions=100):
    """Compare performance of old vs new approaches with corrected deterministic functions"""

    # Create larger chain
    np.random.seed(42)  # For reproducible results
    P_large = np.random.rand(n_states, n_states)
    P_large = P_large / P_large.sum(axis=1, keepdims=True)  # Normalize
    states_large = np.linspace(-2, 2, n_states)
    mc_large = MarkovChain(P_large, states_large)

    # Generate deterministic random functions (corrected version)
    # Note: The original version had a bug with lambda closures and np.random.randn()
    np.random.seed(123)  # Different seed for function coefficients
    random_coeffs = [np.random.randn() for _ in range(n_functions)]
    random_linear_coeffs = [np.random.randn() for _ in range(n_functions)]

    functions = [
        lambda y, c=random_coeffs[i], l=random_linear_coeffs[i]: c * y**2 + l * y
        for i in range(n_functions)
    ]

    # Old approach timing
    start = time.time()
    old_results = []
    for func in functions:
        result = np.zeros(n_states)
        for i in range(n_states):
            for j in range(n_states):
                result[i] += P_large[i, j] * func(states_large[j])
        old_results.append(result)
    old_total_time = time.time() - start

    # New approach timing
    start = time.time()
    new_results = [mc_large.expected(func) for func in functions]
    new_total_time = time.time() - start

    # Verify results are identical
    max_diff = max(
        np.max(np.abs(old - new)) for old, new in zip(old_results, new_results)
    )

    print(f"Performance Benchmark ({n_states} states, {n_functions} functions):")
    print(f"Old approach: {old_total_time:.4f} seconds")
    print(f"New approach: {new_total_time:.4f} seconds")
    print(f"Speedup: {old_total_time / new_total_time:.2f}x")
    print(f"Maximum difference: {max_diff:.2e}")
    print(f"Results identical: {max_diff < 1e-12}")


benchmark_approaches()
```


## Integration with HARK Models

Here's how the new MarkovChain tools integrate with typical HARK economic models:


```python
# Example: Income process for consumption-saving model
def create_income_process(rho_income=0.95, sigma_income=0.2, n_states=7):
    """Create discretized AR(1) income process for HARK model"""

    # Create income Markov chain
    income_mc = create_markov_chain_from_tauchen(
        N=n_states, ar_1=rho_income, sigma=sigma_income
    )

    # Calculate key statistics that models often need
    income_stats = {
        "states": income_mc.state_values,
        "transition_matrix": income_mc.transition_matrix,
        "stationary_dist": income_mc.stationary_dist(),
        "ergodic_mean": income_mc.ergodic_mean(lambda y: y),
        "ergodic_variance": income_mc.ergodic_mean(lambda y: y**2)
        - income_mc.ergodic_mean(lambda y: y) ** 2,
        "autocorrelation_1": income_mc.autocorrelation(1),
        "conditional_means": income_mc.expected(lambda y: y),
        "conditional_variances": income_mc.expected(lambda y: y**2)
        - income_mc.expected(lambda y: y) ** 2,
    }

    return income_mc, income_stats


# Create income process
income_mc, stats = create_income_process()

print("Income Process for HARK Model:")
print(f"States: {stats['states']}")
print(f"Ergodic mean: {stats['ergodic_mean']:.4f}")
print(f"Ergodic variance: {stats['ergodic_variance']:.4f}")
print(f"First-order autocorr: {stats['autocorrelation_1']:.4f}")

# Example calculation that a consumption model might need:
# E[exp(y_{t+1}) | y_t] for each current income state
exp_income_expectations = income_mc.expected(lambda y: np.exp(y))
print(f"\nE[exp(y_{{t+1}}) | y_t] for consumption model:")
for i, exp_exp in enumerate(exp_income_expectations):
    print(f"  State {i}: {exp_exp:.4f}")
```

## Summary and Comparison

```python
print("=== HARK MarkovChain Tools Summary ===")
print()
print("Key Features:")
print("✓ Vectorized conditional expectations: mc.expected(func)")
print("✓ Stationary distribution: mc.stationary_dist()")
print("✓ Multi-step transitions: mc.n_step_transition(n)")
print("✓ Path simulation: mc.simulate() and mc.simulate_many()")
print("✓ Time series analysis: mc.autocorrelation(), mc.ergodic_mean()")
print("✓ AR(1) integration: create_markov_chain_from_tauchen()")
print("✓ Labeled states: LabeledMarkovChain")
print("✓ State lookup flexibility: by index, value, or label")
print()
print("Benefits over previous approach:")
print("• Vectorized operations instead of manual loops")
print("• Unified interface for all Markov operations")
print("• Significant performance improvements")
print("• Cleaner, more readable code")
print("• Same expressiveness as HARK's i.i.d. distribution tools")
print()
print("The MarkovChain tools bring processes with memory to the same")
print("convenience level as HARK's excellent i.i.d. distribution utilities!")
```

This notebook demonstrates how the new `MarkovChain` class transforms working with Markov processes in HARK from a cumbersome, manual process to an intuitive, vectorized experience that matches the elegance of HARK's distribution tools.
