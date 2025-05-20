.. _timing_and_periods:

####################
Timing and Periods
####################

This guide explains how time, horizons, and cyclical patterns are handled in HARK models. Understanding these concepts is crucial for correctly specifying and interpreting your economic models.

Model Horizons
**************

HARK supports both finite and infinite horizon models. The choice of horizon is primarily controlled by the ``cycles`` parameter:

*   **Finite Horizon (``cycles = 1`` or ``cycles > 1``):**
    *   When ``cycles = 1``, the model runs for a specific number of periods, defined by ``T_cycle``. This is the standard setup for lifecycle models where an agent lives for a fixed number of years.
    *   When ``cycles > 1``, the sequence of ``T_cycle`` periods is repeated ``cycles`` times. This is a less common setup but can be useful for models with a fixed number of repeated sequences.
*   **Infinite Horizon (``cycles = 0``):**
    *   When ``cycles = 0``, the model runs indefinitely. The parameters defined for the ``T_cycle`` periods are repeated forever. This is common for theoretical models analyzing long-run behavior or steady states.

Cycles and Periods (``T_cycle``)
********************************

The ``T_cycle`` parameter defines the number of distinct periods within a single cycle of the model. The interpretation of ``T_cycle`` depends on the ``cycles`` parameter:

*   **``cycles = 1, T_cycle = N``:**
    *   This configuration creates a **finite horizon model with N total periods**. Each period from 1 to N can have unique parameters (like income profiles, survival rates, etc.).
    *   Example: A lifecycle model where an agent lives for 80 years would be specified with ``cycles = 1`` and ``T_cycle = 80``.
*   **``cycles = 0, T_cycle = 1``:**
    *   This is the standard **infinite horizon model with time-homogenous parameters**. There's effectively only one type of period that repeats indefinitely.
    *   Example: A basic buffer stock savings model where the interest rate, income shock distribution, and discount factor are constant over time.
*   **``cycles = 0, T_cycle = N``:**
    *   This creates an **infinite horizon model with an N-period repeating cycle**. The sequence of N periods, each potentially with different parameters, repeats indefinitely.
    *   Example: A model with seasonality, such as quarterly data, could be set up with ``cycles = 0`` and ``T_cycle = 4``. Each quarter within the year can have distinct characteristics, and this annual pattern repeats forever.

Time-Varying Parameters
***********************

Many parameters in HARK can be specified as time-varying, allowing them to differ across the periods defined by ``T_cycle``. These parameters are typically provided as Python lists. The length of these lists should correspond to ``T_cycle``.

Common time-varying parameters include:

*   ``LivPrb``: List of survival probabilities for each period in ``T_cycle``.
*   ``PermGroFac``: List of permanent income growth factors.
*   ``Rfree``: List of risk-free interest rates.
*   ``IncShkDstn``: List of income shock distributions. Each element in the list is a distribution object for the corresponding period in ``T_cycle``.
*   ``TranShkDstn``: List of transitory income shock distributions.
*   ``PermShkDstn``: List of permanent income shock distributions.

For example, if ``T_cycle = 4``, then ``LivPrb`` should be a list with 4 elements, e.g., ``LivPrb = [0.99, 0.98, 0.97, 0.96]``. The first element applies to period 0 of the cycle, the second to period 1, and so on.

Simulation Time Tracking
************************

During simulations, HARK agents keep track of time using two key attributes:

*   **``t_cycle``**: This agent-specific attribute indicates the current period the agent is in within the ``T_cycle`` sequence. It is 0-indexed, so it ranges from ``0`` to ``T_cycle - 1``. For example, in a model with ``T_cycle = 4``, ``t_cycle`` will take values ``0, 1, 2, 3, 0, 1, ...``.
*   **``t_age``**: This agent-specific attribute tracks the total number of periods an agent has been alive since birth in the simulation. It increments by 1 each period.

These attributes are crucial for models where agent behavior or parameters depend on their position in a cycle or their overall age.

Examples
********

*   **Lifecycle Model:**
    *   Typically ``cycles = 1`` and ``T_cycle = L``, where ``L`` is the number of periods in the agent's life (e.g., 60 for a model from age 25 to 85).
    *   Parameters like ``PermGroFac`` and ``LivPrb`` would be lists of length ``L``, defining the age-specific income profile and survival probabilities.
*   **Quarterly Infinite Horizon Model:**
    *   Typically ``cycles = 0`` and ``T_cycle = 4``.
    *   Parameters representing quarterly features (e.g., seasonal income components passed via ``IncShkDstn``) would be lists of length 4.
*   **Standard Infinite Horizon Model (Bewley-Huggett-Aiyagari style):**
    *   Typically ``cycles = 0`` and ``T_cycle = 1``.
    *   Parameters like ``Rfree``, ``PermGroFac``, etc., are often lists with a single element (e.g., ``Rfree = [1.03]``).

For detailed examples of how these parameters are implemented, users can refer to the HARK demo notebooks and the documentation for specific agent types like ``IndShockConsumerType`` or ``PerfForesightConsumerType``.
The setup of these parameters is usually done when initializing an agent type. For instance:

.. code-block:: python

    # Example: Finite horizon lifecycle model (simplified)
    lifecycle_params = {
        "cycles": 1,
        "T_cycle": 60, # 60 years of life
        "LivPrb": [0.99]*59 + [0.0], # Dies for sure in last period
        "PermGroFac": [1.02]*35 + [1.0]*25, # Growth during working life, then flat
        # ... other parameters
    }
    lifecycle_agent = IndShockConsumerType(**lifecycle_params)

    # Example: Infinite horizon quarterly model (simplified)
    quarterly_params = {
        "cycles": 0,
        "T_cycle": 4,
        "Rfree": [1.01, 1.01, 1.01, 1.01], # Same interest rate each quarter
        # ... other parameters that might vary by quarter
    }
    quarterly_agent = IndShockConsumerType(**quarterly_params)

Understanding how ``cycles`` and ``T_cycle`` interact with parameter lists and agent state variables (``t_cycle``, ``t_age``) is key to effectively using HARK.
