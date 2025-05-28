"""
Functions to support Monte Carlo simulation of models.
"""

from copy import copy
from typing import Mapping, Sequence
import warnings # For warning in get_history_df

import numpy as np
import pandas as pd # For get_history_df

from HARK.distributions import (
    Distribution,
    IndexDistribution,
    TimeVaryingDiscreteDistribution,
)
from HARK.model import Aggregate
from HARK.model import DBlock, Control # Control needed for type checking
from HARK.model import construct_shocks, simulate_dynamics

from HARK.utilities import apply_fun_to_vals


def draw_shocks(shocks: Mapping[str, Distribution], conditions: Sequence[int]):
    """
    Draw from each shock distribution values, subject to given conditions.

    Parameters
    ------------
    shocks Mapping[str, Distribution]
        A dictionary-like mapping from shock names to distributions from which to draw

    conditions: Sequence[int]
        An array of conditions, one for each agent.
        Typically these will be agent ages.
        # TODO: generalize this to wider range of inputs.

    Parameters
    ------------
    draws : Mapping[str, Sequence]
        A mapping from shock names to drawn shock values.
    """
    draws = {}

    for shock_var in shocks:
        shock = shocks[shock_var]

        if isinstance(shock, (int, float)):
            draws[shock_var] = np.ones(len(conditions)) * shock
        elif isinstance(shock, Aggregate):
            draws[shock_var] = shock.dist.draw(1)[0]
        elif isinstance(shock, IndexDistribution) or isinstance(
            shock, TimeVaryingDiscreteDistribution
        ):
            ## TODO  his type test is awkward. They should share a superclass.
            draws[shock_var] = shock.draw(conditions)
        else:
            draws[shock_var] = shock.draw(len(conditions))
            # this is hacky if there are no conditions.

    return draws


def calibration_by_age(ages, calibration):
    """
    Returns calibration for this model, but with vectorized
    values which map age-varying values to agent ages.

    Parameters
    ----------
    ages: np.array
        An array of agent ages.

    calibration: dict
        A calibration dictionary

    Returns
    --------
    aged_calibration: dict
        A dictionary of parameter values.
        If a parameter is age-varying, the value is a vector
        corresponding to the values for each input age.
    """

    def aged_param(ages, p_value):
        if isinstance(p_value, (float, int)) or callable(p_value):
            return p_value
        elif isinstance(p_value, list) and len(p_value) > 1:
            pv_array = np.array(p_value)

            return np.apply_along_axis(lambda a: pv_array[a], 0, ages)
        else:
            # This case should ideally not be hit if parameters are well-defined.
            # Return a value that's unlikely to cause immediate crashes but signals issues.
            # Or, raise an error if strictness is preferred.
            # For now, returning an empty array of appropriate object type if possible.
            # This might need refinement based on how such params are used.
            if isinstance(p_value, list) and len(p_value) == 1: # If it's a list of one element, use that element
                 if callable(p_value[0]): return p_value[0]
                 if isinstance(p_value[0], (float, int)): return p_value[0]


            # Fallback for safety, though this path indicates a potential issue upstream.
            # An empty array might not be what's expected by consuming code.
            # Consider logging a warning here in a real application.
            return np.array([])


    return {p: aged_param(ages, calibration[p]) for p in calibration}


class Simulator:
    pass


class AgentTypeMonteCarloSimulator(Simulator):
    """
    A Monte Carlo simulation engine based on the HARK.core.AgentType framework.

    Unlike HARK.core.AgentType, this class does not do any model solving,
    and depends on dynamic equations, shocks, and decision rules paased into it.

    The purpose of this class is to provide a way to simulate models without
    relying on inheritance from the AgentType class.

    This simulator makes assumptions about population birth and mortality which
    are not generic. All agents are replaced with newborns when they expire.

    Parameters
    ------------

    calibration: Mapping[str, Any]

    block : DBlock
        Has shocks, dynamics, and rewards

    dr: Mapping[str, Callable]

    initial: dict

    seed : int
        A seed for this instance's random number generator.

    Attributes
    ----------
    agent_count : int
        The number of agents of this type to use in simulation.

    T_sim : int
        The number of periods to simulate.
    
    vars : list
        List of variable names to be tracked. Can be modified by the user
        after initialization to include tuples for summary statistics, e.g.,
        `self.vars = ['var1', ('var2', 'mean', 'std')]`.
    """

    state_vars = [] # Not used by this class, inherited from AgentType for compatibility?

    def __init__(
        self, calibration, block: DBlock, dr, initial, seed=0, agent_count=1, T_sim=10
    ):
        super().__init__()

        self.calibration = calibration
        self.block = block

        raw_shocks = block.get_shocks()
        self.shocks = construct_shocks(raw_shocks, calibration)

        self.dynamics = block.get_dynamics()
        self.dr = dr
        self.initial = initial

        self.seed = seed 
        self.agent_count = agent_count
        self.T_sim = T_sim
        
        # self.vars lists variables to be tracked. User can modify this list
        # after __init__ to include tuples for summary statistics.
        self.vars = block.get_vars() # List of strings from DBlock

        # Initialize vars_now for all base variable names that will be tracked or used for stats
        # This ensures vars_now is prepared before initialize_sim might use it.
        all_base_vars_to_init = set()
        if isinstance(self.vars, list): # self.vars should always be a list from get_vars()
            for var_spec in self.vars:
                if isinstance(var_spec, str):
                    all_base_vars_to_init.add(var_spec)
                elif isinstance(var_spec, tuple) and len(var_spec) > 0:
                    all_base_vars_to_init.add(var_spec[0])
        
        self.vars_now = {var_name: None for var_name in all_base_vars_to_init}
        self.vars_prev = self.vars_now.copy()

        self.read_shocks = False
        self.shock_history = [] 
        self.newborn_init_history = [] 
        self.history = [] 

        self.reset_rng()

    def reset_rng(self):
        self.RNG = np.random.default_rng(self.seed)

    def initialize_sim(self):
        if self.T_sim <= 0:
            raise Exception("T_sim must be a positive number.")

        self.reset_rng()
        self.t_sim = 0
        all_agents = np.ones(self.agent_count, dtype=bool)
        blank_array = np.empty(self.agent_count)
        blank_array[:] = np.nan
        
        # Correctly initialize vars_now based on self.vars (which might have been modified by user)
        # and ensure all variables from the block are also initialized.
        current_vars_keys = set(self.block.get_vars()) # From DBlock
        for var_spec in self.vars: # From potentially user-modified self.vars
            if isinstance(var_spec, str):
                current_vars_keys.add(var_spec)
            elif isinstance(var_spec, tuple) and len(var_spec) > 0:
                current_vars_keys.add(var_spec[0]) # Add base name for stats tuples

        for var_name in current_vars_keys:
            # Initialize if not present or if not a correctly sized array (for agent_count > 0)
            if self.agent_count > 0:
                if not (var_name in self.vars_now and                         isinstance(self.vars_now[var_name], np.ndarray) and                         self.vars_now[var_name].shape == (self.agent_count,)):
                    self.vars_now[var_name] = copy(blank_array)
            else: # agent_count is 0
                 self.vars_now[var_name] = np.array([])


        self.t_age = np.zeros(self.agent_count, dtype=int) 
        self.t_cycle = np.zeros(self.agent_count, dtype=int)  

        # Initialize newborn_init_history as a list of dicts
        self.newborn_init_history = [{} for _ in range(self.T_sim)]

        if self.read_shocks and bool(self.newborn_init_history) and len(self.newborn_init_history) > 0:
            # This part assumes newborn_init_history[0] was pre-populated by make_shock_history
            # or some other means if read_shocks is True from the start.
            first_period_newborn_data = self.newborn_init_history[0] # Should be a dict
            for init_var_name, default_dist in self.initial.items():
                if init_var_name in first_period_newborn_data:
                    val = first_period_newborn_data[init_var_name]
                    # Ensure it's a correctly shaped array if agent_count > 0
                    if self.agent_count > 0:
                        if isinstance(val, np.ndarray) and val.shape == (self.agent_count,):
                            self.vars_now[init_var_name] = val
                        else: # Scalar broadcast
                            self.vars_now[init_var_name] = np.full(self.agent_count, val)
                    else: # agent_count = 0
                        self.vars_now[init_var_name] = np.array([])
                else: # Fallback if not in pre-populated history (should not happen if make_shock_history was run)
                    if self.agent_count > 0:
                        drawn_initials = draw_shocks({init_var_name : default_dist}, np.zeros(self.agent_count, dtype=int))
                        self.vars_now[init_var_name] = drawn_initials[init_var_name]
                    else:
                        self.vars_now[init_var_name] = np.array([])
        
        if self.agent_count > 0 : # sim_birth only makes sense if there are agents
            self.sim_birth(all_agents) 

        self.clear_history() 
        return None

    def sim_one_period(self):
        self.get_mortality()  

        # Correctly copy current state to previous state
        for var_name_key in self.vars_now: 
            current_val = self.vars_now[var_name_key]
            if isinstance(current_val, np.ndarray):
                 self.vars_prev[var_name_key] = copy(current_val)
            else: # Scalars or other types
                 self.vars_prev[var_name_key] = current_val
            
            # Reinitialize array entries in vars_now for the new period's data
            if isinstance(current_val, np.ndarray):
                self.vars_now[var_name_key] = np.empty(self.agent_count)
                self.vars_now[var_name_key][:] = np.nan
        
        shocks_now = {}
        if self.read_shocks:  
            if self.t_sim < len(self.shock_history) and self.shock_history[self.t_sim]: # Check if dict is not empty
                current_period_shocks_data = self.shock_history[self.t_sim]
                for var_name_shock in self.shocks: 
                    if var_name_shock in current_period_shocks_data:
                        shocks_now[var_name_shock] = current_period_shocks_data[var_name_shock]
                    else: # Shock not found in history for this period, draw it (or fill with NaN)
                        if self.agent_count > 0:
                             shocks_now[var_name_shock] = draw_shocks({var_name_shock: self.shocks[var_name_shock]}, self.t_age)[var_name_shock]
                        else:
                             shocks_now[var_name_shock] = np.array([])
            else: # History not long enough or empty dict, draw all shocks
                if self.agent_count > 0:
                    shocks_now = draw_shocks(self.shocks, self.t_age)
                else: # Handle agent_count = 0
                    for var_name_shock in self.shocks: shocks_now[var_name_shock] = np.array([])

        else: # Not reading shocks, draw them
            if self.agent_count > 0:
                shocks_now = draw_shocks(self.shocks, self.t_age)
            else:
                for var_name_shock in self.shocks: shocks_now[var_name_shock] = np.array([])


        pre = calibration_by_age(self.t_age, self.calibration)
        pre.update(self.vars_prev) # Add previous states
        pre.update(shocks_now)   # Add current shocks

        dr = calibration_by_age(self.t_age, self.dr)
        post = simulate_dynamics(self.dynamics, pre, dr)

        # Update self.vars_now with results from post.
        # Only update keys that are expected to be in self.vars_now (base variables).
        for key, value in post.items():
            if key in self.vars_now: 
                self.vars_now[key] = value
            # If a new variable is generated by dynamics not in self.vars, it's not stored in vars_now
            # unless self.vars was manually expanded to include it.

        self.t_age = self.t_age + 1

    def make_shock_history(self):
        # This method populates self.shock_history and self.newborn_init_history
        # It needs to run a simulation itself to generate these.
        
        # Store original state of read_shocks and self.vars
        original_read_shocks = self.read_shocks
        original_vars = self.vars
        original_history = self.history # Store to restore later if needed
        
        self.read_shocks = False # Ensure we generate shocks
        # Ensure all base shock variables are tracked for make_shock_history to work
        # This is a temporary override of self.vars for the purpose of this method
        temp_track_vars = list(self.shocks.keys())
        # Also track initial states if they are part of self.vars or self.initial
        for var_spec in original_vars: # Iterate through original self.vars
            var_name = var_spec if isinstance(var_spec, str) else var_spec[0]
            if var_name in self.initial: # Check if base var_name is an initial condition
                if var_name not in temp_track_vars : temp_track_vars.append(var_name)

        self.vars = temp_track_vars # Temporarily set self.vars for this simulation

        self.initialize_sim() # This will use the temporary self.vars for history structure
        # initialize_sim calls sim_birth, which populates newborn_init_history[0]
        
        # Simulate to generate data in self.history and fill newborn_init_history
        # newborn_init_history is filled during sim_birth calls within simulate
        simulated_history = self.simulate() # Uses the temporary self.vars

        # Populate self.shock_history from the simulated_history
        self.shock_history = []
        for period_data_from_sim_hist in simulated_history:
            current_period_shock_data = {}
            for shock_name_key in self.shocks.keys():
                if shock_name_key in period_data_from_sim_hist and                    isinstance(period_data_from_sim_hist[shock_name_key], np.ndarray):
                    current_period_shock_data[shock_name_key] = period_data_from_sim_hist[shock_name_key]
            if current_period_shock_data: # Only append if there's actual shock data
                self.shock_history.append(current_period_shock_data)
        
        # Restore original state
        self.read_shocks = True # Set to true as per method's contract
        self.vars = original_vars
        self.history = original_history # Restore original history

        # Re-run initialize_sim with original self.vars to set t_sim=0 and clear history correctly
        # for any subsequent user simulation.
        self.initialize_sim()

        return self.shock_history


    def get_mortality(self):
        if 'live' in self.vars_now and isinstance(self.vars_now['live'], np.ndarray) and self.agent_count > 0:
             who_dies = self.vars_now["live"] <= 0
        else:
             who_dies = np.zeros(self.agent_count, dtype=bool)
        
        if self.agent_count > 0:
            self.sim_birth(who_dies)
        self.who_dies = who_dies # Store for potential tracking
        return None

    def sim_birth(self, which_agents):
        # which_agents is a boolean array of size self.agent_count
        num_to_birth = np.sum(which_agents)
        if num_to_birth == 0 and self.agent_count > 0 : return

        current_newborn_data = {}
        if self.read_shocks:
            if self.t_sim < len(self.newborn_init_history) and self.newborn_init_history[self.t_sim]:
                period_newborn_data_source = self.newborn_init_history[self.t_sim]
                # Extract data only for agents being born if source is full array
                for init_var_name in self.initial:
                    if init_var_name in period_newborn_data_source:
                        full_array_val = period_newborn_data_source[init_var_name]
                        if isinstance(full_array_val, np.ndarray) and full_array_val.ndim > 0 and self.agent_count > 0:
                             current_newborn_data[init_var_name] = full_array_val[which_agents]
                        elif self.agent_count > 0 : # Scalar broadcast
                             current_newborn_data[init_var_name] = np.full(num_to_birth, full_array_val)
                        # if agent_count is 0, current_newborn_data remains empty for this var
                    else: # Fallback: draw if missing (should ideally not happen if make_shock_history was used)
                        if self.agent_count > 0:
                            current_newborn_data.update(draw_shocks({init_var_name: self.initial[init_var_name]}, np.zeros(num_to_birth, dtype=int)))
            else: # Fallback: draw all if history is missing/short
                if self.agent_count > 0:
                    current_newborn_data = draw_shocks(self.initial, np.zeros(num_to_birth, dtype=int))
        else: # Not reading shocks, so draw initial states for newborns
            if self.agent_count > 0:
                current_newborn_data = draw_shocks(self.initial, np.zeros(num_to_birth, dtype=int))

            # Store these drawn states in newborn_init_history for the current period
            if self.t_sim < self.T_sim : # Ensure t_sim is within bounds of initialized list
                for var_name_initial, val_array in current_newborn_data.items():
                    # Store the full array (NaNs for non-newborns)
                    if self.agent_count > 0:
                        history_array_for_var = np.full(self.agent_count, np.nan)
                        history_array_for_var[which_agents] = val_array
                        self.newborn_init_history[self.t_sim][var_name_initial] = history_array_for_var
                    # If agent_count is 0, val_array is empty, nothing to store in dict's array.

        # Assign drawn/read states to self.vars_now for the agents being born
        if self.agent_count > 0 and num_to_birth > 0:
            for varn_assign, val_assign_array in current_newborn_data.items():
                if varn_assign in self.vars_now: 
                    self.vars_now[varn_assign][which_agents] = val_assign_array
        
        self.t_age[which_agents] = 0
        self.t_cycle[which_agents] = 0


    def simulate(self, sim_periods=None):
        if not hasattr(self, "t_sim") or self.t_sim == 0 and not self.history : # Fresh start or explicit re-init
            self.initialize_sim()

        if sim_periods is None: sim_periods = self.T_sim
        if self.t_sim + sim_periods > self.T_sim:
            warnings.warn(f"Requested simulation for {sim_periods} periods, but only {self.T_sim - self.t_sim} remain in T_sim. Simulating remaining periods.")
            sim_periods = self.T_sim - self.t_sim
            if sim_periods <=0: return self.history


        with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
            for _t_loop_idx in range(sim_periods):
                if self.t_sim >= self.T_sim: break # Stop if we exceed total simulation horizon
                self.sim_one_period()

                period_data = {"period": self.t_sim}
                for track_spec_item in self.vars: 
                    if isinstance(track_spec_item, str):
                        var_name = track_spec_item
                        val = self.vars_now.get(var_name)
                        period_data[var_name] = copy(val) if isinstance(val, np.ndarray) else val
                    
                    elif isinstance(track_spec_item, tuple) and len(track_spec_item) > 1:
                        var_name = track_spec_item[0]
                        stats_to_calc = track_spec_item[1:]
                        data_array = self.vars_now.get(var_name)

                        if isinstance(data_array, np.ndarray) and data_array.size > 0: # Ensure array has data
                            for stat_str in stats_to_calc:
                                stat_key = f"{var_name}_{stat_str}"
                                if stat_str == 'mean': period_data[stat_key] = np.mean(data_array)
                                elif stat_str == 'std': period_data[stat_key] = np.std(data_array)
                                elif stat_str == 'min': period_data[stat_key] = np.min(data_array)
                                elif stat_str == 'max': period_data[stat_key] = np.max(data_array)
                                elif stat_str == 'median': period_data[stat_key] = np.median(data_array)
                                else: period_data[stat_key] = np.nan # Unknown stat
                        else: # data_array is None, not an array, or empty
                            for stat_str in stats_to_calc:
                                period_data[f"{var_name}_{stat_str}"] = np.nan
                
                self.history.append(period_data)
                self.t_sim += 1
            return self.history

    def clear_history(self):
        self.history = []

    def get_history_df(self, dtype_map=None):
        if not self.history: 
            df_cols = ['period', 'agent_id']
            # Determine columns from self.vars
            for item_spec in getattr(self, 'vars', []):
                if isinstance(item_spec, str):
                    if item_spec not in df_cols: df_cols.append(item_spec)
                elif isinstance(item_spec, tuple) and len(item_spec) > 1:
                    base_name = item_spec[0]
                    for stat in item_spec[1:]:
                        col_name = f"{base_name}_{stat}"
                        if col_name not in df_cols: df_cols.append(col_name)
            # Remove 'period' if it was added from vars, as it's an index level
            if 'period' in df_cols and 'period' not in ['period', 'agent_id']: df_cols.remove('period')

            # Ensure no duplicates if 'agent_id' somehow in self.vars
            if 'agent_id' in df_cols and 'agent_id' not in ['period', 'agent_id']: df_cols.remove('agent_id')
            
            final_cols = [c for c in df_cols if c not in ['period', 'agent_id']]
            
            idx = pd.MultiIndex(levels=[[], []], codes=[[], []], names=['period', 'agent_id'])
            return pd.DataFrame(columns=final_cols, index=idx)

        # Determine all unique column names from the history data (excluding 'period')
        all_keys = set()
        for period_dict in self.history:
            all_keys.update(period_dict.keys())
        if 'period' in all_keys: all_keys.remove('period')
        sorted_columns = sorted(list(all_keys))

        all_records = []
        for period_dict in self.history:
            current_period_id = period_dict.get('period', self.t_sim -1) # Default to last t_sim if 'period' key missing

            # Determine number of agents for this period, robustly
            num_agents_this_period = self.agent_count
            if self.agent_count == 0: # If agent_count is 0
                # Check if there are any scalar (summary) stats to record
                has_scalar_data = any(not isinstance(period_dict.get(k), np.ndarray) for k in sorted_columns if k in period_dict)
                if has_scalar_data:
                    num_agents_this_period = 1 # Create one row for summary stats
                else:
                    num_agents_this_period = 0 # No data to record
            
            for agent_idx in range(num_agents_this_period):
                agent_id_to_record = agent_idx if self.agent_count > 0 else -1 # Use -1 for agent_id if agent_count is 0
                record = {'period': current_period_id, 'agent_id': agent_id_to_record}
                for key in sorted_columns: 
                    value = period_dict.get(key) # Get value, defaults to None if key missing
                    
                    if isinstance(value, np.ndarray):
                        if value.ndim > 0 and agent_idx < len(value): # Ensure it's an array and agent_idx is valid
                            record[key] = value[agent_idx]
                        elif value.ndim == 0 : # 0-dim array (scalar)
                             record[key] = value.item()
                        else: # agent_idx out of bounds or other array issue
                            record[key] = np.nan 
                    elif value is not None: # Scalar value
                        record[key] = value 
                    else: # Key was missing in period_dict or value was None
                        record[key] = np.nan
                all_records.append(record)
        
        if not all_records: # Fallback if loop resulted in no records (e.g. agent_count=0, no summary stats)
            idx = pd.MultiIndex(levels=[[], []], codes=[[], []], names=['period', 'agent_id'])
            return pd.DataFrame(columns=sorted_columns, index=idx)
            
        df = pd.DataFrame.from_records(all_records)
        if not df.empty:
            # Ensure 'period' and 'agent_id' are present before trying to set index
            if 'period' not in df.columns: df['period'] = -1 # Should have been set
            if 'agent_id' not in df.columns: df['agent_id'] = -1 # Should have been set
            df = df.set_index(['period', 'agent_id'])
        else: # DataFrame is empty after from_records, create with proper structure
            idx = pd.MultiIndex(levels=[[], []], codes=[[], []], names=['period', 'agent_id'])
            df = pd.DataFrame(columns=sorted_columns, index=idx)
        
        # Apply dtype mapping if provided
        if dtype_map is not None and isinstance(dtype_map, dict) and not df.empty:
            for col_name, target_type in dtype_map.items():
                if col_name in df.columns:
                    try:
                        # Optional: Coerce to numeric first if target is numeric, to handle strings like 'NA'
                        is_numeric_target = False
                        try:
                            if np.dtype(target_type).kind in 'ifc': # int, float, complex
                                is_numeric_target = True
                        except TypeError: # target_type might be a string like 'Int64' (nullable int)
                            if isinstance(target_type, str) and ('int' in target_type.lower() or 'float' in target_type.lower()):
                                is_numeric_target = True
                        
                        current_col_dtype = df[col_name].dtype
                        # Avoid unnecessary casting if already correct, or if target is object (which pd.to_numeric might alter undesirably)
                        if is_numeric_target and not pd.api.types.is_numeric_dtype(current_col_dtype):
                             df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                        
                        df[col_name] = df[col_name].astype(target_type)
                    except Exception as e:
                        warnings.warn(f"Could not cast column '{col_name}' to type '{target_type}'. Error: {e}")
                else:
                    warnings.warn(f"Column '{col_name}' specified in dtype_map not found in DataFrame.")
            
        return df


class MonteCarloSimulator(Simulator):
    """
    A Monte Carlo simulation engine based.

    Unlike the AgentTypeMonteCarloSimulator HARK.core.AgentType,
    this class does make any assumptions about aging or mortality.
    It operates only on model information passed in as blocks.

    It also does not have read_shocks functionality;
    it is a strict subset of the AgentTypeMonteCarloSimulator functionality.

    Parameters
    ------------

    calibration: Mapping[str, Any]

    block : DBlock
        Has shocks, dynamics, and rewards

    dr: Mapping[str, Callable]

    initial: dict

    seed : int
        A seed for this instance's random number generator.

    Attributes
    ----------
    agent_count : int
        The number of agents of this type to use in simulation.

    T_sim : int
        The number of periods to simulate.
    """

    state_vars = []

    def __init__(
        self, calibration, block: DBlock, dr, initial, seed=0, agent_count=1, T_sim=10
    ):
        super().__init__()

        self.calibration = calibration
        self.block = block

        # shocks are exogenous (but for age) but can depend on calibration
        raw_shocks = block.get_shocks()
        self.shocks = construct_shocks(raw_shocks, calibration)

        self.dynamics = block.get_dynamics()
        self.dr = dr
        self.initial = initial

        self.seed = seed  
        self.agent_count = agent_count  
        self.T_sim = T_sim
        
        self.vars = block.get_vars() # self.vars can be modified by user for summary stats

        # Initialize vars_now based on self.vars (which might include tuples)
        all_base_vars_to_init = set()
        if isinstance(self.vars, list):
            for var_spec in self.vars:
                if isinstance(var_spec, str):
                    all_base_vars_to_init.add(var_spec)
                elif isinstance(var_spec, tuple) and len(var_spec) > 0:
                    all_base_vars_to_init.add(var_spec[0])
        
        self.vars_now = {var_name: None for var_name in all_base_vars_to_init}
        self.vars_prev = self.vars_now.copy()


        self.shock_history = [] 
        self.newborn_init_history = [] 
        self.history = [] 

        self.reset_rng()

    def reset_rng(self):
        self.RNG = np.random.default_rng(self.seed)

    def initialize_sim(self):
        if self.T_sim <= 0:
            raise Exception("T_sim must be a positive number.")

        self.reset_rng()
        self.t_sim = 0
        # Unlike AgentTypeMC, this simpler simulator doesn't manage 'all_agents' for birth automatically
        
        blank_array = np.empty(self.agent_count)
        blank_array[:] = np.nan

        current_vars_keys = set(self.block.get_vars())
        for var_spec in self.vars:
            if isinstance(var_spec, str):
                current_vars_keys.add(var_spec)
            elif isinstance(var_spec, tuple) and len(var_spec) > 0:
                current_vars_keys.add(var_spec[0])

        for var_name in current_vars_keys:
            if self.agent_count > 0:
                if not (var_name in self.vars_now and                         isinstance(self.vars_now[var_name], np.ndarray) and                         self.vars_now[var_name].shape == (self.agent_count,)):
                    self.vars_now[var_name] = copy(blank_array)
            else:
                self.vars_now[var_name] = np.array([])


        self.t_cycle = np.zeros(self.agent_count, dtype=int) 

        # This simulator doesn't use newborn_init_history in the same way as AgentTypeMC
        # It directly initializes states.
        if self.agent_count > 0:
            initial_states = draw_shocks(self.initial, np.zeros(self.agent_count, dtype=int))
            for var_name_initial, val_array in initial_states.items():
                if var_name_initial in self.vars_now:
                    self.vars_now[var_name_initial] = val_array
        
        self.clear_history()
        return None

    def sim_one_period(self):
        for var_name_key in self.vars_now:
            current_val = self.vars_now[var_name_key]
            if isinstance(current_val, np.ndarray):
                 self.vars_prev[var_name_key] = copy(current_val)
            else:
                 self.vars_prev[var_name_key] = current_val
            
            if isinstance(current_val, np.ndarray): # Reinitialize arrays
                self.vars_now[var_name_key] = np.empty(self.agent_count)
                self.vars_now[var_name_key][:] = np.nan
        
        if self.agent_count > 0:
            shocks_now = draw_shocks(self.shocks, np.zeros(self.agent_count, dtype=int))
        else:
            shocks_now = {key:np.array([]) for key in self.shocks}


        pre = self.calibration.copy() # Use a copy to avoid modifying original calibration
        pre.update(self.vars_prev)
        pre.update(shocks_now)

        dr = self.dr 
        post = simulate_dynamics(self.dynamics, pre, dr)

        for r_name in self.block.reward: # Calculate rewards if any
            # Ensure all args for reward func are in 'post' or 'pre' before calling
            reward_func = self.block.reward[r_name]
            # This part might need more robust argument fetching if rewards depend on 'pre'
            # For now, assume rewards depend on 'post' state or shocks already in 'post'
            
            # Simplified: assumes reward func args are in `post`
            # A more robust solution would inspect reward_func signature
            # and pull from `pre` or `post` as needed.
            try:
                eval_context = {}
                eval_context.update(pre) # Shocks and prev states are in pre
                eval_context.update(post) # Current states/controls are in post
                post[r_name] = apply_fun_to_vals(reward_func, eval_context)
            except TypeError as e:
                 warnings.warn(f"Could not compute reward {r_name} due to missing arguments or type error: {e}")
                 if self.agent_count > 0: post[r_name] = np.full(self.agent_count, np.nan)
                 else: post[r_name] = np.array([])


        for key, value in post.items():
            if key in self.vars_now:
                self.vars_now[key] = value


    def simulate(self, sim_periods=None):
        if not hasattr(self, "t_sim") or self.t_sim == 0 and not self.history:
            self.initialize_sim()
            
        if sim_periods is None: sim_periods = self.T_sim
        if self.t_sim + sim_periods > self.T_sim:
            warnings.warn(f"Requested simulation for {sim_periods} periods, but only {self.T_sim - self.t_sim} remain in T_sim. Simulating remaining periods.")
            sim_periods = self.T_sim - self.t_sim
            if sim_periods <=0: return self.history


        with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
            for _t_loop_idx in range(sim_periods):
                if self.t_sim >= self.T_sim: break
                self.sim_one_period()

                period_data = {"period": self.t_sim}
                for var_spec_item in self.vars: 
                    if isinstance(var_spec_item, str):
                        var_name = var_spec_item
                        val = self.vars_now.get(var_name)
                        period_data[var_name] = copy(val) if isinstance(val, np.ndarray) else val
                    
                    elif isinstance(var_spec_item, tuple) and len(var_spec_item) > 1:
                        var_name = var_spec_item[0]
                        stats_to_calc = var_spec_item[1:]
                        data_array = self.vars_now.get(var_name)

                        if isinstance(data_array, np.ndarray) and data_array.size > 0:
                            for stat_str in stats_to_calc:
                                stat_key = f"{var_name}_{stat_str}"
                                if stat_str == 'mean': period_data[stat_key] = np.mean(data_array)
                                elif stat_str == 'std': period_data[stat_key] = np.std(data_array)
                                elif stat_str == 'min': period_data[stat_key] = np.min(data_array)
                                elif stat_str == 'max': period_data[stat_key] = np.max(data_array)
                                elif stat_str == 'median': period_data[stat_key] = np.median(data_array)
                                else: period_data[stat_key] = np.nan 
                        else: 
                            for stat_str in stats_to_calc:
                                period_data[f"{var_name}_{stat_str}"] = np.nan
                
                self.history.append(period_data)
                self.t_sim += 1
            return self.history

    def clear_history(self):
        self.history = []

    def get_history_df(self): # Same as AgentTypeMonteCarloSimulator's get_history_df
        if not self.history: 
            df_cols = ['period', 'agent_id']
            # Determine columns from self.vars
            for item_spec in getattr(self, 'vars', []):
                if isinstance(item_spec, str):
                    if item_spec not in df_cols: df_cols.append(item_spec)
                elif isinstance(item_spec, tuple) and len(item_spec) > 1:
                    base_name = item_spec[0]
                    for stat in item_spec[1:]:
                        col_name = f"{base_name}_{stat}"
                        if col_name not in df_cols: df_cols.append(col_name)
            # Remove 'period' if it was added from vars, as it's an index level
            if 'period' in df_cols and 'period' not in ['period', 'agent_id']: df_cols.remove('period')

            # Ensure no duplicates if 'agent_id' somehow in self.vars
            if 'agent_id' in df_cols and 'agent_id' not in ['period', 'agent_id']: df_cols.remove('agent_id')
            
            final_cols = [c for c in df_cols if c not in ['period', 'agent_id']]
            
            idx = pd.MultiIndex(levels=[[], []], codes=[[], []], names=['period', 'agent_id'])
            return pd.DataFrame(columns=final_cols, index=idx)

        # Determine all unique column names from the history data (excluding 'period')
        all_keys = set()
        for period_dict in self.history:
            all_keys.update(period_dict.keys())
        if 'period' in all_keys: all_keys.remove('period')
        sorted_columns = sorted(list(all_keys))

        all_records = []
        for period_dict in self.history:
            current_period_id = period_dict.get('period', self.t_sim -1)
            num_agents_this_period = self.agent_count
            if self.agent_count == 0:
                has_scalar_data = any(not isinstance(period_dict.get(k), np.ndarray) for k in sorted_columns if k in period_dict)
                num_agents_this_period = 1 if has_scalar_data else 0
            
            for agent_idx in range(num_agents_this_period):
                agent_id_to_record = agent_idx if self.agent_count > 0 else -1
                record = {'period': current_period_id, 'agent_id': agent_id_to_record}
                for key in sorted_columns: 
                    value = period_dict.get(key)
                    if isinstance(value, np.ndarray):
                        if value.ndim > 0 and agent_idx < len(value): record[key] = value[agent_idx]
                        elif value.ndim == 0 : record[key] = value.item()
                        else: record[key] = np.nan 
                    elif value is not None: record[key] = value 
                    else: record[key] = np.nan
                all_records.append(record)
        
        if not all_records: 
            idx = pd.MultiIndex(levels=[[], []], codes=[[], []], names=['period', 'agent_id'])
            return pd.DataFrame(columns=sorted_columns, index=idx)
            
        df = pd.DataFrame.from_records(all_records)
        if not df.empty:
            # Ensure 'period' and 'agent_id' are present before trying to set index
            if 'period' not in df.columns: df['period'] = -1
            if 'agent_id' not in df.columns: df['agent_id'] = -1
            df = df.set_index(['period', 'agent_id'])
        else: 
            idx = pd.MultiIndex(levels=[[], []], codes=[[], []], names=['period', 'agent_id'])
            df = pd.DataFrame(columns=sorted_columns, index=idx)
        return df
