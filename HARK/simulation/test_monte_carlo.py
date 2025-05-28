"""
This file implements unit tests for the Monte Carlo simulation module
"""

import unittest
import pandas as pd # Add pandas import
import numpy as np # Ensure numpy is imported for allclose and other ops

from HARK.distributions import Bernoulli, IndexDistribution, MeanOneLogNormal
from HARK.model import Aggregate, Control, DBlock
from HARK.simulation.monte_carlo import *

cons_shocks = {
    "agg_gro": Aggregate(MeanOneLogNormal(1)),
    "psi": IndexDistribution(MeanOneLogNormal, {"sigma": [1.0, 1.1]}),
    "theta": MeanOneLogNormal(1),
    "live": Bernoulli(p=0.98),
}

cons_pre = {
    "R": 1.05,
    "aNrm": 1,
    "gamma": 1.1,
    "psi": 1.1,  # TODO: draw this from a shock,
    "theta": 1.1,  # TODO: draw this from a shock
}

cons_dynamics = {
    "G": lambda gamma, psi: gamma * psi,
    "Rnrm": lambda R, G: R / G,
    "bNrm": lambda Rnrm, aNrm: Rnrm * aNrm,
    "mNrm": lambda bNrm, theta: bNrm + theta,
    "cNrm": Control(["mNrm"]),
    "aNrm": lambda mNrm, cNrm: mNrm - cNrm,
}

cons_dr = {"cNrm": lambda mNrm: mNrm / 2}


class test_draw_shocks(unittest.TestCase):
    def test_draw_shocks(self):
        drawn = draw_shocks(cons_shocks, np.array([0, 1]))

        self.assertEqual(len(drawn["theta"]), 2)
        self.assertEqual(len(drawn["psi"]), 2)
        self.assertTrue(isinstance(drawn["agg_gro"], float))


class test_simulate_dynamics(unittest.TestCase):
    def test_simulate_dynamics(self):
        post = simulate_dynamics(cons_dynamics, cons_pre, cons_dr)

        self.assertAlmostEqual(post["cNrm"], 0.98388429)


class test_AgentTypeMonteCarloSimulator(unittest.TestCase):
    def setUp(self):
        self.calibration = {  # TODO
            "G": 1.05,
        }
        self.block = DBlock(
            **{
                "shocks": {
                    "theta": MeanOneLogNormal(1),
                    "agg_R": Aggregate(MeanOneLogNormal(1)),
                    "live": Bernoulli(p=0.98),
                },
                "dynamics": {
                    "b": lambda agg_R, G, a: agg_R * G * a,
                    "m": lambda b, theta: b + theta,
                    "c": Control(["m"]),
                    "a": lambda m, c: m - c,
                },
            }
        )

        self.initial = {"a": MeanOneLogNormal(1), "live": 1}

        self.dr = {"c": lambda m: m / 2}

    def test_simulate(self):
        self.simulator = AgentTypeMonteCarloSimulator(
            self.calibration,
            self.block,
            self.dr,
            self.initial,
            agent_count=3,
        )

        self.simulator.initialize_sim()
        self.simulator.simulate() # Default T_sim=10
        history_df = self.simulator.get_history_df()

        # Check for a specific period and agent, e.g., period 5, agent 0
        # The original test checked .all(), implying array comparison.
        # We'll check for each agent in period 5.
        for agent_idx in range(self.simulator.agent_count):
            a1 = history_df.loc[(5, agent_idx), "a"]
            
            # Values from previous period (4) for this agent
            a0 = history_df.loc[(4, agent_idx), "a"]
            
            # Shocks and controls for current period (5) for this agent
            # agg_R is an aggregate shock, so it's the same for all agents in a period.
            # We can get it from period_data in raw history or ensure it's correctly placed in df.
            # For simplicity, let's assume it's correctly populated in the df if tracked.
            # If agg_R is not in self.vars, it won't be in history_df.
            # The DBlock defines 'agg_R' as a shock, so it should be in self.vars.
            agg_R_t5 = history_df.loc[(5, agent_idx), "agg_R"] 
            theta_t5 = history_df.loc[(5, agent_idx), "theta"]
            c_t5 = history_df.loc[(5, agent_idx), "c"]
            
            b1_calc = (
                a0 * agg_R_t5 * self.calibration["G"] # G is scalar from calibration
                + theta_t5
                - c_t5
            )
            self.assertAlmostEqual(a1, b1_calc)

    def test_make_shock_history(self):
        self.simulator = AgentTypeMonteCarloSimulator(
            self.calibration,
            self.block,
            self.dr,
            self.initial,
            agent_count=3, # T_sim defaults to 10
        )

        # make_shock_history calls initialize_sim and simulate internally.
        # It populates self.shock_history (list of dicts) and self.newborn_init_history (list of dicts)
        self.simulator.make_shock_history() 

        # newborn_init_history and shock_history are lists of dicts
        newborn_init_hist_made = deepcopy(self.simulator.newborn_init_history)
        shock_hist_made = deepcopy(self.simulator.shock_history)

        # Re-initialize and simulate again to compare if histories are consistent
        self.simulator.read_shocks = False # Ensure we are not reading the history we just made
        self.simulator.initialize_sim()
        # Simulate again to regenerate newborn_init_history during sim_birth
        # and to have a new self.history to compare against shock_hist_made
        self.simulator.simulate() 
        history_df_after_resim = self.simulator.get_history_df()

        # Compare newborn_init_history (list of dicts of arrays)
        # This checks if sim_birth produces consistent initial states
        self.assertEqual(len(newborn_init_hist_made), len(self.simulator.newborn_init_history))
        for period_idx in range(len(newborn_init_hist_made)):
            made_dict = newborn_init_hist_made[period_idx]
            current_dict = self.simulator.newborn_init_history[period_idx]
            self.assertEqual(made_dict.keys(), current_dict.keys())
            for key in made_dict:
                np.testing.assert_array_almost_equal(made_dict[key], current_dict[key], decimal=6, err_msg=f"newborn_init_history differs for {key} at period {period_idx}")

        # Compare shock_history with the 'theta' shock from the new simulation's history
        # shock_hist_made is a list of dicts. Each dict has shock names as keys.
        # history_df_after_resim contains all vars, including shocks.
        self.assertEqual(len(shock_hist_made), self.simulator.T_sim)
        for t in range(self.simulator.T_sim):
            # Get the array of theta shocks for period t from the dataframe
            theta_from_df = history_df_after_resim.loc[(t, slice(None)), "theta"].values
            # Get the array of theta shocks for period t from the stored shock_history
            theta_from_shock_hist = shock_hist_made[t]["theta"]
            np.testing.assert_array_almost_equal(theta_from_df, theta_from_shock_hist, decimal=6, err_msg=f"Shock 'theta' differs at period {t}")


class test_AgentTypeMonteCarloSimulatorAgeVariance(unittest.TestCase):
    def setUp(self):
        self.calibration = {  # TODO
            "G": 1.05,
        }
        self.block = DBlock(
            **{
                "shocks": {
                    "theta": MeanOneLogNormal(1),
                    "agg_R": Aggregate(MeanOneLogNormal(1)),
                    "live": Bernoulli(p=0.98),
                    "psi": IndexDistribution(MeanOneLogNormal, {"sigma": [1.0, 1.1]}),
                },
                "dynamics": {
                    "b": lambda agg_R, G, a: agg_R * G * a,
                    "m": lambda b, theta: b + theta,
                    "c": Control(["m"]),
                    "a": lambda m, c: m - c,
                },
            }
        )

        self.initial = {"a": MeanOneLogNormal(1), "live": 1}
        self.dr = {"c": [lambda m: m * 0.5, lambda m: m * 0.9]}

    def test_simulate(self):
        self.simulator = AgentTypeMonteCarloSimulator(
            self.calibration,
            self.block,
            self.dr,
            self.initial,
            agent_count=3,
        )

        self.simulator.initialize_sim()
        self.simulator.simulate(sim_periods=2)
        history_df = self.simulator.get_history_df()

        # Example check for period 1, for all agents
        # Age-varying decision rule: self.dr["c"] is a list of functions
        # t_age for period 1 will be 1. So dr["c"][1] is used.
        m_p1 = history_df.loc[(1, slice(None)), "m"].values
        c_p1_expected_dr = self.dr["c"][1](m_p1) # Apply DR to market resources of period 1

        # Compare calculated c with c from history_df for period 1
        c_p1_from_df = history_df.loc[(1, slice(None)), "c"].values
        np.testing.assert_array_almost_equal(c_p1_from_df, c_p1_expected_dr, decimal=6)
        
        # Check a = m - c relationship
        a_p1_from_df = history_df.loc[(1, slice(None)), "a"].values
        calculated_a_p1 = m_p1 - c_p1_from_df
        np.testing.assert_array_almost_equal(a_p1_from_df, calculated_a_p1, decimal=6)


class test_MonteCarloSimulator(unittest.TestCase):
    def setUp(self):
        self.calibration = {  # TODO
            "G": 1.05,
        }
        self.block = DBlock(
            **{
                "shocks": {
                    "theta": MeanOneLogNormal(1),
                    "agg_R": Aggregate(MeanOneLogNormal(1)),
                },
                "dynamics": {
                    "b": lambda agg_R, G, a: agg_R * G * a,
                    "m": lambda b, theta: b + theta,
                    "c": Control(["m"]),
                    "a": lambda m, c: m - c,
                },
            }
        )

        self.initial = {"a": MeanOneLogNormal(1)}

        self.dr = {"c": lambda m: m / 2}

    def test_simulate(self):
        self.simulator = MonteCarloSimulator(
            self.calibration,
            self.block,
            self.dr,
            self.initial,
            agent_count=3,
        )

        self.simulator.initialize_sim()
        self.simulator.simulate() # Default T_sim=10
        history_df = self.simulator.get_history_df()

        # Check for a specific period and agent, e.g., period 5, agent 0
        for agent_idx in range(self.simulator.agent_count):
            a1 = history_df.loc[(5, agent_idx), "a"]
            
            a0 = history_df.loc[(4, agent_idx), "a"]
            agg_R_t5 = history_df.loc[(5, agent_idx), "agg_R"]
            theta_t5 = history_df.loc[(5, agent_idx), "theta"]
            c_t5 = history_df.loc[(5, agent_idx), "c"]
            
            b1_calc = (
                a0 * agg_R_t5 * self.calibration["G"]
                + theta_t5
                - c_t5
            )
            self.assertAlmostEqual(a1, b1_calc)
            self.assertAlmostEqual(a1, b1_calc)


# Helper function to create a basic DBlock for testing get_history_df
def create_test_mc_simulator_block():
    return DBlock(
        shocks={'Shock': MeanOneLogNormal(0.1), 'FixedShock': 1.0},
        dynamics={'StateA': lambda Shock: Shock, 
                  'StateB': lambda FixedShock, StateA: FixedShock + StateA,
                  'Age': lambda Age: Age + 1}, # Age here is a state, not t_age of simulator
        initial={'StateA': 0.0, 'Age': 0} # Initial values for states defined in dynamics
    )

class TestAgentTypeMonteCarloSimulatorHistory(unittest.TestCase):
    def setUp(self):
        self.calibration = {'UnusedCalib': 1.0} # Calibration can be simple if not used by DR/dynamics
        self.block = create_test_mc_simulator_block()
        # Decision rule: consume half of StateA plus half of StateB
        self.dr = {'Consume': lambda StateA, StateB: 0.5 * StateA + 0.5 * StateB}
        # Add 'Consume' to block.vars if it's a control variable that should be tracked
        # For this test, assume 'Consume' is implicitly part of self.vars if it's in dr
        # and its components are in dynamics.
        # Let's ensure 'Consume' is part of the dynamics if we want to track it via self.vars_now
        # A bit circular. Let's assume vars are from block.get_vars() which includes shocks and states.
        # If 'Consume' is a control, it's not automatically a state.
        # The current AgentTypeMonteCarloSimulator tracks self.vars_now, which are `post` from simulate_dynamics.
        # simulate_dynamics output includes states from dynamics. Controls are not automatically states.
        # For simplicity, let's make the DR output a state for testing.
        self.block.dynamics['Consume'] = Control(['StateA', 'StateB']) # Mark as control
        self.block.vars.append('Consume') # Manually add if not picked by get_vars based on Control

        self.initial_sim_states = {'StateA': MeanOneLogNormal(0.2), 'Age': 0, 'live': 1} # live is needed for AgentTypeMCS
        
        self.agent_count = 2
        self.T_sim = 3
        
        self.simulator = AgentTypeMonteCarloSimulator(
            calibration=self.calibration,
            block=self.block,
            dr=self.dr,
            initial=self.initial_sim_states, # These are initial shock-like states for sim_birth
            agent_count=self.agent_count,
            T_sim=self.T_sim
        )
        # AgentTypeMonteCarloSimulator specific setup for t_age, etc.
        self.simulator.t_age = np.zeros(self.agent_count, dtype=int)


    def test_get_history_df_structure_and_content(self):
        self.simulator.initialize_sim()
        self.simulator.simulate()
        df = self.simulator.get_history_df()

        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(df.index, pd.MultiIndex)
        self.assertEqual(df.index.names, ['period', 'agent_id'])
        
        # Columns should be from self.simulator.vars
        # self.vars includes states from dynamics, and all shocks.
        # If 'Consume' (control) became a state via dynamics, it's in. If not, it's not.
        # The current AgentTypeMonteCarloSimulator's self.vars is block.get_vars().
        # block.get_vars() gets keys from shocks and dynamics.
        # Let's assume 'Consume' is not a state, so not in df unless explicitly made a state.
        # The setup for self.block.vars.append('Consume') is a bit of a hack.
        # A cleaner way: define 'Consume' in dynamics if it needs to be recorded in vars_now.
        # For now, let's assume self.simulator.vars is the source of truth for columns.
        expected_cols = sorted([v for v in self.simulator.vars if v != 'period'])
        self.assertListEqual(sorted(list(df.columns)), expected_cols)
        
        self.assertEqual(len(df), self.T_sim * self.agent_count)
        
        # Check some content for a variable we know, e.g. 'Age' state from dynamics
        # This 'Age' is a state variable, not t_age from the simulator itself.
        for p_idx in range(self.T_sim):
            for a_idx in range(self.agent_count):
                # Age state should increment each period from its initial value of 0
                self.assertEqual(df.loc[(p_idx, a_idx), 'Age'], p_idx)


    def test_get_history_df_empty_history(self):
        self.simulator.initialize_sim() # History is now []
        df = self.simulator.get_history_df()

        self.assertTrue(df.empty)
        self.assertIsInstance(df.index, pd.MultiIndex)
        self.assertEqual(df.index.names, ['period', 'agent_id'])
        expected_cols = sorted([v for v in self.simulator.vars if v != 'period'])
        self.assertListEqual(sorted(list(df.columns)), expected_cols)

    def test_get_history_df_empty_vars(self):
        # Re-initialize with a block that has no vars (or minimal)
        minimal_block = DBlock(initial={'live':1}) # Only initial 'live' for sim_birth
        
        simulator_empty_vars = AgentTypeMonteCarloSimulator(
            calibration={}, block=minimal_block, dr={}, initial={'live':1},
            agent_count=self.agent_count, T_sim=self.T_sim
        )
        simulator_empty_vars.initialize_sim()
        simulator_empty_vars.simulate()
        df = simulator_empty_vars.get_history_df()

        self.assertIsInstance(df.index, pd.MultiIndex)
        self.assertEqual(df.index.names, ['period', 'agent_id'])
        
        # simulator_empty_vars.vars might not be completely empty due to 'live' from initial.
        # Let's check against its actual self.vars
        expected_cols = sorted([v for v in simulator_empty_vars.vars if v != 'period'])
        self.assertListEqual(sorted(list(df.columns)),expected_cols)
        self.assertEqual(len(df), self.T_sim * self.agent_count)


    def test_get_history_df_agent_count_zero(self):
        simulator_zero_agents = AgentTypeMonteCarloSimulator(
            calibration=self.calibration, block=self.block, dr=self.dr, initial=self.initial_sim_states,
            agent_count=0, T_sim=self.T_sim
        )
        simulator_zero_agents.initialize_sim()
        # simulate() might not run or might run weirdly. get_history_df should handle it.
        # If initialize_sim sets up agent_count=0, then history should be empty.
        # If simulate() is called, it should produce no records.
        simulator_zero_agents.simulate()
        df = simulator_zero_agents.get_history_df()
        
        self.assertTrue(df.empty)
        self.assertIsInstance(df.index, pd.MultiIndex)
        self.assertEqual(df.index.names, ['period', 'agent_id'])
        expected_cols = sorted([v for v in simulator_zero_agents.vars if v != 'period'])
        self.assertListEqual(sorted(list(df.columns)),expected_cols)


    def test_get_history_df_dtype_map_basic(self):
        """
        Test basic dtype casting using dtype_map for AgentTypeMonteCarloSimulator.
        """
        self.simulator.initialize_sim()
        self.simulator.simulate()
        
        # From create_test_mc_simulator_block:
        # 'Age' is int, 'StateA' is float (from MeanOneLogNormal), 'FixedShock' is float, 'Shock' is float
        # 'StateB' is float. 'Consume' is float.
        dtype_map = {'Age': 'int16', 'StateA': 'float32', 'FixedShock': np.float16}
        df = self.simulator.get_history_df(dtype_map=dtype_map)

        self.assertEqual(df['Age'].dtype, np.dtype('int16'))
        self.assertEqual(df['StateA'].dtype, np.dtype('float32'))
        self.assertEqual(df['FixedShock'].dtype, np.dtype('float16'))
        # Other columns should retain their original dtypes
        self.assertTrue(np.issubdtype(df['StateB'].dtype, np.floating)) # Should be float64
        self.assertTrue(np.issubdtype(df['Consume'].dtype, np.floating)) # Should be float64
        self.assertTrue(np.issubdtype(df['Shock'].dtype, np.floating)) # Should be float64


    def test_get_history_df_dtype_map_invalid_cast(self):
        """
        Test that a warning is issued for invalid type casting attempts
        and the original dtype (or numeric coerced) is retained for AgentTypeMonteCarloSimulator.
        """
        self.simulator.initialize_sim()
        self.simulator.simulate() 
        
        # 'StateA' is float and can have non-integer values.
        original_StateA_dtype = self.simulator.get_history_df()['StateA'].dtype

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always") 
            df = self.simulator.get_history_df(dtype_map={'StateA': 'int8'})
            
            # Check if a UserWarning related to casting was issued
            self.assertTrue(any(issubclass(warn.category, UserWarning) and 
                                "Could not cast column 'StateA'" in str(warn.message) for warn in w),
                            "Expected UserWarning for invalid cast not found.")
        
        # Dtype should ideally be the original, or what pd.to_numeric made it if that step ran.
        # Since 'StateA' is already numeric, pd.to_numeric won't change its float status before astype.
        self.assertEqual(df['StateA'].dtype, original_StateA_dtype)


    def test_get_history_df_dtype_map_column_not_found(self):
        """
        Test that a warning is issued if a column in dtype_map is not found for AgentTypeMonteCarloSimulator.
        """
        self.simulator.initialize_sim()
        self.simulator.simulate()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df = self.simulator.get_history_df(dtype_map={'non_existent_col': 'float32'})
            
            self.assertTrue(any(issubclass(warn.category, UserWarning) and
                                "not found in DataFrame" in str(warn.message) for warn in w),
                            "Expected UserWarning for column not found not issued.")
        # Ensure other columns are still present
        self.assertTrue('StateA' in df.columns)
