"""
This file implements unit tests for core HARK functionality.
"""

import unittest
import warnings # For self.assertWarns

import numpy as np
import pandas as pd # Add pandas import
import pytest
from copy import deepcopy

from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    init_idiosyncratic_shocks,
)
from HARK.core import AgentPopulation, AgentType, Parameters, distribute_params
from HARK.distributions import Uniform
from HARK.metric import MetricObject, distance_metric


class test_distance_metric(unittest.TestCase):
    def setUp(self):
        self.list_a = [1.0, 2.1, 3]
        self.list_b = [3.1, 4, -1.4]
        self.list_c = [8.6, 9]
        self.obj_a = MetricObject()
        self.obj_b = MetricObject()
        self.obj_c = MetricObject()
        self.dict_a = {"a": 1, "b": 2}
        self.dict_b = {"a": 3, "b": 4}
        self.dict_c = {"a": 5, "f": 6}

    def test_list(self):
        # same length
        self.assertEqual(distance_metric(self.list_a, self.list_b), 4.4)
        # different length
        self.assertEqual(distance_metric(self.list_b, self.list_c), 1.0)
        # sanity check, same objects
        self.assertEqual(distance_metric(self.list_b, self.list_b), 0.0)

    def test_array(self):
        # same length
        self.assertEqual(
            distance_metric(np.array(self.list_a), np.array(self.list_b)), 4.4
        )
        # different length
        self.assertEqual(
            distance_metric(np.array(self.list_b).reshape(1, 3), np.array(self.list_c)),
            1.0,
        )
        # sanity check, same objects
        self.assertEqual(
            distance_metric(np.array(self.list_b), np.array(self.list_b)), 0.0
        )

    def test_dict(self):
        # Same keys (max of diffs across keys)
        self.assertEqual(distance_metric(self.dict_a, self.dict_b), 2.0)
        # Different keys
        self.assertEqual(distance_metric(self.dict_a, self.dict_c), 1000.0)

    def test_hark_object_distance(self):
        self.obj_a.distance_criteria = ["var_1", "var_2", "var_3"]
        self.obj_b.distance_criteria = ["var_1", "var_2", "var_3"]
        self.obj_c.distance_criteria = ["var_5"]
        # if attributes don't exist or don't match
        self.assertEqual(distance_metric(self.obj_a, self.obj_b), 1000.0)
        self.assertEqual(distance_metric(self.obj_a, self.obj_c), 1000.0)
        # add single numbers to attributes
        self.obj_a.var_1, self.obj_a.var_2, self.obj_a.var_3 = 0.1, 1, 2.1
        self.obj_b.var_1, self.obj_b.var_2, self.obj_b.var_3 = 1.8, -1, 0.1
        self.assertEqual(distance_metric(self.obj_a, self.obj_b), 2.0)

        # sanity check - same objects
        self.assertEqual(distance_metric(self.obj_a, self.obj_a), 0.0)


class test_MetricObject(unittest.TestCase):
    def setUp(self):
        # similar test to distance_metric
        self.obj_a = MetricObject()
        self.obj_b = MetricObject()
        self.obj_c = MetricObject()

    def test_distance(self):
        self.obj_a.distance_criteria = ["var_1", "var_2", "var_3"]
        self.obj_b.distance_criteria = ["var_1", "var_2", "var_3"]
        self.obj_c.distance_criteria = ["var_5"]
        self.obj_a.var_1, self.obj_a.var_2, self.obj_a.var_3 = [0.1], [1, 2], [2.1]
        self.obj_b.var_1, self.obj_b.var_2, self.obj_b.var_3 = [1.8], [0, 0.1], [1.1]
        self.assertEqual(self.obj_a.distance(self.obj_b), 1.9)
        # change the length of a attribute list
        self.obj_b.var_1, self.obj_b.var_2, self.obj_b.var_3 = [1.8], [0, 0, 0.1], [1.1]
        self.assertEqual(self.obj_a.distance(self.obj_b), 1.7)
        # sanity check
        self.assertEqual(self.obj_b.distance(self.obj_b), 0.0)


class test_AgentType(unittest.TestCase):
    def setUp(self):
        self.agent = AgentType(cycles=1)

    def test_solve(self):
        self.agent.time_vary = ["vary_1"]
        self.agent.time_inv = ["inv_1"]
        self.agent.vary_1 = [1.1, 1.2, 1.3, 1.4]
        self.agent.inv_1 = 1.05
        # to test the superclass we create a dummy solve_one_period function
        # for our agent, which doesn't do anything, instead of using a NullFunc
        self.agent.solve_one_period = lambda vary_1: MetricObject()
        self.agent.solve()
        self.assertEqual(len(self.agent.solution), 4)
        self.assertTrue(isinstance(self.agent.solution[0], MetricObject))

    def test_describe(self):
        self.assertTrue("Parameters" in self.agent.describe())

    def test___eq__(self):
        agent2 = AgentType(cycles=1)
        agent3 = AgentType(cycels=2)

        self.assertEqual(self.agent, agent2)
        self.assertNotEqual(self.agent, agent3)


class test_distribute_params(unittest.TestCase):
    def setUp(self):
        self.agent = AgentType(cycles=1, AgentCount=3)

    def test_distribute_params(self):
        dist = Uniform(bot=0.9, top=0.94)

        self.agents = distribute_params(self.agent, "DiscFac", 3, dist)

        self.assertTrue(all(["DiscFac" in agent.parameters for agent in self.agents]))
        self.assertTrue(
            all(
                [
                    self.agents[i].parameters["DiscFac"]
                    == dist.discretize(3, method="equiprobable").atoms[0, i]
                    for i in range(3)
                ]
            )
        )
        self.assertEqual(self.agents[0].parameters["AgentCount"], 1)


class test_agent_population(unittest.TestCase):
    def setUp(self):
        params = init_idiosyncratic_shocks.copy()
        params["CRRA"] = Uniform(2.0, 10)
        params["DiscFac"] = Uniform(0.9, 0.99)

        self.agent_pop = AgentPopulation(IndShockConsumerType, params)

    def test_distributed_params(self):
        self.assertTrue("CRRA" in self.agent_pop.distributed_params)
        self.assertTrue("DiscFac" in self.agent_pop.distributed_params)

    def test_approx_agents(self):
        self.agent_pop.approx_distributions(
            {
                "CRRA": {"N": 3, "method": "equiprobable"},
                "DiscFac": {"N": 4, "method": "equiprobable"},
            }
        )

        self.assertTrue("CRRA" in self.agent_pop.continuous_distributions)
        self.assertTrue("DiscFac" in self.agent_pop.continuous_distributions)
        self.assertTrue("CRRA" in self.agent_pop.discrete_distributions)
        self.assertTrue("DiscFac" in self.agent_pop.discrete_distributions)

        self.assertEqual(self.agent_pop.agent_type_count, 12)

    def test_create_agents(self):
        self.agent_pop.approx_distributions(
            {
                "CRRA": {"N": 3, "method": "equiprobable"},
                "DiscFac": {"N": 4, "method": "equiprobable"},
            }
        )
        self.agent_pop.create_distributed_agents()

        self.assertEqual(len(self.agent_pop.agents), 12)


@pytest.fixture
def sample_params():
    return Parameters(a=1, b=[2, 3, 4], c=5.0, d=[6.0, 7.0, 8.0], T_cycle=3)


class TestParameters:
    def test_initialization(self, sample_params):
        assert sample_params._length == 3
        assert sample_params._invariant_params == {"a", "c"}
        assert sample_params._varying_params == {"b", "d"}
        assert sample_params._parameters["T_cycle"] == 3

    def test_getitem(self, sample_params):
        assert sample_params["a"] == 1
        assert sample_params["b"] == [2, 3, 4]
        assert sample_params[0]["b"] == 2
        assert sample_params[1]["d"] == 7.0

    def test_setitem(self, sample_params):
        sample_params["e"] = 9
        assert sample_params["e"] == 9
        assert "e" in sample_params._invariant_params

        sample_params["f"] = [10, 11, 12]
        assert sample_params["f"] == [10, 11, 12]
        assert "f" in sample_params._varying_params

    def test_get(self, sample_params):
        assert sample_params.get("a") == 1
        assert sample_params.get("z", 100) == 100

    def test_set_many(self, sample_params):
        sample_params.set_many(g=13, h=[14, 15, 16])
        assert sample_params["g"] == 13
        assert sample_params["h"] == [14, 15, 16]

    def test_is_time_varying(self, sample_params):
        assert sample_params.is_time_varying("b") is True
        assert sample_params.is_time_varying("a") is False

    def test_to_dict(self, sample_params):
        params_dict = sample_params.to_dict()
        assert isinstance(params_dict, dict)
        assert params_dict["a"] == 1
        assert params_dict["b"] == [2, 3, 4]

    def test_update(self, sample_params):
        new_params = Parameters(a=100, e=200)
        sample_params.update(new_params)
        assert sample_params["a"] == 100
        assert sample_params["e"] == 200

    @pytest.mark.parametrize("invalid_key", [1, 2.0, None, []])
    def test_setitem_invalid_key(self, sample_params, invalid_key):
        with pytest.raises(ValueError):
            sample_params[invalid_key] = 42

    def test_setitem_invalid_value_length(self, sample_params):
        with pytest.raises(ValueError):
            sample_params["invalid"] = [1, 2]  # Should be length 1 or 3


class TestSolveFrom(unittest.TestCase):
    def shorten_params(self, params, length):
        par = deepcopy(params)
        for key in params.keys():
            if isinstance(params[key], list):
                par[key] = params[key][:length]
        par["T_cycle"] = length
        return par

    def setUp(self):
        # Create a 3-period parametrization of the IndShockConsumerType model
        self.params = init_idiosyncratic_shocks.copy()
        self.params.update(
            {
                "T_cycle": 3,
                "PermGroFac": [1.05, 1.10, 1.3],
                "LivPrb": [0.95, 0.9, 0.85],
                "TranShkStd": [0.1] * 3,
                "PermShkStd": [0.1] * 3,
                "Rfree": [1.02] * 3,
            }
        )

    def test_solve_from(self):
        # Create an IndShockConsumerType agent
        agent = IndShockConsumerType(**self.params)
        # Solve the model
        agent.solve()
        # Solution must have length 4 (includes terminal)
        assert len(agent.solution) == 4
        # Now create an agent with only the first 2 periods
        agent_2 = IndShockConsumerType(**self.shorten_params(self.params, 2))
        # Solve from the third solution of the previous agent
        agent_2.solve(from_solution=agent.solution[2])
        # The solutions (up to 2) must be the same
        for t, s2 in enumerate(agent_2.solution):
            self.assertEqual(s2.distance(agent.solution[t]), 0.0)


class TestAgentTypeHistory(unittest.TestCase):
    def setUp(self):
        """
        Set up a basic IndShockConsumerType agent for history testing.
        """
        self.params = init_idiosyncratic_shocks.copy()
        self.params.update({
            'AgentCount': 3,
            'T_sim': 2, # Simulate for 2 periods (0 and 1)
            'T_cycle': 1, # To keep it simple, lifecycle of 1 period for model solution
            'track_vars': ['mNrm', 'cNrm', 'pLvl', 't_age'] # Example tracked variables
        })
        self.agent = IndShockConsumerType(**self.params)
        # Solve the model so simulation can run
        self.agent.solve()
        # Initialize simulation (not running it yet, just setting up t_sim etc.)
        self.agent.initialize_sim()


    def test_get_history_df_structure_and_content(self):
        """
        Test the structure and basic content of the DataFrame returned by get_history_df.
        """
        # Simulate the agent
        self.agent.simulate()
        df = self.agent.get_history_df()

        # Test DataFrame structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(df.index, pd.MultiIndex)
        self.assertEqual(df.index.names, ['period', 'agent_id'])
        
        # Test columns - ensure 'period' from dict key is not a column
        expected_columns = [v for v in self.agent.track_vars if v != 'period']
        self.assertListEqual(sorted(list(df.columns)), sorted(expected_columns))

        # Test number of rows
        self.assertEqual(len(df), self.agent.T_sim * self.agent.AgentCount)

        # Test some content (very basic check, assumes simulation runs without error)
        # For period 0, agent 0, check if mNrm exists (it's a tracked var)
        self.assertTrue('mNrm' in df.columns)
        # Example: check a specific value if we had a very predictable model.
        # For IndShockConsumerType, exact values are complex to predict without deep model knowledge.
        # So, we mostly check presence and type.
        first_period_agent_0_mNrm = df.loc[(0, 0), 'mNrm']
        self.assertIsNotNone(first_period_agent_0_mNrm)
        self.assertIsInstance(first_period_agent_0_mNrm, (float, np.float64, np.float32))

        # Check t_age content
        # Period 0, all agents should have t_age = 0 (as per initialize_sim and first period data)
        # Period 1, all agents should have t_age = 1
        for p_idx in range(self.agent.T_sim):
            for a_idx in range(self.agent.AgentCount):
                # t_age in history reflects age at the *start* of the period `p_idx`'s simulation step
                # or rather, the age *during* period `p_idx`.
                # If t_sim is the period index in history:
                # history[0] (period 0) -> t_age is 0
                # history[1] (period 1) -> t_age is 1
                self.assertEqual(df.loc[(p_idx, a_idx), 't_age'], p_idx)


    def test_get_history_df_empty_history(self):
        """
        Test get_history_df when the simulation history is empty.
        """
        # Agent is initialized in setUp, but simulate() is not called here.
        # So, self.agent.history should be []
        df = self.agent.get_history_df()

        self.assertTrue(df.empty)
        self.assertIsInstance(df.index, pd.MultiIndex)
        self.assertEqual(df.index.names, ['period', 'agent_id'])
        # Columns should still be the track_vars
        expected_columns = [v for v in self.agent.track_vars if v != 'period']
        self.assertListEqual(sorted(list(df.columns)), sorted(expected_columns))

    def test_get_history_df_empty_track_vars(self):
        """
        Test get_history_df when agent.track_vars is empty.
        """
        self.agent.track_vars = []
        # Simulate to populate history (even if no vars are tracked, period/agent_id should exist)
        self.agent.simulate() 
        df = self.agent.get_history_df()

        self.assertIsInstance(df.index, pd.MultiIndex)
        self.assertEqual(df.index.names, ['period', 'agent_id'])
        self.assertTrue(df.columns.empty) # No columns other than index
        
        # Should still have rows for each period-agent combination
        self.assertEqual(len(df), self.agent.T_sim * self.agent.AgentCount)

    def test_get_history_df_agent_count_zero(self):
        """
        Test get_history_df when AgentCount is 0.
        """
        self.params_zero_agents = init_idiosyncratic_shocks.copy()
        self.params_zero_agents.update({
            'AgentCount': 0,
            'T_sim': 2,
            'T_cycle': 1,
            'track_vars': ['mNrm', 'cNrm']
        })
        agent_zero = IndShockConsumerType(**self.params_zero_agents)
        agent_zero.solve()
        agent_zero.initialize_sim()
        # agent_zero.simulate() # Simulate might have issues with 0 agents depending on implementation
        
        # If simulate() is called and history is populated with empty arrays for vars:
        # The current get_history_df might create 0 rows.
        # If simulate() itself errors or history is truly empty:
        df = agent_zero.get_history_df()

        self.assertTrue(df.empty)
        self.assertIsInstance(df.index, pd.MultiIndex)
        self.assertEqual(df.index.names, ['period', 'agent_id'])
        expected_columns = [v for v in agent_zero.track_vars if v != 'period']
        self.assertListEqual(sorted(list(df.columns)), sorted(expected_columns))

    def test_get_history_df_dtype_map_basic(self):
        """
        Test basic dtype casting using dtype_map.
        """
        self.agent.simulate()
        dtype_map = {'mNrm': 'float32', 't_age': 'int16', 'cNrm': np.float16}
        df = self.agent.get_history_df(dtype_map=dtype_map)

        self.assertEqual(df['mNrm'].dtype, np.dtype('float32'))
        self.assertEqual(df['t_age'].dtype, np.dtype('int16'))
        self.assertEqual(df['cNrm'].dtype, np.dtype('float16'))
        # pLvl was in track_vars but not in dtype_map, should retain its original type (likely float64)
        self.assertTrue(np.issubdtype(df['pLvl'].dtype, np.floating))


    def test_get_history_df_dtype_map_invalid_cast(self):
        """
        Test that a warning is issued for invalid type casting attempts
        and the original dtype (or numeric coerced) is retained.
        """
        self.agent.simulate() # mNrm will have float values

        # Attempt to cast float 'mNrm' to 'int8' which might fail or lose precision significantly
        # and is a good candidate for pd.to_numeric to coerce to NaN if it were a string.
        # Here, we expect astype to potentially raise an error caught by the warning system.
        original_mNrm_dtype = self.agent.get_history_df()['mNrm'].dtype

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always") # Capture all warnings
            df = self.agent.get_history_df(dtype_map={'mNrm': 'int8'})
            
            # Check if a UserWarning related to casting was issued
            self.assertTrue(any(issubclass(warn.category, UserWarning) and 
                                "Could not cast column 'mNrm'" in str(warn.message) for warn in w))
        
        # Dtype should ideally be the original, or what pd.to_numeric made it if that step ran.
        # Since 'mNrm' is already numeric, pd.to_numeric won't change its float status before astype.
        # If astype('int8') fails due to values (e.g. NaN, Inf, or too large/small),
        # it would raise an error, caught by get_history_df, and the column remains float.
        self.assertEqual(df['mNrm'].dtype, original_mNrm_dtype)


    def test_get_history_df_dtype_map_column_not_found(self):
        """
        Test that a warning is issued if a column in dtype_map is not found.
        """
        self.agent.simulate()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df = self.agent.get_history_df(dtype_map={'non_existent_col': 'float32'})
            
            self.assertTrue(any(issubclass(warn.category, UserWarning) and
                                "not found in DataFrame" in str(warn.message) for warn in w))
        # Ensure other columns are still present
        self.assertTrue('mNrm' in df.columns)
