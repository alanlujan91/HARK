import unittest

import pandas as pd # Add import for pd.Series
import numpy as np # Add import for np.integer if needed for type check

from HARK.Calibration import load_SCF_wealth_weights
from HARK.Calibration.cpi.us.CPITools import cpi_deflator
from HARK.Calibration.SCF.WealthIncomeDist.SCFDistTools import (
    income_wealth_dists_from_scf,
    parse_scf_distr_stats, # Import the function to be tested
)


class test_load_SCF_wealth_weights(unittest.TestCase):
    def setUp(self):
        self.SCF_wealth, self.SCF_weights = load_SCF_wealth_weights()

    def test_shape(self):
        self.assertEqual(self.SCF_wealth.shape, (3553,))
        self.assertEqual(self.SCF_weights.shape, (3553,))


# %% US CPI tests
class test_cpi_deflators(unittest.TestCase):
    def test_month_deflators(self):
        # Same year test
        defl_same_year = cpi_deflator(2000, 2000, "SEP")
        self.assertEqual(defl_same_year[0], 1.0)

        # Different year test
        defl_diff_year = cpi_deflator(1998, 2019, "SEP")
        self.assertAlmostEqual(defl_diff_year[0], 1.57279534)

    def test_avg_deflators(self):
        # Same year test
        defl_same_year = cpi_deflator(2000, 2000)
        self.assertEqual(defl_same_year[0], 1.0)

        # Different year test
        defl_diff_year = cpi_deflator(1998, 2019)
        self.assertAlmostEqual(defl_diff_year[0], 1.57202505)


# %% Tests for Survey of Consumer finances initial distributions
class test_SCF_dists(unittest.TestCase):
    def setUp(self):
        self.BaseYear = 1992

    def test_at_21(self):
        # Get stats for various groups and test them
        NoHS = income_wealth_dists_from_scf(
            self.BaseYear, age=21, education="NoHS", wave=1995
        )
        self.assertAlmostEqual(NoHS["aNrmInitMean"], -1.0611984728537684)
        self.assertAlmostEqual(NoHS["aNrmInitStd"], 1.475816500147777)
        self.assertAlmostEqual(NoHS["pLvlInitMean"], 2.5413398571226233)
        self.assertAlmostEqual(NoHS["pLvlInitStd"], 0.7264931123240703)

        HS = income_wealth_dists_from_scf(
            self.BaseYear, age=21, education="HS", wave=2013
        )
        self.assertAlmostEqual(HS["aNrmInitMean"], -1.0812342937817578)
        self.assertAlmostEqual(HS["aNrmInitStd"], 1.7526704743231725)
        self.assertAlmostEqual(HS["pLvlInitMean"], 2.806605268756435)
        self.assertAlmostEqual(HS["pLvlInitStd"], 0.6736467457859727)

        Coll = income_wealth_dists_from_scf(
            self.BaseYear, age=21, education="College", wave=2019
        )
        self.assertAlmostEqual(Coll["aNrmInitMean"], -0.6837248150760165)
        self.assertAlmostEqual(Coll["aNrmInitStd"], 0.8813676761170798)
        self.assertAlmostEqual(Coll["pLvlInitMean"], 3.2790838587291127)
        self.assertAlmostEqual(Coll["pLvlInitStd"], 0.746362502979793)

    def test_at_60(self):
        # Get stats for various groups and test them
        NoHS = income_wealth_dists_from_scf(
            self.BaseYear, age=60, education="NoHS", wave=1995
        )
        self.assertAlmostEqual(NoHS["aNrmInitMean"], 0.1931578281432479)
        self.assertAlmostEqual(NoHS["aNrmInitStd"], 1.6593916577375334)
        self.assertAlmostEqual(NoHS["pLvlInitMean"], 3.3763953392998705)
        self.assertAlmostEqual(NoHS["pLvlInitStd"], 0.61810580085094993)

        HS = income_wealth_dists_from_scf(
            self.BaseYear, age=60, education="HS", wave=2013
        )
        self.assertAlmostEqual(HS["aNrmInitMean"], 0.6300862955841334)
        self.assertAlmostEqual(HS["aNrmInitStd"], 1.7253736778036055)
        self.assertAlmostEqual(HS["pLvlInitMean"], 3.462790681398899)
        self.assertAlmostEqual(HS["pLvlInitStd"], 0.8179188962937205)

        Coll = income_wealth_dists_from_scf(
            self.BaseYear, age=60, education="College", wave=2019
        )
        self.assertAlmostEqual(Coll["aNrmInitMean"], 1.643936802283761)
        self.assertAlmostEqual(Coll["aNrmInitStd"], 1.2685135110865389)
        self.assertAlmostEqual(Coll["pLvlInitMean"], 4.278905678818748)
        self.assertAlmostEqual(Coll["pLvlInitStd"], 1.0776403992280614)

    def test_parse_scf_distr_stats_output(self):
        # Test that parse_scf_distr_stats returns a pandas Series with correct info
        # Using a known data point from WealthIncomeStats.csv
        # For (education='College', wave_str='2019', age_bracket='(55,60]'):
        # BASE_YR should be 2019
        # PermIncome_median should be 101.3600006
        
        result_series = parse_scf_distr_stats(age=60, education="College", wave=2019)
        
        # Check type
        self.assertIsInstance(result_series, pd.Series)
        
        # Check name of series (index of the original table)
        # Expected name tuple: (education, wave_str, age_bracket)
        # age = 60 -> u_bound = 60, l_bound = 55, age_bracket = "(55,60]"
        expected_name = ("College", "2019", "(55,60]")
        self.assertEqual(result_series.name, expected_name)
        
        # Check BASE_YR type and value
        self.assertIsInstance(result_series["BASE_YR"], (int, np.integer)) # np.integer for numpy's int types
        self.assertEqual(result_series["BASE_YR"], 2019)
        
        # Check a float value
        # Taking PermIncome_median as an example. Value from CSV for this group.
        # Make sure to use a value that is less likely to change due to floating point issues in future pandas versions.
        self.assertAlmostEqual(result_series["PermIncome_median"], 101.3600006, places=5)
        
        # Check another value, e.g., a share
        self.assertAlmostEqual(result_series["PermIncome_shares_percentile_90"], 0.294, places=3)

        # Test with "All" categories
        result_all_educ = parse_scf_distr_stats(age=40, education=None, wave=2019)
        self.assertIsInstance(result_all_educ, pd.Series)
        expected_name_all_educ = ("All", "2019", "(35,40]")
        self.assertEqual(result_all_educ.name, expected_name_all_educ)
        self.assertEqual(result_all_educ["BASE_YR"], 2019)

        result_all_age = parse_scf_distr_stats(age=None, education="HS", wave=2007)
        self.assertIsInstance(result_all_age, pd.Series)
        expected_name_all_age = ("HS", "2007", "All")
        self.assertEqual(result_all_age.name, expected_name_all_age)
        self.assertEqual(result_all_age["BASE_YR"], 2007)

        result_all_wave = parse_scf_distr_stats(age=30, education="NoHS", wave=None) # Wave "All"
        self.assertIsInstance(result_all_wave, pd.Series)
        expected_name_all_wave = ("NoHS", "All", "(25,30]")
        self.assertEqual(result_all_wave.name, expected_name_all_wave)
        # BASE_YR for "All" waves is not uniquely defined in the CSV structure provided.
        # The original CSV has BASE_YR varying by actual wave year.
        # If wave is "All", parse_scf_distr_stats would average or otherwise aggregate.
        # The current code structure for parse_scf_distr_stats using index_col and .loc
        # implies that "All" for YEAR (wave) must be an explicit entry in the CSV.
        # Let's assume for "All" waves, a representative BASE_YR or NaN might be present.
        # The test data `WealthIncomeStats.csv` does have rows where YEAR is "All".
        # For ("NoHS", "All", "(25,30]"), BASE_YR is 2007 in the CSV.
        self.assertEqual(result_all_wave["BASE_YR"], 2007)
