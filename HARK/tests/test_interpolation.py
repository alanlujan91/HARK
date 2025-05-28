"""
This file implements unit tests for interpolation methods
"""

from HARK.interpolation import (
    IdentityFunction,
    LinearInterp,
    BilinearInterp,
    TrilinearInterp,
    QuadlinearInterp,
)
from HARK.interpolation import CubicHermiteInterp as CubicInterp
# Make HARK.interpolation available for HAS_NUMBA patching
from HARK import interpolation as HARK_interpolation

import numpy as np
import unittest
import timeit # For performance testing
# numpy already imported, no need for second import


class testsLinearInterp(unittest.TestCase):
    """tests for LinearInterp, currently tests for uneven length of
    x and y with user input as lists, arrays, arrays with column orientation
    """

    def setUp(self):
        self.x_list = [1, 2, 3]
        self.y_list = [3, 4]
        self.z_list = [3, 4, 5]
        self.x_array = np.array(self.x_list)
        self.y_array = np.array(self.y_list)
        self.z_array = np.array(self.z_list)
        self.x_array_t = self.x_array.reshape(len(self.x_array), 1)
        self.y_array_t = self.y_array.reshape(len(self.y_array), 1)
        self.z_array_t = self.z_array.reshape(len(self.z_array), 1)

    def test_uneven_length(self):
        self.assertRaises(ValueError, LinearInterp, self.x_list, self.y_list)
        self.assertRaises(ValueError, LinearInterp, self.x_array, self.y_array)
        self.assertRaises(ValueError, LinearInterp, self.x_array_t, self.y_array_t)

    def test_same_length(self):
        linear = LinearInterp(self.x_list, self.z_list)
        self.assertEqual(linear(1.5), 3.5)
        linear = LinearInterp(self.x_array, self.z_array)
        self.assertEqual(linear(1.5), 3.5)
        linear = LinearInterp(self.x_array_t, self.z_array_t)
        self.assertEqual(linear(1.5), 3.5)


class testsCubicInterp(unittest.TestCase):
    """tests for CubicInterp, currently tests for uneven length of
    x, y and derivative with user input as lists, arrays, arrays with column orientation
    """

    def setUp(self):
        self.x_list = [1, 2, 3]
        self.y_list = [1, 4]
        self.dydx_list = [2, 4, 6]
        self.z_list = [1, 4, 9]
        self.x_array = np.array(self.x_list)
        self.y_array = np.array(self.y_list)
        self.dydx_array = np.array(self.dydx_list)
        self.z_array = np.array(self.z_list)
        self.x_array_t = self.x_array.reshape(len(self.x_array), 1)
        self.y_array_t = self.y_array.reshape(len(self.y_array), 1)
        self.dydx_array_t = self.dydx_array.reshape(len(self.dydx_array), 1)
        self.z_array_t = self.z_array.reshape(len(self.z_array), 1)

    def test_uneven_length(self):
        self.assertRaises(
            ValueError, CubicInterp, self.x_list, self.y_list, self.dydx_list
        )
        self.assertRaises(
            ValueError, CubicInterp, self.x_array, self.y_array, self.dydx_array
        )
        self.assertRaises(
            ValueError, CubicInterp, self.x_array_t, self.y_array_t, self.dydx_array_t
        )

    def test_same_length(self):
        cube = CubicInterp(self.x_list, self.z_list, self.dydx_list)
        self.assertEqual(cube(1.5), 2.25)
        cube = CubicInterp(self.x_array, self.z_array, self.dydx_array)
        self.assertEqual(cube(1.5), 2.25)
        cube = CubicInterp(self.x_array_t, self.z_array_t, self.dydx_array_t)
        self.assertEqual(cube(1.5), 2.25)


class testsBilinearInterp(unittest.TestCase):
    """tests for BilinearInterp, currently tests for uneven length of
    x, y, f(x,y) with user input as arrays, arrays with column orientation
    """

    def setUp(self):
        self.f_array = np.array([[2, 4], [4, 6]])
        self.x_array = np.array([1, 3])
        self.y_array = np.array([1, 3])
        self.z_array = np.array([1, 2, 3])
        self.z_array_t = self.z_array.reshape(len(self.z_array), 1)
        self.y_array_t = self.y_array.reshape(len(self.y_array), 1)

    def test_uneven_length(self):
        self.assertRaises(
            ValueError, BilinearInterp, self.f_array, self.x_array, self.z_array
        )
        self.assertRaises(
            ValueError, BilinearInterp, self.f_array, self.x_array, self.z_array_t
        )

    def test_same_length(self):
        bilinear = BilinearInterp(self.f_array, self.x_array, self.y_array)
        self.assertEqual(bilinear(2, 2), 4.0)
        bilinear = BilinearInterp(self.f_array, self.x_array, self.y_array_t)
        self.assertEqual(bilinear(2, 2), 4.0)


class testsTrilinearInterp(unittest.TestCase):
    """tests for TrilinearInterp, currently tests for uneven length of
    x, y, z, f(x, y, z) with user input as arrays, arrays with column orientation
    """

    def setUp(self):
        self.f_array = np.array([[[3, 5], [5, 7]], [[5, 7], [7, 10]]])
        self.x_array = np.array([1, 3])
        self.y_array = np.array([1, 3])
        self.z_array = np.array([1, 3])
        self.fail_array = np.array([1, 2, 3])
        self.fail_array_t = self.z_array.reshape(len(self.z_array), 1)
        self.y_array_t = self.y_array.reshape(len(self.y_array), 1)

    def test_uneven_length(self):
        self.assertRaises(
            ValueError,
            TrilinearInterp,
            self.f_array,
            self.x_array,
            self.y_array,
            self.fail_array,
        )
        self.assertRaises(
            ValueError,
            TrilinearInterp,
            self.f_array,
            self.x_array,
            self.fail_array,
            self.fail_array_t,
        )

    def test_same_length(self):
        bilinear = TrilinearInterp(
            self.f_array, self.x_array, self.y_array, self.z_array
        )
        self.assertEqual(bilinear(1, 2, 2), 5.0)
        bilinear = TrilinearInterp(
            self.f_array, self.x_array, self.y_array_t, self.z_array
        )
        self.assertEqual(bilinear(1, 2, 2), 5.0)


class testsQuadlinearInterp(unittest.TestCase):
    """tests for TrilinearInterp, currently tests for uneven length of
    w, x, y, z, f(w, x, y, z) with user input as arrays, arrays with column orientation
    """

    def setUp(self):
        self.f_array = np.array(
            [
                [[[4, 6], [6, 8]], [[6, 8], [8, 11]]],
                [[[6, 8], [8, 10]], [[8, 10], [10, 13]]],
            ]
        )
        self.x_array = np.array([1, 3])
        self.y_array = np.array([1, 3])
        self.z_array = np.array([1, 3])
        self.w_array = np.array([1, 3])
        self.fail_array = np.array([1, 2, 3])
        self.fail_array_t = self.z_array.reshape(len(self.z_array), 1)
        self.y_array_t = self.y_array.reshape(len(self.y_array), 1)

    def test_uneven_length(self):
        self.assertRaises(
            ValueError,
            QuadlinearInterp,
            self.f_array,
            self.x_array,
            self.y_array,
            self.fail_array,
            self.w_array,
        )
        self.assertRaises(
            ValueError,
            QuadlinearInterp,
            self.f_array,
            self.x_array,
            self.fail_array,
            self.fail_array_t,
            self.w_array,
        )

    def test_same_length(self):
        bilinear = QuadlinearInterp(
            self.f_array, self.w_array, self.x_array, self.y_array, self.z_array
        )
        self.assertEqual(bilinear(1, 2, 1, 2), 6.0)
        bilinear = QuadlinearInterp(
            self.f_array, self.w_array, self.x_array, self.y_array_t, self.z_array
        )
        self.assertEqual(bilinear(1, 2, 1, 2), 6.0)


class test_IdentityFunction(unittest.TestCase):
    """
    Tests evaluation and derivatives of IdentityFunction class.
    """

    def setUp(self):
        self.IF1D = IdentityFunction()
        self.IF2Da = IdentityFunction(i_dim=0, n_dims=2)
        self.IF2Db = IdentityFunction(i_dim=1, n_dims=2)
        self.IF3Da = IdentityFunction(i_dim=0, n_dims=3)
        self.IF3Db = IdentityFunction(i_dim=2, n_dims=3)
        self.X = 3 * np.ones(100)
        self.Y = 4 * np.ones(100)
        self.Z = 5 * np.ones(100)
        self.zero = np.zeros(100)
        self.one = np.ones(100)

    def test_eval(self):
        assert np.all(self.X == self.IF1D(self.X))
        assert np.all(self.X == self.IF2Da(self.X, self.Y))
        assert np.all(self.Y == self.IF2Db(self.X, self.Y))
        assert np.all(self.X == self.IF3Da(self.X, self.Y, self.Z))
        assert np.all(self.Z == self.IF3Db(self.X, self.Y, self.Z))

    def test_der(self):
        assert np.all(self.one == self.IF1D.derivative(self.X))

        assert np.all(self.one == self.IF2Da.derivativeX(self.X, self.Y))
        assert np.all(self.zero == self.IF2Da.derivativeY(self.X, self.Y))

        assert np.all(self.zero == self.IF2Db.derivativeX(self.X, self.Y))
        assert np.all(self.one == self.IF2Db.derivativeY(self.X, self.Y))

        assert np.all(self.one == self.IF3Da.derivativeX(self.X, self.Y, self.Z))
        assert np.all(self.zero == self.IF3Da.derivativeY(self.X, self.Y, self.Z))
        assert np.all(self.zero == self.IF3Da.derivativeZ(self.X, self.Y, self.Z))

        assert np.all(self.zero == self.IF3Db.derivativeX(self.X, self.Y, self.Z))
        assert np.all(self.zero == self.IF3Db.derivativeY(self.X, self.Y, self.Z))
        assert np.all(self.one == self.IF3Db.derivativeZ(self.X, self.Y, self.Z))


class TestLinearInterpNumba(unittest.TestCase):
    def _run_comparative_test(self, x_grid, y_grid, test_points, **kwargs):
        """
        Helper function to compare Numba and Pure Python LinearInterp results.
        """
        # Ensure test_points is a numpy array
        test_points_np = np.asarray(test_points)

        # --- Numba version (or default if Numba not available/forced off) ---
        # Ensure Numba is enabled for this part if available globally
        original_has_numba_state = HARK_interpolation.HAS_NUMBA
        HARK_interpolation.HAS_NUMBA = True # Try to enable Numba path
        
        interpolator_numba = LinearInterp(x_grid, y_grid, **kwargs)
        y_numba = interpolator_numba(test_points_np)
        dydx_numba = interpolator_numba.derivative(test_points_np)
        
        # Restore global Numba state
        HARK_interpolation.HAS_NUMBA = original_has_numba_state

        # --- Pure Python version ---
        # Force HARK.interpolation.HAS_NUMBA to False to ensure pure Python path
        HARK_interpolation.HAS_NUMBA = False
        interpolator_py = LinearInterp(x_grid, y_grid, **kwargs)
        y_py = interpolator_py(test_points_np)
        dydx_py = interpolator_py.derivative(test_points_np)
        
        # Restore global Numba state
        HARK_interpolation.HAS_NUMBA = original_has_numba_state

        # Compare results
        np.testing.assert_allclose(y_numba, y_py, rtol=1e-7, atol=1e-9, err_msg="Mismatch in __call__ output")
        np.testing.assert_allclose(dydx_numba, dydx_py, rtol=1e-7, atol=1e-9, err_msg="Mismatch in derivative output")

    def test_numerical_equivalence_simple(self):
        x_grid = np.array([0.0, 1.0, 2.0, 3.0])
        y_grid = np.array([0.0, 0.5, 1.5, 1.0])
        test_points = [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.75, 3.0, 3.5]
        self._run_comparative_test(x_grid, y_grid, test_points)
        self._run_comparative_test(x_grid, y_grid, 0.75) # Scalar test

    def test_numerical_equivalence_lower_extrap(self):
        x_grid = np.array([1.0, 2.0, 3.0])
        y_grid = np.array([1.0, 0.5, 0.0])
        test_points = [0.5, 1.0, 1.5, 2.5, 3.0, 3.5]
        self._run_comparative_test(x_grid, y_grid, test_points, lower_extrap=True)

    def test_numerical_equivalence_decay_extrap(self):
        x_grid = np.array([0.0, 1.0, 2.0])
        y_grid = np.array([0.0, 1.0, 1.5])
        # For decay_extrap_A to be non-zero, level_diff must be non-zero.
        # level_diff = intercept_limit + slope_limit * x_list[-1] - y_list[-1]
        # slope_at_top = (1.5 - 1.0) / (2.0 - 1.0) = 0.5
        # Let slope_limit = 0.2. Let intercept_limit = 1.0.
        # level_diff = 1.0 + 0.2 * 2.0 - 1.5 = 1.0 + 0.4 - 1.5 = -0.1
        # slope_diff = 0.2 - 0.5 = -0.3
        # decay_extrap_A = -0.1
        # decay_extrap_B = -(-0.3) / (-0.1) = 0.3 / -0.1 = -3.0
        test_points = [-0.5, 0.5, 1.5, 2.0, 2.5, 3.0]
        self._run_comparative_test(x_grid, y_grid, test_points, 
                                   intercept_limit=1.0, slope_limit=0.2, lower_extrap=True)
        
        # Test case where decay_extrap would be false due to slope_limit == slope_at_top
        self._run_comparative_test(x_grid, y_grid, test_points,
                                   intercept_limit=y_grid[-1] - 0.5 * x_grid[-1], # intercept that matches slope_at_top
                                   slope_limit=0.5, lower_extrap=True)


    def test_numerical_equivalence_single_segment(self):
        x_grid = np.array([1.0, 2.0])
        y_grid = np.array([5.0, 10.0])
        test_points = [0.5, 1.0, 1.5, 2.0, 2.5]
        self._run_comparative_test(x_grid, y_grid, test_points, lower_extrap=True)

    def test_numerical_equivalence_single_point_grid(self):
        x_grid = np.array([1.0])
        y_grid = np.array([5.0])
        test_points = [0.5, 1.0, 1.5] # Should be constant function
        self._run_comparative_test(x_grid, y_grid, test_points, lower_extrap=True)

    def test_numerical_equivalence_empty_grid(self):
        x_grid = np.array([])
        y_grid = np.array([])
        test_points = [0.5, 1.0, 1.5] # Should result in NaNs
        self._run_comparative_test(x_grid, y_grid, test_points, lower_extrap=True)

    @unittest.skipIf(not HARK_interpolation.HAS_NUMBA, "Numba not available, skipping performance test.")
    def test_linear_interp_numba_performance(self):
        print("\nRunning LinearInterp Numba Performance Comparison (higher is better for Numba speedup):")
        x_grid = np.linspace(0, 100, 1000)
        y_grid = np.sin(x_grid/10.0) # Some non-trivial function
        x_test_points = np.random.rand(1000000) * 100
        
        number_of_reps = 10

        # Time Numba path (if Numba is available)
        # Ensure HARK_interpolation.HAS_NUMBA is its true detected state
        original_has_numba_state = HARK_interpolation.HAS_NUMBA
        HARK_interpolation.HAS_NUMBA = True # Force to true for this measurement if available globally
        
        interpolator_numba = LinearInterp(x_grid, y_grid)
        
        # Make sure Numba kernel is compiled before timing critical section
        if HARK_interpolation.HAS_NUMBA and HARK_interpolation._numba_linear_interp_eval_or_der_jitted is not None:
            interpolator_numba(np.array([x_grid[0],x_grid[-1]])) # Pre-compile
            
            numba_time = timeit.timeit(lambda: interpolator_numba(x_test_points), number=number_of_reps)
            print(f"  Numba time ({number_of_reps} reps): {numba_time:.4f}s")
        else:
            numba_time = float('inf') # Should not happen if skipIf works
            print("  Numba path not executed (should have been skipped).")

        HARK_interpolation.HAS_NUMBA = original_has_numba_state # Restore state

        # Time Pure Python path
        HARK_interpolation.HAS_NUMBA = False 
        interpolator_py = LinearInterp(x_grid, y_grid) 
        # Pre-call to mimic any first-call overhead if relevant, though less so for Python
        interpolator_py(np.array([x_grid[0], x_grid[-1]]))
        
        python_time = timeit.timeit(lambda: interpolator_py(x_test_points), number=number_of_reps)
        print(f"  Python time ({number_of_reps} reps): {python_time:.4f}s")
        HARK_interpolation.HAS_NUMBA = original_has_numba_state # Restore

        if HARK_interpolation.HAS_NUMBA and HARK_interpolation._numba_linear_interp_eval_or_der_jitted is not None:
            print(f"  Speedup (Python/Numba): {python_time/numba_time:.2f}x")
            # Allowing Numba to be slightly slower in some edge cases or due to overhead,
            # but expecting it to be faster for large arrays.
            self.assertTrue(numba_time < python_time * 1.2, "Numba version was significantly slower than Python version.")
        else:
            # This part of the test effectively means Numba is not being tested for performance.
            # The skipIf should prevent this from being a failure.
            pass
