"""
High-level functions and classes for solving a wide variety of economic models.
The "core" of HARK is a framework for "microeconomic" and "macroeconomic"
models.  A micro model concerns the dynamic optimization problem for some type
of agents, where agents take the inputs to their problem as exogenous.  A macro
model adds an additional layer, endogenizing some of the inputs to the micro
problem by finding a general equilibrium dynamic rule.
"""

# Set logging and define basic functions
import inspect
import logging
import sys
from collections import namedtuple
from copy import copy, deepcopy
from dataclasses import dataclass, field
from time import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union
from warnings import warn

import numpy as np
import pandas as pd
from xarray import DataArray

from HARK.distributions import (
    Distribution,
    IndexDistribution,
    TimeVaryingDiscreteDistribution,
    combine_indep_dstns,
)
from HARK.parallel import multi_thread_commands, multi_thread_commands_fake
from HARK.utilities import NullFunc, get_arg_names

logging.basicConfig(format="%(message)s")
_log = logging.getLogger("HARK")
_log.setLevel(logging.ERROR)


def disable_logging():
    _log.disabled = True


def enable_logging():
    _log.disabled = False


def warnings():
    _log.setLevel(logging.WARNING)


def quiet():
    _log.setLevel(logging.ERROR)


def verbose():
    _log.setLevel(logging.INFO)


def set_verbosity_level(level):
    _log.setLevel(level)


class Parameters:
    """
    A smart container for model parameters that handles age-varying dynamics.

    This class stores parameters as an internal dictionary and manages their
    age-varying properties, providing both attribute-style and dictionary-style
    access. It is designed to handle the time-varying dynamics of parameters
    in economic models.

    Attributes
    ----------
    _length : int
        The terminal age of the agents in the model.
    _invariant_params : Set[str]
        A set of parameter names that are invariant over time.
    _varying_params : Set[str]
        A set of parameter names that vary over time.
    _parameters : Dict[str, Any]
        The internal dictionary storing all parameters.
    """

    __slots__ = ("_length", "_invariant_params", "_varying_params", "_parameters")

    def __init__(self, **parameters: Any) -> None:
        """
        Initialize a Parameters object and parse the age-varying dynamics of parameters.

        Parameters
        ----------
        **parameters : Any
            Any number of parameters in the form key=value.
        """
        self._length: int = parameters.pop("T_cycle", 1)
        self._invariant_params: Set[str] = set()
        self._varying_params: Set[str] = set()
        self._parameters: Dict[str, Any] = {"T_cycle": self._length}

        for key, value in parameters.items():
            self[key] = value

    def __getitem__(self, item_or_key: Union[int, str]) -> Union["Parameters", Any]:
        """
        Access parameters by age index or parameter name.

        If item_or_key is an integer, returns a Parameters object with the parameters
        that apply to that age. This includes all invariant parameters and the
        `item_or_key`th element of all age-varying parameters. If item_or_key is a
        string, it returns the value of the parameter with that name.

        Parameters
        ----------
        item_or_key : Union[int, str]
            Age index or parameter name.

        Returns
        -------
        Union[Parameters, Any]
            A new Parameters object for the specified age, or the value of the
            specified parameter.

        Raises
        ------
        ValueError:
            If the age index is out of bounds.
        KeyError:
            If the parameter name is not found.
        TypeError:
            If the key is neither an integer nor a string.
        """
        if isinstance(item_or_key, int):
            if item_or_key >= self._length:
                raise ValueError(
                    f"Age {item_or_key} is out of bounds (max: {self._length - 1})."
                )

            params = {key: self._parameters[key] for key in self._invariant_params}
            params.update(
                {
                    key: (
                        self._parameters[key][item_or_key]
                        if isinstance(self._parameters[key], (list, tuple, np.ndarray))
                        else self._parameters[key]
                    )
                    for key in self._varying_params
                }
            )
            return Parameters(**params)
        elif isinstance(item_or_key, str):
            return self._parameters[item_or_key]
        else:
            raise TypeError("Key must be an integer (age) or string (parameter name).")

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set parameter values, automatically inferring time variance.

        If the parameter is a scalar, numpy array, boolean, distribution, callable
        or None, it is assumed to be invariant over time. If the parameter is a
        list or tuple, it is assumed to be varying over time. If the parameter
        is a list or tuple of length greater than 1, the length of the list or
        tuple must match the `_length` attribute of the Parameters object.

        Parameters
        ----------
        key : str
            Name of the parameter.
        value : Any
            Value of the parameter.

        Raises
        ------
        ValueError:
            If the parameter name is not a string or if the value type is unsupported.
            If the parameter value is inconsistent with the current model length.
        """
        if not isinstance(key, str):
            raise ValueError(f"Parameter name must be a string, got {type(key)}")

        if isinstance(
            value, (int, float, np.ndarray, type(None), Distribution, bool, Callable)
        ):
            self._invariant_params.add(key)
            self._varying_params.discard(key)
        elif isinstance(value, (list, tuple)):
            if len(value) == 1:
                value = value[0]
                self._invariant_params.add(key)
                self._varying_params.discard(key)
            elif self._length is None or self._length == 1:
                self._length = len(value)
                self._varying_params.add(key)
                self._invariant_params.discard(key)
            elif len(value) == self._length:
                self._varying_params.add(key)
                self._invariant_params.discard(key)
            else:
                raise ValueError(
                    f"Parameter {key} must have length 1 or {self._length}, not {len(value)}"
                )
        else:
            raise ValueError(f"Unsupported type for parameter {key}: {type(value)}")

        self._parameters[key] = value

    def __iter__(self) -> Iterator[str]:
        """Allow iteration over parameter names."""
        return iter(self._parameters)

    def __len__(self) -> int:
        """Return the number of parameters."""
        return len(self._parameters)

    def keys(self) -> Iterator[str]:
        """Return a view of parameter names."""
        return self._parameters.keys()

    def values(self) -> Iterator[Any]:
        """Return a view of parameter values."""
        return self._parameters.values()

    def items(self) -> Iterator[Tuple[str, Any]]:
        """Return a view of parameter (name, value) pairs."""
        return self._parameters.items()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert parameters to a plain dictionary.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing all parameters.
        """
        return dict(self._parameters)

    def to_namedtuple(self) -> namedtuple:
        """
        Convert parameters to a namedtuple.

        Returns
        -------
        namedtuple
            A namedtuple containing all parameters.
        """
        return namedtuple("Parameters", self.keys())(**self.to_dict())

    def update(self, other: Union["Parameters", Dict[str, Any]]) -> None:
        """
        Update parameters from another Parameters object or dictionary.

        Parameters
        ----------
        other : Union[Parameters, Dict[str, Any]]
            The source of parameters to update from.

        Raises
        ------
        TypeError
            If the input is neither a Parameters object nor a dictionary.
        """
        if isinstance(other, Parameters):
            for key, value in other._parameters.items():
                self[key] = value
        elif isinstance(other, dict):
            for key, value in other.items():
                self[key] = value
        else:
            raise TypeError(
                "Update source must be a Parameters object or a dictionary."
            )

    def __repr__(self) -> str:
        """Return a detailed string representation of the Parameters object."""
        return (
            f"Parameters(_length={self._length}, "
            f"_invariant_params={self._invariant_params}, "
            f"_varying_params={self._varying_params}, "
            f"_parameters={self._parameters})"
        )

    def __str__(self) -> str:
        """Return a simple string representation of the Parameters object."""
        return f"Parameters({str(self._parameters)})"

    def __getattr__(self, name: str) -> Any:
        """
        Allow attribute-style access to parameters.

        Parameters
        ----------
        name : str
            Name of the parameter to access.

        Returns
        -------
        Any
            The value of the specified parameter.

        Raises
        ------
        AttributeError:
            If the parameter name is not found.
        """
        if name.startswith("_"):
            return super().__getattribute__(name)
        try:
            return self._parameters[name]
        except KeyError:
            raise AttributeError(f"'Parameters' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Allow attribute-style setting of parameters.

        Parameters
        ----------
        name : str
            Name of the parameter to set.
        value : Any
            Value to set for the parameter.
        """
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self[name] = value

    def __contains__(self, item: str) -> bool:
        """Check if a parameter exists in the Parameters object."""
        return item in self._parameters

    def copy(self) -> "Parameters":
        """
        Create a deep copy of the Parameters object.

        Returns
        -------
        Parameters
            A new Parameters object with the same contents.
        """
        return deepcopy(self)

    def add_to_time_vary(self, *params: str) -> None:
        """
        Adds any number of parameters to the time-varying set.

        Parameters
        ----------
        *params : str
            Any number of strings naming parameters to be added to time_vary.
        """
        for param in params:
            if param in self._parameters:
                self._varying_params.add(param)
                self._invariant_params.discard(param)
            else:
                warn(
                    f"Parameter '{param}' does not exist and cannot be added to time_vary."
                )

    def add_to_time_inv(self, *params: str) -> None:
        """
        Adds any number of parameters to the time-invariant set.

        Parameters
        ----------
        *params : str
            Any number of strings naming parameters to be added to time_inv.
        """
        for param in params:
            if param in self._parameters:
                self._invariant_params.add(param)
                self._varying_params.discard(param)
            else:
                warn(
                    f"Parameter '{param}' does not exist and cannot be added to time_inv."
                )

    def del_from_time_vary(self, *params: str) -> None:
        """
        Removes any number of parameters from the time-varying set.

        Parameters
        ----------
        *params : str
            Any number of strings naming parameters to be removed from time_vary.
        """
        for param in params:
            self._varying_params.discard(param)

    def del_from_time_inv(self, *params: str) -> None:
        """
        Removes any number of parameters from the time-invariant set.

        Parameters
        ----------
        *params : str
            Any number of strings naming parameters to be removed from time_inv.
        """
        for param in params:
            self._invariant_params.discard(param)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a parameter value, returning a default if not found.

        Parameters
        ----------
        key : str
            The parameter name.
        default : Any, optional
            The default value to return if the key is not found.

        Returns
        -------
        Any
            The parameter value or the default.
        """
        return self._parameters.get(key, default)

    def set_many(self, **kwargs: Any) -> None:
        """
        Set multiple parameters at once.

        Parameters
        ----------
        **kwargs : Keyword arguments representing parameter names and values.
        """
        for key, value in kwargs.items():
            self[key] = value

    def is_time_varying(self, key: str) -> bool:
        """
        Check if a parameter is time-varying.

        Parameters
        ----------
        key : str
            The parameter name.

        Returns
        -------
        bool
            True if the parameter is time-varying, False otherwise.
        """
        return key in self._varying_params


class Model:
    """
    A class with special handling of parameters assignment.
    """

    def __init__(self):
        if not hasattr(self, "parameters"):
            self.parameters = {}
        if not hasattr(self, "constructors"):
            self.constructors = {}

    def assign_parameters(self, **kwds):
        """
        Assign an arbitrary number of attributes to this agent.

        Parameters
        ----------
        **kwds : keyword arguments
            Any number of keyword arguments of the form key=value.
            Each value will be assigned to the attribute named in self.

        Returns
        -------
            None
        """
        self.parameters.update(kwds)
        for key in kwds:
            setattr(self, key, kwds[key])

    def get_parameter(self, name):
        """
        Returns a parameter of this model

        Parameters
        ----------
        name : str
            The name of the parameter to get

        Returns
        -------
        value : The value of the parameter
        """
        return self.parameters[name]

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.parameters == other.parameters

        return NotImplemented

    def __str__(self):
        type_ = type(self)
        module = type_.__module__
        qualname = type_.__qualname__

        s = f"<{module}.{qualname} object at {hex(id(self))}.\n"
        s += "Parameters:"

        for p in self.parameters:
            s += f"\n{p}: {self.parameters[p]}"

        s += ">"
        return s

    def describe(self):
        return self.__str__()

    def del_param(self, param_name):
        """
        Deletes a parameter from this instance, removing it both from the object's
        namespace (if it's there) and the parameters dictionary (likewise).

        Parameters
        ----------
        param_name : str
            A string naming a parameter or data to be deleted from this instance.
            Removes information from self.parameters dictionary and own namespace.

        Returns
        -------
        None
        """
        if param_name in self.parameters:
            del self.parameters[param_name]
        if hasattr(self, param_name):
            delattr(self, param_name)

    def construct(self, *args, force=False):
        """
        Top-level method for building constructed inputs. If called without any
        inputs, construct builds each of the objects named in the keys of the
        constructors dictionary; it draws inputs for the constructors from the
        parameters dictionary and adds its results to the same. If passed one or
        more strings as arguments, the method builds only the named keys. The
        method will do multiple "passes" over the requested keys, as some cons-
        tructors require inputs built by other constructors. If any requested
        constructors failed to build due to missing data, those keys (and the
        missing data) will be named in self._missing_key_data. Other errors are
        recorded in the dictionary attribute _constructor_errors.

        Parameters
        ----------
        *args : str, optional
            Keys of self.constructors that are requested to be constructed.
            If no arguments are passed, *all* elements of the dictionary are implied.
        force : bool, optional
            When True, the method will force its way past any errors, including
            missing constructors, missing arguments for constructors, and errors
            raised during execution of constructors. Information about all such
            errors is stored in the dictionary attributes described above. When
            False (default), any errors or exception will be raised.

        Returns
        -------
        None
        """
        # Set up the requested work
        if len(args) > 0:
            keys = args
        else:
            keys = list(self.constructors.keys())
        N_keys = len(keys)
        keys_complete = np.zeros(N_keys, dtype=bool)
        if N_keys == 0:
            return  # Do nothing if there are no constructed objects

        # Get the dictionary of constructor errors
        if not hasattr(self, "_constructor_errors"):
            self._constructor_errors = {}
        errors = self._constructor_errors

        # As long as the work isn't complete and we made some progress on the last
        # pass, repeatedly perform passes of trying to construct objects
        any_keys_incomplete = np.any(np.logical_not(keys_complete))
        go = any_keys_incomplete
        while go:
            anything_accomplished_this_pass = False  # Nothing done yet!
            missing_key_data = []  # Keep this up-to-date on each pass

            # Loop over keys to be constructed
            for i in range(N_keys):
                if keys_complete[i]:
                    continue  # This key has already been built

                # Get this key and its constructor function
                key = keys[i]
                try:
                    constructor = self.constructors[key]
                except Exception as not_found:
                    errors[key] = "No constructor found for " + str(not_found)
                    self.del_param(key)
                    if force:
                        continue
                    else:
                        raise ValueError("No constructor found for " + key) from None

                # If this constructor is None, do nothing and mark it as completed
                if constructor is None:
                    keys_complete[i] = True
                    anything_accomplished_this_pass = True  # We did something!
                    continue

                # Get the names of arguments for this constructor and try to gather them
                args_needed = get_arg_names(constructor)
                has_no_default = {
                    k: v.default is inspect.Parameter.empty
                    for k, v in inspect.signature(constructor).parameters.items()
                }
                temp_dict = {}
                any_missing = False
                missing_args = []
                for j in range(len(args_needed)):
                    this_arg = args_needed[j]
                    if hasattr(self, this_arg):
                        temp_dict[this_arg] = getattr(self, this_arg)
                    else:
                        try:
                            temp_dict[this_arg] = self.parameters[this_arg]
                        except:
                            if has_no_default[this_arg]:
                                # Record missing key-data pair
                                any_missing = True
                                missing_key_data.append((key, this_arg))
                                missing_args.append(this_arg)

                # If all of the required data was found, run the constructor and
                # store the result in parameters (and on self)
                if not any_missing:
                    try:
                        temp = constructor(**temp_dict)
                    except Exception as problem:
                        errors[key] = str(type(problem)) + ": " + str(problem)
                        self.del_param(key)
                        if force:
                            continue
                        else:
                            raise
                    setattr(self, key, temp)
                    self.parameters[key] = temp
                    if key in errors:
                        del errors[key]
                    keys_complete[i] = True
                    anything_accomplished_this_pass = True  # We did something!
                else:
                    msg = "Missing required arguments:"
                    for arg in missing_args:
                        msg += " " + arg + ","
                    msg = msg[:-1]
                    errors[key] = msg
                    self.del_param(key)
                    # Never raise exceptions here, as the arguments might be filled in later

            # Check whether another pass should be performed
            any_keys_incomplete = np.any(np.logical_not(keys_complete))
            go = any_keys_incomplete and anything_accomplished_this_pass

        # Store missing key-data pairs and exit
        self._missing_key_data = missing_key_data
        if any_keys_incomplete:
            msg = "Did not construct these objects:"
            for i in range(N_keys):
                if keys_complete[i]:
                    continue
                msg += " " + keys[i] + ","
            msg = msg[:-1]
            if not force:
                raise ValueError(msg)
        return

    def describe_constructors(self, *args):
        """
        Prints to screen a string describing this instance's constructed objects,
        including their names, the function that constructs them, the names of
        those functions inputs, and whether those inputs are present.

        Parameters
        ----------
        *args : str, optional
            Optional list of strings naming constructed inputs to be described.
            If none are passed, all constructors are described.

        Returns
        -------
        None
        """
        if len(args) > 0:
            keys = args
        else:
            keys = list(self.constructors.keys())
        yes = "\u2713"
        no = "X"
        maybe = "*"
        noyes = [no, yes]

        out = ""
        for key in keys:
            has_val = hasattr(self, key) or (key in self.parameters)

            # Get the constructor function if possible
            try:
                constructor = self.constructors[key]
                out += (
                    noyes[int(has_val)]
                    + " "
                    + key
                    + " : "
                    + constructor.__name__
                    + "\n"
                )
            except:
                out += noyes[int(has_val)] + " " + key + " : NO CONSTRUCTOR FOUND\n"
                continue

            # Get constructor argument names
            arg_names = get_arg_names(constructor)
            has_no_default = {
                k: v.default is inspect.Parameter.empty
                for k, v in inspect.signature(constructor).parameters.items()
            }

            # Check whether each argument existd
            for j in range(len(arg_names)):
                this_arg = arg_names[j]
                if hasattr(self, this_arg) or this_arg in self.parameters:
                    symb = yes
                elif not has_no_default[this_arg]:
                    symb = maybe
                else:
                    symb = no
                out += "    " + symb + " " + this_arg + "\n"

        # Print the string to screen
        print(out)
        return

    # This is a "synonym" method so that old calls to update() still work
    def update(self, *args):
        self.construct(*args)


class AgentType(Model):
    """
    A superclass for economic agents in the HARK framework. Each model should
    specify its own subclass of AgentType, inheriting its methods and overwriting
    as necessary.  Critically, every subclass of AgentType should define class-
    specific static values of the attributes time_vary and time_inv as lists of
    strings.  Each element of time_vary is the name of a field in AgentSubType
    that varies over time in the model.  Each element of time_inv is the name of
    a field in AgentSubType that is constant over time in the model.

    Parameters
    ----------
    solution_terminal : Solution
        A representation of the solution to the terminal period problem of
        this AgentType instance, or an initial guess of the solution if this
        is an infinite horizon problem.
    cycles : int
        The number of times the sequence of periods is experienced by this
        AgentType in their "lifetime".  cycles=1 corresponds to a lifecycle
        model, with a certain sequence of one period problems experienced
        once before terminating.  cycles=0 corresponds to an infinite horizon
        model, with a sequence of one period problems repeating indefinitely.
    pseudo_terminal : bool
        Indicates whether solution_terminal isn't actually part of the
        solution to the problem (as a known solution to the terminal period
        problem), but instead represents a "scrap value"-style termination.
        When True, solution_terminal is not included in the solution; when
        False, solution_terminal is the last element of the solution.
    tolerance : float
        Maximum acceptable "distance" between successive solutions to the
        one period problem in an infinite horizon (cycles=0) model in order
        for the solution to be considered as having "converged".  Inoperative
        when cycles>0.
    verbose : int
        Level of output to be displayed by this instance, default is 1.
    quiet : bool
        Indicator for whether this instance should operate "quietly", default False.
    seed : int
        A seed for this instance's random number generator.
    construct : bool
        Indicator for whether this instance's construct() method should be run
        when initialized (default True). When False, an instance of the class
        can be created even if not all of its attributes can be constructed.

    Attributes
    ----------
    AgentCount : int
        The number of agents of this type to use in simulation.

    state_vars : list of string
        The string labels for this AgentType's model state variables.
    """

    time_vary_ = []
    time_inv_ = []
    shock_vars_ = []
    state_vars = []
    poststate_vars = []
    default_ = {"params": {}, "solver": NullFunc()}

    def __init__(
        self,
        solution_terminal=None,
        pseudo_terminal=True,
        tolerance=0.000001,
        verbose=1,
        quiet=False,
        seed=0,
        construct=True,
        **kwds,
    ):
        super().__init__()
        params = deepcopy(self.default_["params"])
        params.update(kwds)

        if solution_terminal is None:
            solution_terminal = NullFunc()

        self.solve_one_period = self.default_["solver"]  # NOQA
        self.solution_terminal = solution_terminal  # NOQA
        self.pseudo_terminal = pseudo_terminal  # NOQA
        self.tolerance = tolerance  # NOQA
        self.verbose = verbose
        self.quiet = quiet
        set_verbosity_level((4 - verbose) * 10)
        self.seed = seed  # NOQA
        self.track_vars = []  # NOQA
        self.state_now = {sv: None for sv in self.state_vars}
        self.state_prev = self.state_now.copy()
        self.controls = {}
        self.shocks = {}
        self.read_shocks = False  # NOQA
        self.shock_history = []  # MODIFIED: Changed to list
        self.newborn_init_history = []  # MODIFIED: Changed to list
        self.history = []  # MODIFIED: Changed to list
        self.assign_parameters(**params)  # NOQA
        self.reset_rng()  # NOQA
        self.bilt = {}
        if construct:
            self.construct()

        # Add instance-level lists and objects
        self.time_vary = deepcopy(self.time_vary_)
        self.time_inv = deepcopy(self.time_inv_)
        self.shock_vars = deepcopy(self.shock_vars_)

    def add_to_time_vary(self, *params):
        """
        Adds any number of parameters to time_vary for this instance.

        Parameters
        ----------
        params : string
            Any number of strings naming attributes to be added to time_vary

        Returns
        -------
        None
        """
        for param in params:
            if param not in self.time_vary:
                self.time_vary.append(param)

    def add_to_time_inv(self, *params):
        """
        Adds any number of parameters to time_inv for this instance.

        Parameters
        ----------
        params : string
            Any number of strings naming attributes to be added to time_inv

        Returns
        -------
        None
        """
        for param in params:
            if param not in self.time_inv:
                self.time_inv.append(param)

    def del_from_time_vary(self, *params):
        """
        Removes any number of parameters from time_vary for this instance.

        Parameters
        ----------
        params : string
            Any number of strings naming attributes to be removed from time_vary

        Returns
        -------
        None
        """
        for param in params:
            if param in self.time_vary:
                self.time_vary.remove(param)

    def del_from_time_inv(self, *params):
        """
        Removes any number of parameters from time_inv for this instance.

        Parameters
        ----------
        params : string
            Any number of strings naming attributes to be removed from time_inv

        Returns
        -------
        None
        """
        for param in params:
            if param in self.time_inv:
                self.time_inv.remove(param)

    def unpack(self, parameter):
        """
        Unpacks a parameter from a solution object for easier access.
        After the model has been solved, the parameters (like consumption function)
        reside in the attributes of each element of `ConsumerType.solution` (e.g. `cFunc`).  This method creates a (time varying) attribute of the given
        parameter name that contains a list of functions accessible by `ConsumerType.parameter`.

        Parameters
        ----------
        parameter: str
            Name of the function to unpack from the solution

        Returns
        -------
        none
        """
        setattr(self, parameter, list())
        for solution_t in self.solution:
            self.__dict__[parameter].append(solution_t.__dict__[parameter])
        self.add_to_time_vary(parameter)

    def solve(self, verbose=False, presolve=True, from_solution=None):
        """
        Solve the model for this instance of an agent type by backward induction.
        Loops through the sequence of one period problems, passing the solution
        from period t+1 to the problem for period t.

        Parameters
        ----------
        verbose : bool, optional
            If True, solution progress is printed to screen. Default False.
        presolve : bool, optional
            If True (default), the pre_solve method is run before solving.
        from_solution: Solution
            If different from None, will be used as the starting point of backward
            induction, instead of self.solution_terminal

        Returns
        -------
        none
        """

        # Ignore floating point "errors". Numpy calls it "errors", but really it's excep-
        # tions with well-defined answers such as 1.0/0.0 that is np.inf, -1.0/0.0 that is
        # -np.inf, np.inf/np.inf is np.nan and so on.
        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            if presolve:
                self.pre_solve()  # Do pre-solution stuff
            self.solution = solve_agent(
                self, verbose, from_solution
            )  # Solve the model by backward induction
            self.post_solve()  # Do post-solution stuff

    def reset_rng(self):
        """
        Reset the random number generator for this type.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        self.RNG = np.random.default_rng(self.seed)

    def check_elements_of_time_vary_are_lists(self):
        """
        A method to check that elements of time_vary are lists.
        """
        for param in self.time_vary:
            if not hasattr(self, param):
                continue
            if not isinstance(
                getattr(self, param),
                (TimeVaryingDiscreteDistribution, IndexDistribution),
            ):
                assert type(getattr(self, param)) == list, (
                    param
                    + " is not a list or time varying distribution,"
                    + " but should be because it is in time_vary"
                )

    def check_restrictions(self):
        """
        A method to check that various restrictions are met for the model class.
        """
        return

    def pre_solve(self):
        """
        A method that is run immediately before the model is solved, to check inputs or to prepare
        the terminal solution, perhaps.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        self.check_restrictions()
        self.check_elements_of_time_vary_are_lists()
        return None

    def post_solve(self):
        """
        A method that is run immediately after the model is solved, to finalize
        the solution in some way.  Does nothing here.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        return None

    def initialize_sim(self):
        """
        Prepares this AgentType for a new simulation.  Resets the internal random number generator,
        makes initial states for all agents (using sim_birth), clears histories of tracked variables.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if not hasattr(self, "T_sim"):
            raise Exception(
                "To initialize simulation variables it is necessary to first "
                + "set the attribute T_sim to the largest number of observations "
                + "you plan to simulate for each agent including re-births."
            )
        elif self.T_sim <= 0:
            raise Exception(
                "T_sim represents the largest number of observations "
                + "that can be simulated for an agent, and must be a positive number."
            )

        self.reset_rng()
        self.t_sim = 0
        all_agents = np.ones(self.AgentCount, dtype=bool)
        blank_array = np.empty(self.AgentCount)
        blank_array[:] = np.nan
        for var in self.state_now:
            if self.state_now[var] is None:
                self.state_now[var] = copy(blank_array)

            # elif self.state_prev[var] is None:
            #    self.state_prev[var] = copy(blank_array)
        self.t_age = np.zeros(
            self.AgentCount, dtype=int
        )  # Number of periods since agent entry
        self.t_cycle = np.zeros(
            self.AgentCount, dtype=int
        )  # Which cycle period each agent is on
        self.sim_birth(all_agents)

        # If we are asked to use existing shocks and a set of initial conditions
        # exist, use them
        if self.read_shocks and bool(self.newborn_init_history) and len(self.newborn_init_history) > 0:
            # Access the first period's data (dictionary), then the variable from it.
            first_period_newborn_history = self.newborn_init_history[0]
            for var_name in self.state_now:
                # Check that we are actually given a value for the variable
                if var_name in first_period_newborn_history:
                    # Copy only array-like idiosyncratic states. Aggregates should
                    # not be set by newborns
                    idio = (
                        isinstance(self.state_now[var_name], np.ndarray)
                        and len(self.state_now[var_name]) == self.AgentCount
                    )
                    if idio:
                        # Ensure the stored value is an array before assigning
                        new_born_val = first_period_newborn_history[var_name]
                        if isinstance(new_born_val, np.ndarray):
                            self.state_now[var_name] = new_born_val
                        else: # If it's scalar (e.g. for aggregate), it will be broadcast.
                              # This part might need care if an aggregate var was stored as scalar but idio state expects array.
                              # However, sim_birth should handle types correctly.
                            self.state_now[var_name].fill(new_born_val) 
                else:
                    warn(
                        "The option for reading shocks was activated but "
                        + "the model requires state "
                        + var_name
                        + ", not contained in "
                        + "newborn_init_history for the first period (t=0)."
                    )
        self.clear_history()
        return None

    def sim_one_period(self):
        """
        Simulates one period for this type.  Calls the methods get_mortality(), get_shocks() or
        read_shocks, get_states(), get_controls(), and get_poststates().  These should be defined for
        AgentType subclasses, except get_mortality (define its components sim_death and sim_birth
        instead) and read_shocks.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if not hasattr(self, "solution"):
            raise Exception(
                "Model instance does not have a solution stored. To simulate, it is necessary"
                " to run the `solve()` method of the class first."
            )

        # Mortality adjusts the agent population
        self.get_mortality()  # Replace some agents with "newborns"

        # state_{t-1}
        for var in self.state_now:
            self.state_prev[var] = self.state_now[var]

            if isinstance(self.state_now[var], np.ndarray):
                self.state_now[var] = np.empty(self.AgentCount)
            else:
                # Probably an aggregate variable. It may be getting set by the Market.
                pass

        if self.read_shocks:  # If shock histories have been pre-specified, use those
            self.read_shocks_from_history()
        else:  # Otherwise, draw shocks as usual according to subclass-specific method
            self.get_shocks()
        self.get_states()  # Determine each agent's state at decision time
        self.get_controls()  # Determine each agent's choice or control variables based on states
        self.get_poststates()  # Move now state_now to state_prev

        # Advance time for all agents
        self.t_age = self.t_age + 1  # Age all consumers by one period
        self.t_cycle = self.t_cycle + 1  # Age all consumers within their cycle
        self.t_cycle[self.t_cycle == self.T_cycle] = (
            0  # Resetting to zero for those who have reached the end
        )

    def make_shock_history(self):
        """
        Makes a pre-specified history of shocks for the simulation.  Shock variables should be named
        in self.shock_vars, a list of strings that is subclass-specific.  This method runs a subset
        of the standard simulation loop by simulating only mortality and shocks; each variable named
        in shock_vars is stored in a T_sim x AgentCount array in history dictionary self.history[X].
        Automatically sets self.read_shocks to True so that these pre-specified shocks are used for
        all subsequent calls to simulate().

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Re-initialize the simulation. self.history, self.shock_history, self.newborn_init_history are now [].
        # self.t_sim is 0.
        self.initialize_sim()

        # Pre-allocate lists of dictionaries for T_sim periods
        self.shock_history = [{} for _ in range(self.T_sim)]
        self.newborn_init_history = [{} for _ in range(self.T_sim)]

        # Record the initial conditions (t=0) of all agents as if they are "newborn"
        # These are set by initialize_sim() -> sim_birth(all_agents)
        # This populates newborn_init_history[0]
        if self.T_sim > 0:
            initial_newborn_data_t0 = {}
            for var_name in self.state_vars:
                current_state_val = self.state_now[var_name]
                if isinstance(current_state_val, np.ndarray):
                    initial_newborn_data_t0[var_name] = copy(current_state_val)
                else: # Should be aggregate, store as an array for consistency if needed by reader
                    initial_newborn_data_t0[var_name] = np.full(self.AgentCount, current_state_val)
            self.newborn_init_history[0] = initial_newborn_data_t0

        # Loop through each simulation period to generate and store shock & newborn history
        for t_idx in range(self.T_sim):
            self.t_sim = t_idx  # Set current simulation time for methods called below

            current_period_shock_data = {}
            current_period_newborn_data = {} # Data for agents born in *this* period t_idx

            # Simulate mortality for this period.
            # If read_shocks is True, get_mortality uses self.shock_history[t_idx] (which is being built).
            # This implies make_shock_history should generally be run with read_shocks=False initially.
            # If read_shocks is True here, it means we are potentially re-generating based on a loaded history,
            # which could be complex. Assuming standard use: read_shocks=False when first creating.
            self.get_mortality()  # This updates self.who_dies and calls sim_birth for new agents
            current_period_shock_data["who_dies"] = copy(self.who_dies)

            # For agents who were born in this period (self.who_dies is True),
            # record their state_now values (as set by sim_birth).
            if np.sum(self.who_dies) > 0:
                for var_name in self.state_vars:
                    state_val = self.state_now[var_name]
                    if isinstance(state_val, np.ndarray) and state_val.size == self.AgentCount:
                        # Store only for those who died and were replaced, others NaN
                        value_to_store = np.full_like(state_val, np.nan) 
                        value_to_store[self.who_dies] = state_val[self.who_dies]
                        current_period_newborn_data[var_name] = value_to_store
                    else: # Aggregate variable or scalar
                        # Store the single value that applies to newborns, as an array for consistency
                        current_val_for_newborns = state_val
                        if isinstance(current_val_for_newborns, np.ndarray) and current_val_for_newborns.size == 1:
                            current_val_for_newborns = current_val_for_newborns.item()
                        current_period_newborn_data[var_name] = np.full(self.AgentCount, current_val_for_newborns)
            
            # If it's t=0, all agents are "newborn" effectively from initialize_sim.
            # The newborn_init_history[0] should reflect these initial states.
            # If read_shocks=True caused some to "die" at t=0, current_period_newborn_data will capture their new state.
            if t_idx == 0:
                # Merge/update initial_newborn_data_t0 with any specific data from deaths at t0
                self.newborn_init_history[0].update(current_period_newborn_data)
            elif np.sum(self.who_dies) > 0 : # For t_idx > 0, only if there were actual deaths
                 self.newborn_init_history[t_idx] = current_period_newborn_data
            # If no deaths at t_idx > 0, newborn_init_history[t_idx] remains an empty dict.


            # Simulate other shocks for this period
            self.get_shocks()  # Populates self.shocks dictionary
            for var_name in self.shock_vars: # shock_vars does not include 'who_dies'
                if var_name in self.shocks:
                    current_period_shock_data[var_name] = copy(self.shocks[var_name])
                else: # Should not happen if shock_vars is defined correctly
                    warn(f"Shock variable {var_name} not found in self.shocks after get_shocks() call.")


            # Store the collected shock data for the current period t_idx
            self.shock_history[t_idx].update(current_period_shock_data)


            # Advance agent ages and cycle for the next iteration
            self.t_age = self.t_age + 1
            self.t_cycle = self.t_cycle + 1
            self.t_cycle[self.t_cycle == self.T_cycle] = 0
        
        # After the loop, set self.t_sim to indicate completion of T_sim periods
        self.t_sim = self.T_sim

        # Flag that these histories can now be read
        self.read_shocks = True

    def get_mortality(self):
        """
        Simulates mortality or agent turnover according to some model-specific rules named sim_death
        and sim_birth (methods of an AgentType subclass).  sim_death takes no arguments and returns
        a Boolean array of size AgentCount, indicating which agents of this type have "died" and
        must be replaced.  sim_birth takes such a Boolean array as an argument and generates initial
        post-decision states for those agent indices.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.read_shocks:
            # Ensure t_sim is a valid index for pre-made histories
            if self.t_sim >= len(self.shock_history) or self.t_sim >= len(self.newborn_init_history):
                # This case should ideally be prevented by how T_sim and loop bounds are handled.
                # If it occurs, it implies an issue with t_sim advancement or history length.
                raise IndexError(
                    f"t_sim = {self.t_sim} is out of bounds for shock_history (len {len(self.shock_history)}) "
                    f"or newborn_init_history (len {len(self.newborn_init_history)})."
                )

            current_period_shocks = self.shock_history[self.t_sim]
            if "who_dies" not in current_period_shocks:
                 # This might happen if make_shock_history was not completed or cleared.
                warn(f"'who_dies' not found in shock_history for t_sim = {self.t_sim}. Assuming no deaths.")
                who_dies = np.zeros(self.AgentCount, dtype=bool)
            else:
                who_dies = current_period_shocks["who_dies"]

            # Instead of simulating births, assign the saved newborn initial conditions
            if np.sum(who_dies) > 0:
                current_period_newborn_states = self.newborn_init_history[self.t_sim]
                if not current_period_newborn_states: # Check if the dictionary is empty
                    warn(f"Newborn history for t_sim = {self.t_sim} is empty, but there are deaths. States won't be updated from history.")
                else:
                    for var_name in self.state_now:
                        if var_name in current_period_newborn_states:
                            idio = (
                                isinstance(self.state_now[var_name], np.ndarray)
                                and len(self.state_now[var_name]) == self.AgentCount
                            )
                            if idio:
                                new_vals_for_var = current_period_newborn_states[var_name]
                                if isinstance(new_vals_for_var, np.ndarray):
                                    # We need to assign only the values for agents who_dies
                                    self.state_now[var_name][who_dies] = new_vals_for_var[who_dies]
                                else:
                                    warn(f"Idiosyncratic state {var_name} in newborn_init_history is not an ndarray for t_sim={self.t_sim}.")
                                    # Attempt to broadcast if it's a scalar; might fail or be incorrect.
                                    self.state_now[var_name][who_dies] = new_vals_for_var
                            else: # Aggregate variable.
                                # The value in newborn_init_history should be the one for newborns.
                                # It might be stored as a scalar or a full array.
                                agg_new_val = current_period_newborn_states[var_name]
                                if isinstance(agg_new_val, np.ndarray) and agg_new_val.size == self.AgentCount:
                                    self.state_now[var_name] = agg_new_val[0] # Assign the scalar value
                                elif isinstance(agg_new_val, np.ndarray) and agg_new_val.size == 1:
                                     self.state_now[var_name] = agg_new_val.item()
                                else: # scalar
                                     self.state_now[var_name] = agg_new_val

                        else:
                            warn(
                                f"The option for reading shocks was activated but "
                                f"the model requires state {var_name} for newborns at t={self.t_sim}, "
                                f"not contained in newborn_init_history[{self.t_sim}]."
                            )
                # Reset ages of newborns
                self.t_age[who_dies] = 0
                self.t_cycle[who_dies] = 0
        else:
            who_dies = self.sim_death()
            self.sim_birth(who_dies)
        self.who_dies = who_dies
        return None

    def sim_death(self):
        """
        Determines which agents in the current population "die" or should be replaced.  Takes no
        inputs, returns a Boolean array of size self.AgentCount, which has True for agents who die
        and False for those that survive. Returns all False by default, must be overwritten by a
        subclass to have replacement events.

        Parameters
        ----------
        None

        Returns
        -------
        who_dies : np.array
            Boolean array of size self.AgentCount indicating which agents die and are replaced.
        """
        who_dies = np.zeros(self.AgentCount, dtype=bool)
        return who_dies

    def sim_birth(self, which_agents):
        """
        Makes new agents for the simulation.  Takes a boolean array as an input, indicating which
        agent indices are to be "born".  Does nothing by default, must be overwritten by a subclass.

        Parameters
        ----------
        which_agents : np.array(Bool)
            Boolean array of size self.AgentCount indicating which agents should be "born".

        Returns
        -------
        None
        """
        print("AgentType subclass must define method sim_birth!")
        return None

    def get_shocks(self):
        """
        Gets values of shock variables for the current period.  Does nothing by default, but can
        be overwritten by subclasses of AgentType.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        return None

    def read_shocks_from_history(self):
        """
        Reads values of shock variables for the current period from history arrays.
        For each variable X named in self.shock_vars, this attribute of self is
        set to self.shock_history[self.t_sim][X]. (Note: self.shocks is a dictionary on the agent)

        This method is only ever called if self.read_shocks is True.  This can
        be achieved by using the method make_shock_history() (or manually after
        storing a "handcrafted" shock history).

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.t_sim >= len(self.shock_history):
            raise IndexError(f"t_sim = {self.t_sim} is out of bounds for shock_history (len {len(self.shock_history)}).")

        current_period_shocks_from_history = self.shock_history[self.t_sim]
        for var_name in self.shock_vars: # self.shock_vars is a list of shock names (strings)
            if var_name in current_period_shocks_from_history:
                self.shocks[var_name] = current_period_shocks_from_history[var_name]
            else:
                # This implies a mismatch between shock_vars and what's in shock_history.
                # This could happen if shock_history was not properly populated for this var_name.
                warn(f"Shock variable '{var_name}' not found in shock_history for period {self.t_sim}.")
                # Initialize to NaN or some default to avoid AttributeError if code expects it in self.shocks
                self.shocks[var_name] = np.full(self.AgentCount, np.nan)

    def get_states(self):
        """
        Gets values of state variables for the current period.
        By default, calls transition function and assigns values
        to the state_now dictionary.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        new_states = self.transition()

        for i, var in enumerate(self.state_now):
            # a hack for now to deal with 'post-states'
            if i < len(new_states):
                self.state_now[var] = new_states[i]

        return None

    def transition(self):
        """

        Parameters
        ----------
        None

        [Eventually, to match dolo spec:
        exogenous_prev, endogenous_prev, controls, exogenous, parameters]

        Returns
        -------

        endogenous_state: ()
            Tuple with new values of the endogenous states
        """

        return ()

    def get_controls(self):
        """
        Gets values of control variables for the current period, probably by using current states.
        Does nothing by default, but can be overwritten by subclasses of AgentType.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        return None

    def get_poststates(self):
        """
        Gets values of post-decision state variables for the current period,
        probably by current
        states and controls and maybe market-level events or shock variables.
        Does nothing by
        default, but can be overwritten by subclasses of AgentType.

        DEPRECATED: New models should use the state now/previous rollover
        functionality instead of poststates.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        return None

    def simulate(self, sim_periods=None):
        """
        Simulates this agent type for a given number of periods. Defaults to
        self.T_sim if no input.
        Records histories of attributes named in self.track_vars in
        self.history[varname].

        Parameters
        ----------
        None

        Returns
        -------
        history : dict
            The history tracked during the simulation.
        """
        if not hasattr(self, "t_sim"):
            raise Exception(
                "It seems that the simulation variables were not initialize before calling "
                + "simulate(). Call initialize_sim() to initialize the variables before calling simulate() again."
            )

        if not hasattr(self, "T_sim"):
            raise Exception(
                "This agent type instance must have the attribute T_sim set to a positive integer."
                + "Set T_sim to match the largest dataset you might simulate, and run this agent's"
                + "initalizeSim() method before running simulate() again."
            )

        if sim_periods is not None and self.T_sim < sim_periods:
            raise Exception(
                "To simulate, sim_periods has to be larger than the maximum data set size "
                + "T_sim. Either increase the attribute T_sim of this agent type instance "
                + "and call the initialize_sim() method again, or set sim_periods <= T_sim."
            )

        # Ignore floating point "errors". Numpy calls it "errors", but really it's excep-
        # tions with well-defined answers such as 1.0/0.0 that is np.inf, -1.0/0.0 that is
        # -np.inf, np.inf/np.inf is np.nan and so on.
        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            if sim_periods is None:
                sim_periods = self.T_sim

            if sim_periods is None:
                sim_periods = self.T_sim
            
            # t_sim at the start of simulate() is the first period to simulate.
            # It increments up to t_sim + sim_periods -1.
            # The loop for t goes from 0 to sim_periods-1.
            # self.t_sim is the *current* period index.

            for _t_loop_idx in range(sim_periods): # Loop for the number of periods to simulate
                self.sim_one_period() # Advances agent states for self.t_sim

                # Record data for the period self.t_sim that just completed
                period_data = {"period": self.t_sim}
                for track_spec_item in self.track_vars:
                    if isinstance(track_spec_item, str):
                        var_name = track_spec_item
                        value_to_store = np.nan # Default if not found
                        if var_name in self.state_now:
                            value_to_store = self.state_now[var_name]
                        elif var_name in self.shocks:
                            value_to_store = self.shocks[var_name]
                        elif var_name in self.controls:
                            value_to_store = self.controls[var_name]
                        elif hasattr(self, var_name):
                            value_to_store = getattr(self, var_name)
                        else:
                            warn(f"Tracked variable '{var_name}' not found on agent instance for period {self.t_sim}.")

                        if isinstance(value_to_store, np.ndarray):
                            period_data[var_name] = copy(value_to_store)
                        elif isinstance(value_to_store, (list, dict)):
                            period_data[var_name] = deepcopy(value_to_store)
                        else:
                            period_data[var_name] = value_to_store
                    
                    elif isinstance(track_spec_item, tuple) and len(track_spec_item) > 1:
                        var_name = track_spec_item[0]
                        stats_to_calc = track_spec_item[1:]
                        
                        data_array = np.nan
                        if var_name in self.state_now:
                            data_array = self.state_now[var_name]
                        elif var_name in self.shocks:
                            data_array = self.shocks[var_name]
                        elif var_name in self.controls:
                            data_array = self.controls[var_name]
                        elif hasattr(self, var_name):
                            data_array = getattr(self, var_name)
                        else:
                            warn(f"Variable '{var_name}' for summary statistics not found on agent instance for period {self.t_sim}.")

                        if isinstance(data_array, np.ndarray):
                            for stat_str in stats_to_calc:
                                stat_key = f"{var_name}_{stat_str}"
                                if stat_str == 'mean':
                                    period_data[stat_key] = np.mean(data_array)
                                elif stat_str == 'std':
                                    period_data[stat_key] = np.std(data_array)
                                elif stat_str == 'min':
                                    period_data[stat_key] = np.min(data_array)
                                elif stat_str == 'max':
                                    period_data[stat_key] = np.max(data_array)
                                elif stat_str == 'median':
                                    period_data[stat_key] = np.median(data_array)
                                # Can add more stats like 'q25', 'q75' here later
                                else:
                                    warn(f"Unknown statistic '{stat_str}' requested for variable '{var_name}'.")
                        else: # data_array was not found or not an array
                            for stat_str in stats_to_calc:
                                stat_key = f"{var_name}_{stat_str}"
                                period_data[stat_key] = np.nan # Store NaN if data_array is not available
                    else:
                        warn(f"Invalid item in track_vars: {track_spec_item}")

                self.history.append(period_data)
                
                # self.t_sim was the period *just simulated*. Now advance it for the next loop iteration.
                self.t_sim += 1

            return self.history

    def clear_history(self):
        """
        Clears the main simulation history.
        self.history will be an empty list.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.history = []
        # Note: This does not clear self.shock_history or self.newborn_init_history,
        # as clear_history is typically for simulation results, not pre-specified shocks.
        # If those also need clearing, separate methods or an extended clear_all_histories() might be needed.

    def get_history_df(self, dtype_map=None):
        """
        Converts the agent's simulation history into a pandas DataFrame,
        optionally casting columns to specified data types.

        The history is stored in `self.history` as a list of dictionaries,
        where each dictionary represents a period and contains arrays of
        variable values for all agents. This method transforms that structure
        into a DataFrame with a MultiIndex (period, agent_id).

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the simulation history, indexed by period
            and agent_id. Columns are the tracked variables. Returns an empty
            DataFrame with the correct structure if history is empty.
        """
        if not hasattr(self, "AgentCount"):
            warn("AgentCount attribute is not set. Cannot determine number of agents for history DataFrame.")
            # Construct column names for an empty DataFrame based on track_vars parsing
            empty_df_cols = ['period', 'agent_id']
            for item in getattr(self, 'track_vars', []):
                if isinstance(item, str):
                    if item != 'period': empty_df_cols.append(item)
                elif isinstance(item, tuple) and len(item) > 1:
                    var_name = item[0]
                    for stat_str in item[1:]:
                        empty_df_cols.append(f"{var_name}_{stat_str}")
            return pd.DataFrame(columns=empty_df_cols).set_index(['period', 'agent_id'])

        all_records = []
        if not self.history:
            empty_df_cols = ['period', 'agent_id']
            for item in self.track_vars: # Use self.track_vars directly
                if isinstance(item, str):
                    if item != 'period': empty_df_cols.append(item)
                elif isinstance(item, tuple) and len(item) > 1:
                    var_name = item[0]
                    for stat_str in item[1:]:
                        empty_df_cols.append(f"{var_name}_{stat_str}")
            return pd.DataFrame(columns=empty_df_cols).set_index(['period', 'agent_id'])

        # Determine column names from the first period's data if history is not empty
        # This is more robust if track_vars definition leads to complex column names
        # or if not all tracked items appear in every period_dict (though they should).
        # The keys in period_dict are now the definitive source for columns other than 'agent_id'.
        # 'period' is already a key in period_dict.
        
        # Get a representative set of column names from the data itself, excluding 'period'
        # as it's handled as an index.
        potential_columns = set()
        for period_dict_scan in self.history:
            for key in period_dict_scan:
                if key != 'period':
                    potential_columns.add(key)
        sorted_columns = sorted(list(potential_columns))


        for period_dict in self.history:
            current_period_id = period_dict.get('period', -1)

            # Determine number of agents for this period.
            # If all entries are scalars (e.g. only summary stats tracked), num_agents might be ambiguous.
            # Default to self.AgentCount. If a full array exists, use its length.
            num_agents_in_period = self.AgentCount 
            found_array_for_agent_count = False
            for key in sorted_columns: # Check based on actual data columns
                if key in period_dict and isinstance(period_dict[key], np.ndarray):
                    num_agents_in_period = len(period_dict[key])
                    found_array_for_agent_count = True
                    break
            
            if num_agents_in_period == 0 and self.AgentCount > 0: # If AgentCount > 0 but no arrays found, still loop once for summary stats
                num_agents_to_loop = 1 # Create one "representative" row for summary stats
            elif num_agents_in_period == 0 and self.AgentCount == 0:
                 num_agents_to_loop = 0 # Truly no agents
            else:
                 num_agents_to_loop = num_agents_in_period


            for agent_idx in range(num_agents_to_loop):
                record = {'period': current_period_id, 'agent_id': agent_idx if self.AgentCount > 0 else -1} # Use -1 if AgentCount is 0
                for key in sorted_columns: # Iterate using the derived column set
                    value = period_dict.get(key, np.nan) # Get value, default to NaN if key missing in this specific period_dict
                    
                    if isinstance(value, np.ndarray):
                        if agent_idx < len(value):
                            record[key] = value[agent_idx]
                        else: # Should not happen if num_agents_in_period was derived correctly
                            record[key] = np.nan 
                    else: # Scalar (summary statistic or broadcasted value)
                        record[key] = value 
                all_records.append(record)
        
        if not all_records:
            # Re-construct empty_df_cols as before for safety, if all_records is empty.
            empty_df_cols = ['period', 'agent_id']
            for item in getattr(self, 'track_vars', []):
                if isinstance(item, str):
                    if item != 'period': empty_df_cols.append(item)
                elif isinstance(item, tuple) and len(item) > 1:
                    var_name = item[0]
                    for stat_str in item[1:]:
                        empty_df_cols.append(f"{var_name}_{stat_str}")
            return pd.DataFrame(columns=empty_df_cols).set_index(['period', 'agent_id'])


        df = pd.DataFrame.from_records(all_records)
        
        if 'period' in df.columns and 'agent_id' in df.columns:
            # Handle cases where agent_id might be -1 if AgentCount is 0.
            # This can lead to a non-unique index if not careful.
            # If agent_id contains -1 and other valid ids, it might be better to not set index or clean.
            # For now, assume if AgentCount is 0, all_records is empty or df.empty will be true.
            if not df.empty:
                 df = df.set_index(['period', 'agent_id'])
            else: # If df is empty, create it with proper index and columns
                empty_df_cols = ['period', 'agent_id'] + sorted_columns
                df = pd.DataFrame(columns=empty_df_cols).set_index(['period', 'agent_id'])

        elif not df.empty:
            warn("DataFrame created from history is missing 'period' or 'agent_id' columns for index setting.")
            
        return df


def solve_agent(agent, verbose, from_solution=None):
    """
    Solve the dynamic model for one agent type
    using backwards induction.
    This function iterates on "cycles"
    of an agent's model either a given number of times
    or until solution convergence
    if an infinite horizon model is used
    (with agent.cycles = 0).

    Parameters
    ----------
    agent : AgentType
        The microeconomic AgentType whose dynamic problem
        is to be solved.
    verbose : boolean
        If True, solution progress is printed to screen (when cycles != 1).
    from_solution: Solution
        If different from None, will be used as the starting point of backward
        induction, instead of self.solution_terminal

    Returns
    -------
    solution : [Solution]
        A list of solutions to the one period problems that the agent will
        encounter in his "lifetime".
    """
    # Check to see whether this is an (in)finite horizon problem
    cycles_left = agent.cycles  # NOQA
    infinite_horizon = cycles_left == 0  # NOQA

    if from_solution is None:
        solution_last = agent.solution_terminal  # NOQA
    else:
        solution_last = from_solution

    # Initialize the solution, which includes the terminal solution if it's not a pseudo-terminal period
    solution = []
    if not agent.pseudo_terminal:
        solution.insert(0, deepcopy(solution_last))

    # Initialize the process, then loop over cycles
    go = True  # NOQA
    completed_cycles = 0  # NOQA
    max_cycles = 5000  # NOQA  - escape clause
    if verbose:
        t_last = time()
    while go:
        # Solve a cycle of the model, recording it if horizon is finite
        solution_cycle = solve_one_cycle(agent, solution_last)
        if not infinite_horizon:
            solution = solution_cycle + solution

        # Check for termination: identical solutions across
        # cycle iterations or run out of cycles
        solution_now = solution_cycle[0]
        if infinite_horizon:
            if completed_cycles > 0:
                solution_distance = solution_now.distance(solution_last)
                agent.solution_distance = (
                    solution_distance  # Add these attributes so users can
                )
                agent.completed_cycles = (
                    completed_cycles  # query them to see if solution is ready
                )
                go = (
                    solution_distance > agent.tolerance
                    and completed_cycles < max_cycles
                )
            else:  # Assume solution does not converge after only one cycle
                solution_distance = 100.0
                go = True
        else:
            cycles_left += -1
            go = cycles_left > 0

        # Update the "last period solution"
        solution_last = solution_now
        completed_cycles += 1

        # Display progress if requested
        if verbose:
            t_now = time()
            if infinite_horizon:
                print(
                    "Finished cycle #"
                    + str(completed_cycles)
                    + " in "
                    + str(t_now - t_last)
                    + " seconds, solution distance = "
                    + str(solution_distance)
                )
            else:
                print(
                    "Finished cycle #"
                    + str(completed_cycles)
                    + " of "
                    + str(agent.cycles)
                    + " in "
                    + str(t_now - t_last)
                    + " seconds."
                )
            t_last = t_now

    # Record the last cycle if horizon is infinite (solution is still empty!)
    if infinite_horizon:
        solution = (
            solution_cycle  # PseudoTerminal=False impossible for infinite horizon
        )

    return solution


def solve_one_cycle(agent, solution_last):
    """
    Solve one "cycle" of the dynamic model for one agent type.  This function
    iterates over the periods within an agent's cycle, updating the time-varying
    parameters and passing them to the single period solver(s).

    Parameters
    ----------
    agent : AgentType
        The microeconomic AgentType whose dynamic problem is to be solved.
    solution_last : Solution
        A representation of the solution of the period that comes after the
        end of the sequence of one period problems.  This might be the term-
        inal period solution, a "pseudo terminal" solution, or simply the
        solution to the earliest period from the succeeding cycle.

    Returns
    -------
    solution_cycle : [Solution]
        A list of one period solutions for one "cycle" of the AgentType's
        microeconomic model.
    """

    # Check if the agent has a 'Parameters' attribute of the 'Parameters' class
    # if so, take advantage of it. Else, use the old method
    if hasattr(agent, "params") and isinstance(agent.params, Parameters):
        T = agent.params._length

        # Initialize the solution for this cycle, then iterate on periods
        solution_cycle = []
        solution_next = solution_last

        cycles_range = [0] + list(range(T - 1, 0, -1))
        for k in range(T - 1, -1, -1) if agent.cycles == 1 else cycles_range:
            # Update which single period solver to use (if it depends on time)
            if hasattr(agent.solve_one_period, "__getitem__"):
                solve_one_period = agent.solve_one_period[k]
            else:
                solve_one_period = agent.solve_one_period

            if hasattr(solve_one_period, "solver_args"):
                these_args = solve_one_period.solver_args
            else:
                these_args = get_arg_names(solve_one_period)

            # Make a temporary dictionary for this period
            temp_pars = agent.params[k]
            temp_dict = {
                name: solution_next if name == "solution_next" else temp_pars[name]
                for name in these_args
            }

            # Solve one period, add it to the solution, and move to the next period
            solution_t = solve_one_period(**temp_dict)
            solution_cycle.insert(0, solution_t)
            solution_next = solution_t

    else:
        # Calculate number of periods per cycle, defaults to 1 if all variables are time invariant
        if len(agent.time_vary) > 0:
            T = len(agent.__dict__[agent.time_vary[0]])
        else:
            T = 1

        solve_dict = {
            parameter: agent.__dict__[parameter] for parameter in agent.time_inv
        }
        solve_dict.update({parameter: None for parameter in agent.time_vary})

        # Initialize the solution for this cycle, then iterate on periods
        solution_cycle = []
        solution_next = solution_last

        cycles_range = [0] + list(range(T - 1, 0, -1))
        for k in range(T - 1, -1, -1) if agent.cycles == 1 else cycles_range:
            # Update which single period solver to use (if it depends on time)
            if hasattr(agent.solve_one_period, "__getitem__"):
                solve_one_period = agent.solve_one_period[k]
            else:
                solve_one_period = agent.solve_one_period

            if hasattr(solve_one_period, "solver_args"):
                these_args = solve_one_period.solver_args
            else:
                these_args = get_arg_names(solve_one_period)

            # Update time-varying single period inputs
            for name in agent.time_vary:
                if name in these_args:
                    solve_dict[name] = agent.__dict__[name][k]
            solve_dict["solution_next"] = solution_next

            # Make a temporary dictionary for this period
            temp_dict = {name: solve_dict[name] for name in these_args}

            # Solve one period, add it to the solution, and move to the next period
            solution_t = solve_one_period(**temp_dict)
            solution_cycle.insert(0, solution_t)
            solution_next = solution_t

    # Return the list of per-period solutions
    return solution_cycle


def make_one_period_oo_solver(solver_class):
    """
    Returns a function that solves a single period consumption-saving
    problem.
    Parameters
    ----------
    solver_class : Solver
        A class of Solver to be used.
    -------
    solver_function : function
        A function for solving one period of a problem.
    """

    def one_period_solver(**kwds):
        solver = solver_class(**kwds)

        # not ideal; better if this is defined in all Solver classes
        if hasattr(solver, "prepare_to_solve"):
            solver.prepare_to_solve()

        solution_now = solver.solve()
        return solution_now

    one_period_solver.solver_class = solver_class
    # This can be revisited once it is possible to export parameters
    one_period_solver.solver_args = get_arg_names(solver_class.__init__)[1:]

    return one_period_solver


# ========================================================================
# ========================================================================


class Market(Model):
    """
    A superclass to represent a central clearinghouse of information.  Used for
    dynamic general equilibrium models to solve the "macroeconomic" model as a
    layer on top of the "microeconomic" models of one or more AgentTypes.

    Parameters
    ----------
    agents : [AgentType]
        A list of all the AgentTypes in this market.
    sow_vars : [string]
        Names of variables generated by the "aggregate market process" that should
        "sown" to the agents in the market.  Aggregate state, etc.
    reap_vars : [string]
        Names of variables to be collected ("reaped") from agents in the market
        to be used in the "aggregate market process".
    const_vars : [string]
        Names of attributes of the Market instance that are used in the "aggregate
        market process" but do not come from agents-- they are constant or simply
        parameters inherent to the process.
    track_vars : [string]
        Names of variables generated by the "aggregate market process" that should
        be tracked as a "history" so that a new dynamic rule can be calculated.
        This is often a subset of sow_vars.
    dyn_vars : [string]
        Names of variables that constitute a "dynamic rule".
    mill_rule : function
        A function that takes inputs named in reap_vars and returns a tuple the same size and order as sow_vars.  The "aggregate market process" that
        transforms individual agent actions/states/data into aggregate data to
        be sent back to agents.
    calc_dynamics : function
        A function that takes inputs named in track_vars and returns an object
        with attributes named in dyn_vars.  Looks at histories of aggregate
        variables and generates a new "dynamic rule" for agents to believe and
        act on.
    act_T : int
        The number of times that the "aggregate market process" should be run
        in order to generate a history of aggregate variables.
    tolerance: float
        Minimum acceptable distance between "dynamic rules" to consider the
        Market solution process converged.  Distance is a user-defined metric.
    """

    def __init__(
        self,
        agents=None,
        sow_vars=None,
        reap_vars=None,
        const_vars=None,
        track_vars=None,
        dyn_vars=None,
        mill_rule=None,
        calc_dynamics=None,
        act_T=1000,
        tolerance=0.000001,
        **kwds,
    ):
        super().__init__()
        self.agents = agents if agents is not None else list()  # NOQA

        self.reap_vars = reap_vars if reap_vars is not None else list()  # NOQA
        self.reap_state = {var: [] for var in self.reap_vars}

        self.sow_vars = sow_vars if sow_vars is not None else list()  # NOQA
        # dictionaries for tracking initial and current values
        # of the sow variables.
        self.sow_init = {var: None for var in self.sow_vars}
        self.sow_state = {var: None for var in self.sow_vars}

        const_vars = const_vars if const_vars is not None else list()  # NOQA
        self.const_vars = {var: None for var in const_vars}

        self.track_vars = track_vars if track_vars is not None else list()  # NOQA
        self.dyn_vars = dyn_vars if dyn_vars is not None else list()  # NOQA

        if mill_rule is not None:  # To prevent overwriting of method-based mill_rules
            self.mill_rule = mill_rule
        if calc_dynamics is not None:  # Ditto for calc_dynamics
            self.calc_dynamics = calc_dynamics
        self.act_T = act_T  # NOQA
        self.tolerance = tolerance  # NOQA
        self.max_loops = 1000  # NOQA
        self.history = []  # MODIFIED: Initialized as an empty list
        self.assign_parameters(**kwds)

        self.print_parallel_error_once = True
        # Print the error associated with calling the parallel method
        # "solve_agents" one time. If set to false, the error will never
        # print. See "solve_agents" for why this prints once or never.

    def solve_agents(self):
        """
        Solves the microeconomic problem for all AgentTypes in this market.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        try:
            multi_thread_commands(self.agents, ["solve()"])
        except Exception as err:
            if self.print_parallel_error_once:
                # Set flag to False so this is only printed once.
                self.print_parallel_error_once = False
                print(
                    "**** WARNING: could not execute multi_thread_commands in HARK.core.Market.solve_agents() ",
                    "so using the serial version instead. This will likely be slower. "
                    "The multiTreadCommands() functions failed with the following error:",
                    "\n",
                    sys.exc_info()[0],
                    ":",
                    err,
                )  # sys.exc_info()[0])
            multi_thread_commands_fake(self.agents, ["solve()"])

    def solve(self):
        """
        "Solves" the market by finding a "dynamic rule" that governs the aggregate
        market state such that when agents believe in these dynamics, their actions
        collectively generate the same dynamic rule.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        go = True
        max_loops = self.max_loops  # Failsafe against infinite solution loop
        completed_loops = 0
        old_dynamics = None

        while go:  # Loop until the dynamic process converges or we hit the loop cap
            self.solve_agents()  # Solve each AgentType's micro problem
            self.make_history()  # "Run" the model while tracking aggregate variables
            new_dynamics = self.update_dynamics()  # Find a new aggregate dynamic rule

            # Check to see if the dynamic rule has converged (if this is not the first loop)
            if completed_loops > 0:
                distance = new_dynamics.distance(old_dynamics)
            else:
                distance = 1000000.0

            # Move to the next loop if the terminal conditions are not met
            old_dynamics = new_dynamics
            completed_loops += 1
            go = distance >= self.tolerance and completed_loops < max_loops

        self.dynamics = new_dynamics  # Store the final dynamic rule in self

    def reap(self):
        """
        Collects attributes named in reap_vars from each AgentType in the market,
        storing them in respectively named attributes of self.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        for var in self.reap_state:
            harvest = []

            for agent in self.agents:
                # TODO: generalized variable lookup across namespaces
                if var in agent.state_now:
                    # or state_now ??
                    harvest.append(agent.state_now[var])

            self.reap_state[var] = harvest

    def sow(self):
        """
        Distributes attrributes named in sow_vars from self to each AgentType
        in the market, storing them in respectively named attributes.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        for sow_var in self.sow_state:
            for this_type in self.agents:
                if sow_var in this_type.state_now:
                    this_type.state_now[sow_var] = self.sow_state[sow_var]
                if sow_var in this_type.shocks:
                    this_type.shocks[sow_var] = self.sow_state[sow_var]
                else:
                    setattr(this_type, sow_var, self.sow_state[sow_var])

    def mill(self):
        """
        Processes the variables collected from agents using the function mill_rule,
        storing the results in attributes named in aggr_sow.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        # Make a dictionary of inputs for the mill_rule
        mill_dict = copy(self.reap_state)
        mill_dict.update(self.const_vars)

        # Run the mill_rule and store its output in self
        product = self.mill_rule(**mill_dict)

        for i, sow_var in enumerate(self.sow_state):
            self.sow_state[sow_var] = product[i]

    def cultivate(self):
        """
        Has each AgentType in agents perform their market_action method, using
        variables sown from the market (and maybe also "private" variables).
        The market_action method should store new results in attributes named in
        reap_vars to be reaped later.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        for this_type in self.agents:
            this_type.market_action()

    def reset(self):
        """
        Reset the state of the market (attributes in sow_vars, etc) to some
        user-defined initial state, and erase the histories of tracked variables.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        # Reset the history of tracked variables
        self.history = [] # MODIFIED: history is now a list of dictionaries

        # Set the sow variables to their initial levels
        for var_name in self.sow_state:
            self.sow_state[var_name] = self.sow_init[var_name]

        # Reset each AgentType in the market
        for this_type in self.agents:
            this_type.reset()

    def store(self):
        """
        Record the current value of each variable X named in track_vars in an
        dictionary field named history[X].

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        period_market_data = {}
        for var_name in self.track_vars:
            if var_name in self.sow_state:
                value_now = self.sow_state[var_name]
            elif var_name in self.reap_state:
                value_now = self.reap_state[var_name]
            elif var_name in self.const_vars:
                value_now = self.const_vars[var_name]
            else:
                value_now = getattr(self, var_name)
            
            # Market variables are often scalars or small arrays, direct copy is usually fine.
            # If they can be large mutable objects, a deepcopy might be considered.
            period_market_data[var_name] = copy(value_now) 

        self.history.append(period_market_data)

    def make_history(self):
        """
        Runs a loop of sow-->cultivate-->reap-->mill act_T times, tracking the
        evolution of variables X named in track_vars in dictionary fields
        history[X].

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        self.reset()  # Initialize the state of the market
        for t in range(self.act_T):
            self.sow()  # Distribute aggregated information/state to agents
            self.cultivate()  # Agents take action
            self.reap()  # Collect individual data from agents
            self.mill()  # Process individual data into aggregate data
            self.store()  # Record variables of interest

    def update_dynamics(self):
        """
        Calculates a new "aggregate dynamic rule" using the history of variables
        named in track_vars, and distributes this rule to AgentTypes in agents.

        Parameters
        ----------
        none

        Returns
        -------
        dynamics : instance
            The new "aggregate dynamic rule" that agents believe in and act on.
            Should have attributes named in dyn_vars.
        """
        # Make a dictionary of inputs for the dynamics calculator
        # Convert self.history (list of dicts) to a pandas DataFrame
        history_df = pd.DataFrame(self.history)

        arg_names = list(get_arg_names(self.calc_dynamics))
        if "self" in arg_names:
            arg_names.remove("self")
        
        update_dict = {}
        for name in arg_names:
            if name in history_df:
                update_dict[name] = history_df[name] # Pass the pandas Series directly
            else:
                # This case should be handled based on how calc_dynamics expects missing data.
                # It might expect None, or raise an error. For now, we replicate previous behavior
                # of potentially raising KeyError if a tracked var was expected but not found.
                # However, history_df[name] will raise KeyError if name is not a column.
                # A more robust way could be: update_dict[name] = history_df.get(name)
                # which would pass None if the column is missing.
                # For now, direct access implies 'name' must be in history_df.
                pass # Or: raise KeyError(f"Tracked variable {name} not found in history_df for calc_dynamics.")

        # Calculate a new dynamic rule and distribute it to the agents in agent_list
        dynamics = self.calc_dynamics(**update_dict)  # User-defined dynamics calculator
        for var_name in self.dyn_vars:
            this_obj = getattr(dynamics, var_name)
            for this_type in self.agents:
                setattr(this_type, var_name, this_obj)
        return dynamics


def distribute_params(agent, param_name, param_count, distribution):
    """
    Distributes heterogeneous values of one parameter to the AgentTypes in self.agents.
    Parameters
    ----------
    agent: AgentType
        An agent to clone.
    param_name : string
        Name of the parameter to be assigned.
    param_count : int
        Number of different values the parameter will take on.
    distribution : Distribution
        A 1-D distribution.

    Returns
    -------
    agent_set : [AgentType]
        A list of param_count agents, ex ante heterogeneous with
        respect to param_name. The AgentCount of the original
        will be split between the agents of the returned
        list in proportion to the given distribution.
    """
    param_dist = distribution.discretize(N=param_count)

    agent_set = [deepcopy(agent) for i in range(param_count)]

    for j in range(param_count):
        agent_set[j].assign_parameters(
            **{"AgentCount": int(agent.AgentCount * param_dist.pmv[j])}
        )
        # agent_set[j].__dict__[param_name] = param_dist.atoms[j]

        agent_set[j].assign_parameters(**{param_name: param_dist.atoms[0, j]})

    return agent_set


@dataclass
class AgentPopulation:
    """
    A class for representing a population of ex-ante heterogeneous agents.
    """

    agent_type: AgentType  # type of agent in the population
    parameters: dict  # dictionary of parameters
    seed: int = 0  # random seed
    time_var: List[str] = field(init=False)
    time_inv: List[str] = field(init=False)
    distributed_params: List[str] = field(init=False)
    agent_type_count: Optional[int] = field(init=False)
    term_age: Optional[int] = field(init=False)
    continuous_distributions: Dict[str, Distribution] = field(init=False)
    discrete_distributions: Dict[str, Distribution] = field(init=False)
    population_parameters: List[Dict[str, Union[List[float], float]]] = field(
        init=False
    )
    agents: List[AgentType] = field(init=False)
    agent_database: pd.DataFrame = field(init=False)
    solution: List[Any] = field(init=False)

    def __post_init__(self):
        """
        Initialize the population of agents, determine distributed parameters,
        and infer `agent_type_count` and `term_age`.
        """
        # create a dummy agent and obtain its time-varying
        # and time-invariant attributes
        dummy_agent = self.agent_type()
        self.time_var = dummy_agent.time_vary
        self.time_inv = dummy_agent.time_inv

        # create list of distributed parameters
        # these are parameters that differ across agents
        self.distributed_params = [
            key
            for key, param in self.parameters.items()
            if (isinstance(param, list) and isinstance(param[0], list))
            or isinstance(param, Distribution)
            or (isinstance(param, DataArray) and param.dims[0] == "agent")
        ]

        self.__infer_counts__()

    def __infer_counts__(self):
        """
        Infer `agent_type_count` and `term_age` from the parameters.
        If parameters include a `Distribution` type, a list of lists,
        or a `DataArray` with `agent` as the first dimension, then
        the AgentPopulation contains ex-ante heterogenous agents.
        """

        # infer agent_type_count from distributed parameters
        agent_type_count = 1
        for key in self.distributed_params:
            param = self.parameters[key]
            if isinstance(param, Distribution):
                agent_type_count = None
                warn(
                    "Cannot infer agent_type_count from a Distribution. "
                    "Please provide approximation parameters."
                )
                break
            elif isinstance(param, list):
                agent_type_count = max(agent_type_count, len(param))
            elif isinstance(param, DataArray) and param.dims[0] == "agent":
                agent_type_count = max(agent_type_count, param.shape[0])

        self.agent_type_count = agent_type_count

        # infer term_age from all parameters
        term_age = 1
        for param in self.parameters.values():
            if isinstance(param, Distribution):
                term_age = None
                warn(
                    "Cannot infer term_age from a Distribution. "
                    "Please provide approximation parameters."
                )
                break
            elif isinstance(param, list) and isinstance(param[0], list):
                term_age = max(term_age, len(param[0]))
            elif isinstance(param, DataArray) and param.dims[-1] == "age":
                term_age = max(term_age, param.shape[-1])

        self.term_age = term_age

    def approx_distributions(self, approx_params: dict):
        """
        Approximate continuous distributions with discrete ones. If the initial
        parameters include a `Distribution` type, then the AgentPopulation is
        not ready to solve, and stands for an abstract population. To solve the
        AgentPopulation, we need discretization parameters for each continuous
        distribution. This method approximates the continuous distributions with
        discrete ones, and updates the parameters dictionary.
        """
        self.continuous_distributions = {}
        self.discrete_distributions = {}

        for key, args in approx_params.items():
            param = self.parameters[key]
            if key in self.distributed_params and isinstance(param, Distribution):
                self.continuous_distributions[key] = param
                self.discrete_distributions[key] = param.discretize(**args)
            else:
                raise ValueError(
                    f"Warning: parameter {key} is not a Distribution found "
                    f"in agent type {self.agent_type}"
                )

        if len(self.discrete_distributions) > 1:
            joint_dist = combine_indep_dstns(*self.discrete_distributions.values())
        else:
            joint_dist = list(self.discrete_distributions.values())[0]

        for i, key in enumerate(self.discrete_distributions):
            self.parameters[key] = DataArray(joint_dist.atoms[i], dims=("agent"))

        self.__infer_counts__()

    def __parse_parameters__(self) -> None:
        """
        Creates distributed dictionaries of parameters for each ex-ante
        heterogeneous agent in the parameterized population. The parameters
        are stored in a list of dictionaries, where each dictionary contains
        the parameters for one agent. Expands parameters that vary over time
        to a list of length `term_age`.
        """

        population_parameters = []  # container for dictionaries of each agent subgroup
        for agent in range(self.agent_type_count):
            agent_parameters = {}
            for key, param in self.parameters.items():
                if key in self.time_var:
                    # parameters that vary over time have to be repeated
                    if isinstance(param, (int, float)):
                        parameter_per_t = [param] * self.term_age
                    elif isinstance(param, list):
                        if isinstance(param[0], list):
                            parameter_per_t = param[agent]
                        else:
                            parameter_per_t = param
                    elif isinstance(param, DataArray):
                        if param.dims[0] == "agent":
                            if param.dims[-1] == "age":
                                parameter_per_t = param[agent].item()
                            else:
                                parameter_per_t = param.item()
                        elif param.dims[0] == "age":
                            parameter_per_t = param.item()

                    agent_parameters[key] = parameter_per_t

                elif key in self.time_inv:
                    if isinstance(param, (int, float)):
                        agent_parameters[key] = param
                    elif isinstance(param, list):
                        if isinstance(param[0], list):
                            agent_parameters[key] = param[agent]
                        else:
                            agent_parameters[key] = param
                    elif isinstance(param, DataArray) and param.dims[0] == "agent":
                        agent_parameters[key] = param[agent].item()

                else:
                    if isinstance(param, (int, float)):
                        agent_parameters[key] = param  # assume time inv
                    elif isinstance(param, list):
                        if isinstance(param[0], list):
                            agent_parameters[key] = param[agent]  # assume agent vary
                        else:
                            agent_parameters[key] = param  # assume time vary
                    elif isinstance(param, DataArray):
                        if param.dims[0] == "agent":
                            if param.dims[-1] == "age":
                                agent_parameters[key] = param[
                                    agent
                                ].item()  # assume agent vary
                            else:
                                agent_parameters[key] = param.item()  # assume time vary
                        elif param.dims[0] == "age":
                            agent_parameters[key] = param.item()  # assume time vary

            population_parameters.append(agent_parameters)

        self.population_parameters = population_parameters

    def create_distributed_agents(self):
        """
        Parses the parameters dictionary and creates a list of agents with the
        appropriate parameters. Also sets the seed for each agent.
        """

        self.__parse_parameters__()

        rng = np.random.default_rng(self.seed)

        self.agents = [
            self.agent_type(seed=rng.integers(0, 2**31 - 1), **agent_dict)
            for agent_dict in self.population_parameters
        ]

    def create_database(self):
        """
        Optionally creates a pandas DataFrame with the parameters for each agent.
        """
        database = pd.DataFrame(self.population_parameters)
        database["agents"] = self.agents

        self.agent_database = database

    def solve(self):
        """
        Solves each agent of the population serially.
        """

        # see Market class for an example of how to solve distributed agents in parallel

        for agent in self.agents:
            agent.solve()

    def unpack_solutions(self):
        """
        Unpacks the solutions of each agent into an attribute of the population.
        """
        self.solution = [agent.solution for agent in self.agents]

    def initialize_sim(self):
        """
        Initializes the simulation for each agent.
        """
        for agent in self.agents:
            agent.initialize_sim()

    def simulate(self):
        """
        Simulates each agent of the population serially.
        """
        for agent in self.agents:
            agent.simulate()

    def __iter__(self):
        """
        Allows for iteration over the agents in the population.
        """
        return iter(self.agents)

    def __getitem__(self, idx):
        """
        Allows for indexing into the population.
        """
        return self.agents[idx]
