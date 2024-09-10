"""
HARK Metric Module
==================

This module provides utilities for calculating distances between various types of objects
in the HARK (Heterogeneous Agents Resources and toolKit) framework. It includes a universal
distance metric function and a base class for objects that can calculate distances between
instances.

Key Components:
- distance_metric: A singledispatch function for calculating distances between objects.
- MetricObject: A base class for objects that can calculate distances between instances.

The module supports distance calculations for numbers, lists, dictionaries, numpy arrays,
and custom objects that inherit from MetricObject.

Usage:
    from HARK.metric import distance_metric, MetricObject

    # Calculate distance between numbers
    dist = distance_metric(1, 4)  # Returns 3

    # Calculate distance between lists
    dist = distance_metric([1, 2, 3], [4, 5, 6])  # Returns 3

    # Create custom objects with distance calculation
    class MyObject(MetricObject):
        distance_criteria = ['attr1', 'attr2']

    obj1, obj2 = MyObject(), MyObject()
    obj1.attr1, obj1.attr2 = 1, 2
    obj2.attr1, obj2.attr2 = 4, 6
    dist = obj1.distance(obj2)  # Returns 4
"""

from warnings import warn
from typing import Any, List, Dict, Set, Tuple, Union, TypeVar
import numpy as np
from functools import singledispatch

# Type aliases for clarity
Number = Union[int, float]
Array = np.ndarray
T = TypeVar("T")


@singledispatch
def distance_metric(thing_a: Any, thing_b: Any) -> float:
    """A "universal distance" metric that can be used as a default in many settings."""
    if isinstance(thing_a, type) and isinstance(thing_b, type):
        return distance_class(thing_a, thing_b)
    warn(f"No specific distance metric for types {type(thing_a)} and {type(thing_b)}.")
    return 1000.0


@distance_metric.register(int)
@distance_metric.register(float)
def distance_numbers(a: Number, b: Number) -> float:
    """Calculate the distance between two numbers."""
    return abs(a - b)


@distance_metric.register(list)
def distance_lists(list_a: List[T], list_b: List[T]) -> float:
    """Calculate the distance between two lists."""
    len_a, len_b = len(list_a), len(list_b)
    if len_a == len_b:
        return max((distance_metric(a, b) for a, b in zip(list_a, list_b)), default=0.0)
    warn(
        f"Lists of different lengths: {len_a} and {len_b}. Returning difference in lengths."
    )
    return abs(len_a - len_b)


@distance_metric.register(dict)
def distance_dicts(dict_a: Dict[Any, T], dict_b: Dict[Any, T]) -> float:
    """Calculate the distance between two dictionaries."""
    if not isinstance(dict_a, dict) or not isinstance(dict_b, dict):
        warn(f"Incompatible types: {type(dict_a)} and {type(dict_b)}")
        return 1000.0
    keys_a, keys_b = set(dict_a.keys()), set(dict_b.keys())
    if keys_a != keys_b:
        warn(f"Dictionaries with different keys: {keys_a.symmetric_difference(keys_b)}")
        return 1000.0
    return max(
        (distance_metric(dict_a[key], dict_b[key]) for key in keys_a), default=0.0
    )


@distance_metric.register(np.ndarray)
def distance_arrays(arr_a: Array, arr_b: Array) -> float:
    """Calculate the distance between two numpy arrays."""
    if arr_a.shape != arr_b.shape:
        warn(f"Arrays of different shapes: {arr_a.shape} and {arr_b.shape}")
        return float(abs(arr_a.size - arr_b.size))
    if arr_a.size == 0 and arr_b.size == 0:
        return 0.0
    return float(np.max(np.abs(arr_a - arr_b)))


@distance_metric.register(set)
def distance_sets(set_a: Set[T], set_b: Set[T]) -> float:
    """
    Calculate the distance between two sets using the Jaccard distance.

    The Jaccard distance is defined as 1 - (size of intersection / size of union).
    If both sets are empty, the distance is 0.

    Parameters:
    -----------
    set_a : Set[T]
        The first set to compare.
    set_b : Set[T]
        The second set to compare.

    Returns:
    --------
    float
        The Jaccard distance between the two sets.

    Examples:
    ---------
    >>> distance_sets({1, 2, 3}, {2, 3, 4})
    0.5
    >>> distance_sets({1, 2}, {3, 4})
    1.0
    >>> distance_sets(set(), set())
    0.0
    """
    union = set_a.union(set_b)
    if not union:
        return 0.0
    intersection = set_a.intersection(set_b)
    return 1 - len(intersection) / len(union)


@distance_metric.register(tuple)
def distance_tuples(tuple_a: Tuple[Any, ...], tuple_b: Tuple[Any, ...]) -> float:
    """
    Calculate the distance between two tuples.

    If the tuples have the same length, return the maximum distance between
    corresponding elements. Otherwise, return the difference in lengths.

    Parameters:
    -----------
    tuple_a : Tuple[Any, ...]
        The first tuple to compare.
    tuple_b : Tuple[Any, ...]
        The second tuple to compare.

    Returns:
    --------
    float
        The calculated distance between the tuples.

    Examples:
    ---------
    >>> distance_tuples((1, 2, 3), (4, 5, 6))
    3.0
    >>> distance_tuples((1, 2), (1, 2, 3))
    1.0
    >>> distance_tuples((), ())
    0.0
    """
    len_a, len_b = len(tuple_a), len(tuple_b)
    if len_a == len_b:
        return max(
            (distance_metric(a, b) for a, b in zip(tuple_a, tuple_b)), default=0.0
        )
    warn(
        f"Tuples of different lengths: {len_a} and {len_b}. Returning difference in lengths."
    )
    return abs(len_a - len_b)


class MetricObject:
    """A superclass for object classes in HARK with distance calculation capabilities."""

    distance_criteria: List[str] = []  # Ensure this is set in subclasses

    def distance(self, other: Any) -> float:
        """Calculate the distance between this instance and another instance or object."""
        if not isinstance(other, MetricObject):
            warn(f"Cannot compare MetricObject with {type(other)}")
            return 1000.0
        if not self.distance_criteria:  # Ensure this is defined
            warn("distance_criteria must be set for distance calculation")
            return 1000.0

        max_distance = 0.0
        for attr_name in self.distance_criteria:
            try:
                attr_a = getattr(self, attr_name)  # Access instance attribute
                attr_b = getattr(other, attr_name)  # Access instance attribute
                attr_distance = distance_metric(attr_a, attr_b)
                max_distance = max(max_distance, attr_distance)
            except AttributeError as e:
                warn(f"AttributeError in distance calculation: {e}")
                return 1000.0

        return max_distance


@distance_metric.register(MetricObject)
def distance_class(cls_a: Any, cls_b: Any) -> float:
    """Calculate the distance between two objects of the same class."""
    if not isinstance(cls_a, type(cls_b)):
        warn(
            f"Cannot compare objects of different types: {type(cls_a)} and {type(cls_b)}"
        )
        return 1000.0
    if isinstance(cls_a, type):
        if hasattr(cls_a, "distance") and callable(getattr(cls_a, "distance")):
            return cls_a.distance(cls_b)
        warn(f"{cls_a.__name__} does not have a callable 'distance' class method.")
        return 1000.0
    if not hasattr(cls_a, "distance") or not callable(getattr(cls_a, "distance")):
        warn(f"{type(cls_a).__name__} does not have a callable 'distance' method.")
        return 1000.0
    return cls_a.distance(cls_b)
