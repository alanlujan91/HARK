from warnings import warn

import numpy as np


def distance_lists(list_a, list_b):
    """
    If both inputs are lists, then the distance between
    them is the maximum distance between corresponding
    elements in the lists.  If they differ in length,
    the distance is the difference in lengths.
    """
    len_a = len(list_a)
    len_b = len(list_b)
    if len_a == len_b:
        return np.max([distance_metric(list_a[n], list_b[n]) for n in range(len_a)])
    warn("Objects of different lengths. Returning difference in lengths.")
    return np.abs(len_a - len_b)


def distance_dicts(dict_a, dict_b):
    """
    If both inputs are dictionaries, call distance on the list of its elements
    If keys don't match, print a warning.
    If they have different lengths, log a warning and return the
    difference in lengths.
    """
    len_a = len(dict_a)
    len_b = len(dict_b)

    if len_a == len_b:
        if set(dict_a.keys()) != set(dict_b.keys()):
            warn("Dictionaries with keys that do not match are being compared.")
            return 1000.0
        return np.max(
            [distance_metric(dict_a[key], dict_b[key]) for key in dict_a.keys()]
        )
    warn("Objects of different lengths. Returning difference in lengths.")
    return np.abs(len_a - len_b)


def distance_arrays(arr_a, arr_b):
    """
    If both inputs are array-like, return the maximum absolute difference b/w
    corresponding elements (if same shape); return largest difference in dimensions
    if shapes do not align.
    Flatten arrays so they have the same dimensions
    """

    if arr_a.shape == arr_b.shape:
        return np.max(np.abs(arr_a - arr_b))
    warn("Arrays of different shapes. Returning differences in size.")
    return np.abs(arr_a.size - arr_b.size)


def distance_class(cls_a, cls_b):
    """
    If none of the above cases, but the objects are of the same class,
    call the distance method of one on the other
    """
    if isinstance(cls_a, type(lambda: None)):
        warn("Cannot compare functions. Returning large distance.")
        return 1000.0
    return cls_a.distance(cls_b)


def distance_metric(thing_a, thing_b):
    """
    A "universal distance" metric that can be used as a default in many settings.

    Parameters
    ----------
    thing_a : object
        A generic object.
    thing_b : object
        Another generic object.

    Returns:
    ------------
    distance : float
        The "distance" between thing_a and thing_b.
    """

    # If both inputs are numbers, return their difference
    if isinstance(thing_a, (int, float)) and isinstance(thing_b, (int, float)):
        return np.abs(thing_a - thing_b)

    if isinstance(thing_a, list) and isinstance(thing_b, list):
        return distance_lists(thing_a, thing_b)

    if isinstance(thing_a, np.ndarray) and isinstance(thing_b, np.ndarray):
        return distance_arrays(thing_a, thing_b)

    if isinstance(thing_a, dict) and isinstance(thing_b, dict):
        return distance_dicts(thing_a, thing_b)

    if isinstance(thing_a, type(thing_b)):
        return distance_class(thing_a, thing_b)

    # Failsafe: the inputs are very far apart
    return 1000.0


class MetricObject:
    """
    A superclass for object classes in HARK.  Comes with two useful methods:
    a generic/universal distance method and an attribute assignment method.
    """

import logging # Import logging module

# Configure a specific logger for distance messages, if not already configured elsewhere
# This basicConfig is a fallback; typically, HARK's main entry point would configure logging.
# logging.basicConfig(level=logging.INFO) # Basic configuration if needed
hark_distance_logger = logging.getLogger("HARK.distance")


class MetricObject:
    """
    A superclass for object classes in HARK.  Comes with two useful methods:
    a generic/universal distance method and an attribute assignment method.
    """

    distance_criteria = []  # This should be overwritten by subclasses.

    def distance(self, other, metric_name: str = None, report_metric_flag: bool = False):
        """
        A generic distance method.
        Checks for a `distance_override` method on `self` first.
        If `metric_name` is provided, it uses that as the sole criterion.
        Otherwise, it uses self.distance_criteria (a list of strings naming attributes).

        Parameters
        ----------
        other : object
            Another object to compare this instance to.
        metric_name : str, optional
            A specific attribute name or a predefined metric keyword
            (e.g., "cFunc", "vFunc", "mNrmStE", "mNrmTrg") to use for
            distance calculation. If None, uses the class's distance_criteria.
        report_metric_flag : bool, optional
            If True, logs the metric determination path.

        Returns
        -------
        (unnamed) : float
            The distance between this object and another.
        """
        log_message_detail = None # Initialize for potential logging

        # Check for a distance_override method on the instance first.
        if hasattr(self, "distance_override") and callable(self.distance_override):
            if report_metric_flag:
                log_message_detail = f"Using custom `distance_override` method for {type(self).__name__}."
                hark_distance_logger.info(f"HARK.distance: {log_message_detail}")
            return self.distance_override(other)

        try:
            if metric_name is not None:
                if metric_name == "cFunc":
                    if report_metric_flag:
                        log_message_detail = f"Comparing solutions using specific metric_name: '{metric_name}' for {type(self).__name__}."
                    if hasattr(self, 'cFunc') and hasattr(other, 'cFunc'):
                        result = distance_metric(self.cFunc, other.cFunc)
                    else:
                        warn(f"Metric '{metric_name}' requested, but one or both objects lack 'cFunc' attribute. Returning large distance.")
                        result = float('inf')
                elif metric_name == "vFunc":
                    if report_metric_flag:
                        log_message_detail = f"Comparing solutions using specific metric_name: '{metric_name}' for {type(self).__name__}."
                    if hasattr(self, 'vFunc') and hasattr(other, 'vFunc'):
                        result = distance_metric(self.vFunc, other.vFunc)
                    else:
                        warn(f"Metric '{metric_name}' requested, but one or both objects lack 'vFunc' attribute. Returning large distance.")
                        result = float('inf')
                elif metric_name == "mNrmStE":
                    if report_metric_flag:
                        log_message_detail = f"Comparing solutions using specific metric_name: '{metric_name}' for {type(self).__name__}."
                    self_val = getattr(self, 'mNrmStE', None)
                    other_val = getattr(other, 'mNrmStE', None)
                    if self_val is not None and other_val is not None and isinstance(self_val, (int, float)) and isinstance(other_val, (int, float)):
                        result = abs(self_val - other_val)
                    else:
                        warn(f"Metric '{metric_name}' requested, but 'mNrmStE' is None, missing, or not a number on one or both objects. Returning large distance.")
                        result = float('inf')
                elif metric_name == "mNrmTrg":
                    if report_metric_flag:
                        log_message_detail = f"Comparing solutions using specific metric_name: '{metric_name}' for {type(self).__name__}."
                    self_val = getattr(self, 'mNrmTrg', None)
                    other_val = getattr(other, 'mNrmTrg', None)
                    if self_val is not None and other_val is not None and isinstance(self_val, (int, float)) and isinstance(other_val, (int, float)):
                        result = abs(self_val - other_val)
                    else:
                        warn(f"Metric '{metric_name}' requested, but 'mNrmTrg' is None, missing, or not a number on one or both objects. Returning large distance.")
                        result = float('inf')
                else:
                    # Fallback to treating metric_name as a direct attribute name
                    criteria_to_use = [metric_name]
                    if report_metric_flag:
                        log_message_detail = f"Comparing solutions using attribute specified by metric_name: '{metric_name}' for {type(self).__name__}."
                    result = np.max(
                        [
                            distance_metric(getattr(self, attr_name), getattr(other, attr_name))
                            for attr_name in criteria_to_use
                        ]
                    )
            else:
                # Fallback to the class-defined distance_criteria
                criteria_to_use = self.distance_criteria
                if not criteria_to_use and hasattr(self, 'distance_criteria') and self.distance_criteria:
                     criteria_to_use = self.distance_criteria
                
                if not criteria_to_use:
                     warn("No distance criteria specified or found. Returning large distance.")
                     result = 1000.0
                else:
                    if report_metric_flag:
                        criteria_str = str(criteria_to_use)
                        log_message_detail = f"Comparing solutions using default distance_criteria: {criteria_str} from {type(self).__name__}."
                    result = np.max(
                        [
                            distance_metric(getattr(self, attr_name), getattr(other, attr_name))
                            for attr_name in criteria_to_use
                        ]
                    )

            if report_metric_flag and log_message_detail:
                hark_distance_logger.info(f"HARK.distance: {log_message_detail}")
            return result

        except (AttributeError, ValueError, TypeError) as e: # Catching TypeError as well
            warn(f"Error during distance calculation for metric '{metric_name}': {e}. Returning large distance.")
            if report_metric_flag: # Log error if reporting is on
                hark_distance_logger.error(f"HARK.distance: Error during distance calculation for metric '{metric_name}' on {type(self).__name__}: {e}")
            return float('inf')

    def describe_distance_metric(self, metric_name: str = None, other_for_type_ref=None, _indent_level: int = 0) -> str:
        indent = "  " * _indent_level
        description_lines = []

        # 1. Check for distance_override
        if hasattr(self, "distance_override") and callable(self.distance_override):
            description_lines.append(f"{indent}Distance calculation is overridden by custom `self.distance_override` method for {type(self).__name__}.")
            return "\n".join(description_lines)

        # 2. Handle specific metric_name keywords or determine criteria_to_use
        criteria_to_use = None
        if metric_name is not None:
            if metric_name == "cFunc":
                description_lines.append(f"{indent}Distance will be calculated by comparing the 'cFunc' attributes of the two solution objects using `distance_metric`.")
                # Optionally, inspect self.cFunc type
                self_cfunc_type = type(getattr(self, 'cFunc', None)).__name__
                description_lines.append(f"{indent}  - Self.cFunc type: {self_cfunc_type}")
                other_cfunc_type = "N/A"
                if other_for_type_ref and hasattr(other_for_type_ref, 'cFunc'):
                    other_cfunc_type = type(getattr(other_for_type_ref, 'cFunc', None)).__name__
                description_lines.append(f"{indent}  - Other.cFunc type for comparison: {other_cfunc_type}")
                return "\n".join(description_lines)
            elif metric_name == "vFunc":
                description_lines.append(f"{indent}Distance will be calculated by comparing the 'vFunc' attributes of the two solution objects using `distance_metric`.")
                self_vfunc_type = type(getattr(self, 'vFunc', None)).__name__
                description_lines.append(f"{indent}  - Self.vFunc type: {self_vfunc_type}")
                other_vfunc_type = "N/A"
                if other_for_type_ref and hasattr(other_for_type_ref, 'vFunc'):
                    other_vfunc_type = type(getattr(other_for_type_ref, 'vFunc', None)).__name__
                description_lines.append(f"{indent}  - Other.vFunc type for comparison: {other_vfunc_type}")
                return "\n".join(description_lines)
            elif metric_name == "mNrmStE":
                description_lines.append(f"{indent}Distance will be calculated as the absolute difference between the 'mNrmStE' (numeric) attributes.")
                return "\n".join(description_lines)
            elif metric_name == "mNrmTrg":
                description_lines.append(f"{indent}Distance will be calculated as the absolute difference between the 'mNrmTrg' (numeric) attributes.")
                return "\n".join(description_lines)
            else:
                # metric_name is provided but not a special keyword
                criteria_to_use = [metric_name]
                description_lines.append(f"{indent}Distance will be calculated based on the single specified criterion: '{metric_name}'.")
        else:
            # metric_name is None, use self.distance_criteria
            # Ensure distance_criteria is a list, even if empty
            criteria_to_use = getattr(self, 'distance_criteria', [])
            if not isinstance(criteria_to_use, list): # Should not happen if properly defined
                criteria_to_use = []

            if not criteria_to_use :
                description_lines.append(f"{indent}No default `distance_criteria` specified for {type(self).__name__}. Comparison would likely result in a large distance or error if not overridden.")
                return "\n".join(description_lines)
            
            description_lines.append(f"{indent}Distance will be calculated based on the default `distance_criteria` of {type(self).__name__}: {criteria_to_use}.")

        # 3. Describe comparison for each criterion in criteria_to_use
        if criteria_to_use is None: # Should have been handled by specific metric name returns
             description_lines.append(f"{indent}Error in logic: criteria_to_use is None unexpectedly.")
             return "\n".join(description_lines)

        for attr_name in criteria_to_use:
            attr_value_self = getattr(self, attr_name, None)
            attr_value_other = getattr(other_for_type_ref, attr_name, None) if other_for_type_ref else None
            
            type_self_str = type(attr_value_self).__name__
            type_other_str = type(attr_value_other).__name__ if other_for_type_ref else "N/A"

            desc_line = f"{indent}- Attribute '{attr_name}' (self type: {type_self_str}, other type: {type_other_str}):"
            
            if isinstance(attr_value_self, MetricObject):
                # Pass other_for_type_ref's attribute for more accurate recursive description
                sub_desc = attr_value_self.describe_distance_metric(metric_name=None, other_for_type_ref=attr_value_other, _indent_level=_indent_level + 1)
                desc_line += f"\n{indent}  Recursively described as:\n{sub_desc}"
            elif isinstance(attr_value_self, np.ndarray):
                desc_line += f" Compared as numerical arrays (max absolute difference if shapes match)."
            elif isinstance(attr_value_self, (float, int)):
                desc_line += f" Compared as numerical values (absolute difference)."
            elif isinstance(attr_value_self, list):
                desc_line += f" Compared as lists (element-wise, max distance of corresponding elements)."
                if attr_value_self and isinstance(attr_value_self[0], MetricObject) and \
                   other_for_type_ref and isinstance(attr_value_other, list) and attr_value_other and isinstance(attr_value_other[0], MetricObject):
                     sub_desc = attr_value_self[0].describe_distance_metric(metric_name=None, other_for_type_ref=attr_value_other[0], _indent_level=_indent_level + 1)
                     desc_line += f"\n{indent}  Element e.g., first element (MetricObject) described as:\n{sub_desc}"
                elif attr_value_self and not isinstance(attr_value_self[0], MetricObject):
                     desc_line += f" Element type: {type(attr_value_self[0]).__name__}."
            elif isinstance(attr_value_self, dict):
                desc_line += f" Compared as dictionaries (value-wise by key, max distance of corresponding values)."
                if attr_value_self:
                    first_key = next(iter(attr_value_self.keys()), None)
                    if first_key is not None:
                        first_val_self = attr_value_self[first_key]
                        first_val_other = attr_value_other.get(first_key) if isinstance(attr_value_other, dict) else None
                        if isinstance(first_val_self, MetricObject):
                            sub_desc = first_val_self.describe_distance_metric(metric_name=None, other_for_type_ref=first_val_other, _indent_level=_indent_level + 1)
                            desc_line += f"\n{indent}  Value e.g., for key '{first_key}' (MetricObject) described as:\n{sub_desc}"
                        else:
                            desc_line += f" Value type for key '{first_key}': {type(first_val_self).__name__}."
            elif callable(attr_value_self) and not isinstance(attr_value_self, type): # Check if it's a function/method but not a class
                if hasattr(attr_value_self, '__name__') and 'NullFunc' in attr_value_self.__name__:
                     desc_line += " This is a NullFunc; comparison depends on the other object's attribute type."
                else:
                     desc_line += " Compared as functions/callables. `distance_metric` typically returns a large distance (1000.0) unless a specific class handler exists or they are identical."
            elif attr_value_self is None:
                desc_line += " Attribute is None on self."
            else: # Default catch-all
                desc_line += f" Comparison via `distance_metric` generic rules for type {type_self_str}."
                if hasattr(attr_value_self, 'distance') and callable(getattr(attr_value_self, 'distance')):
                    desc_line += " This type has its own `.distance()` method, which would be called by `distance_class`."

            description_lines.append(desc_line)

        return "\n".join(description_lines)
