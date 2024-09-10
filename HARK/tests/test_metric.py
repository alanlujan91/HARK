import pytest
import numpy as np
from HARK.metric import MetricObject, distance_metric


@pytest.fixture
def sample_data():
    return {
        "list_a": [1.0, 2.1, 3],
        "list_b": [3.1, 4, -1.4],
        "list_c": [8.6, 9],
        "dict_a": {"a": 1, "b": 2},
        "dict_b": {"a": 3, "b": 4},
        "dict_c": {"a": 5, "f": 6},
        "array_a": np.array([1, 2, 3]),
        "array_b": np.array([4, 5, 6]),
        "array_c": np.array([[1, 2], [3, 4]]),
    }


def test_distance_metric_numbers():
    assert distance_metric(1, 4) == 3
    assert distance_metric(1.5, -2.5) == 4.0


def test_distance_metric_lists(sample_data):
    assert distance_metric(
        sample_data["list_a"], sample_data["list_b"]
    ) == pytest.approx(4.4)
    assert distance_metric(sample_data["list_b"], sample_data["list_c"]) == 1.0
    assert distance_metric(sample_data["list_b"], sample_data["list_b"]) == 0.0


def test_distance_metric_arrays(sample_data):
    assert distance_metric(sample_data["array_a"], sample_data["array_b"]) == 3.0
    with pytest.warns(UserWarning):
        assert (
            distance_metric(sample_data["array_a"], sample_data["array_c"]) == 1.0
        )  # Changed from 2.0 to 1.0


def test_distance_metric_dicts(sample_data):
    assert distance_metric(sample_data["dict_a"], sample_data["dict_b"]) == 2.0
    with pytest.warns(UserWarning):
        assert distance_metric(sample_data["dict_a"], sample_data["dict_c"]) == 1000.0


def test_distance_metric_unsupported():
    with pytest.warns(UserWarning):
        assert distance_metric("a", "b") == 1000.0


def test_distance_metric_empty_structures():
    assert distance_metric([], []) == 0
    assert distance_metric({}, {}) == 0
    assert distance_metric(np.array([]), np.array([])) == 0


def test_distance_metric_nested_structures():
    nested_a = [{"a": 1, "b": [2, 3]}, {"c": 4}]
    nested_b = [{"a": 2, "b": [3, 4]}, {"c": 5}]
    assert distance_metric(nested_a, nested_b) == 1


@pytest.mark.parametrize("size", [100, 1000, 10000])
def test_distance_metric_performance(size):
    import time

    large_list_a = list(range(size))
    large_list_b = list(range(size, 2 * size))
    start_time = time.time()
    distance_metric(large_list_a, large_list_b)
    end_time = time.time()
    assert end_time - start_time < 1.0  # Adjust threshold as needed


def test_distance_metric_sets():
    assert distance_metric({1, 2, 3}, {2, 3, 4}) == pytest.approx(0.5)
    assert distance_metric({1, 2}, {3, 4}) == 1.0
    assert distance_metric(set(), set()) == 0.0
    assert distance_metric({1, 2}, {1, 2}) == 0.0


def test_distance_metric_tuples():
    assert distance_metric((1, 2, 3), (4, 5, 6)) == 3.0
    with pytest.warns(UserWarning):
        assert distance_metric((1, 2), (1, 2, 3)) == 1.0
    assert distance_metric((), ()) == 0.0
    assert distance_metric((1, 2), (1, 2)) == 0.0


def test_distance_metric_mixed_types():
    assert distance_metric([1, 2, 3], (1, 2, 3)) == 0.0
    assert distance_metric({1, 2, 3}, [1, 2, 3]) == 0.0
    with pytest.warns(UserWarning):
        assert distance_metric({1: "a", 2: "b"}, {1, 2}) == 1000.0


def test_distance_metric_warning_message():
    with pytest.warns(UserWarning, match="Dictionaries with different keys"):
        distance_metric({"a": 1}, {"b": 2})


class TestMetricObject:
    @pytest.fixture
    def metric_objects(self):
        obj_a, obj_b, obj_c = MetricObject(), MetricObject(), MetricObject()
        obj_a.distance_criteria = obj_b.distance_criteria = ["var_1", "var_2", "var_3"]
        obj_c.distance_criteria = ["var_5"]
        return obj_a, obj_b, obj_c

    def test_attribute_assignment(self, metric_objects):
        obj_a, obj_b, _ = metric_objects
        obj_a.var_1, obj_a.var_2, obj_a.var_3 = 0.1, 1, 2.1
        obj_b.var_1, obj_b.var_2, obj_b.var_3 = 1.8, -1, 0.1

        assert obj_a.var_1 == 0.1
        assert obj_b.var_2 == -1

    def test_distance_calculation(self, metric_objects):
        obj_a, obj_b, _ = metric_objects
        obj_a.var_1, obj_a.var_2, obj_a.var_3 = [0.1], [1, 2], [2.1]
        obj_b.var_1, obj_b.var_2, obj_b.var_3 = [1.8], [0, 0.1], [1.1]

        assert obj_a.distance(obj_b) == pytest.approx(1.9)

        obj_b.var_2 = [0, 0, 0.1]
        assert obj_a.distance(obj_b) == pytest.approx(1.7)

    def test_distance_same_object(self, metric_objects):
        obj_a, _, _ = metric_objects
        obj_a.var_1, obj_a.var_2, obj_a.var_3 = 1, 2, 3
        assert obj_a.distance(obj_a) == 0.0

    def test_distance_invalid_input(self, metric_objects):
        obj_a, _, _ = metric_objects
        with pytest.raises(TypeError):
            obj_a.distance("not a MetricObject")

    def test_distance_missing_criteria(self):
        obj = MetricObject()
        with pytest.raises(ValueError):
            obj.distance(MetricObject())

    def test_distance_missing_attribute(self, metric_objects):
        obj_a, obj_b, _ = metric_objects
        obj_a.var_1 = 1
        # obj_a.var_2 and obj_a.var_3 are not set
        obj_b.var_1, obj_b.var_2, obj_b.var_3 = 1, 2, 3

        with pytest.warns(UserWarning):
            assert obj_a.distance(obj_b) == 1000.0

    def test_distance_with_arrays(self, metric_objects):
        obj_a, obj_b, _ = metric_objects
        obj_a.var_1 = np.array([1, 2, 3])
        obj_b.var_1 = np.array([4, 5, 6])
        obj_a.var_2 = obj_b.var_2 = 0
        obj_a.var_3 = obj_b.var_3 = 0

        assert obj_a.distance(obj_b) == 3.0

    def test_distance_with_dicts(self, metric_objects):
        obj_a, obj_b, _ = metric_objects
        obj_a.var_1 = {"a": 1, "b": 2}
        obj_b.var_1 = {"a": 3, "b": 4}
        obj_a.var_2 = obj_b.var_2 = 0
        obj_a.var_3 = obj_b.var_3 = 0

        assert obj_a.distance(obj_b) == 2.0

    def test_distance_different_types(self):
        obj_a = MetricObject()
        obj_b = "not a MetricObject"
        with pytest.warns(UserWarning):
            assert distance_metric(obj_a, obj_b) == 1000.0

    def test_distance_same_type_no_criteria(self):
        obj_a = MetricObject()
        obj_b = MetricObject()
        with pytest.raises(ValueError):
            obj_a.distance(obj_b)

    def test_distance_same_type_with_criteria(self):
        class CustomMetricObject(MetricObject):
            distance_criteria = ["attr1", "attr2"]

        obj_a = CustomMetricObject()
        obj_b = CustomMetricObject()
        obj_a.attr1, obj_a.attr2 = 1, 2
        obj_b.attr1, obj_b.attr2 = 4, 6

        assert obj_a.distance(obj_b) == 4.0
