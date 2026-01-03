"""
Unit tests for panther.tuner.SkAutoTuner.Configs.ParamSpec

Covers:
- Categorical: empty choices error, normal creation
- Int: low>high, step<1, log with low<=0
- Float: low>high, step<=0, log with low<=0
- get_param_choices: expansion logic for Categorical, list, Int (small range vs large)
"""

import pytest

from panther.tuner.SkAutoTuner.Configs.ParamSpec import (
    Categorical,
    Float,
    Int,
    get_param_choices,
    is_param_spec,
    to_categorical,
)


# ============================================================================
# Categorical tests
# ============================================================================


class TestCategorical:
    def test_valid_int_choices(self):
        cat = Categorical([1, 2, 3])
        assert cat.choices == [1, 2, 3]

    def test_valid_string_choices(self):
        cat = Categorical(["relu", "gelu", "silu"])
        assert cat.choices == ["relu", "gelu", "silu"]

    def test_valid_bool_choices(self):
        cat = Categorical([True, False])
        assert cat.choices == [True, False]

    def test_tuple_converted_to_list(self):
        cat = Categorical((1, 2, 3))
        assert isinstance(cat.choices, list)
        assert cat.choices == [1, 2, 3]

    def test_empty_choices_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            Categorical([])

    def test_repr(self):
        cat = Categorical([1, 2])
        assert repr(cat) == "Categorical([1, 2])"


# ============================================================================
# Int tests
# ============================================================================


class TestInt:
    def test_valid_range(self):
        i = Int(1, 100)
        assert i.low == 1
        assert i.high == 100
        assert i.step == 1
        assert i.log is False

    def test_valid_with_step(self):
        i = Int(8, 512, step=8)
        assert i.step == 8

    def test_valid_with_log(self):
        i = Int(1, 1000, log=True)
        assert i.log is True

    def test_low_greater_than_high_raises(self):
        with pytest.raises(ValueError, match="low.*must be <= high"):
            Int(100, 1)

    def test_step_less_than_one_raises(self):
        with pytest.raises(ValueError, match="step.*must be >= 1"):
            Int(1, 10, step=0)

    def test_log_with_non_positive_low_raises(self):
        with pytest.raises(ValueError, match="log=True requires low > 0"):
            Int(0, 10, log=True)

    def test_repr_basic(self):
        i = Int(1, 10)
        assert repr(i) == "Int(1, 10)"

    def test_repr_with_step(self):
        i = Int(8, 64, step=8)
        assert repr(i) == "Int(8, 64, step=8)"

    def test_repr_with_log(self):
        i = Int(1, 100, log=True)
        assert repr(i) == "Int(1, 100, log=True)"


# ============================================================================
# Float tests
# ============================================================================


class TestFloat:
    def test_valid_range(self):
        f = Float(0.0, 1.0)
        assert f.low == 0.0
        assert f.high == 1.0
        assert f.step is None
        assert f.log is False

    def test_valid_with_step(self):
        f = Float(0.1, 1.0, step=0.1)
        assert f.step == 0.1

    def test_valid_log_scale(self):
        f = Float(1e-5, 1e-1, log=True)
        assert f.log is True

    def test_low_greater_than_high_raises(self):
        with pytest.raises(ValueError, match="low.*must be <= high"):
            Float(1.0, 0.5)

    def test_step_zero_raises(self):
        with pytest.raises(ValueError, match="step.*must be > 0"):
            Float(0.0, 1.0, step=0.0)

    def test_step_negative_raises(self):
        with pytest.raises(ValueError, match="step.*must be > 0"):
            Float(0.0, 1.0, step=-0.1)

    def test_log_with_non_positive_low_raises(self):
        with pytest.raises(ValueError, match="log=True requires low > 0"):
            Float(0.0, 1.0, log=True)

    def test_repr_basic(self):
        f = Float(0.0, 1.0)
        assert repr(f) == "Float(0.0, 1.0)"

    def test_repr_with_step(self):
        f = Float(0.0, 1.0, step=0.1)
        assert repr(f) == "Float(0.0, 1.0, step=0.1)"


# ============================================================================
# Helper function tests
# ============================================================================


class TestIsParamSpec:
    def test_categorical(self):
        assert is_param_spec(Categorical([1, 2]))

    def test_int(self):
        assert is_param_spec(Int(1, 10))

    def test_float(self):
        assert is_param_spec(Float(0.0, 1.0))

    def test_list(self):
        assert is_param_spec([1, 2, 3])

    def test_dict_is_not_param_spec(self):
        assert not is_param_spec({"a": 1})

    def test_string_is_not_param_spec(self):
        assert not is_param_spec("hello")


class TestToCategorical:
    def test_list_to_categorical(self):
        cat = to_categorical([1, 2, 3])
        assert isinstance(cat, Categorical)
        assert cat.choices == [1, 2, 3]

    def test_categorical_passthrough(self):
        original = Categorical([1, 2])
        result = to_categorical(original)
        assert result is original

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Cannot convert"):
            to_categorical("not a list")


class TestGetParamChoices:
    def test_categorical_returns_list(self):
        cat = Categorical([1, 2, 3])
        choices = get_param_choices(cat)
        assert choices == [1, 2, 3]

    def test_list_returns_list(self):
        choices = get_param_choices([4, 5, 6])
        assert choices == [4, 5, 6]

    def test_int_small_range_expands(self):
        # Int(1, 5) should expand to [1, 2, 3, 4, 5]
        i = Int(1, 5)
        choices = get_param_choices(i)
        assert choices == [1, 2, 3, 4, 5]

    def test_int_with_step_expands(self):
        # Int(8, 32, step=8) should expand to [8, 16, 24, 32]
        i = Int(8, 32, step=8)
        choices = get_param_choices(i)
        assert choices == [8, 16, 24, 32]

    def test_int_large_range_returns_none(self):
        # Range > 100 values should return None
        i = Int(1, 1000)
        choices = get_param_choices(i)
        assert choices is None

    def test_float_returns_none(self):
        # Float is continuous, should return None
        f = Float(0.0, 1.0)
        choices = get_param_choices(f)
        assert choices is None
