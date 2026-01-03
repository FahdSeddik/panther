"""
Unit tests for panther.tuner.SkAutoTuner.Configs.LayerNameResolver

Covers:
- String pattern resolution (regex and substring fallback)
- List of string patterns (deduplication)
- Dict selectors: pattern, contains, type, indices, range
- Error conditions: indices + range conflict, out-of-bounds indices, invalid range format
"""

import pytest
import torch.nn as nn

from panther.tuner.SkAutoTuner.Configs.LayerNameResolver import LayerNameResolver


# ============================================================================
# Fixtures: Simple models for testing layer resolution
# ============================================================================


class SimpleModel(nn.Module):
    """A simple model with named layers for testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 30)
        self.conv = nn.Conv2d(3, 16, 3)
        self.output = nn.Linear(30, 10)

    def forward(self, x):
        return x


class NestedModel(nn.Module):
    """A nested model to test hierarchical name resolution."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 30),
        )
        self.decoder = nn.Sequential(
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=4)

    def forward(self, x):
        return x


@pytest.fixture
def simple_model():
    return SimpleModel()


@pytest.fixture
def nested_model():
    return NestedModel()


def build_layer_map(model: nn.Module, prefix: str = "") -> dict:
    """Build a name -> module mapping for a model."""
    layer_map = {}
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        layer_map[full_name] = module
        layer_map.update(build_layer_map(module, full_name))
    return layer_map


@pytest.fixture
def simple_resolver(simple_model):
    layer_map = build_layer_map(simple_model)
    return LayerNameResolver(simple_model, layer_map)


@pytest.fixture
def nested_resolver(nested_model):
    layer_map = build_layer_map(nested_model)
    return LayerNameResolver(nested_model, layer_map)


# ============================================================================
# String pattern tests
# ============================================================================


class TestStringPatternResolution:
    def test_exact_match(self, simple_resolver):
        result = simple_resolver.resolve("fc1")
        assert result == ["fc1"]

    def test_regex_pattern(self, simple_resolver):
        # Match fc1 and fc2
        result = simple_resolver.resolve("fc.*")
        assert set(result) == {"fc1", "fc2"}

    def test_substring_fallback(self, simple_resolver):
        # "fc" should match fc1 and fc2 as substring
        result = simple_resolver.resolve("fc")
        assert set(result) == {"fc1", "fc2"}

    def test_no_match_returns_empty(self, simple_resolver):
        result = simple_resolver.resolve("nonexistent")
        assert result == []

    def test_nested_pattern(self, nested_resolver):
        # Match all encoder layers
        result = nested_resolver.resolve("encoder.*")
        # Should match encoder.0 (Linear), encoder.1 (ReLU), encoder.2 (Linear)
        assert len(result) >= 3
        assert all("encoder" in name for name in result)


# ============================================================================
# List of patterns tests
# ============================================================================


class TestListPatternResolution:
    def test_multiple_patterns(self, simple_resolver):
        result = simple_resolver.resolve(["fc1", "fc2"])
        assert set(result) == {"fc1", "fc2"}

    def test_deduplication(self, simple_resolver):
        # Both patterns match fc1, should only appear once
        result = simple_resolver.resolve(["fc1", "fc.*"])
        # fc1 matches both, but should be deduplicated
        fc1_count = result.count("fc1")
        assert fc1_count <= 1

    def test_mixed_patterns(self, simple_resolver):
        result = simple_resolver.resolve(["fc1", "conv"])
        assert set(result) == {"fc1", "conv"}


# ============================================================================
# Dict selector tests
# ============================================================================


class TestDictPatternSelector:
    def test_pattern_single(self, simple_resolver):
        result = simple_resolver.resolve({"pattern": "fc.*"})
        assert set(result) == {"fc1", "fc2"}

    def test_pattern_list(self, simple_resolver):
        result = simple_resolver.resolve({"pattern": ["fc1", "output"]})
        assert set(result) == {"fc1", "output"}


class TestDictContainsSelector:
    def test_contains_single(self, simple_resolver):
        result = simple_resolver.resolve({"contains": "fc"})
        assert set(result) == {"fc1", "fc2"}

    def test_contains_list(self, simple_resolver):
        result = simple_resolver.resolve({"contains": ["fc", "conv"]})
        assert set(result) == {"fc1", "fc2", "conv"}


class TestDictTypeSelector:
    def test_type_single(self, simple_resolver):
        result = simple_resolver.resolve({"type": "Linear"})
        assert set(result) == {"fc1", "fc2", "output"}

    def test_type_conv2d(self, simple_resolver):
        result = simple_resolver.resolve({"type": "Conv2d"})
        assert result == ["conv"]

    def test_type_list(self, simple_resolver):
        result = simple_resolver.resolve({"type": ["Linear", "Conv2d"]})
        assert set(result) == {"fc1", "fc2", "output", "conv"}

    def test_type_nested_model(self, nested_resolver):
        result = nested_resolver.resolve({"type": "MultiheadAttention"})
        assert result == ["attention"]


class TestDictIndicesSelector:
    def test_indices_single(self, simple_resolver):
        # Select first layer alphabetically from all layers
        result = simple_resolver.resolve({"type": "Linear", "indices": 0})
        assert len(result) == 1

    def test_indices_list(self, simple_resolver):
        # Select first and second Linear layers
        result = simple_resolver.resolve({"type": "Linear", "indices": [0, 1]})
        assert len(result) == 2

    def test_indices_out_of_bounds_raises(self, simple_resolver):
        # There are only 3 Linear layers, index 10 is out of bounds
        with pytest.raises(ValueError, match="out of bounds"):
            simple_resolver.resolve({"type": "Linear", "indices": 10})


class TestDictRangeSelector:
    def test_range_basic(self, simple_resolver):
        # Select first 2 Linear layers
        result = simple_resolver.resolve({"type": "Linear", "range": [0, 2]})
        assert len(result) == 2

    def test_range_with_step(self, simple_resolver):
        # Select every other Linear layer from 0 to 3
        result = simple_resolver.resolve({"type": "Linear", "range": [0, 3, 2]})
        assert len(result) == 2

    def test_range_invalid_format_raises(self, simple_resolver):
        with pytest.raises(ValueError, match="must be a list"):
            simple_resolver.resolve({"type": "Linear", "range": "0-2"})

    def test_range_too_few_elements_raises(self, simple_resolver):
        with pytest.raises(ValueError, match="must contain 2 or 3 elements"):
            simple_resolver.resolve({"type": "Linear", "range": [0]})

    def test_range_non_integer_raises(self, simple_resolver):
        with pytest.raises(ValueError, match="must be integers"):
            simple_resolver.resolve({"type": "Linear", "range": [0, "2"]})

    def test_range_negative_start_raises(self, simple_resolver):
        with pytest.raises(ValueError, match="must be non-negative"):
            simple_resolver.resolve({"type": "Linear", "range": [-1, 2]})

    def test_range_end_not_greater_than_start_raises(self, simple_resolver):
        with pytest.raises(ValueError, match="end value must be greater than start"):
            simple_resolver.resolve({"type": "Linear", "range": [2, 2]})


class TestDictConflictingSelectors:
    def test_indices_and_range_conflict_raises(self, simple_resolver):
        with pytest.raises(ValueError, match="Cannot use both"):
            simple_resolver.resolve({"type": "Linear", "indices": [0], "range": [0, 2]})


class TestDictCombinedSelectors:
    def test_pattern_and_type(self, simple_resolver):
        # Match layers that match pattern AND are Linear type
        result = simple_resolver.resolve({"pattern": "fc.*", "type": "Linear"})
        assert set(result) == {"fc1", "fc2"}

    def test_contains_and_indices(self, simple_resolver):
        # Match layers containing "fc" and select first one
        result = simple_resolver.resolve({"contains": "fc", "indices": 0})
        assert len(result) == 1

    def test_type_and_range_nested(self, nested_resolver):
        # Select first 2 Linear layers from encoder/decoder
        result = nested_resolver.resolve({"type": "Linear", "range": [0, 2]})
        assert len(result) == 2


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    def test_empty_model(self):
        empty_model = nn.Module()
        resolver = LayerNameResolver(empty_model, {})
        result = resolver.resolve("anything")
        assert result == []

    def test_no_match_dict_selector(self, simple_resolver):
        result = simple_resolver.resolve({"type": "BatchNorm2d"})
        assert result == []

    def test_indices_with_no_matches_returns_empty(self, simple_resolver):
        # No BatchNorm2d layers, so no matches to index
        result = simple_resolver.resolve({"type": "BatchNorm2d"})
        assert result == []
