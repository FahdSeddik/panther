"""
Unit tests for panther.tuner.SkAutoTuner.Visualizer.ModelVisualizer

Covers:
- print_module_tree with simple models
- print_module_tree with nested/sequential models
- print_module_tree with empty models
- Output format verification
"""

import io
import sys

import pytest
import torch.nn as nn

from panther.tuner.SkAutoTuner.Visualizer.ModelVisualizer import ModelVisualizer


# ============================================================================
# Fixtures: Test models
# ============================================================================


class SimpleModel(nn.Module):
    """A simple model with a few named layers."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class NestedModel(nn.Module):
    """A nested model with hierarchical structure."""

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

    def forward(self, x):
        return self.decoder(self.encoder(x))


class DeeplyNestedModel(nn.Module):
    """A deeply nested model to test tree rendering."""

    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Sequential(
                nn.Linear(10, 10),
                nn.BatchNorm1d(10),
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class EmptyModel(nn.Module):
    """An empty model with no child modules."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class ModelWithModuleList(nn.Module):
    """A model using ModuleList."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@pytest.fixture
def simple_model():
    return SimpleModel()


@pytest.fixture
def nested_model():
    return NestedModel()


@pytest.fixture
def deeply_nested_model():
    return DeeplyNestedModel()


@pytest.fixture
def empty_model():
    return EmptyModel()


@pytest.fixture
def model_with_module_list():
    return ModelWithModuleList()


# ============================================================================
# Helper to capture stdout
# ============================================================================


def capture_print_output(func, *args, **kwargs):
    """Capture stdout from a function call."""
    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        func(*args, **kwargs)
    finally:
        sys.stdout = old_stdout
    return captured.getvalue()


# ============================================================================
# Tests for print_module_tree
# ============================================================================


class TestPrintModuleTreeBasic:
    def test_simple_model_output_contains_layer_names(self, simple_model):
        output = capture_print_output(ModelVisualizer.print_module_tree, simple_model)

        # Should contain the layer names
        assert "fc1" in output
        assert "relu" in output
        assert "fc2" in output

    def test_simple_model_output_contains_layer_types(self, simple_model):
        output = capture_print_output(ModelVisualizer.print_module_tree, simple_model)

        # Should contain the layer types
        assert "Linear" in output
        assert "ReLU" in output

    def test_simple_model_root_name_displayed(self, simple_model):
        output = capture_print_output(ModelVisualizer.print_module_tree, simple_model)

        # Default root name is "model"
        assert output.startswith("model")

    def test_custom_root_name(self, simple_model):
        output = capture_print_output(
            ModelVisualizer.print_module_tree, simple_model, root_name="my_network"
        )

        assert output.startswith("my_network")

    def test_root_shows_model_type(self, simple_model):
        output = capture_print_output(ModelVisualizer.print_module_tree, simple_model)

        # Root should show the model's class name
        assert "SimpleModel" in output


class TestPrintModuleTreeNested:
    def test_nested_model_shows_hierarchy(self, nested_model):
        output = capture_print_output(ModelVisualizer.print_module_tree, nested_model)

        # Should contain parent module names
        assert "encoder" in output
        assert "decoder" in output
        assert "Sequential" in output

    def test_nested_model_shows_indexed_children(self, nested_model):
        output = capture_print_output(ModelVisualizer.print_module_tree, nested_model)

        # Sequential children are indexed
        assert "0" in output
        assert "1" in output
        assert "2" in output

    def test_deeply_nested_shows_all_levels(self, deeply_nested_model):
        output = capture_print_output(
            ModelVisualizer.print_module_tree, deeply_nested_model
        )

        # Should show all nested levels
        assert "block" in output
        assert "Linear" in output
        assert "BatchNorm1d" in output
        assert "ReLU" in output


class TestPrintModuleTreeEdgeCases:
    def test_empty_model_shows_root_only(self, empty_model):
        output = capture_print_output(ModelVisualizer.print_module_tree, empty_model)

        # Should show the root
        assert "model" in output
        assert "EmptyModel" in output

        # Output should be minimal (just the root line)
        lines = [line for line in output.strip().split("\n") if line.strip()]
        assert len(lines) == 1

    def test_module_list_shows_indexed_layers(self, model_with_module_list):
        output = capture_print_output(
            ModelVisualizer.print_module_tree, model_with_module_list
        )

        # Should show layers container and indexed children
        assert "layers" in output
        assert "ModuleList" in output
        assert "0" in output
        assert "1" in output
        assert "2" in output


class TestPrintModuleTreeFormatting:
    def test_output_uses_tree_characters(self, nested_model):
        output = capture_print_output(ModelVisualizer.print_module_tree, nested_model)

        # Should use tree drawing characters
        assert "└─" in output or "├─" in output

    def test_output_ends_with_newline(self, simple_model):
        output = capture_print_output(ModelVisualizer.print_module_tree, simple_model)

        # Last character should be a newline for proper terminal output
        assert output.endswith("\n")

    def test_containers_have_trailing_slash(self, nested_model):
        output = capture_print_output(ModelVisualizer.print_module_tree, nested_model)

        # Container modules should have trailing slash to indicate they have children
        # The root line should have a trailing slash
        first_line = output.split("\n")[0]
        assert first_line.endswith("/")
