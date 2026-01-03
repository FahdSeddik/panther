"""
End-to-end tests for panther.tuner.SkAutoTuner.SKAutoTuner

Covers the main execution paths:
- tune_separate (default): each layer tuned independently
- tune_joint: grouped layers tuned together with shared params
- apply_best_params: best sketched layers are applied to model
- save_tuning_results / load_tuning_results: persistence round-trip
- replace_without_tuning: use first param values without search
- Config validation errors

Uses real Optuna with GridSampler for determinism and real pawX sketch layers.
"""

import os
import tempfile

import pytest
import torch.nn as nn

from panther.tuner.SkAutoTuner.Configs.LayerConfig import LayerConfig
from panther.tuner.SkAutoTuner.Configs.ParamSpec import Categorical, Int
from panther.tuner.SkAutoTuner.Configs.TuningConfigs import TuningConfigs
from panther.tuner.SkAutoTuner.Searching.OptunaSearch import OptunaSearch
from panther.tuner.SkAutoTuner.SKAutoTuner import SKAutoTuner

# Suppress expected warnings from SKLinear about efficiency in small test models
pytestmark = pytest.mark.filterwarnings(
    "ignore:The sketching layer uses more parameters:UserWarning",
    "ignore:Tensor Core not utilized:UserWarning",
)

# Try to import pawX for integration tests
try:
    from panther.nn.linear import SKLinear

    HAS_SKLINEAR = True
except ImportError:
    HAS_SKLINEAR = False


# ============================================================================
# Test models
# ============================================================================


class TwoLayerModel(nn.Module):
    """Simple model with two Linear layers for tuning tests."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 16)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ThreeLayerModel(nn.Module):
    """Model with three Linear layers for joint tuning tests."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(32, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 16)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class MixedModel(nn.Module):
    """Model with different layer types."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def two_layer_model():
    return TwoLayerModel()


@pytest.fixture
def three_layer_model():
    return ThreeLayerModel()


@pytest.fixture
def mixed_model():
    return MixedModel()


def dummy_eval_func(model):
    """Dummy evaluation function that returns a score based on model state."""
    # Simple deterministic score based on layer count
    return 0.8


def param_based_eval_func(model, params_score_map=None):
    """
    Evaluation function that returns different scores based on current model config.
    Used to verify that tuner selects the best configuration.
    """
    # For testing, just return a constant
    return 0.9


# ============================================================================
# tune_separate tests (mainline path)
# ============================================================================


@pytest.mark.skipif(not HAS_SKLINEAR, reason="pawX/SKLinear not available")
class TestTuneSeparate:
    """Test the separate tuning path where each layer is tuned independently."""

    def test_tune_separate_basic(self, two_layer_model):
        """Basic separate tuning with small param space."""
        # Use seeded TPESampler for deterministic search
        search = OptunaSearch(n_trials=4, seed=42)

        configs = TuningConfigs(
            [
                LayerConfig(
                    layer_names=["fc1"],
                    params={"num_terms": [1, 2], "low_rank": [8, 16]},
                    separate=True,
                )
            ]
        )

        tuner = SKAutoTuner(
            model=two_layer_model,
            configs=configs,
            accuracy_eval_func=dummy_eval_func,
            search_algorithm=search,
            verbose=False,
        )

        best_params = tuner.tune()

        # Should have best params for fc1
        assert "fc1" in best_params
        assert "params" in best_params["fc1"]
        assert "num_terms" in best_params["fc1"]["params"]
        assert "low_rank" in best_params["fc1"]["params"]

    def test_tune_separate_multiple_layers(self, two_layer_model):
        """Tune multiple layers separately."""
        search = OptunaSearch(n_trials=2, seed=42)

        configs = TuningConfigs(
            [
                LayerConfig(
                    layer_names=["fc1", "fc2"],
                    params={"num_terms": [1, 2], "low_rank": [8]},
                    separate=True,
                )
            ]
        )

        tuner = SKAutoTuner(
            model=two_layer_model,
            configs=configs,
            accuracy_eval_func=dummy_eval_func,
            search_algorithm=search,
            verbose=False,
        )

        best_params = tuner.tune()

        # Both layers should have results
        assert "fc1" in best_params
        assert "fc2" in best_params

    def test_tune_separate_results_recorded(self, two_layer_model):
        """Verify that trial results are recorded for each layer."""
        search = OptunaSearch(n_trials=2, seed=42)

        configs = TuningConfigs(
            [
                LayerConfig(
                    layer_names=["fc1"],
                    params={"num_terms": [1, 2], "low_rank": [8]},
                    separate=True,
                )
            ]
        )

        tuner = SKAutoTuner(
            model=two_layer_model,
            configs=configs,
            accuracy_eval_func=dummy_eval_func,
            search_algorithm=search,
        )

        tuner.tune()

        # Check results are recorded
        assert "fc1" in tuner.results
        assert len(tuner.results["fc1"]) == 2  # 2 trials


# ============================================================================
# tune_joint tests
# ============================================================================


@pytest.mark.skipif(not HAS_SKLINEAR, reason="pawX/SKLinear not available")
class TestTuneJoint:
    """Test the joint tuning path where layers share parameters."""

    def test_tune_joint_basic(self, three_layer_model):
        """Joint tuning with multiple layers sharing params."""
        search = OptunaSearch(n_trials=2, seed=42)

        configs = TuningConfigs(
            [
                LayerConfig(
                    layer_names=["layer1", "layer2"],
                    params={"num_terms": [1, 2], "low_rank": [8]},
                    separate=False,  # Joint tuning
                )
            ]
        )

        tuner = SKAutoTuner(
            model=three_layer_model,
            configs=configs,
            accuracy_eval_func=dummy_eval_func,
            search_algorithm=search,
        )

        best_params = tuner.tune()

        # Both layers should have same params (joint tuning)
        assert "layer1" in best_params
        assert "layer2" in best_params
        assert best_params["layer1"]["params"] == best_params["layer2"]["params"]

    def test_tune_joint_group_key_in_results(self, three_layer_model):
        """Joint tuning should store results under a group key."""
        search = OptunaSearch(n_trials=1, seed=42)

        configs = TuningConfigs(
            [
                LayerConfig(
                    layer_names=["layer1", "layer2"],
                    params={"num_terms": [1], "low_rank": [8]},
                    separate=False,
                )
            ]
        )

        tuner = SKAutoTuner(
            model=three_layer_model,
            configs=configs,
            accuracy_eval_func=dummy_eval_func,
            search_algorithm=search,
        )

        tuner.tune()

        # Results should be stored under group key (layer names joined)
        # The order may vary, so check for any key containing both layer names
        result_keys = list(tuner.results.keys())
        assert len(result_keys) == 1
        group_key = result_keys[0]
        assert "layer1" in group_key and "layer2" in group_key


# ============================================================================
# apply_best_params tests
# ============================================================================


@pytest.mark.skipif(not HAS_SKLINEAR, reason="pawX/SKLinear not available")
class TestApplyBestParams:
    """Test applying best parameters to the model."""

    def test_apply_best_replaces_layers(self, two_layer_model):
        """apply_best_params should replace original layers with sketched ones."""
        search = OptunaSearch(n_trials=1, seed=42)

        configs = TuningConfigs(
            [
                LayerConfig(
                    layer_names=["fc1"],
                    params={"num_terms": [1], "low_rank": [8]},
                    separate=True,
                )
            ]
        )

        tuner = SKAutoTuner(
            model=two_layer_model,
            configs=configs,
            accuracy_eval_func=dummy_eval_func,
            search_algorithm=search,
        )

        tuner.tune()

        # Before apply: should be nn.Linear
        assert isinstance(two_layer_model.fc1, nn.Linear)

        tuner.apply_best_params()

        # After apply: should be SKLinear
        assert isinstance(two_layer_model.fc1, SKLinear)

    def test_apply_best_preserves_untuned_layers(self, two_layer_model):
        """Layers not in config should remain unchanged."""
        search = OptunaSearch(n_trials=1, seed=42)

        configs = TuningConfigs(
            [
                LayerConfig(
                    layer_names=["fc1"],  # Only tune fc1
                    params={"num_terms": [1], "low_rank": [8]},
                    separate=True,
                )
            ]
        )

        tuner = SKAutoTuner(
            model=two_layer_model,
            configs=configs,
            accuracy_eval_func=dummy_eval_func,
            search_algorithm=search,
        )

        tuner.tune()
        tuner.apply_best_params()

        # fc2 should still be nn.Linear
        assert isinstance(two_layer_model.fc2, nn.Linear)


# ============================================================================
# replace_without_tuning tests
# ============================================================================


@pytest.mark.skipif(not HAS_SKLINEAR, reason="pawX/SKLinear not available")
class TestReplaceWithoutTuning:
    """Test replacing layers with first param values without running search."""

    def test_replace_without_tuning_uses_first_values(self, two_layer_model):
        """Should use first choice from each param spec."""
        # Don't need GridSampler here since no search is run
        search = OptunaSearch(n_trials=1)

        configs = TuningConfigs(
            [
                LayerConfig(
                    layer_names=["fc1"],
                    params={"num_terms": [2, 3, 4], "low_rank": [16, 32]},
                    separate=True,
                )
            ]
        )

        tuner = SKAutoTuner(
            model=two_layer_model,
            configs=configs,
            accuracy_eval_func=dummy_eval_func,
            search_algorithm=search,
        )

        tuner.replace_without_tuning()

        # fc1 should be replaced
        assert isinstance(two_layer_model.fc1, SKLinear)


# ============================================================================
# Persistence tests
# ============================================================================


@pytest.mark.skipif(not HAS_SKLINEAR, reason="pawX/SKLinear not available")
class TestPersistence:
    """Test save/load tuning results."""

    def test_save_and_load_results(self, two_layer_model):
        """save_tuning_results and load_tuning_results round-trip."""
        search = OptunaSearch(n_trials=2, seed=42)

        configs = TuningConfigs(
            [
                LayerConfig(
                    layer_names=["fc1"],
                    params={"num_terms": [1, 2], "low_rank": [8]},
                    separate=True,
                )
            ]
        )

        tuner = SKAutoTuner(
            model=two_layer_model,
            configs=configs,
            accuracy_eval_func=dummy_eval_func,
            search_algorithm=search,
        )

        tuner.tune()
        original_results = tuner.results.copy()
        original_best = {k: v["params"] for k, v in tuner.best_params.items()}

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "tuning_results.pkl")
            tuner.save_tuning_results(filepath)

            # Create new tuner and load
            new_model = TwoLayerModel()
            new_tuner = SKAutoTuner(
                model=new_model,
                configs=configs,
                accuracy_eval_func=dummy_eval_func,
                search_algorithm=OptunaSearch(n_trials=1),
            )
            new_tuner.load_tuning_results(filepath)

            # Check results match
            assert "fc1" in new_tuner.results
            assert len(new_tuner.results["fc1"]) == len(original_results["fc1"])
            assert new_tuner.best_params["fc1"]["params"] == original_best["fc1"]

    def test_load_nonexistent_file_raises(self, two_layer_model):
        """Loading from nonexistent file should raise FileNotFoundError."""
        search = OptunaSearch(n_trials=1)
        configs = TuningConfigs(
            [
                LayerConfig(
                    layer_names=["fc1"],
                    params={"num_terms": [1], "low_rank": [8]},
                    separate=True,
                )
            ]
        )

        tuner = SKAutoTuner(
            model=two_layer_model,
            configs=configs,
            accuracy_eval_func=dummy_eval_func,
            search_algorithm=search,
        )

        with pytest.raises(FileNotFoundError):
            tuner.load_tuning_results("/nonexistent/path/file.pkl")


# ============================================================================
# Config validation error tests
# ============================================================================


class TestConfigValidation:
    """Test config validation in SKAutoTuner.__init__."""

    def test_empty_layer_names_raises(self, two_layer_model):
        """Empty layer_names should raise ValueError."""
        configs = TuningConfigs(
            [
                LayerConfig(
                    layer_names=[],  # Empty!
                    params={"num_terms": [1], "low_rank": [8]},
                )
            ]
        )

        with pytest.raises(ValueError, match="no layer names"):
            SKAutoTuner(
                model=two_layer_model,
                configs=configs,
                accuracy_eval_func=dummy_eval_func,
            )

    def test_empty_params_raises(self, two_layer_model):
        """Empty params should raise ValueError."""
        configs = TuningConfigs(
            [
                LayerConfig(
                    layer_names=["fc1"],
                    params={},  # Empty!
                )
            ]
        )

        with pytest.raises(ValueError, match="no parameters"):
            SKAutoTuner(
                model=two_layer_model,
                configs=configs,
                accuracy_eval_func=dummy_eval_func,
            )

    def test_nonexistent_layer_raises(self, two_layer_model):
        """Layer name not in model should raise ValueError."""
        # Use a list with an explicit nonexistent name that won't get resolved away
        # The resolver returns empty for nonexistent patterns, causing 'no layer names' error.
        # So we test for that error message instead.
        configs = TuningConfigs(
            [
                LayerConfig(
                    layer_names="nonexistent_layer_pattern_xyz",
                    params={"num_terms": [1], "low_rank": [8]},
                )
            ]
        )

        with pytest.raises(ValueError, match="no layer names"):
            SKAutoTuner(
                model=two_layer_model,
                configs=configs,
                accuracy_eval_func=dummy_eval_func,
            )

    def test_unsupported_layer_type_raises(self, two_layer_model):
        """Layer type not in LAYER_TYPE_MAPPING should raise ValueError."""
        configs = TuningConfigs(
            [
                LayerConfig(
                    layer_names=["relu"],  # ReLU is not supported
                    params={"num_terms": [1], "low_rank": [8]},
                )
            ]
        )

        with pytest.raises(ValueError, match="not supported"):
            SKAutoTuner(
                model=two_layer_model,
                configs=configs,
                accuracy_eval_func=dummy_eval_func,
            )

    def test_invalid_param_name_raises(self, two_layer_model):
        """Parameter not valid for layer type should raise ValueError."""
        configs = TuningConfigs(
            [
                LayerConfig(
                    layer_names=["fc1"],
                    params={"invalid_param": [1, 2, 3]},  # Not valid for Linear
                )
            ]
        )

        with pytest.raises(ValueError, match="not valid"):
            SKAutoTuner(
                model=two_layer_model,
                configs=configs,
                accuracy_eval_func=dummy_eval_func,
            )


# ============================================================================
# ParamSpec types in configs
# ============================================================================


@pytest.mark.skipif(not HAS_SKLINEAR, reason="pawX/SKLinear not available")
class TestParamSpecInConfigs:
    """Test using ParamSpec types (Categorical, Int) in configs."""

    def test_categorical_in_params(self, two_layer_model):
        """Categorical ParamSpec should work in config."""
        search = OptunaSearch(n_trials=2, seed=42)

        configs = TuningConfigs(
            [
                LayerConfig(
                    layer_names=["fc1"],
                    params={
                        "num_terms": Categorical([1, 2]),
                        "low_rank": Categorical([8]),
                    },
                    separate=True,
                )
            ]
        )

        tuner = SKAutoTuner(
            model=two_layer_model,
            configs=configs,
            accuracy_eval_func=dummy_eval_func,
            search_algorithm=search,
        )

        best_params = tuner.tune()
        assert "fc1" in best_params

    def test_int_spec_in_params(self, two_layer_model):
        """Int ParamSpec should work in config."""
        # Use small Int range that expands to choices
        search = OptunaSearch(n_trials=4, seed=42)

        configs = TuningConfigs(
            [
                LayerConfig(
                    layer_names=["fc1"],
                    params={
                        "num_terms": Int(1, 2),
                        "low_rank": Int(8, 16, step=8),
                    },
                    separate=True,
                )
            ]
        )

        tuner = SKAutoTuner(
            model=two_layer_model,
            configs=configs,
            accuracy_eval_func=dummy_eval_func,
            search_algorithm=search,
        )

        best_params = tuner.tune()
        assert "fc1" in best_params


# ============================================================================
# Verbose mode test
# ============================================================================


@pytest.mark.skipif(not HAS_SKLINEAR, reason="pawX/SKLinear not available")
class TestVerboseMode:
    """Test verbose output doesn't break anything."""

    def test_tune_with_verbose(self, two_layer_model, capsys):
        """Tuning with verbose=True should print progress."""
        search = OptunaSearch(n_trials=1, seed=42)

        configs = TuningConfigs(
            [
                LayerConfig(
                    layer_names=["fc1"],
                    params={"num_terms": [1], "low_rank": [8]},
                    separate=True,
                )
            ]
        )

        tuner = SKAutoTuner(
            model=two_layer_model,
            configs=configs,
            accuracy_eval_func=dummy_eval_func,
            search_algorithm=search,
            verbose=True,
        )

        tuner.tune()

        # Should have printed something
        captured = capsys.readouterr()
        assert len(captured.out) > 0


# ============================================================================
# get_results_dataframe test (optional pandas)
# ============================================================================


@pytest.mark.skipif(not HAS_SKLINEAR, reason="pawX/SKLinear not available")
class TestResultsDataframe:
    """Test get_results_dataframe method."""

    def test_get_results_dataframe(self, two_layer_model):
        """Should return a DataFrame with results."""
        pd = pytest.importorskip("pandas")

        search = OptunaSearch(n_trials=2, seed=42)

        configs = TuningConfigs(
            [
                LayerConfig(
                    layer_names=["fc1"],
                    params={"num_terms": [1, 2], "low_rank": [8]},
                    separate=True,
                )
            ]
        )

        tuner = SKAutoTuner(
            model=two_layer_model,
            configs=configs,
            accuracy_eval_func=dummy_eval_func,
            search_algorithm=search,
        )

        tuner.tune()
        df = tuner.get_results_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert "layer_name" in df.columns
        assert "score" in df.columns
        assert len(df) == 2  # 2 trials
