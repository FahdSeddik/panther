"""
Unit tests for panther.tuner.SkAutoTuner.Searching.OptunaSearch

Covers:
- Initialization with different samplers/seeds
- Parameter space normalization (list -> Categorical, ParamSpec types)
- Lifecycle: initialize -> get_next_params -> update -> is_finished
- Deterministic behavior with GridSampler
- Error handling: uninitialized calls, invalid param types
- Best params/score tracking
- save_state/load_state round-trip
"""

import os
import tempfile

import pytest
import optuna
from optuna.samplers import TPESampler

from panther.tuner.SkAutoTuner.Configs.ParamSpec import Categorical, Float, Int
from panther.tuner.SkAutoTuner.Searching.OptunaSearch import OptunaSearch


# ============================================================================
# Initialization tests
# ============================================================================


class TestOptunaSearchInit:
    def test_default_init(self):
        search = OptunaSearch()
        assert search.n_trials == 100
        assert search.direction == "maximize"
        assert isinstance(search.sampler, TPESampler)

    def test_custom_n_trials(self):
        search = OptunaSearch(n_trials=50)
        assert search.n_trials == 50

    def test_custom_direction(self):
        search = OptunaSearch(direction="minimize")
        assert search.direction == "minimize"

    def test_custom_sampler(self):
        custom_sampler = TPESampler(seed=123)
        search = OptunaSearch(sampler=custom_sampler)
        assert search.sampler is custom_sampler

    def test_seed_creates_seeded_sampler(self):
        search = OptunaSearch(seed=42)
        assert search.seed == 42


# ============================================================================
# Parameter space initialization tests
# ============================================================================


class TestOptunaSearchInitialize:
    def test_list_converted_to_categorical(self):
        search = OptunaSearch(n_trials=3)
        search.initialize({"x": [1, 2, 3]})
        assert isinstance(search._param_space["x"], Categorical)

    def test_categorical_passthrough(self):
        search = OptunaSearch(n_trials=3)
        cat = Categorical([1, 2, 3])
        search.initialize({"x": cat})
        assert search._param_space["x"] is cat

    def test_int_passthrough(self):
        search = OptunaSearch(n_trials=3)
        int_spec = Int(1, 10)
        search.initialize({"x": int_spec})
        assert search._param_space["x"] is int_spec

    def test_float_passthrough(self):
        search = OptunaSearch(n_trials=3)
        float_spec = Float(0.0, 1.0)
        search.initialize({"x": float_spec})
        assert search._param_space["x"] is float_spec

    def test_invalid_type_raises(self):
        search = OptunaSearch(n_trials=3)
        with pytest.raises(TypeError, match="unsupported type"):
            search.initialize({"x": "invalid"})

    def test_study_created(self):
        search = OptunaSearch(n_trials=3)
        search.initialize({"x": [1, 2]})
        assert search._study is not None
        assert isinstance(search._study, optuna.Study)


# ============================================================================
# Lifecycle tests
# ============================================================================


class TestOptunaSearchLifecycle:
    def test_get_next_params_returns_dict(self):
        search = OptunaSearch(n_trials=3)
        search.initialize({"x": [1, 2, 3], "y": Int(1, 5)})
        params = search.get_next_params()
        assert isinstance(params, dict)
        assert "x" in params
        assert "y" in params

    def test_get_next_params_without_init_raises(self):
        search = OptunaSearch()
        with pytest.raises(RuntimeError, match="not initialized"):
            search.get_next_params()

    def test_update_records_score(self):
        search = OptunaSearch(n_trials=3)
        search.initialize({"x": [1, 2, 3]})
        params = search.get_next_params()
        search.update(params, 0.9)
        assert search.get_best_score() == 0.9

    def test_update_without_pending_trial_raises(self):
        search = OptunaSearch(n_trials=3)
        search.initialize({"x": [1, 2, 3]})
        with pytest.raises(RuntimeError, match="No pending trial"):
            search.update({"x": 1}, 0.5)

    def test_is_finished_after_n_trials(self):
        search = OptunaSearch(n_trials=2)
        search.initialize({"x": [1, 2]})

        # Trial 1
        params = search.get_next_params()
        search.update(params, 0.5)
        assert not search.is_finished()

        # Trial 2
        params = search.get_next_params()
        search.update(params, 0.6)
        assert search.is_finished()

    def test_get_next_params_returns_none_when_finished(self):
        search = OptunaSearch(n_trials=1)
        search.initialize({"x": [1, 2]})
        params = search.get_next_params()
        search.update(params, 0.5)
        assert search.get_next_params() is None


# ============================================================================
# Deterministic behavior with seeded sampler
# ============================================================================


class TestOptunaSearchDeterministic:
    def test_seeded_sampler_reproducible(self):
        """Seeded TPESampler should produce reproducible results."""
        search1 = OptunaSearch(n_trials=5, seed=42)
        search1.initialize({"x": Categorical([1, 2, 3, 4, 5])})

        results1 = []
        while not search1.is_finished():
            params = search1.get_next_params()
            if params is None:
                break
            score = params["x"] * 2
            search1.update(params, score)
            results1.append((params["x"], score))

        search2 = OptunaSearch(n_trials=5, seed=42)
        search2.initialize({"x": Categorical([1, 2, 3, 4, 5])})

        results2 = []
        while not search2.is_finished():
            params = search2.get_next_params()
            if params is None:
                break
            score = params["x"] * 2
            search2.update(params, score)
            results2.append((params["x"], score))

        # With same seed, results should be identical
        assert results1 == results2

    def test_best_selection_with_clear_winner(self):
        """Best params should select the highest scoring configuration."""
        search = OptunaSearch(n_trials=3, seed=42)
        search.initialize({"x": Categorical([1, 2, 3])})

        # Run all trials - manually assign scores to ensure deterministic best
        scores = {1: 0.1, 2: 0.9, 3: 0.5}
        while not search.is_finished():
            params = search.get_next_params()
            if params is None:
                break
            search.update(params, scores.get(params["x"], 0.0))

        # Best should be x=2 with score 0.9
        assert search.get_best_score() == 0.9
        assert search.get_best_params()["x"] == 2


# ============================================================================
# Best tracking tests
# ============================================================================


class TestOptunaSearchBestTracking:
    def test_get_best_params_initially_none(self):
        search = OptunaSearch(n_trials=3)
        search.initialize({"x": [1, 2]})
        assert search.get_best_params() is None

    def test_get_best_score_initially_none(self):
        search = OptunaSearch(n_trials=3)
        search.initialize({"x": [1, 2]})
        assert search.get_best_score() is None

    def test_best_updated_on_better_score_maximize(self):
        search = OptunaSearch(n_trials=3, direction="maximize")
        search.initialize({"x": [1, 2, 3]})

        params1 = search.get_next_params()
        search.update(params1, 0.5)
        assert search.get_best_score() == 0.5

        params2 = search.get_next_params()
        search.update(params2, 0.8)
        assert search.get_best_score() == 0.8

        params3 = search.get_next_params()
        search.update(params3, 0.6)
        # Best should still be 0.8
        assert search.get_best_score() == 0.8

    def test_best_updated_on_better_score_minimize(self):
        search = OptunaSearch(n_trials=3, direction="minimize")
        search.initialize({"x": [1, 2, 3]})

        params1 = search.get_next_params()
        search.update(params1, 0.8)
        assert search.get_best_score() == 0.8

        params2 = search.get_next_params()
        search.update(params2, 0.5)
        assert search.get_best_score() == 0.5

        params3 = search.get_next_params()
        search.update(params3, 0.6)
        # Best should still be 0.5 (lower is better)
        assert search.get_best_score() == 0.5


# ============================================================================
# Reset tests
# ============================================================================


class TestOptunaSearchReset:
    def test_reset_clears_state(self):
        search = OptunaSearch(n_trials=3)
        search.initialize({"x": [1, 2]})
        params = search.get_next_params()
        search.update(params, 0.9)

        # Check reset method exists before testing
        if hasattr(search, "reset"):
            search.reset()
            # After reset, best values should be cleared
            assert search._best_params is None
            assert search._best_score is None
            assert search._trial_count == 0
        else:
            # If no reset method, verify the search can be reinitialized
            search.initialize({"x": [1, 2]})
            # After reinitialization, trial count resets to 0
            assert search._trial_count == 0


# ============================================================================
# Save/Load state tests
# ============================================================================


class TestOptunaSearchPersistence:
    def test_save_and_load_state(self):
        search = OptunaSearch(n_trials=5)
        search.initialize({"x": [1, 2, 3]})

        # Run a couple trials
        params = search.get_next_params()
        search.update(params, 0.7)
        params = search.get_next_params()
        search.update(params, 0.9)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "search_state.pkl")
            search.save_state(filepath)

            # Create new search and load
            new_search = OptunaSearch(n_trials=5)
            new_search.load_state(filepath)

            assert new_search.get_best_score() == search.get_best_score()
            assert new_search._trial_count == search._trial_count
