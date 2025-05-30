import random

import pytest

from panther.tuner.SkAutoTuner.Searching.TreeParzenEstimator import TreeParzenEstimator


class TestTreeParzenEstimator:
    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        self.param_space = {
            "param1": [1, 2, 3, 4, 5],
            "param2": ["a", "b", "c"],
            "param3": [0.1, 0.5, 1.0, 1.5],
        }
        self.tpe = TreeParzenEstimator(
            n_initial_points=3,
            max_trials=10,
            gamma_ratio=0.25,
            n_ei_candidates=5,
            seed=42,
        )
        self.tpe.initialize(self.param_space)
        self.test_filepath = tmp_path / "test_tpe_state.pkl"

        # Store original random state to restore it after each test
        self.original_random_state = random.getstate()
        yield
        # Teardown logic (formerly in tearDown)
        # No need to explicitly delete self.test_filepath as tmp_path handles it
        random.setstate(self.original_random_state)  # Restore random state

    def test_initialization_valid(self):
        assert self.tpe.n_initial_points == 3
        assert self.tpe.max_trials == 10
        assert self.tpe.gamma_ratio == 0.25
        assert self.tpe.n_ei_candidates == 5
        assert self.tpe.param_space == self.param_space
        assert self.tpe.trial_count == 0
        assert self.tpe.best_score == -float("inf")
        assert self.tpe.best_params is not None
        assert all(p in self.param_space for p in self.tpe.best_params.keys())

    def test_initialization_invalid_gamma(self):
        with pytest.raises(ValueError):
            TreeParzenEstimator(gamma_ratio=0)
        with pytest.raises(ValueError):
            TreeParzenEstimator(gamma_ratio=1)
        with pytest.raises(ValueError):
            TreeParzenEstimator(gamma_ratio=1.1)

    def test_initialization_low_n_initial_points(self):
        # n_initial_points is adjusted to be at least 2
        tpe_low_init = TreeParzenEstimator(
            n_initial_points=1, verbose=True
        )  # verbose to check warning potentially
        assert tpe_low_init.n_initial_points == 2
        tpe_low_init_0 = TreeParzenEstimator(n_initial_points=0, verbose=True)
        assert tpe_low_init_0.n_initial_points == 2

    def test_initialize_empty_param_space(self):
        tpe_empty = TreeParzenEstimator()
        with pytest.raises(ValueError):
            tpe_empty.initialize({})

    def test_initialize_param_space_with_empty_values(self):
        tpe_empty_vals = TreeParzenEstimator()
        with pytest.raises(ValueError):
            tpe_empty_vals.initialize({"param1": []})

    def test_generate_random_params(self):
        params = self.tpe._generate_random_params()
        assert isinstance(params, dict)
        for p_name, p_value in params.items():
            assert p_name in self.param_space
            assert p_value in self.param_space[p_name]

    def test_reset(self):
        self.tpe.update({"param1": 1, "param2": "a", "param3": 0.1}, 10)
        self.tpe.trial_count = 5
        original_param_space = self.tpe.param_space.copy()
        self.tpe.reset()
        assert self.tpe.trial_count == 0
        assert self.tpe.history == []
        assert self.tpe.scores == []
        assert self.tpe.best_score == -float("inf")
        # best_params should be re-initialized to a random configuration if param_space was set
        assert self.tpe.best_params is not None
        assert (
            self.tpe.param_space == original_param_space
        )  # param_space itself is not cleared by reset, but kept

        # Test reset without param_space initialized (e.g. if called directly after __init__)
        tpe_no_space = TreeParzenEstimator()
        tpe_no_space.reset()
        assert tpe_no_space.best_params is None

    # def test_get_next_params_initial_phase(self):
    #     for i in range(self.tpe.n_initial_points):
    #         params = self.tpe.get_next_params()
    #         assert isinstance(params, dict)
    #         # In initial phase, trial_count is not yet incremented by get_next_params
    #         assert (
    #             self.tpe.trial_count == 0
    #         )  # trial_count is updated in `update`
    #         # We simulate update for testing purposes here
    #         self.tpe.update(params, random.uniform(0, 10))

    def test_get_next_params_tpe_phase_not_enough_trials_fallback(self):
        # Test case where n_initial_points is met, but not enough trials for good/bad split (e.g. 1 trial)
        tpe_few = TreeParzenEstimator(n_initial_points=1, max_trials=5, seed=43)
        tpe_few.initialize(self.param_space)

        params_init = tpe_few.get_next_params()  # Initial random point
        tpe_few.update(params_init, 5.0)
        assert tpe_few.trial_count == 1

        # Now trial_count is 1, which is >= n_initial_points (1).
        # However, len(scores) is 1, which is < 2, so TPE cannot split. Should fallback to random.
        params_next = tpe_few.get_next_params()
        assert isinstance(params_next, dict)
        # Here, we can't definitively check if it *was* random, but it shouldn't crash.
        # The internal logic should handle the fallback.

    def test_update(self):
        params = {"param1": 1, "param2": "a", "param3": 0.1}
        score = 10.0
        self.tpe.update(params, score)

        assert self.tpe.trial_count == 1
        assert len(self.tpe.history) == 1
        assert self.tpe.history[0] == params
        assert len(self.tpe.scores) == 1
        assert self.tpe.scores[0] == score
        assert self.tpe.best_score == score
        assert self.tpe.best_params == params

        # Update with worse score
        params_worse = {"param1": 2, "param2": "b", "param3": 0.5}
        score_worse = 5.0
        self.tpe.update(params_worse, score_worse)
        assert self.tpe.trial_count == 2
        assert self.tpe.best_score == score  # Best should not change
        assert self.tpe.best_params == params

        # Update with better score
        params_better = {"param1": 3, "param2": "c", "param3": 1.0}
        score_better = 15.0
        self.tpe.update(params_better, score_better)
        assert self.tpe.trial_count == 3
        assert self.tpe.best_score == score_better
        assert self.tpe.best_params == params_better

    def test_update_first_call_sets_best_params(self):
        tpe_first = TreeParzenEstimator(n_initial_points=1, max_trials=2, seed=100)
        tpe_first.initialize(self.param_space)
        assert tpe_first.best_params is not None  # Initialized randomly
        initial_random_best_score = tpe_first.best_score  # Should be -inf
        assert initial_random_best_score == -float("inf")

        p = tpe_first.get_next_params()
        s = 5.0
        tpe_first.update(p, s)
        assert tpe_first.best_params == p
        assert tpe_first.best_score == s

    def test_save_and_load_state(self):
        # Run some trials to populate history
        for _ in range(self.tpe.n_initial_points + 2):
            if self.tpe.is_finished():
                break
            params = self.tpe.get_next_params()
            self.tpe.update(params, random.uniform(0, 20))

        # Manually set a value to check if it's saved/loaded
        self.tpe.verbose = True
        # original_random_state is handled by the fixture now

        self.tpe.save_state(
            str(self.test_filepath)
        )  # Ensure path is string for save_state
        assert self.test_filepath.exists()

        new_tpe = TreeParzenEstimator(
            seed=123
        )  # Use a different seed to ensure loaded state overrides it
        new_tpe.initialize(
            self.param_space
        )  # Initialize with same space, but state should override history etc.
        new_tpe.load_state(
            str(self.test_filepath)
        )  # Ensure path is string for load_state

        assert new_tpe.param_space == self.tpe.param_space
        assert new_tpe.history == self.tpe.history
        assert new_tpe.scores == self.tpe.scores
        assert new_tpe.best_params == self.tpe.best_params
        assert new_tpe.best_score == self.tpe.best_score
        assert new_tpe.trial_count == self.tpe.trial_count
        assert new_tpe.n_initial_points == self.tpe.n_initial_points
        assert new_tpe.max_trials == self.tpe.max_trials
        assert new_tpe.gamma_ratio == self.tpe.gamma_ratio
        assert new_tpe.n_ei_candidates == self.tpe.n_ei_candidates
        assert new_tpe.verbose == self.tpe.verbose
        # Random state is restored by the fixture after this test,
        # so checking random.getstate() here against original_random_state might be tricky
        # if load_state also restores it. Assuming load_state correctly restores the state *it* saved.

    def test_load_state_file_not_found(self):
        tpe_load = TreeParzenEstimator()
        # Expect a print message and no crash
        tpe_load.load_state("non_existent_file.pkl")
        # State should remain as default initialized
        assert tpe_load.trial_count == 0

    def test_get_best_params_and_score_no_trials(self):
        tpe_no_trials = TreeParzenEstimator(seed=42)
        tpe_no_trials.initialize(self.param_space)
        # Before any updates, best_params is a random configuration, best_score is -inf
        assert tpe_no_trials.get_best_params() is not None
        assert all(
            p in self.param_space for p in tpe_no_trials.get_best_params().keys()
        )
        assert tpe_no_trials.get_best_score() == -float("inf")

    # def test_get_best_params_uninitialized(self):
    #     tpe_uninit = TreeParzenEstimator()
    #     with pytest.raises(RuntimeError, match="Optimizer not initialized"): # Check for specific error message
    #         tpe_uninit.get_best_params()

    def test_is_finished(self):
        assert not self.tpe.is_finished()
        self.tpe.trial_count = self.tpe.max_trials
        assert self.tpe.is_finished()
        self.tpe.trial_count = self.tpe.max_trials + 1  # Exceed max_trials
        assert self.tpe.is_finished()

    def test_tpe_logic_runs(self):
        # This test aims to ensure the TPE part of get_next_params runs without error.
        # It doesn't validate the statistical correctness of TPE, just that the code path executes.

        # Fill initial points
        for _ in range(self.tpe.n_initial_points):
            params = self.tpe.get_next_params()
            self.tpe.update(params, random.uniform(1, 10))

        assert self.tpe.trial_count >= self.tpe.n_initial_points

        # Next call should use TPE
        try:
            tpe_params = self.tpe.get_next_params()
            assert isinstance(tpe_params, dict)
            for p_name, p_value in tpe_params.items():
                assert p_name in self.param_space
                assert p_value in self.param_space[p_name]
        except Exception as e:
            pytest.fail(f"TPE get_next_params failed after initial points: {e}")

        # Ensure it can run for the full budget
        while not self.tpe.is_finished():
            try:
                params = self.tpe.get_next_params()
                assert isinstance(params, dict)
                # Simulate evaluation
                self.tpe.update(params, random.uniform(0, 20))
            except Exception as e:
                pytest.fail(
                    f"get_next_params or update failed during budgeted run: {e}"
                )
        assert self.tpe.trial_count == self.tpe.max_trials

    def test_calculate_param_prob_laplace_smoothing(self):
        # Test _calculate_param_prob directly to ensure smoothing works
        all_values = ["x", "y", "z"]
        # Case 1: value seen
        observed = ["x", "x", "y"]
        prob_x = self.tpe._calculate_param_prob("x", observed, all_values)
        # count(x)=2, total_obs=3, alpha=1, num_cat=3. (2+1)/(3+1*3) = 3/6 = 0.5
        assert prob_x == pytest.approx((2 + 1) / (3 + 3 * 1))

        # Case 2: value not seen (zero count)
        prob_z = self.tpe._calculate_param_prob("z", observed, all_values)
        # count(z)=0. (0+1)/(3+1*3) = 1/6
        assert prob_z == pytest.approx((0 + 1) / (3 + 3 * 1))

        # Case 3: no observations (should rely on smoothing entirely, or handled by _sample_param_value)
        # _calculate_param_prob assumes observed_param_values is what it gets from good/bad trials
        # if that list is empty, it means e.g. no good trials for a parameter.
        # This is tested by _sample_param_value, which should use uniform from all_possible_values
        prob_x_empty_obs = self.tpe._calculate_param_prob("x", [], all_values)
        # (0+1)/(0+1*3) = 1/3
        assert prob_x_empty_obs == pytest.approx((0 + 1) / (0 + 3 * 1))

    def test_sample_param_value(self):
        all_possible = [10, 20, 30]
        # Sample from observed
        observed = [10, 20, 10]
        samples_from_observed = [
            self.tpe._sample_param_value(observed, all_possible) for _ in range(30)
        ]
        assert all(s in observed for s in samples_from_observed)
        assert 10 in samples_from_observed
        assert 20 in samples_from_observed
        assert 30 not in samples_from_observed  # 30 is not in observed

        # Sample from all_possible if observed is empty
        samples_from_all = [
            self.tpe._sample_param_value([], all_possible) for _ in range(30)
        ]
        assert all(s in all_possible for s in samples_from_all)
        # Check if all values appear with some frequency (probabilistic)
        assert any(s == 10 for s in samples_from_all)
        assert any(s == 20 for s in samples_from_all)
        assert any(s == 30 for s in samples_from_all)

    def test_gamma_split_logic(self):
        # Test how n_good is calculated and indices are split
        tpe_gamma_test = TreeParzenEstimator(
            gamma_ratio=0.25
        )  # Default n_initial_points=10
        tpe_gamma_test.initialize(self.param_space)

        # Simulate 10 trials with scores
        scores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        params_dummy = {"param1": 0, "param2": "z", "param3": 0.0}
        for score in scores:
            tpe_gamma_test.update(params_dummy, float(score))

        num_total_trials = len(tpe_gamma_test.scores)
        assert num_total_trials == 10

        # gamma = 0.25, num_total = 10. n_good_ideal = ceil(0.25*10) = ceil(2.5) = 3
        # n_good = max(1, min(3, 10-1)) = max(1, min(3,9)) = 3
        # Scores are 1..10. Sorted indices (reverse=True) will be for [10,9,8,7,6,5,4,3,2,1]
        # Good indices: first 3 (for scores 10,9,8)
        # Bad indices: remaining 7

        # We need to call get_next_params to trigger the internal logic
        # To do this, we must be past n_initial_points.
        # Here, n_initial_points defaults to 10, and we have 10 trials.
        # The *next* call to get_next_params will be the 11th trial overall if max_trials allows,
        # and it will try to use TPE logic based on the 10 existing trials.
        tpe_gamma_test.max_trials = 11  # Allow one more trial
        tpe_gamma_test.n_initial_points = (
            0  # Force TPE mode for next call based on existing trials
        )
        # by making sure trial_count (10) >= n_initial_points (0)

        # Spy on the split (this is a bit white-boxy, but necessary to check counts)
        # The actual good/bad params are not directly accessible from outside.
        # We can infer by providing scores and checking resulting candidate.
        # However, an easier check is the n_good calculation part if it were a helper method.
        # Since it's embedded, we just run get_next_params and ensure it doesn't fail.
        try:
            next_p = tpe_gamma_test.get_next_params()
            assert next_p is not None
        except Exception as e:
            pytest.fail(f"get_next_params failed in gamma_split_logic test: {e}")

        # Test edge case: n_good becomes 1 if gamma is small
        tpe_gamma_low = TreeParzenEstimator(gamma_ratio=0.01)  # Expect n_good = 1
        tpe_gamma_low.initialize(self.param_space)
        for i in range(5):  # e.g. 5 trials
            tpe_gamma_low.update(params_dummy, float(i))
        # n_good_ideal = ceil(0.01*5) = ceil(0.05) = 1.
        # n_good = max(1, min(1, 5-1)) = 1.
        tpe_gamma_low.max_trials = 6
        tpe_gamma_low.n_initial_points = 0
        try:
            next_p_low = tpe_gamma_low.get_next_params()
            assert next_p_low is not None
        except Exception as e:
            pytest.fail(
                f"get_next_params failed in gamma_split_logic low gamma test: {e}"
            )

        # Test edge case: n_good becomes num_total_trials - 1 if gamma is high
        tpe_gamma_high = TreeParzenEstimator(
            gamma_ratio=0.99
        )  # Expect n_good = num_trials - 1
        tpe_gamma_high.initialize(self.param_space)
        for i in range(5):  # e.g. 5 trials
            tpe_gamma_high.update(params_dummy, float(i))
        # n_good_ideal = ceil(0.99*5) = ceil(4.95) = 5.
        # n_good = max(1, min(5, 5-1)) = max(1, 4) = 4.
        tpe_gamma_high.max_trials = 6
        tpe_gamma_high.n_initial_points = 0
        try:
            next_p_high = tpe_gamma_high.get_next_params()
            assert next_p_high is not None
        except Exception as e:
            pytest.fail(
                f"get_next_params failed in gamma_split_logic high gamma test: {e}"
            )
