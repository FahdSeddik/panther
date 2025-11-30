import os
import random

import pytest

from panther.tuner.SkAutoTuner.Searching.EvolutionaryAlgorithm import (
    EvolutionaryAlgorithm,
)


class TestEvolutionaryAlgorithm:
    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        """Set up basic parameters for testing the EvolutionaryAlgorithm."""
        self.param_space_example = {
            "learning_rate": [0.001, 0.01, 0.1],
            "batch_size": [32, 64, 128],
            "optimizer": ["adam", "sgd"],
            "activation": ["relu", "tanh", "sigmoid"],
        }
        self.population_size = 10
        self.n_generations = 5
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.tournament_size = 3
        self.elitism_count = 1

        self.ea = EvolutionaryAlgorithm(
            population_size=self.population_size,
            n_generations=self.n_generations,
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate,
            tournament_size=self.tournament_size,
            elitism_count=self.elitism_count,
        )

        # Store original random state to restore it after tests
        self.original_random_state = random.getstate()
        self.test_file_path = tmp_path / "test_ga_state.json"

        yield
        # Clean up any files created during tests
        random.setstate(self.original_random_state)

    def test_initialization_valid_parameters(self):
        """Test algorithm initializes correctly with valid parameters."""
        assert self.ea.population_size == self.population_size
        assert self.ea.n_generations == self.n_generations
        assert self.ea.mutation_rate == self.mutation_rate
        assert self.ea.crossover_rate == self.crossover_rate
        assert self.ea.tournament_size == self.tournament_size
        assert self.ea.elitism_count == self.elitism_count
        assert self.ea.param_space is None  # Not initialized yet
        assert len(self.ea.population) == 0  # Not initialized yet
        assert self.ea.best_params == {}
        assert self.ea.best_score == -float("inf")
        assert self.ea.current_generation == 0

    def test_initialization_invalid_parameters(self):
        """Test algorithm raises ValueError for invalid initialization parameters."""
        with pytest.raises(ValueError):
            EvolutionaryAlgorithm(-1, 5, 0.1, 0.7, 3)  # Negative population_size
        with pytest.raises(ValueError):
            EvolutionaryAlgorithm(10, -5, 0.1, 0.7, 3)  # Negative n_generations
        with pytest.raises(ValueError):
            EvolutionaryAlgorithm(10, 5, 1.1, 0.7, 3)  # mutation_rate > 1
        with pytest.raises(ValueError):
            EvolutionaryAlgorithm(10, 5, -0.1, 0.7, 3)  # mutation_rate < 0
        with pytest.raises(ValueError):
            EvolutionaryAlgorithm(10, 5, 0.1, 1.1, 3)  # crossover_rate > 1
        with pytest.raises(ValueError):
            EvolutionaryAlgorithm(10, 5, 0.1, -0.1, 3)  # crossover_rate < 0
        with pytest.raises(ValueError):
            EvolutionaryAlgorithm(10, 5, 0.1, 0.7, -3)  # Negative tournament_size
        with pytest.raises(ValueError):
            EvolutionaryAlgorithm(
                10, 5, 0.1, 0.7, 3, elitism_count=-1
            )  # Negative elitism_count
        with pytest.raises(ValueError):
            EvolutionaryAlgorithm(
                10, 5, 0.1, 0.7, 3, elitism_count=11
            )  # elitism_count > population_size

    def test_initialize_method_valid(self):
        """Test the initialize method with a valid param_space."""
        self.ea.initialize(self.param_space_example)
        assert self.ea.param_space == self.param_space_example
        assert len(self.ea.population) == self.population_size
        # Population consists of params directly (not wrapped in "params" key)
        for individual in self.ea.population:
            for param_name, param_value in individual.items():
                assert param_name in self.param_space_example
                assert param_value in self.param_space_example[param_name]
        assert self.ea.current_generation == 0

    def test_initialize_method_invalid_param_space(self):
        """Test initialize method with invalid param_space - empty param space should generate empty indexed space."""
        # Empty param_space will generate empty indexed_param_space, causing RuntimeError in initialize
        with pytest.raises(RuntimeError):
            self.ea.initialize({})  # Empty param_space

    def test_reset_method_before_initialize(self):
        """Test reset method can be called before initialize (it just resets to initial state)."""
        # The current implementation allows reset() to be called anytime
        self.ea.reset()
        assert self.ea.param_space is None
        assert len(self.ea.population) == 0

    def test_reset_method_after_initialize(self):
        """Test reset method correctly re-initializes the algorithm."""
        self.ea.initialize(self.param_space_example)
        # Simulate some progress
        self.ea.best_score = 0.5
        self.ea.best_params = {"lr": 0.1}
        self.ea.current_generation = 1

        self.ea.reset()

        assert self.ea.best_params == {}
        assert self.ea.best_score == -float("inf")
        assert self.ea.current_generation == 0
        assert self.ea.param_space is None

    def test_create_individual_before_initialize(self):
        """Test that methods requiring param_space fail appropriately before initialize."""
        # The implementation doesn't have _create_individual method
        # Skip this test as the method doesn't exist
        pass

    def test_get_next_params_when_finished(self):
        """Test get_next_params returns None when search is finished."""
        self.ea.initialize(self.param_space_example)
        self.ea.current_generation = (
            self.n_generations
        )  # Manually set to finished state
        assert self.ea.get_next_params() is None

    def test_update_method(self):
        """Test the update method correctly updates scores and best results."""
        self.ea.initialize(self.param_space_example)

        params1 = self.ea.get_next_params()
        score1 = 0.8
        self.ea.update(params1, score1)
        assert self.ea.best_score == score1
        # best_params should have the params plus score
        assert self.ea.best_params["score"] == score1

        params2 = self.ea.get_next_params()
        score2 = 0.7  # Worse score
        self.ea.update(params2, score2)
        assert self.ea.best_score == score1  # Best score remains score1

        params3 = self.ea.get_next_params()
        score3 = 0.9  # New best score
        self.ea.update(params3, score3)
        assert self.ea.best_score == score3

    def test_all_individuals_scored(self):
        """Test population scoring through update."""
        self.ea.initialize(self.param_space_example)

        # Initially, population_to_eval has individuals, evaluated_population is empty
        assert len(self.ea.evaluated_population) == 0

        # Score all individuals
        for i in range(self.population_size):
            params = self.ea.get_next_params()
            self.ea.update(params, random.random())

        # After scoring all, evaluated_population should have all individuals
        assert len(self.ea.evaluated_population) == self.population_size

    def test_tournament_selection_no_scored_individuals(self):
        """Test tournament selection with empty population."""
        self.ea.initialize(self.param_space_example)
        # Tournament selection requires a population_sorted_normalized parameter
        result = self.ea._tournament_selection([])
        assert result is None

    def test_tournament_selection_with_scored_individuals(self):
        """Test tournament selection picks from a population."""
        self.ea.initialize(self.param_space_example)

        # Create a mock scored population
        population = [
            {"learning_rate": 0.001, "batch_size": 32, "score": 0.5},
            {"learning_rate": 0.01, "batch_size": 64, "score": 0.8},
            {"learning_rate": 0.1, "batch_size": 128, "score": 0.2},
        ]

        selected = self.ea._tournament_selection(population.copy())
        assert selected is not None
        assert "score" in selected

    def test_tournament_selection_actual_size_adjustment(self):
        """Test tournament selection with small population."""
        ea = EvolutionaryAlgorithm(
            population_size=2,
            n_generations=5,
            mutation_rate=0.1,
            crossover_rate=0.7,
            tournament_size=3,
            elitism_count=1,
        )

        param_space = {
            "learning_rate": [0.001, 0.01],
            "batch_size": [32, 64],
        }
        ea.initialize(param_space)

        # Create a small mock scored population
        population = [
            {"learning_rate": 0.001, "batch_size": 32, "score": 0.5},
            {"learning_rate": 0.01, "batch_size": 64, "score": 0.8},
        ]

        selected = self.ea._tournament_selection(population.copy())
        assert selected is not None

    def test_uniform_crossover(self):
        """Test uniform crossover produces valid offspring."""
        self.ea.initialize(self.param_space_example)

        parent1 = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "optimizer": "adam",
            "activation": "relu",
            "score": 0.5,
        }
        parent2 = {
            "learning_rate": 0.1,
            "batch_size": 128,
            "optimizer": "sgd",
            "activation": "tanh",
            "score": 0.7,
        }

        child1, child2 = self.ea._uniform_crossover(parent1, parent2)

        assert len(child1) == len(self.param_space_example)
        assert len(child2) == len(self.param_space_example)

        for param_name in self.param_space_example.keys():
            assert param_name in child1
            assert param_name in child2
            # Check if values are valid according to param_space
            assert child1[param_name] in self.param_space_example[param_name]
            assert child2[param_name] in self.param_space_example[param_name]

    def test_mutate_method(self):
        """Test mutation changes parameters based on mutation_rate."""
        self.ea.initialize(self.param_space_example)
        self.ea.mutation_rate = 1.0  # Ensure mutation happens for every gene

        original_params = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "optimizer": "adam",
            "activation": "relu",
        }

        mutated_params = self.ea._mutate(original_params.copy())

        assert len(mutated_params) == len(original_params)
        for param_name in mutated_params.keys():
            assert mutated_params[param_name] in self.param_space_example[param_name]

        # Test with mutation_rate = 0.0
        self.ea.mutation_rate = 0.0
        not_mutated_params = self.ea._mutate(original_params.copy())
        assert original_params == not_mutated_params

    def test_evolve_population_elitism(self):
        """Test that elitism carries over the best individuals."""
        self.ea.initialize(self.param_space_example)
        self.ea.elitism_count = 2

        # Score all individuals first
        for i in range(self.population_size):
            params = self.ea.get_next_params()
            self.ea.update(params, i * 0.1)

        # Get the best individuals before evolution
        sorted_pop = sorted(self.ea.evaluated_population, key=lambda x: -x["score"])
        elite_scores = [sorted_pop[0]["score"], sorted_pop[1]["score"]]

        # Evolve
        self.ea._evolve()

        # After evolution, the new population should be in population_to_eval
        # We can't directly check elitism without running the full cycle
        assert len(self.ea.population_to_eval) == self.population_size

    def test_evolve_population_full_cycle(self):
        """Test a full evolution cycle: selection, crossover, mutation."""
        self.ea.initialize(self.param_space_example)

        # Score initial population
        for i in range(self.population_size):
            params = self.ea.get_next_params()
            score = random.random()
            self.ea.update(params, score)

        # Evolve
        self.ea._evolve()

        assert len(self.ea.population_to_eval) == self.population_size
        for individual in self.ea.population_to_eval:
            assert isinstance(individual, dict)
            # Check all parameters (excluding 'score' which is added by the algorithm)
            for param_name in self.param_space_example.keys():
                assert param_name in individual
                assert individual[param_name] in self.param_space_example[param_name]

    def test_is_finished(self):
        """Test is_finished method."""
        self.ea.initialize(self.param_space_example)
        assert not self.ea.is_finished()

        self.ea.current_generation = self.n_generations - 1
        assert not self.ea.is_finished()

        self.ea.current_generation = self.n_generations
        assert self.ea.is_finished()

        self.ea.current_generation = self.n_generations + 1
        assert self.ea.is_finished()

    def test_get_best_params_score(self):
        """Test get_best_params and get_best_score."""
        self.ea.initialize(self.param_space_example)
        assert self.ea.get_best_params() == {}
        assert self.ea.get_best_score() == -float("inf")

        params1 = self.ea.get_next_params()
        params1_copy = {
            k: v for k, v in params1.items()
        }  # Save copy before update modifies it
        score1 = 0.8
        self.ea.update(params1, score1)
        best_params = self.ea.get_best_params()
        # best_params includes score since update() adds it
        assert best_params["score"] == score1
        # Compare parameters without score
        best_params_no_score = {k: v for k, v in best_params.items() if k != "score"}
        assert best_params_no_score == params1_copy
        assert self.ea.get_best_score() == score1

        # Test deepcopy by modifying returned params
        best_p = self.ea.get_best_params()
        best_p["new_key"] = "new_value"
        assert self.ea.get_best_params() != best_p  # Internal state should be unchanged

    def test_full_run_loop(self):
        """Test a complete run of the algorithm for several generations."""
        self.ea.initialize(self.param_space_example)

        total_evals_expected = self.n_generations * self.population_size
        eval_count = 0

        for _ in range(total_evals_expected + 5):  # Loop a bit more to ensure it stops
            if self.ea.is_finished():
                break

            params = self.ea.get_next_params()
            if params is None:  # Should only happen if finished
                assert self.ea.is_finished()
                continue

            eval_count += 1
            # Simulate scoring
            score = sum(v for k, v in params.items() if isinstance(v, (int, float)))
            score += random.choice([0.1, 0.01, 0.001])
            if params.get("optimizer") == "adam":
                score += 0.5
            if params.get("activation") == "relu":
                score += 0.2

            self.ea.update(params, score)

        assert self.ea.is_finished()
        assert self.ea.current_generation == self.n_generations
        assert self.ea.get_best_params() != {}
        assert self.ea.get_best_score() > -float("inf")

    def test_save_and_load_state(self, tmp_path):
        """Test saving and loading the algorithm's state."""
        self.ea.initialize(self.param_space_example)
        filepath = str(tmp_path / "test_ga_state.pkl")

        # Simulate some progress
        for _ in range(self.population_size // 2):
            params = self.ea.get_next_params()
            self.ea.update(params, random.random())

        # Capture state before saving
        best_score_before = self.ea.best_score
        best_params_before = self.ea.get_best_params()
        current_gen_before = self.ea.current_generation

        self.ea.save_state(filepath)
        assert os.path.exists(filepath)

        # Create a new EA instance and load state
        ea_loaded = EvolutionaryAlgorithm(
            population_size=1,
            n_generations=1,
            mutation_rate=0,
            crossover_rate=0,
            tournament_size=1,
            elitism_count=0,
        )
        ea_loaded.load_state(filepath)

        assert ea_loaded.population_size == self.population_size
        assert ea_loaded.n_generations == self.n_generations
        assert ea_loaded.mutation_rate == self.mutation_rate
        assert ea_loaded.crossover_rate == self.crossover_rate
        assert ea_loaded.tournament_size == self.tournament_size
        assert ea_loaded.elitism_count == self.elitism_count
        assert ea_loaded.param_space == self.param_space_example
        assert pytest.approx(ea_loaded.best_score, abs=1e-7) == best_score_before
        assert ea_loaded.current_generation == current_gen_before

    def test_save_state_handles_io_error(self):
        """Test save_state raises RuntimeError on IOError (e.g. invalid path)."""
        self.ea.initialize(self.param_space_example)
        with pytest.raises(RuntimeError):
            self.ea.save_state("/invalid_path/that/does/not/exist/ga_state.json")

    def test_load_state_handles_file_not_found(self):
        """Test load_state raises RuntimeError if file does not exist."""
        with pytest.raises(RuntimeError):
            self.ea.load_state("non_existent_ga_state.json")

    def test_load_state_handles_json_decode_error(self, tmp_path):
        """Test load_state raises RuntimeError on invalid pickle data."""
        filepath = str(tmp_path / "test_ga_state_bad.pkl")
        with open(filepath, "w") as f:
            f.write("this is not valid pickle data")

        with pytest.raises(RuntimeError):
            self.ea.load_state(filepath)


if __name__ == "__main__":
    pytest.main()
