import random

import pytest

from panther.tuner.SkAutoTuner.Searching.ParticleSwarmOptimization import (
    ParticleSwarmOptimization,
)


# Define a dummy evaluation function for testing purposes
def dummy_evaluation_function(params):
    # Example: a simple function that rewards higher values for 'x' and 'y'
    # and has a specific preference for a category.
    score = 0
    if "x" in params:
        score += params["x"]
    if "y" in params:
        score += params["y"] * 2
    if "category" in params:
        if params["category"] == "A":
            score += 10
        elif params["category"] == "B":
            score += 5
    return score


class TestParticleSwarmOptimization:
    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        self.param_space_continuous = {
            "x": [0, 10],  # Representing min/max for continuous
            "y": [-5, 5],
        }
        self.min_values_continuous = {"x": 0, "y": -5}
        self.max_values_continuous = {"x": 10, "y": 5}

        self.param_space_categorical = {
            "cat1": ["A", "B", "C"],
            "cat2": ["X", "Y"],
        }
        self.param_space_mixed = {
            "x": [0, 10],
            "cat1": ["A", "B"],
        }
        self.min_values_mixed = {"x": 0}
        self.max_values_mixed = {"x": 10}

        self.test_dir = tmp_path

        self.original_random_state = random.getstate()
        yield
        random.setstate(self.original_random_state)

    def test_initialization_basic(self):
        pso = ParticleSwarmOptimization(num_particles=10, max_iterations=5, seed=42)
        assert pso.num_particles == 10
        assert pso.max_iterations == 5
        assert pso.seed == 42
        assert not pso._initialized

    def test_initialization_with_param_space(self):
        pso = ParticleSwarmOptimization(
            num_particles=5,
            max_iterations=3,
            seed=42,
        )
        pso.initialize(self.param_space_continuous)
        assert pso._initialized
        assert len(pso.particles) == 5
        assert len(pso._params_pending_evaluation) == 5
        for particle in pso.particles:
            assert "x" in particle["position"]
            assert "y" in particle["position"]
            assert (
                self.min_values_continuous["x"]
                <= particle["position"]["x"]
                <= self.max_values_continuous["x"]
            )
            assert (
                self.min_values_continuous["y"]
                <= particle["position"]["y"]
                <= self.max_values_continuous["y"]
            )
            assert "x" in particle["velocity"]
            assert "y" in particle["velocity"]

    def test_initialization_with_categorical_param_space(self):
        # PSO doesn't support categorical parameters - they must be [min, max] ranges
        # This test should verify that non-continuous params raise an error
        pso = ParticleSwarmOptimization(num_particles=3, max_iterations=2, seed=123)
        with pytest.raises(ValueError, match="must be a list of two numbers"):
            pso.initialize(self.param_space_categorical)

    def test_initialization_empty_param_space(self):
        pso = ParticleSwarmOptimization()
        with pytest.raises(ValueError):
            pso.initialize({})

    def test_get_next_params_before_initialize(self):
        pso = ParticleSwarmOptimization()
        with pytest.raises(RuntimeError):
            pso.get_next_params()

    def test_update_before_initialize(self):
        pso = ParticleSwarmOptimization()
        with pytest.raises(RuntimeError):
            pso.update({"x": 1}, 10)

    def test_get_next_params_flow(self):
        pso = ParticleSwarmOptimization(
            num_particles=2,
            max_iterations=1,
            seed=42,
        )
        pso.initialize(self.param_space_continuous)

        params1 = pso.get_next_params()
        assert params1 is not None
        assert pso.current_particle_idx == 0
        assert len(pso._params_pending_evaluation) == 1

        params2 = pso.get_next_params()
        assert params2 is not None
        assert pso.current_particle_idx == 1
        assert len(pso._params_pending_evaluation) == 0

        pso.update(params1, dummy_evaluation_function(params1))
        pso.update(params2, dummy_evaluation_function(params2))

        assert pso.current_iteration == 0
        params_after_iter_0_update = pso.get_next_params()

        assert pso.current_iteration == 1
        assert pso.is_finished()
        assert params_after_iter_0_update is None

    def test_update_and_best_score(self):
        pso = ParticleSwarmOptimization(
            num_particles=3,
            max_iterations=2,
            seed=42,
        )
        pso.initialize(self.param_space_continuous)

        assert pso.get_best_score() == -float("inf")
        assert pso.get_best_params() is not None

        params_list = []
        scores_list = []

        for i in range(pso.num_particles):
            params = pso.get_next_params()
            assert params is not None
            params_list.append(params)
            if i == 0:
                scores_list.append(10)
            elif i == 1:
                scores_list.append(5)
            elif i == 2:
                scores_list.append(15)

        pso.update(params_list[0], scores_list[0])
        assert pso.gbest_score == 10
        assert pso.gbest_position == params_list[0]
        assert len(pso._scores_to_process) == 1

        pso.update(params_list[1], scores_list[1])
        assert pso.gbest_score == 10
        assert pso.gbest_position == params_list[0]
        assert len(pso._scores_to_process) == 2

        pso.update(params_list[2], scores_list[2])
        assert pso.gbest_score == 15
        assert pso.gbest_position == params_list[2]
        assert len(pso._scores_to_process) == 3

        next_round_params = pso.get_next_params()
        assert next_round_params is not None
        assert pso.current_iteration == 1
        assert len(pso._scores_to_process) == 0

        assert pso.particles[0]["pbest_score"] == scores_list[0]
        assert pso.particles[0]["pbest_position"] == params_list[0]
        assert pso.particles[1]["pbest_score"] == scores_list[1]
        assert pso.particles[1]["pbest_position"] == params_list[1]
        assert pso.particles[2]["pbest_score"] == scores_list[2]
        assert pso.particles[2]["pbest_position"] == params_list[2]

        assert pso.get_best_score() == 15
        assert pso.get_best_params() == params_list[2]

    def test_is_finished(self):
        pso = ParticleSwarmOptimization(num_particles=1, max_iterations=1, seed=1)
        # PSO requires continuous parameters, so we use continuous param_space
        pso.initialize(self.param_space_continuous)
        assert not pso.is_finished()

        params = pso.get_next_params()
        pso.update(params, 1.0)

        next_params = pso.get_next_params()
        assert next_params is None
        assert pso.is_finished()

    def test_reset(self):
        pso = ParticleSwarmOptimization(
            num_particles=2,
            max_iterations=1,
            seed=42,
        )
        pso.initialize(self.param_space_continuous)
        params1 = pso.get_next_params()
        pso.update(params1, 10)
        assert pso._initialized
        assert pso.gbest_score != -float("inf")

        pso.reset()
        assert not pso._initialized
        assert pso.gbest_score == -float("inf")
        assert pso.current_iteration == 0
        assert len(pso.particles) == 0
        assert len(pso._params_pending_evaluation) == 0
        with pytest.raises(RuntimeError):
            pso.get_next_params()

    def test_save_and_load_state_continuous(self):
        filepath = self.test_dir / "pso_state_continuous.json"
        pso1 = ParticleSwarmOptimization(
            num_particles=3,
            max_iterations=2,
            seed=10,
        )
        pso1.initialize(self.param_space_continuous)

        for _ in range(pso1.num_particles):
            params = pso1.get_next_params()
            pso1.update(params, dummy_evaluation_function(params))

        pso1.get_next_params()
        assert pso1.current_iteration == 1

        pso1.save_state(str(filepath))
        assert filepath.exists()

        pso2 = ParticleSwarmOptimization(num_particles=1, max_iterations=1, seed=1)
        pso2.load_state(str(filepath))

        assert pso2._initialized
        assert pso2.num_particles == pso1.num_particles
        assert pso2.max_iterations == pso1.max_iterations
        assert pso2.current_iteration == pso1.current_iteration
        assert pso2.gbest_score == pso1.gbest_score
        assert pso2.gbest_position == pso1.gbest_position
        assert len(pso2.particles) == len(pso1.particles)
        for i in range(len(pso1.particles)):
            assert pso2.particles[i]["position"] == pso1.particles[i]["position"]
            assert pso2.particles[i]["velocity"] == pso1.particles[i]["velocity"]
            assert (
                pso2.particles[i]["pbest_position"]
                == pso1.particles[i]["pbest_position"]
            )
            assert pso2.particles[i]["pbest_score"] == pso1.particles[i]["pbest_score"]
        assert pso2.param_space == pso1.param_space
        assert pso2.min_values == pso1.min_values
        assert pso2.max_values == pso1.max_values
        assert len(pso2._params_pending_evaluation) == len(
            pso1._params_pending_evaluation
        )
        assert pso2.current_particle_idx == pso1.current_particle_idx

    def test_save_and_load_state_categorical(self):
        # PSO doesn't support categorical params - test should verify this raises error
        filepath = self.test_dir / "pso_state_categorical.json"
        pso1 = ParticleSwarmOptimization(num_particles=2, max_iterations=1, seed=20)
        with pytest.raises(ValueError, match="must be a list of two numbers"):
            pso1.initialize(self.param_space_categorical)

    def test_save_and_load_state_mixed(self):
        # PSO requires pure continuous parameters - mixed categorical will fail
        filepath = self.test_dir / "pso_state_mixed.json"
        pso1 = ParticleSwarmOptimization(
            num_particles=2,
            max_iterations=1,
            seed=30,
        )
        with pytest.raises(ValueError, match="must be a list of two numbers"):
            pso1.initialize(self.param_space_mixed)

    def test_get_best_params_score_initial_state(self):
        pso = ParticleSwarmOptimization(num_particles=5, max_iterations=3, seed=42)
        assert pso.get_best_score() == -float("inf")
        assert pso.get_best_params() is not None

        pso.initialize(self.param_space_continuous)
        assert pso.get_best_score() == -float("inf")
        assert pso.get_best_params() is not None

    def test_get_best_params_uninitialized(self):
        pso = ParticleSwarmOptimization()
        assert pso.get_best_params() is not None
        assert pso.get_best_score() == float("-inf")

    def test_invalid_param_space_format_values_not_list(self):
        pso = ParticleSwarmOptimization()
        # PSO expects lists with [min, max], tuples should fail validation
        with pytest.raises(ValueError, match="must be a list of two numbers"):
            pso.initialize({"x": (0, 10)})

    def test_invalid_param_space_format_continuous_wrong_length(self):
        pso = ParticleSwarmOptimization()
        params_x_single = {"x": [0]}
        # PSO requires exactly 2 values [min, max]
        with pytest.raises(ValueError, match="must be a list of two numbers"):
            pso.initialize(params_x_single)

        params_y_triple = {"y": [0, 1, 2]}
        pso = ParticleSwarmOptimization()
        with pytest.raises(ValueError, match="must be a list of two numbers"):
            pso.initialize(params_y_triple)

    def test_missing_min_max_for_continuous_param(self):
        pso = ParticleSwarmOptimization()
        pso.initialize({"x": [0, 10]})
        assert pso.param_space == {"x": [0, 10]}
        # PSO derives min/max from param_space during initialize
        assert pso.min_values == {"x": 0}
        assert pso.max_values == {"x": 10}

    def test_edge_case_one_particle_one_iteration(self):
        pso = ParticleSwarmOptimization(
            num_particles=1,
            max_iterations=1,
            seed=77,
        )
        pso.initialize(self.param_space_continuous)

        params1 = pso.get_next_params()
        assert params1 is not None
        score1 = dummy_evaluation_function(params1)
        pso.update(params1, score1)

        assert pso.get_best_score() == score1
        assert pso.get_best_params() == params1

        params2 = pso.get_next_params()
        assert params2 is None
        assert pso.is_finished()
        assert pso.current_iteration == 1

        assert pso.get_best_score() == score1
        assert pso.get_best_params() == params1
