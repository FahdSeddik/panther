import os
import shutil  # For cleaning up test directories/files
import unittest

from panther.utils.SkAutoTuner.Searching.ParticleSwarmOptimization import (
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


class TestParticleSwarmOptimization(unittest.TestCase):
    def setUp(self):
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

        self.test_dir = "test_pso_states"
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

    def tearDown(self):
        # Clean up any files created during tests
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_initialization_basic(self):
        pso = ParticleSwarmOptimization(num_particles=10, max_iterations=5, seed=42)
        self.assertEqual(pso.num_particles, 10)
        self.assertEqual(pso.max_iterations, 5)
        self.assertEqual(pso.seed, 42)
        self.assertFalse(pso._initialized)

    def test_initialization_with_param_space(self):
        pso = ParticleSwarmOptimization(
            num_particles=5,
            max_iterations=3,
            min_values=self.min_values_continuous,
            max_values=self.max_values_continuous,
            seed=42,
        )
        pso.initialize(self.param_space_continuous)
        self.assertTrue(pso._initialized)
        self.assertEqual(len(pso.particles), 5)
        self.assertEqual(len(pso._params_pending_evaluation), 5)
        for particle in pso.particles:
            self.assertIn("x", particle["position"])
            self.assertIn("y", particle["position"])
            self.assertTrue(
                self.min_values_continuous["x"]
                <= particle["position"]["x"]
                <= self.max_values_continuous["x"]
            )
            self.assertTrue(
                self.min_values_continuous["y"]
                <= particle["position"]["y"]
                <= self.max_values_continuous["y"]
            )
            self.assertIn("x", particle["velocity"])
            self.assertIn("y", particle["velocity"])

    def test_initialization_with_categorical_param_space(self):
        pso = ParticleSwarmOptimization(num_particles=3, max_iterations=2, seed=123)
        pso.initialize(self.param_space_categorical)
        self.assertTrue(pso._initialized)
        self.assertEqual(len(pso.particles), 3)
        for particle in pso.particles:
            self.assertIn("cat1", particle["position"])
            self.assertIn(
                particle["position"]["cat1"], self.param_space_categorical["cat1"]
            )
            self.assertIn("cat2", particle["position"])
            self.assertIn(
                particle["position"]["cat2"], self.param_space_categorical["cat2"]
            )
            # Velocity for categorical might be 0 or handled differently
            self.assertEqual(particle["velocity"]["cat1"], 0.0)
            self.assertEqual(particle["velocity"]["cat2"], 0.0)

    def test_initialization_empty_param_space(self):
        pso = ParticleSwarmOptimization()
        with self.assertRaises(ValueError):
            pso.initialize({})

    def test_get_next_params_before_initialize(self):
        pso = ParticleSwarmOptimization()
        with self.assertRaises(RuntimeError):
            pso.get_next_params()

    def test_update_before_initialize(self):
        pso = ParticleSwarmOptimization()
        with self.assertRaises(RuntimeError):
            pso.update({"x": 1}, 10)

    def test_get_next_params_flow(self):
        pso = ParticleSwarmOptimization(
            num_particles=2,
            max_iterations=1,
            min_values=self.min_values_continuous,
            max_values=self.max_values_continuous,
            seed=42,
        )
        pso.initialize(self.param_space_continuous)

        params1 = pso.get_next_params()
        self.assertIsNotNone(params1)
        self.assertEqual(pso.current_particle_idx, 0)
        self.assertEqual(len(pso._params_pending_evaluation), 1)

        params2 = pso.get_next_params()
        self.assertIsNotNone(params2)
        self.assertEqual(pso.current_particle_idx, 1)
        self.assertEqual(len(pso._params_pending_evaluation), 0)

        # After getting all params for the first iteration,
        # _params_pending_evaluation should be empty.
        # Calling get_next_params again should trigger swarm update and prepare for next iter (or finish)
        # We need to call update for these params before the swarm state updates.
        pso.update(params1, dummy_evaluation_function(params1))
        pso.update(params2, dummy_evaluation_function(params2))

        # Now, get_next_params should trigger _update_swarm_state and potentially finish
        # if max_iterations is reached. Here max_iterations is 1.
        self.assertEqual(pso.current_iteration, 0)  # Iteration 0 params were given
        params_after_iter_0_update = pso.get_next_params()

        # Since max_iterations is 1, current_iteration becomes 1 after _update_swarm_state,
        # and is_finished() becomes true.
        self.assertEqual(pso.current_iteration, 1)
        self.assertTrue(pso.is_finished())
        self.assertIsNone(params_after_iter_0_update)

    # def test_update_and_best_score(self):
    #     pso = ParticleSwarmOptimization(
    #         num_particles=3,
    #         max_iterations=2,
    #         min_values=self.min_values_continuous,
    #         max_values=self.max_values_continuous,
    #         seed=42,
    #     )
    #     pso.initialize(self.param_space_continuous)

    #     self.assertEqual(pso.get_best_score(), -float("inf"))
    #     self.assertEqual(pso.get_best_params(), {})  # Initially empty

    #     params1 = pso.get_next_params()  # Particle 0
    #     score1 = 10
    #     pso.update(params1, score1)
    #     self.assertEqual(pso.gbest_score, 10)
    #     self.assertEqual(pso.gbest_position, params1)
    #     self.assertEqual(len(pso._scores_to_process), 1)

    #     params2 = pso.get_next_params()  # Particle 1
    #     score2 = 5
    #     pso.update(params2, score2)
    #     self.assertEqual(pso.gbest_score, 10)  # Still 10
    #     self.assertEqual(pso.gbest_position, params1)  # Still params1
    #     self.assertEqual(len(pso._scores_to_process), 2)

    #     params3 = pso.get_next_params()  # Particle 2
    #     score3 = 15
    #     pso.update(params3, score3)
    #     self.assertEqual(pso.gbest_score, 15)
    #     self.assertEqual(pso.gbest_position, params3)
    #     self.assertEqual(len(pso._scores_to_process), 3)

    #     # This will trigger _update_swarm_state internally
    #     next_round_params = pso.get_next_params()
    #     self.assertIsNotNone(next_round_params)  # Should start next iteration

    #     # Check pbest for the first particle (assuming it was particle 0)
    #     # _update_swarm_state processes _scores_to_process
    #     self.assertEqual(pso.particles[0]["pbest_score"], score1)
    #     self.assertEqual(pso.particles[0]["pbest_position"], params1)
    #     self.assertEqual(pso.particles[1]["pbest_score"], score2)
    #     self.assertEqual(pso.particles[1]["pbest_position"], params2)
    #     self.assertEqual(pso.particles[2]["pbest_score"], score3)
    #     self.assertEqual(pso.particles[2]["pbest_position"], params3)

    #     self.assertEqual(pso.get_best_score(), 15)
    #     self.assertEqual(pso.get_best_params(), params3)

    def test_is_finished(self):
        pso = ParticleSwarmOptimization(num_particles=1, max_iterations=1, seed=1)
        pso.initialize(self.param_space_categorical)  # Use any valid space
        self.assertFalse(pso.is_finished())

        params = pso.get_next_params()
        pso.update(params, 1.0)  # Score doesn't matter here

        # Calling get_next_params again will trigger end of iteration 0
        # and increment current_iteration to 1.
        pso.get_next_params()
        self.assertTrue(pso.is_finished())

    def test_reset(self):
        pso = ParticleSwarmOptimization(
            num_particles=2,
            max_iterations=1,
            min_values=self.min_values_continuous,
            max_values=self.max_values_continuous,
            seed=42,
        )
        pso.initialize(self.param_space_continuous)
        params1 = pso.get_next_params()
        pso.update(params1, 10)
        self.assertTrue(pso._initialized)
        self.assertNotEqual(pso.gbest_score, -float("inf"))

        pso.reset()
        self.assertFalse(pso._initialized)
        self.assertEqual(pso.gbest_score, -float("inf"))
        self.assertEqual(pso.current_iteration, 0)
        self.assertEqual(len(pso.particles), 0)
        self.assertEqual(len(pso._params_pending_evaluation), 0)
        with self.assertRaises(RuntimeError):  # Need to initialize again
            pso.get_next_params()

    def test_save_and_load_state_continuous(self):
        filepath = os.path.join(self.test_dir, "pso_state_continuous.json")
        pso1 = ParticleSwarmOptimization(
            num_particles=3,
            max_iterations=5,
            min_values=self.min_values_continuous,
            max_values=self.max_values_continuous,
            seed=42,
        )
        pso1.initialize(self.param_space_continuous)

        # Run a few steps
        for _ in range(2):  # Evaluate 2 particles
            params = pso1.get_next_params()
            pso1.update(params, dummy_evaluation_function(params))

        pso1.save_state(filepath)
        self.assertTrue(os.path.exists(filepath))

        pso2 = ParticleSwarmOptimization()  # Create a new instance
        pso2.load_state(filepath)

        self.assertEqual(pso1.num_particles, pso2.num_particles)
        self.assertEqual(pso1.max_iterations, pso2.max_iterations)
        self.assertEqual(pso1.w, pso2.w)
        self.assertEqual(pso1.c1, pso2.c1)
        self.assertEqual(pso1.c2, pso2.c2)
        self.assertEqual(pso1.min_values, pso2.min_values)
        self.assertEqual(pso1.max_values, pso2.max_values)
        self.assertEqual(pso1.seed, pso2.seed)
        self.assertEqual(pso1.param_space, pso2.param_space)
        self.assertEqual(pso1.gbest_score, pso2.gbest_score)
        # Comparing entire particle list can be tricky due to float precision / dict order in older pythons
        # Let's check key attributes
        self.assertEqual(len(pso1.particles), len(pso2.particles))
        if pso1.particles and pso2.particles:  # if particles exist
            self.assertEqual(
                pso1.particles[0]["pbest_score"], pso2.particles[0]["pbest_score"]
            )

        self.assertEqual(pso1._initialized, pso2._initialized)
        self.assertEqual(pso1.current_iteration, pso2.current_iteration)
        # _params_pending_evaluation might differ if one is a copy
        self.assertEqual(
            len(pso1._params_pending_evaluation), len(pso2._params_pending_evaluation)
        )

        # Continue with pso2
        params_from_pso2 = pso2.get_next_params()
        self.assertIsNotNone(params_from_pso2)

    def test_save_and_load_state_mixed(self):
        filepath = os.path.join(self.test_dir, "pso_state_mixed.json")
        pso1 = ParticleSwarmOptimization(
            num_particles=2,
            max_iterations=3,
            min_values=self.min_values_mixed,
            max_values=self.max_values_mixed,
            seed=123,
        )
        pso1.initialize(self.param_space_mixed)

        params_p1 = pso1.get_next_params()
        pso1.update(params_p1, dummy_evaluation_function(params_p1))

        pso1.save_state(filepath)
        self.assertTrue(os.path.exists(filepath))

        pso2 = ParticleSwarmOptimization()
        pso2.load_state(filepath)

        self.assertEqual(pso1.param_space, pso2.param_space)
        self.assertEqual(pso1.gbest_score, pso2.gbest_score)
        self.assertEqual(
            pso1.get_best_params().get("x"), pso2.get_best_params().get("x")
        )
        self.assertEqual(
            pso1.get_best_params().get("cat1"), pso2.get_best_params().get("cat1")
        )

        params_p2_from_loaded = (
            pso2.get_next_params()
        )  # Should be the second particle from iter 0
        self.assertIsNotNone(params_p2_from_loaded)
        pso2.update(
            params_p2_from_loaded, dummy_evaluation_function(params_p2_from_loaded)
        )

        # Trigger next iteration
        next_iter_params = pso2.get_next_params()
        self.assertIsNotNone(next_iter_params)
        self.assertEqual(pso2.current_iteration, 1)

    def test_get_best_params_score_initial_state(self):
        pso = ParticleSwarmOptimization(
            num_particles=2,
            min_values=self.min_values_continuous,
            max_values=self.max_values_continuous,
            seed=1,
        )
        # Before initialize
        self.assertEqual(pso.get_best_params(), {})
        self.assertEqual(pso.get_best_score(), -float("inf"))

        pso.initialize(self.param_space_continuous)
        # After initialize but before any updates
        # gbest_position is empty, gbest_score is -inf.
        # The method should return the first particle's position and its pbest_score (-inf initially)
        # or a copy of the first particle's position.
        self.assertNotEqual(pso.get_best_params(), {})
        self.assertTrue(
            all(k in pso.get_best_params() for k in self.param_space_continuous.keys())
        )
        self.assertEqual(pso.get_best_score(), -float("inf"))  # No scores recorded yet

        params1 = pso.get_next_params()
        pso.update(params1, 5.0)
        self.assertEqual(pso.get_best_params(), params1)
        self.assertEqual(pso.get_best_score(), 5.0)

        params2 = pso.get_next_params()
        pso.update(params2, 2.0)  # Lower score
        self.assertEqual(pso.get_best_params(), params1)  # Best is still params1
        self.assertEqual(pso.get_best_score(), 5.0)

    def test_edge_case_one_particle_one_iteration(self):
        pso = ParticleSwarmOptimization(
            num_particles=1,
            max_iterations=1,
            min_values=self.min_values_continuous,
            max_values=self.max_values_continuous,
            seed=42,
        )
        pso.initialize(self.param_space_continuous)

        params1 = pso.get_next_params()
        self.assertIsNotNone(params1)
        self.assertEqual(pso.current_particle_idx, 0)

        # Only one particle, so getting next params again before update should not yield another
        # It should return None after swarm update, as max_iterations is 1
        pso.update(params1, dummy_evaluation_function(params1))
        self.assertEqual(pso.current_iteration, 0)

        next_params = pso.get_next_params()  # This triggers _update_swarm_state
        self.assertEqual(pso.current_iteration, 1)
        self.assertTrue(pso.is_finished())
        self.assertIsNone(next_params)

        self.assertEqual(pso.get_best_params(), params1)
        self.assertEqual(pso.get_best_score(), dummy_evaluation_function(params1))

    # TODO: Add more specific tests for categorical parameter updates if complex logic is implemented.
    # The current categorical update is probabilistic and might be harder to assert specific outcomes
    # without more controlled scenarios or by inspecting internal probabilities (if they were exposed).


if __name__ == "__main__":
    unittest.main()
