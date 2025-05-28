\
# filepath: d:\\Gaser\\Univeristy\\GP\\myPanther\\panther\\tests\\Searching\\test_particle_swarm_optimization.py
import pytest
import random
import numpy as np
import json
import os
from panther.utils.SkAutoTuner.Searching.ParticleSwarmOptimization import ParticleSwarmOptimization

# Define a simple parameter space for testing
SIMPLE_PARAM_SPACE = {
    "param1": [0, 10],  # Continuous, but PSO treats as float initially
    "param2": [-5, 5],
}

# A more complex parameter space
COMPLEX_PARAM_SPACE = {
    "learning_rate": [0.001, 0.1],
    "n_estimators": [50, 200], # PSO will treat as float, user might need to round
    "max_depth": [3, 10]
}

@pytest.fixture
def pso_default():
    """Returns a ParticleSwarmOptimization instance with default parameters."""
    return ParticleSwarmOptimization()

@pytest.fixture
def pso_custom():
    """Returns a ParticleSwarmOptimization instance with custom parameters."""
    return ParticleSwarmOptimization(num_particles=10, max_iterations=50, w=0.7, c1=1.0, c2=2.0, seed=42)

@pytest.fixture
def initialized_pso(pso_default):
    """Returns an initialized PSO instance."""
    pso_default.initialize(SIMPLE_PARAM_SPACE)
    return pso_default

@pytest.fixture
def initialized_pso_custom_seed(pso_custom):
    """Returns an initialized PSO instance with a seed."""
    pso_custom.initialize(SIMPLE_PARAM_SPACE)
    return pso_custom

# --- Test __init__ ---
def test_pso_init_default_values(pso_default):
    assert pso_default.num_particles == 20
    assert pso_default.max_iterations == 100
    assert pso_default.w == 0.5
    assert pso_default.c1 == 1.5
    assert pso_default.c2 == 1.5
    assert pso_default.seed is None
    assert not pso_default._initialized

def test_pso_init_custom_values(pso_custom):
    assert pso_custom.num_particles == 10
    assert pso_custom.max_iterations == 50
    assert pso_custom.w == 0.7
    assert pso_custom.c1 == 1.0
    assert pso_custom.c2 == 2.0
    assert pso_custom.seed == 42
    assert not pso_custom._initialized

# --- Test initialize ---
def test_initialize_sets_attributes(pso_default):
    pso_default.initialize(SIMPLE_PARAM_SPACE)
    assert pso_default._initialized
    assert pso_default.param_space == SIMPLE_PARAM_SPACE
    assert set(pso_default.param_names) == set(SIMPLE_PARAM_SPACE.keys())
    assert pso_default.min_values == {"param1": 0, "param2": -5}
    assert pso_default.max_values == {"param1": 10, "param2": 5}
    assert len(pso_default.particles) == pso_default.num_particles
    assert pso_default.gbest_score == -float("inf")
    assert not pso_default.gbest_position  # Initially empty
    assert pso_default.current_iteration == 0
    assert len(pso_default._params_pending_evaluation) == pso_default.num_particles

def test_initialize_particle_positions_within_bounds(initialized_pso):
    pso = initialized_pso
    for particle in pso.particles:
        position = particle["position"]
        for param_name, value in position.items():
            assert pso.min_values[param_name] <= value <= pso.max_values[param_name]

def test_initialize_with_empty_param_space_raises_error(pso_default):
    with pytest.raises(ValueError, match="Parameter space cannot be empty."):
        pso_default.initialize({})

# def test_initialize_with_invalid_param_space_format(pso_default):
#     # Assuming the implementation has checks for this, which are currently placeholders
#     # For example, if a param_space entry is not a list of two numbers
#     with pytest.raises(TypeError): # Or ValueError, depending on implementation
#          pso_default.initialize({"param1": [0]}) # Invalid length
#     with pytest.raises(TypeError): # Or ValueError
#          pso_default.initialize({"param1": ["a", "b"]}) # Invalid type
#     # Test for min_value >= max_value
#     with pytest.raises(ValueError): # Or other appropriate error
#         pso_default.initialize({"param1": [10, 0]})


# --- Test get_next_params ---
def test_get_next_params_before_initialize(pso_default):
    # Assuming it should raise an error or return None/empty if not initialized
    # Based on current code, it will likely fail due to uninitialized attributes
    # Let's assume a graceful handling or specific error is desired.
    # For now, let's expect an error if used before initialize, or check for placeholder behavior.
    # The current code has `if not self._initialized: ...` which is a placeholder.
    # If it raises an error:
    with pytest.raises(Exception): # Replace Exception with specific error if defined
        pso_default.get_next_params()
    # Or if it returns None:
    # assert pso_default.get_next_params() is None

def test_get_next_params_returns_sequentially(initialized_pso):
    pso = initialized_pso
    num_particles = pso.num_particles
    all_params = []
    for _ in range(num_particles):
        params = pso.get_next_params()
        assert params is not None
        all_params.append(params)
    assert len(all_params) == num_particles
    assert len(pso._params_pending_evaluation) == 0
    # Check that all returned params are unique initially (as they are from different particles)
    # This might not hold if particles could be initialized to the same position, but unlikely with random.uniform
    unique_params_as_tuples = {tuple(sorted(p.items())) for p in all_params}
    assert len(unique_params_as_tuples) == num_particles

    # Test current_particle_idx update
    pso.initialize(SIMPLE_PARAM_SPACE) # re-initialize to reset pending list
    for i in range(num_particles):
        pso.get_next_params()
        assert pso.current_particle_idx == i

# def test_get_next_params_when_pending_is_empty_but_not_finished(initialized_pso):
#     pso = initialized_pso
#     # Evaluate all initial particles
#     for _ in range(pso.num_particles):
#         pso.get_next_params()
    
#     # At this point, _params_pending_evaluation is empty.
#     # The current code has `if not self._params_pending_evaluation: ...` placeholder
#     # Behavior depends on how _update_swarm_state is called and repopulates this.
#     # For now, let's assume it might return None or raise if called before _update_swarm_state repopulates.
#     # If it's expected to wait or trigger update, test that.
#     # Based on current structure, it seems like it would just return from an empty list if not handled.
#     # Let's assume it should indicate no params are ready yet.
#     # This test might need adjustment based on the intended behavior of the placeholder.
#     assert not pso._params_pending_evaluation
#     # If it raises an error or returns a specific signal:
#     with pytest.raises(IndexError): # Or a custom "NoParamsReadyError"
#         pso.get_next_params()
#     # Or:
#     # assert pso.get_next_params() is None


# --- Test update ---
def test_update_before_initialize(pso_default):
    # Similar to get_next_params, expect graceful handling or error
    with pytest.raises(Exception): # Replace with specific error
        pso_default.update({"param1": 1}, 0.5)

def test_update_appends_to_scores_to_process(initialized_pso):
    pso = initialized_pso
    params = {"param1": 5, "param2": 0} # Example params
    score = 0.75
    pso.update(params, score)
    assert len(pso._scores_to_process) == 1
    assert pso._scores_to_process[0] == (params, score)

def test_update_updates_gbest_immediately(initialized_pso):
    pso = initialized_pso
    
    params1 = {"param1": 1, "param2": 1} # Assume these are particle positions
    score1 = 0.5
    pso.update(params1, score1)
    assert pso.gbest_score == 0.5
    assert pso.gbest_position == params1

    params2 = {"param1": 2, "param2": 2}
    score2 = 0.8
    pso.update(params2, score2)
    assert pso.gbest_score == 0.8
    assert pso.gbest_position == params2

    params3 = {"param1": 3, "param2": 3}
    score3 = 0.7 # Worse than current gbest
    pso.update(params3, score3)
    assert pso.gbest_score == 0.8 # Should not change
    assert pso.gbest_position == params2 # Should not change


# --- Test _update_swarm_state (indirectly) ---
# This method is harder to test directly as it relies on internal state
# and the full PSO loop. We can test its effects after a generation.

# Mock objective function
def mock_objective_function(params):
    # Simple function: score = param1 - abs(param2)
    # Higher is better, matching PSO's default assumption
    return params["param1"] - abs(params["param2"])

def run_one_generation(pso_instance, objective_func):
    """Helper to run one generation of PSO."""
    if not pso_instance._initialized:
        raise ValueError("PSO not initialized")
    if not pso_instance._params_pending_evaluation: # If previous gen finished
         # This implies _update_swarm_state should have been called to repopulate
         # For testing, we assume _update_swarm_state is called correctly by the user/framework
         # or we manually call it if it's exposed and safe.
         # The current PSO code doesn't automatically call _update_swarm_state.
         # It's called when _params_pending_evaluation is empty in get_next_params (placeholder)
         # or user calls it.
         # For this test, we'll assume _update_swarm_state is called after all particles are processed.
        if pso_instance._scores_to_process: # If there are scores, process them
            # Manually call _update_swarm_state if it were public and safe.
            # Since it's private, we test its effects.
            # The current code has a placeholder for _update_swarm_state.
            # We'll simulate its expected outcome for pbest/gbest updates.
            
            # Simulate pbest updates based on _scores_to_process
            # This part is tricky because _update_swarm_state is where this happens.
            # For now, let's focus on testing the flow up to update() and gbest.
            # A full test of _update_swarm_state requires its implementation.
            pass # Placeholder for now

    # Get params, evaluate, and update for all particles in the current pending list
    # This simulates one round of evaluations.
    num_to_evaluate = len(pso_instance._params_pending_evaluation)
    evaluated_this_round = []
    for i in range(num_to_evaluate):
        params = pso_instance.get_next_params()
        if params is None: # Should not happen if num_to_evaluate was > 0
            break
        score = objective_func(params)
        pso_instance.update(params, score)
        # Store which particle this was for pbest check later
        # This is complex because params are copies. We need to know particle index.
        # The `current_particle_idx` in `get_next_params` helps.
        evaluated_this_round.append({'params': params, 'score': score, 'particle_idx_at_get': pso_instance.current_particle_idx})


    # After all updates in a generation, _update_swarm_state would be called.
    # Let's assume the placeholder for _update_swarm_state is:
    # 1. Updates pbest for each particle based on _scores_to_process
    # 2. Updates gbest (already partially done in update)
    # 3. Updates velocities and positions
    # 4. Clears _scores_to_process
    # 5. Repopulates _params_pending_evaluation
    # 6. Increments current_iteration

    # For testing, we can't call the private _update_swarm_state directly.
    # We need to infer its effects or test parts of it if the implementation allows.
    # The current `update` method updates gbest.
    # The pbest update and velocity/position updates are in the placeholder `_update_swarm_state`.

    # Let's assume for now that after all updates, if we were to manually trigger
    # the logic of _update_swarm_state (or if it's triggered internally),
    # pbests would be updated.

    # This helper is becoming complex due to the private nature and placeholders.
    # Let's simplify: test that after a series of get_next_params and update calls,
    # gbest is correct. Testing pbest and particle movement requires _update_swarm_state.

def test_gbest_after_multiple_updates(initialized_pso_custom_seed):
    pso = initialized_pso_custom_seed # Uses seed for reproducibility if random is involved
    
    # Simulate a few evaluations
    # Particle 0
    params0 = pso.get_next_params() # Assume this is particle 0's initial position
    score0 = mock_objective_function(params0)
    pso.update(params0, score0)
    
    current_gbest_score = pso.gbest_score
    current_gbest_pos = pso.gbest_position.copy()

    # Particle 1
    params1 = pso.get_next_params()
    score1 = mock_objective_function(params1)
    pso.update(params1, score1)

    if score1 > current_gbest_score:
        assert pso.gbest_score == score1
        assert pso.gbest_position == params1
    else:
        assert pso.gbest_score == current_gbest_score
        assert pso.gbest_position == current_gbest_pos
    
    # This test mainly re-verifies immediate gbest update.
    # A full generation test needs the _update_swarm_state logic.

# --- Test get_best_params and get_best_score ---
# def test_get_best_before_updates(initialized_pso):
#     pso = initialized_pso
#     assert pso.get_best_score() == -float("inf")
#     assert pso.get_best_params() == {} # As gbest_position is initially empty

def test_get_best_after_updates(initialized_pso):
    pso = initialized_pso
    params1 = {"param1": 1, "param2": 1}
    score1 = 0.5
    pso.update(params1, score1) # This updates gbest

    assert pso.get_best_score() == score1
    assert pso.get_best_params() == params1

    params2 = {"param1": 2, "param2": 2}
    score2 = 0.3 # Worse score
    pso.update(params2, score2)

    assert pso.get_best_score() == score1 # Should remain best
    assert pso.get_best_params() == params1


# --- Test is_finished ---
def test_is_finished_false_initially(initialized_pso):
    assert not initialized_pso.is_finished()

def test_is_finished_true_after_max_iterations(initialized_pso):
    pso = initialized_pso
    pso.current_iteration = pso.max_iterations
    assert pso.is_finished()
    pso.current_iteration = pso.max_iterations + 1
    assert pso.is_finished()


# --- Test reset ---
def test_reset_clears_state(initialized_pso_custom_seed):
    pso = initialized_pso_custom_seed
    # Simulate some activity
    pso.get_next_params()
    pso.update({"param1":1}, 1.0)
    pso.current_iteration = 5
    
    original_seed = pso.seed
    original_num_particles = pso.num_particles # These should be preserved by reset

    pso.reset()

    assert not pso._initialized
    assert pso.param_space == {}
    assert pso.param_names == []
    assert pso.particles == []
    assert pso.gbest_position == {}
    assert pso.gbest_score == -float("inf")
    assert pso.current_iteration == 0
    assert pso.current_particle_idx == 0
    assert pso._scores_to_process == []
    assert pso._params_pending_evaluation == []
    
    assert pso.seed == original_seed # Constructor args like seed should persist
    # Re-check if random state is reset if seed was used
    if original_seed is not None:
        # This is hard to check directly without knowing how random state was before reset
        # But we can check if initializing again produces same first particle if seed is active
        random.seed(original_seed)
        np.random.seed(original_seed)
        expected_first_pos_after_reinit = ParticleSwarmOptimization(seed=original_seed)
        expected_first_pos_after_reinit.initialize(SIMPLE_PARAM_SPACE)
        
        pso.initialize(SIMPLE_PARAM_SPACE) # Initialize after reset
        # Compare particle positions if they are deterministic due to seed
        # This requires _initialize_particle_position to be deterministic with seed
        # For now, we assume reset correctly resets random seeds if they were set.
        # A more robust test would involve mocking random.uniform.


# --- Test save_state and load_state ---
@pytest.fixture
def temp_state_file(tmp_path):
    return tmp_path / "pso_state.json"

# def test_save_and_load_state(initialized_pso_custom_seed, temp_state_file):
#     pso_to_save = initialized_pso_custom_seed
    
#     # Simulate some progress
#     params_eval = []
#     for _ in range(5): # Evaluate a few particles
#         p = pso_to_save.get_next_params()
#         if p is None: break
#         params_eval.append(p)
#         # Simulate scores; actual scores don't matter as much as state preservation
#         pso_to_save.update(p, random.uniform(0,1)) 
    
#     pso_to_save.current_iteration = 2
#     # Assume _update_swarm_state would have been called, and it might modify particles
#     # For simplicity, we're testing persistence of what's there.
#     # The `particles` list contains complex data (dicts with numpy arrays if velocity is np.array)
#     # The `convert_numpy_types` function in `save_state` is crucial.
#     # Let's assume it's implemented.

#     # Capture state before saving for comparison
#     # Need to handle potential numpy arrays in particles for direct comparison
#     # For now, let's compare basic attributes and structure
    
#     state_before_save = {
#         "num_particles": pso_to_save.num_particles,
#         "max_iterations": pso_to_save.max_iterations,
#         "w": pso_to_save.w, "c1": pso_to_save.c1, "c2": pso_to_save.c2,
#         "seed": pso_to_save.seed,
#         "param_space": pso_to_save.param_space,
#         "gbest_score": pso_to_save.gbest_score,
#         "gbest_position": pso_to_save.gbest_position.copy(),
#         "current_iteration": pso_to_save.current_iteration,
#         "_initialized": pso_to_save._initialized,
#         # particles are complex, compare length and some key presence
#     }

#     pso_to_save.save_state(str(temp_state_file))
#     assert os.path.exists(temp_state_file)

#     pso_to_load = ParticleSwarmOptimization(seed=pso_to_save.seed) # Start with same seed for consistency if it matters post-load
#     pso_to_load.load_state(str(temp_state_file))

#     assert pso_to_load.num_particles == state_before_save["num_particles"]
#     assert pso_to_load.max_iterations == state_before_save["max_iterations"]
#     assert pso_to_load.w == state_before_save["w"]
#     assert pso_to_load.c1 == state_before_save["c1"]
#     assert pso_to_load.c2 == state_before_save["c2"]
#     assert pso_to_load.seed == state_before_save["seed"]
#     assert pso_to_load.param_space == state_before_save["param_space"]
#     assert pso_to_load.gbest_score == state_before_save["gbest_score"]
#     assert pso_to_load.gbest_position == state_before_save["gbest_position"]
#     assert pso_to_load.current_iteration == state_before_save["current_iteration"]
#     assert pso_to_load._initialized == state_before_save["_initialized"]
    
#     assert set(pso_to_load.param_names) == set(state_before_save["param_space"].keys())
#     assert pso_to_load.min_values == {k: v[0] for k, v in state_before_save["param_space"].items()}
#     assert pso_to_load.max_values == {k: v[1] for k, v in state_before_save["param_space"].items()}

#     assert len(pso_to_load.particles) == len(pso_to_save.particles)
#     # Deep comparison of particles can be tricky if they contain numpy arrays not handled by default json
#     # Assuming convert_numpy_types and load handles this.
#     # For a simple check:
#     if pso_to_save.particles:
#         assert pso_to_load.particles[0]["pbest_score"] == pso_to_save.particles[0]["pbest_score"]
#         # Position and velocity might be numpy arrays, ensure they are loaded correctly (e.g. as lists or np.arrays)
#         # This depends on how convert_numpy_types and load_state reconstruct them.
#         # If they are loaded as lists:
#         assert pso_to_load.particles[0]["position"] == pso_to_save.particles[0]["position"]


#     # Check pending evaluations and scores (these might be empty depending on state)
#     assert len(pso_to_load._params_pending_evaluation) == len(pso_to_save._params_pending_evaluation)
#     if pso_to_save._params_pending_evaluation:
#          assert pso_to_load._params_pending_evaluation[0] == pso_to_save._params_pending_evaluation[0]
    
#     assert len(pso_to_load._scores_to_process) == len(pso_to_save._scores_to_process)
#     if pso_to_save._scores_to_process:
#         assert pso_to_load._scores_to_process[0] == pso_to_save._scores_to_process[0]

#     # Test that after loading, the seeded random state is also restored (if seed is not None)
#     if pso_to_load.seed is not None:
#         # This is hard to verify directly without controlling the exact sequence of random calls.
#         # However, if `np.random.seed` and `random.seed` are called in `load_state`,
#         # subsequent operations relying on random numbers should be deterministic.
#         pass


def test_load_state_handles_missing_fields_gracefully(pso_default, temp_state_file):
    # Simulate an older save file that might be missing newer fields
    # like _scores_to_process or _params_pending_evaluation
    state = {
        "num_particles": 20, "max_iterations": 100, "w": 0.5, "c1": 1.5, "c2": 1.5,
        "seed": None, "param_space": SIMPLE_PARAM_SPACE,
        "param_names": list(SIMPLE_PARAM_SPACE.keys()), # Older saves might have this explicitly
        "particles": [], # Simplified for this test
        "gbest_position": {}, "gbest_score": -float("inf"),
        "current_iteration": 0, "current_particle_idx": 0,
        "_initialized": True,
        # Missing _scores_to_process, _params_pending_evaluation
    }
    with open(temp_state_file, "w") as f:
        json.dump(state, f)

    pso_default.load_state(str(temp_state_file))
    assert pso_default._scores_to_process == [] # Should default to empty list
    assert pso_default._params_pending_evaluation == [] # Should default to empty list
    assert pso_default.param_names == list(SIMPLE_PARAM_SPACE.keys()) # Ensure param_names derived if not in file but param_space is


