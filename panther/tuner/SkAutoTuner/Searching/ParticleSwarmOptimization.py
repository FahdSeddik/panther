import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .SearchAlgorithm import SearchAlgorithm


class ParticleSwarmOptimization(SearchAlgorithm):
    """
    Particle Swarm Optimization algorithm.
    """

    def __init__(
        self,
        num_particles: int = 20,
        max_iterations: int = 100,
        w: float = 0.5,
        c1: float = 1.5,
        c2: float = 1.5,
        min_values: Optional[Dict[str, Any]] = None,
        max_values: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize PSO.

        Args:
            num_particles: Number of particles in the swarm.
            max_iterations: Maximum number of iterations.
            w: Inertia weight.
            c1: Cognitive coefficient.
            c2: Social coefficient.
            min_values: Dictionary of minimum values for each parameter (for continuous params).
            max_values: Dictionary of maximum values for each parameter (for continuous params).
            seed: Random seed for reproducibility.
        """
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.min_values = min_values if min_values else {}
        self.max_values = max_values if max_values else {}
        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        self.param_space: Dict[str, List] = {}
        self.param_names: List[str] = []
        self.particles: List[Dict[str, Any]] = []
        self.gbest_position: Dict[str, Any] = {}
        self.gbest_score: float = -float("inf")
        self.current_iteration: int = 0
        self.current_particle_idx: int = 0
        self._initialized: bool = False
        self._scores_to_process: List[Tuple[Dict[str, Any], float]] = []
        self._params_pending_evaluation: List[Dict[str, Any]] = []

    def initialize(self, param_space: Dict[str, List]):
        """
        Initialize the search algorithm with the parameter space.
        """
        if not param_space:
            raise ValueError("Parameter space cannot be empty.")

        self.param_space = param_space
        self.param_names = list(param_space.keys())
        self.particles = []
        self.gbest_position = {}
        self.gbest_score = -float("inf")  # Assuming higher is better
        self.current_iteration = 0
        self.current_particle_idx = 0
        self._scores_to_process = []
        self._params_pending_evaluation = []

        for _ in range(self.num_particles):
            position = self._initialize_particle_position()
            velocity = self._initialize_particle_velocity()

            particle = {
                "position": position,
                "velocity": velocity,
                "pbest_position": position.copy(),
                "pbest_score": -float("inf"),
            }
            self.particles.append(particle)

        # Initial gbest will be set after first evaluation
        self._initialized = True
        self._params_pending_evaluation = [p["position"] for p in self.particles]

    def _initialize_particle_position(self) -> Dict[str, Any]:
        position = {}
        for param_name in self.param_names:
            values = self.param_space[param_name]
            if isinstance(values, list) and values:  # Categorical
                position[param_name] = random.choice(values)
            elif (
                param_name in self.min_values and param_name in self.max_values
            ):  # Continuous
                position[param_name] = random.uniform(
                    self.min_values[param_name], self.max_values[param_name]
                )
            else:  # Fallback or error
                raise ValueError(
                    f"Parameter '{param_name}' is not well-defined for initialization. "
                    "Provide a list of categorical values or min/max for continuous values."
                )
        return position

    def _initialize_particle_velocity(
        self,
    ) -> Union[Dict[str, float], Dict[str, int], None]:
        velocity = {}
        for param_name in self.param_names:
            # For categorical parameters, velocity concept might not directly apply or needs adaptation.
            # For simplicity, we'll focus on continuous parameters for velocity.
            # If it's categorical, we can assign a default or skip.
            if param_name in self.min_values and param_name in self.max_values:
                # Initialize velocity typically between some range related to the parameter's range
                v_max = (
                    self.max_values[param_name] - self.min_values[param_name]
                ) * 0.1  # e.g., 10% of range
                velocity[param_name] = random.uniform(-v_max, v_max)
            else:  # Categorical or not specified for continuous treatment
                velocity[param_name] = 0.0  # Or handle differently
        return velocity

    def get_next_params(self) -> Any:
        """
        Get the next set of parameters to try.
        """
        if not self._initialized:
            raise RuntimeError("PSO not initialized. Call initialize() first.")

        if not self._params_pending_evaluation:
            # All particles of the current iteration evaluated, update swarm for next iteration
            self._update_swarm_state()
            self.current_iteration += 1
            if self.is_finished():
                return None  # Search finished
            self._params_pending_evaluation = [p["position"] for p in self.particles]

        if not self._params_pending_evaluation:  # Should not happen if not finished
            return None

        next_params = self._params_pending_evaluation.pop(0)
        self.current_particle_idx = (
            self.num_particles - len(self._params_pending_evaluation) - 1
        )  # Update based on popped
        return next_params

    def _update_swarm_state(self):
        """
        Called after all particles in a generation have been evaluated.
        Processes the scores and updates velocities and positions for all particles.
        """
        if not self._scores_to_process:
            # This can happen in the very first call if update() isn't called before _update_swarm_state
            # Or if get_next_params was called multiple times without update
            return

        # Apply updates based on collected scores
        for particle_idx, (params, score) in enumerate(self._scores_to_process):
            # Ensure we are updating the correct particle if scores came in out of order (though not expected with current flow)
            # For now, assume scores are processed for particles 0 to N-1 in order
            particle = self.particles[
                particle_idx % self.num_particles
            ]  # Use modulo if processing more scores than particles

            if score > particle["pbest_score"]:
                particle["pbest_score"] = score
                particle["pbest_position"] = (
                    params.copy()
                )  # params is the position that was evaluated

            if score > self.gbest_score:
                self.gbest_score = score
                self.gbest_position = params.copy()

        self._scores_to_process = []  # Clear processed scores

        # Now update velocities and positions for all particles
        for particle in self.particles:
            new_velocity = {}
            new_position = {}

            for param_name in self.param_names:
                if (
                    param_name in self.min_values and param_name in self.max_values
                ):  # Continuous
                    r1, r2 = random.random(), random.random()

                    current_pos = particle["position"][param_name]
                    current_vel = particle["velocity"][param_name]
                    pbest_pos = particle["pbest_position"][param_name]
                    gbest_pos = self.gbest_position.get(
                        param_name, current_pos
                    )  # Fallback if gbest not set for param

                    cognitive_component = self.c1 * r1 * (pbest_pos - current_pos)
                    social_component = self.c2 * r2 * (gbest_pos - current_pos)

                    new_vel = (
                        self.w * current_vel + cognitive_component + social_component
                    )

                    # Optional: Velocity clamping
                    # v_max = (self.max_values[param_name] - self.min_values[param_name]) * 0.5
                    # new_vel = max(min(new_vel, v_max), -v_max)

                    new_pos = current_pos + new_vel

                    # Clamping position to bounds
                    new_pos = max(
                        min(new_pos, self.max_values[param_name]),
                        self.min_values[param_name],
                    )

                    new_velocity[param_name] = new_vel
                    new_position[param_name] = new_pos

                else:  # Categorical - position updated by different means (e.g. probability, or no direct velocity update)
                    # For simplicity, categorical params might change based on gbest or pbest directly,
                    # or use a probability to switch based on velocity-like "influence"
                    # Here we just keep the old position/velocity for categorical params if not handled by velocity update.
                    new_velocity[param_name] = particle["velocity"].get(
                        param_name, 0.0
                    )  # Keep old or 0

                    # For categorical, a common approach is to not use velocity but to probabilistically switch
                    # to pbest or gbest. Or, one might discretize velocity.
                    # Simplest: if influence (e.g. from social/cognitive) is high, pick from pbest/gbest.
                    # For now, let's try a simple approach for categorical:
                    # If random number < some_prob, take from pbest, else if < some_other_prob, take from gbest, else keep.
                    # This is a placeholder for a more sophisticated categorical PSO handling.

                    # Simplified: if particle is "moving" towards a better categorical, it might adopt it.
                    # A more robust way: model probability of choosing a category.
                    # For now, keep it simple: update based on a mix strategy for categorical values if needed.
                    # Let's try a probabilistic switch for categorical parameters
                    rand_val = random.random()
                    if (
                        rand_val < 0.33 and param_name in particle["pbest_position"]
                    ):  # Probability to switch to pbest
                        new_position[param_name] = particle["pbest_position"][
                            param_name
                        ]
                    elif (
                        rand_val < 0.66 and param_name in self.gbest_position
                    ):  # Probability to switch to gbest
                        new_position[param_name] = self.gbest_position[param_name]
                    else:  # Keep current
                        new_position[param_name] = particle["position"][param_name]

            particle["velocity"] = new_velocity
            particle["position"] = new_position

    def update(self, params: Dict[str, Any], score: float):
        """
        Update the search algorithm with the results of the latest trial.
        This method is called for each particle after its evaluation.
        The actual swarm update (velocities, positions) happens in _update_swarm_state
        once all particles in a generation are evaluated.
        """
        if not self._initialized:
            raise RuntimeError("PSO not initialized. Call initialize() first.")

        # Store the score and params to be processed together at the end of the generation
        self._scores_to_process.append((params, score))

        # Find which particle these params belong to for immediate pbest/gbest update
        # This is a bit tricky if params are copied. Assuming params is the *exact* dict
        # from a particle's position that was returned by get_next_params.
        # A better way would be to track particle_idx alongside params.
        # For now, let's assume the update calls correspond to the order of get_next_params calls within a generation.

        # The current_particle_idx in the class tracks the *next* particle to be given out by get_next_params,
        # or rather, (total_evaluated_in_current_generation -1).
        # When update is called, it's for the particle whose params were *just* returned.

        # For simplicity, let's update pbest and gbest immediately.
        # The full velocity/position update will still happen in _update_swarm_state.
        # This allows gbest to be updated as soon as a better score is found.

        # Identify the particle that was just evaluated.
        # This logic assumes `update` is called for the particle whose parameters were most recently returned by `get_next_params`.
        # `self.current_particle_idx` at the time `get_next_params` returned `params` is the correct index.
        # However, `self.current_particle_idx` might have advanced if `get_next_params` was called again before this `update`.
        # A safer way: `get_next_params` should return a (particle_idx, params) tuple or an identifier.
        # For now, let's assume a sequential call pattern: get_next -> evaluate -> update.

        # Find the particle that generated these params.
        # This can be error-prone if params are modified or copied.
        # A robust approach: `get_next_params` tags returned params with particle_id or uses `current_particle_idx`
        # when it was called.
        # For now, let's search by `params` object identity or value, assuming params is `particle['position']`.

        found_particle = None  # noqa: F841
        particle_index_for_update = -1  # noqa: F841

        # Attempt to find the particle that corresponds to 'params'
        # This is not robust if 'params' is a copy.
        # Let's assume 'update' is called for the particle at `self.current_particle_idx`
        # which was set when `get_next_params` was last called for this particle.
        # `self.current_particle_idx` refers to the particle that was *just evaluated*.
        # When `get_next_params()` is called, it sets `self.current_particle_idx` to the index of the particle
        # whose parameters are being returned.

        # Simpler: The `_scores_to_process` list will be processed in order.
        # Immediate update of pbest/gbest:
        target_particle = None  # noqa: F841
        # We need to identify which particle these 'params' belong to to update its pbest.
        # If `get_next_params` returns `particle['position']`, and that dict is passed to `update`,
        # we can compare. However, `params` might be a copy.
        # Let's assume `_scores_to_process` handles this sequentially for now.
        # The logic in `_update_swarm_state` will handle pbest for each particle based on `_scores_to_process`.

        # However, gbest can be updated immediately.
        if score > self.gbest_score:
            self.gbest_score = score
            self.gbest_position = params.copy()
            # print(f"New gbest: {self.gbest_score} with params {self.gbest_position}")

        # The pbest update for the specific particle will happen in _update_swarm_state
        # when _scores_to_process is iterated. This is because we need to associate
        # 'params' and 'score' with the correct particle, and _scores_to_process
        # implies an order.

    def save_state(self, filepath: str):
        """
        Save the current state of the search algorithm to a file.
        """
        import json

        state = {
            "num_particles": self.num_particles,
            "max_iterations": self.max_iterations,
            "w": self.w,
            "c1": self.c1,
            "c2": self.c2,
            "min_values": self.min_values,
            "max_values": self.max_values,
            "seed": self.seed,
            "param_space": self.param_space,
            "param_names": self.param_names,
            "particles": self.particles,  # Note: np.float might not be JSON serializable directly
            "gbest_position": self.gbest_position,
            "gbest_score": self.gbest_score,
            "current_iteration": self.current_iteration,
            "current_particle_idx": self.current_particle_idx,  # refers to next particle from _params_pending_evaluation
            "_initialized": self._initialized,
            "_scores_to_process": self._scores_to_process,
            "_params_pending_evaluation": self._params_pending_evaluation,
        }

        # Convert numpy types to standard python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(i) for i in obj]
            elif isinstance(
                obj,
                (
                    np.int_,
                    np.intc,
                    np.intp,
                    np.int8,
                    np.int16,
                    np.int32,
                    np.int64,
                    np.uint8,
                    np.uint16,
                    np.uint32,
                    np.uint64,
                ),
            ):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        serializable_state = convert_numpy_types(state)

        with open(filepath, "w") as f:
            json.dump(serializable_state, f, indent=4)

    def load_state(self, filepath: str):
        """
        Load the state of the search algorithm from a file.
        """
        import json

        with open(filepath, "r") as f:
            state = json.load(f)

        self.num_particles = state["num_particles"]
        self.max_iterations = state["max_iterations"]
        self.w = state["w"]
        self.c1 = state["c1"]
        self.c2 = state["c2"]
        self.min_values = state.get("min_values", {})  # Handle older saves
        self.max_values = state.get("max_values", {})  # Handle older saves
        self.seed = state["seed"]
        if self.seed is not None:  # Re-apply seed if loaded
            random.seed(self.seed)
            np.random.seed(self.seed)

        self.param_space = state["param_space"]
        self.param_names = state["param_names"]
        self.particles = state["particles"]
        self.gbest_position = state["gbest_position"]
        self.gbest_score = state["gbest_score"]
        self.current_iteration = state["current_iteration"]
        self.current_particle_idx = state["current_particle_idx"]
        self._initialized = state["_initialized"]
        self._scores_to_process = state.get(
            "_scores_to_process", []
        )  # Handle older saves
        self._params_pending_evaluation = state.get("_params_pending_evaluation", [])

        if (
            not self._initialized
        ):  # If loading a state that wasn't fully initialized somehow
            print(
                "Warning: Loaded state was not marked as initialized. Re-check logic if issues arise."
            )
        # Ensure param_names is consistent if it was missing from an old save file
        if not self.param_names and self.param_space:
            self.param_names = list(self.param_space.keys())

    def get_best_params(self) -> Dict[str, Any]:
        """
        Get the best set of parameters found so far.
        """
        if not self.gbest_position and self._initialized:
            # If gbest is empty but we have particles, pick the best pbest as a fallback
            # This might happen if no updates occurred yet or only one particle.
            best_p_score = -float("inf")
            best_p_pos = None
            for p in self.particles:
                if p["pbest_score"] > best_p_score:
                    best_p_score = p["pbest_score"]
                    best_p_pos = p["pbest_position"]
            if best_p_pos:
                return best_p_pos.copy()
            # If still no best, and particles exist, return first particle's initial position
            elif self.particles:
                return self.particles[0]["position"].copy()

        return self.gbest_position.copy() if self.gbest_position else {}

    def get_best_score(self) -> float:
        """
        Get the best score achieved so far.
        """
        if self.gbest_score == -float("inf") and self._initialized:
            best_p_score = -float("inf")
            for p in self.particles:
                if p["pbest_score"] > best_p_score:
                    best_p_score = p["pbest_score"]
            return (
                best_p_score if best_p_score != -float("inf") else -float("inf")
            )  # Or appropriate default

        return self.gbest_score

    def reset(self):
        """
        Reset the search algorithm to its initial state.
        """
        # Re-initialize attributes as in __init__
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        # Keep original constructor args, but reset dynamic state
        self.param_space = {}  # Will be set by initialize
        self.param_names = []
        self.particles = []
        self.gbest_position = {}
        self.gbest_score = -float("inf")
        self.current_iteration = 0
        self.current_particle_idx = 0
        self._initialized = False
        self._scores_to_process = []
        self._params_pending_evaluation = []
        # Note: self.param_space should be re-provided via initialize() call after reset

    def is_finished(self) -> bool:
        """
        Check if the search algorithm has finished its search.
        """
        return self.current_iteration >= self.max_iterations
