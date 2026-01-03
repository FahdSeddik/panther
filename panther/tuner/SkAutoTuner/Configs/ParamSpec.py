"""
First-class parameter distribution specifications for modern HPO.

Enables expressing categorical, integer, and float distributions explicitly,
which can be mapped to Optuna distributions or other HPO frameworks.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union


@dataclass
class Categorical:
    """
    A categorical parameter that takes values from a fixed set of choices.

    Examples:
        Categorical([1, 2, 3])  # Integer choices
        Categorical(["relu", "gelu", "silu"])  # String choices
        Categorical([True, False])  # Boolean choices
    """

    choices: Sequence[Any]

    def __post_init__(self):
        if not self.choices:
            raise ValueError("Categorical choices must not be empty")
        # Convert to list for consistency
        self.choices = list(self.choices)

    def __repr__(self) -> str:
        return f"Categorical({self.choices})"


@dataclass
class Int:
    """
    An integer parameter within a range [low, high].

    Args:
        low: Lower bound (inclusive)
        high: Upper bound (inclusive)
        step: Step size for discrete values (default: 1)
        log: Whether to sample in log scale (useful for parameters like learning rate)

    Examples:
        Int(1, 100)  # Integer from 1 to 100
        Int(8, 512, step=8)  # Multiples of 8 from 8 to 512
        Int(1, 1000, log=True)  # Log-scale integer sampling
    """

    low: int
    high: int
    step: int = 1
    log: bool = False

    def __post_init__(self):
        if self.low > self.high:
            raise ValueError(f"Int: low ({self.low}) must be <= high ({self.high})")
        if self.step < 1:
            raise ValueError(f"Int: step ({self.step}) must be >= 1")
        if self.log and self.low <= 0:
            raise ValueError(f"Int: log=True requires low > 0, got {self.low}")

    def __repr__(self) -> str:
        parts = [f"{self.low}", f"{self.high}"]
        if self.step != 1:
            parts.append(f"step={self.step}")
        if self.log:
            parts.append("log=True")
        return f"Int({', '.join(parts)})"


@dataclass
class Float:
    """
    A floating-point parameter within a range [low, high].

    Args:
        low: Lower bound (inclusive)
        high: Upper bound (inclusive)
        step: Step size for discrete values (None for continuous)
        log: Whether to sample in log scale

    Examples:
        Float(0.0, 1.0)  # Continuous float from 0 to 1
        Float(1e-5, 1e-1, log=True)  # Log-scale float (e.g., learning rate)
        Float(0.1, 1.0, step=0.1)  # Discrete float values
    """

    low: float
    high: float
    step: Optional[float] = None
    log: bool = False

    def __post_init__(self):
        if self.low > self.high:
            raise ValueError(f"Float: low ({self.low}) must be <= high ({self.high})")
        if self.step is not None and self.step <= 0:
            raise ValueError(f"Float: step ({self.step}) must be > 0")
        if self.log and self.low <= 0:
            raise ValueError(f"Float: log=True requires low > 0, got {self.low}")

    def __repr__(self) -> str:
        parts = [f"{self.low}", f"{self.high}"]
        if self.step is not None:
            parts.append(f"step={self.step}")
        if self.log:
            parts.append("log=True")
        return f"Float({', '.join(parts)})"


# Type alias for any parameter specification
ParamSpec = Union[Categorical, Int, Float, List[Any]]


def is_param_spec(value: Any) -> bool:
    """Check if a value is a ParamSpec type."""
    return isinstance(value, (Categorical, Int, Float, list))


def to_categorical(value: Any) -> Categorical:
    """
    Convert a list to a Categorical spec.

    This is useful for backward compatibility with legacy list-based param definitions.
    """
    if isinstance(value, Categorical):
        return value
    if isinstance(value, list):
        return Categorical(value)
    raise TypeError(f"Cannot convert {type(value).__name__} to Categorical")


def get_param_choices(spec: ParamSpec) -> Optional[List[Any]]:
    """
    Get discrete choices from a param spec, if applicable.

    Returns:
        List of choices for Categorical, None for continuous Int/Float ranges.
    """
    if isinstance(spec, Categorical):
        return list(spec.choices)
    if isinstance(spec, list):
        return spec
    if isinstance(spec, Int):
        # Generate discrete choices if range is small enough
        num_values = (spec.high - spec.low) // spec.step + 1
        if num_values <= 100:  # Only expand small ranges
            return list(range(spec.low, spec.high + 1, spec.step))
        return None
    if isinstance(spec, Float):
        if spec.step is not None:
            # Generate discrete choices for stepped float
            num_values = int((spec.high - spec.low) / spec.step) + 1
            if num_values <= 100:
                return [spec.low + i * spec.step for i in range(num_values)]
        return None
    return None
