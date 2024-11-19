# Panther

Panther is a streamlined Python library offering optimized RandNLA, automatic differentiation, and GPU acceleration for advanced numerical computing and machine learning.

## Setup

### Prerequisites

- Python 3.7+
- `pip` (Python package installer)

### Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/your-username/panther.git
    cd panther
    ```

2. Create and activate a virtual environment:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:

    ```sh
    pip install .
    ```

4. Install the development dependencies:

    ```sh
    pip install .[dev]
    ```

### Pre-commit Hooks

This project uses `pre-commit` to manage pre-commit hooks. To set it up:

1. Install `pre-commit`:

    ```sh
    pip install pre-commit
    ```

2. Install the pre-commit hooks:

    ```sh
    pre-commit install
    ```

3. Run the pre-commit hooks on all files:

    ```sh
    pre-commit run --all-files
    ```

### Ruff

Ruff is used for linting and formatting. To use Ruff:

1. Run Ruff to check for linting issues:

    ```sh
    ruff check
    ```

2. Run Ruff to automatically fix issues:

    ```sh
    ruff format
    ```

### Running Tests

This project uses `pytest` for testing. To run the tests:

    ```sh
    pytest
    ```

### Mypy
Mypy is used for type checking. To run Mypy:

    ```sh
    mypy panther
    ```
