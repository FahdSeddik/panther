import os
import sys

# Add the path to your library to sys.path
sys.path.insert(0, os.path.abspath("../panther"))

project = "Panther"
copyright = "2024, The Panther Authors"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # For Google and NumPy style docstrings
    "sphinx_rtd_theme",
]

templates_path = ["_templates"]
exclude_patterns = []

# The theme used for HTML output
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
