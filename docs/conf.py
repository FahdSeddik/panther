# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../"))


project = "Panther"
copyright = "2024, The Panther Team"
author = "The Panther Team"
version = "0.1.2"
release = "0.1.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "panther.rst",
    "modules.rst",
    "panther.linalg.rst",
    "panther.nn.rst",
    "panther.sketch.rst",
    "panther.tuner.rst",
    "panther.utils.rst",
]
source_suffix = [".rst", ".md"]

# Autodoc configuration
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Handle import errors gracefully
autodoc_typehints = "none"
autodoc_class_signature = "mixed"

# Suppress warnings for mocked objects and duplicate descriptions
suppress_warnings = [
    "autodoc.mock_object",
    "autodoc.duplicate_object",
    "toc.not_included",
    "toc.excluded",
    "autodoc.import_object",
    "autodoc.failed_import",
    "ref.python",  # Suppress reference warnings
]

# Mock only the C++ extension and problematic dependencies that can't be built on Read the Docs
autodoc_mock_imports = [
    "pawX",
    "pawX.pawX",
    "triton",
    # Mock specific submodules that depend on pawX
    "panther.nn.pawXimpl",
    "panther.nn.linear_kernels",
    "panther.nn.linear_kernels.backward",
    "panther.nn.linear_kernels.forward",
    # Mock entire nn module components that have import issues
    "panther.nn.attention",
    "panther.nn.conv2d",
    "panther.nn.linear",
    "panther.nn.linear_tr",
    # Mock specific classes/enums from pawX
    "DistributionFamily",
    "Axis",
    "torch"
]

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_logo = "_static/panther-logo.png"
html_title = "Panther Documentation"
html_css_files = [
    "custom.css",
]

html_theme_options = {
    "repository_url": "https://github.com/FahdSeddik/panther",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "path_to_docs": "docs/",
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "show_navbar_depth": 2,
    "collapse_navigation": True,
    "navigation_depth": 4,
    "use_sidenotes": True,
    "show_prev_next": False,
}
