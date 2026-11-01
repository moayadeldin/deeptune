"""Sphinx configuration for the DeepTune documentation."""

import os
import sys


sys.path.insert(0, os.path.abspath('../..'))

# -- Project information

project = 'DeepTune'
copyright = '2026, Moayadeldin Hussain'
author = 'Moayadeldin Hussain'

release = '1.1.0'
version = '1.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

autodoc_mock_imports = [
    'PIL',
    'cv2',
    'datasets',
    'numpy',
    'pandas',
    'peft',
    'pyarrow',
    'pytorch_forecasting',
    'pytorch_lightning',
    'pytorch_tabular',
    'sklearn',
    'torch',
    'torchmetrics',
    'torchvision',
    'tqdm',
    'tabpfn',
    'tabulate',
    'transformers',
    'yaml',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']


# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']

# -- Options for EPUB output
epub_show_urls = 'footnote'
