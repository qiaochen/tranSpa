# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys
import os
from datetime import datetime
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
from pathlib import Path
from sphinx.application import Sphinx
from sphinx.ext import autosummary
from urllib.request import urlretrieve
import logging

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
sys.path.insert(0, f"{HERE.parent.parent}")
sys.path.insert(0, os.path.abspath("_ext"))

# -- Project information -----------------------------------------------------

project = 'TransImp'
author = 'Chen Qiao, Yuanhua Huang'
copyright = f"{datetime.now():%Y}, {author}"
# The full version, including alpha/beta/rc tags
release = '0.1.0'

notebooks_url = "https://github.com/qiaochen/tranSpa/raw/main/demo/"
notebooks = [
    "seqfish.ipynb",
    "seqfish_unprobed_genes.ipynb",
]
for nb in notebooks:
    try:
        urlretrieve(notebooks_url + nb, nb)
    except:
        raise ValueError(f'{nb} cannot be retrieved.')

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
needs_sphinx = "1.7"

extensions = [
  "sphinx.ext.autodoc",
  "sphinx.ext.doctest",
  "sphinx.ext.coverage",
  "sphinx.ext.mathjax",
  "sphinx.ext.autosummary",
  "sphinx.ext.napoleon",
  "sphinx.ext.intersphinx",
  "sphinx.ext.githubpages",
  "sphinx_autodoc_typehints",
  "nbsphinx",
]



autosummary_generate = True
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
source_suffix = ['.rst']
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
master_doc = 'index'
enable_pdf_build = False
enable_epub_build = False

def setup(app):
    """TODO."""
    app.add_css_file("custom.css")

def process_generate_options(app: Sphinx):
    """TODO."""
    genfiles = app.config.autosummary_generate

    if genfiles and not hasattr(genfiles, "__len__"):
        env = app.builder.env
        genfiles = [
            env.doc2path(x, base=None)
            for x in env.found_docs
            if Path(env.doc2path(x)).is_file()
        ]
    if not genfiles:
        return

    from sphinx.ext.autosummary.generate import generate_autosummary_docs

    ext = app.config.source_suffix
    genfiles = [
        genfile + (not genfile.endswith(tuple(ext)) and ext[0] or "")
        for genfile in genfiles
    ]

    suffix = autosummary.get_rst_suffix(app)
    if suffix is None:
        return

    generate_autosummary_docs(
        genfiles,
        # builder=app.builder,
        # warn=logger.warning,
        # info=logger.info,
        suffix=suffix,
        base_path=app.srcdir,
        imported_members=True,
        app=app,
    )


autosummary.process_generate_options = process_generate_options

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

pygments_style = 'sphinx'
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'
github_repo = 'tranSpa'
github_nb_repo = 'tranSpa'
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_theme_options = dict(navigation_depth=1, titles_only=True)
htmlhelp_basename = 'tranSpa'
