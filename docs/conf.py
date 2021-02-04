# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
import re
import textwrap
import pprint

# add ROOT of directory to PYTHONPATH
ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))
print("> Adding ROOT of RL-Scope repository to PYTHONPATH: {path}".format(path=ROOT))
sys.path.insert(0, ROOT)

from rlscope import py_config

DEBUG_SPHINX = py_config.EnvVars.get_bool('DEBUG_SPHINX', dflt=False)

from rlscope.profiler.rlscope_logging import logger
from rlscope.profiler import rlscope_logging

if DEBUG_SPHINX:
    rlscope_logging.enable_debug_logging()


# -- Project information -----------------------------------------------------

project = 'rlscope'
copyright = '2020, James Gleeson'
author = 'James Gleeson'

# The full version, including alpha/beta/rc tags
release = '1.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_rtd_theme",
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    # https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#configuration
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    # refer to setions using :ref:`Section Title`
    'sphinx.ext.autosectionlabel',
    # https://www.sphinx-doc.org/en/master/usage/extensions/extlinks.html
    'sphinx.ext.extlinks',
]

# extlinks = {'github': ('https://github.com/sphinx-doc/sphinx/issues/%s',
#                       'issue ')}
extlinks = {
    # e.g. :github:`dockerfiles/sh/test_cupti_profiler_api.sh`
    'github': ('https://github.com/UofT-EcoSystem/rlscope/blob/master/%s',
               ''),
    # https://colab.research.google.com/github/UofT-Ecosystem/rlscope/blob/tutorial/jupyter/01_rlscope_getting_started.ipynb
    # e.g. :colab:`jupyter/01_rlscope_getting_started.ipynb`
    'colab': (
        'https://colab.research.google.com/github/UofT-Ecosystem/rlscope/blob/master/%s',
        ''),
}

html_logo = 'images/rlscope_logo_bordered.svg'

# If this is True, todo and todolist produce output, else they produce nothing.
# The default is False.
todo_include_todos = True

# I prefer the numpy style.
napoleon_google_docstring = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']
html_static_path = []

# https://stackoverflow.com/questions/39249466/how-to-exclude-pytest-test-functions-from-sphinx-autodoc

# This is the expected signature of the handler for this event, cf doc
def autodoc_skip_member_handler(app, what, name, obj, skip, options):
    # Basic approach; you might want a regex instead
    # import pdb; pdb.set_trace()
    if re.search(r'^test_|^Test|^rlscope_plot_index\.py', name):
        if DEBUG_SPHINX:
            logger.debug("SKIP: {name} {msg}".format(
                name=name,
                msg=pprint_msg(locals())))
        return True
    if DEBUG_SPHINX:
        logger.debug("DOCS: {name} {msg}".format(
            name=name,
            msg=pprint_msg(locals())))
    # Use "default" handling behaviour of whether to skip docs for this symbol.
    # NOTE: we don't return False since it results in weird symbols like
    # __dict__ getting documented.
    return None

# Automatically called by sphinx at startup
def setup(app):
    # Connect the autodoc-skip-member event from apidoc to the callback
    app.connect('autodoc-skip-member', autodoc_skip_member_handler)

def pprint_msg(dic, prefix='  '):
    """
    Give logger.info a string for neatly printing a dictionary.

    Usage:
    logger.info(pprint_msg(arbitrary_object))
    """
    return "\n" + textwrap.indent(pprint.pformat(dic), prefix=prefix)
