# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from datetime import datetime
import mock
from sphinx.builders.html import StandaloneHTMLBuilder
sys.path.insert(0, os.path.abspath('../..'))
# sys.path.insert(0, os.path.abspath('.'))

# pylint: disable=line-too-long

# -- Project information -----------------------------------------------------

project = 'DeepCell'
copyright = ('2016-{currentyear}, Van Valen Lab at the '
             'California Institute of Technology (Caltech)').format(
                 currentyear=datetime.now().year)
author = 'Van Valen Lab at Caltech'

# The short X.Y version
version = '0.6.0'
# The full version, including alpha/beta/rc tags
release = '0.6.0'

import subprocess
try:
    git_rev = subprocess.check_output(['git', 'describe', '--exact-match', 'HEAD'], universal_newlines=True)
except subprocess.CalledProcessError:
    try:
        git_rev = subprocess.check_output(['git', 'rev-parse', 'HEAD'], universal_newlines=True)
    except subprocess.CalledProcessError:
        git_rev = ''
if git_rev:
    git_rev = git_rev.splitlines()[0] + '/'

# -- RTD configuration ------------------------------------------------

# on_rtd is whether we are on readthedocs.org, this line of code grabbed from docs.readthedocs.org
on_rtd = os.environ.get("READTHEDOCS", None) == "True"

# This is used for linking and such so we link to the thing we're building
rtd_version = os.environ.get("READTHEDOCS_VERSION", "latest")
if rtd_version not in ["stable", "latest"]:
    rtd_version = "stable"

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
needs_sphinx = '2.3.1'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'm2r',
    'IPython.sphinxext.ipython_console_highlighting',
    'nbsphinx',
    'nbsphinx_link',
    'sphinx.ext.todo',
    'sphinx.ext.autosectionlabel'
]

napoleon_google_docstring = True

default_role = 'py:obj'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = ['.rst', '.md']
# source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'DeepCelldoc'


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'DeepCell.tex', 'DeepCell Documentation',
     'Van Valen Lab at Caltech', 'manual'),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'deepcell', 'DeepCell Documentation',
     [author], 1)
]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'DeepCell', 'DeepCell Documentation',
     author, 'DeepCell', 'One line description of project.',
     'Miscellaneous'),
]


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']

# -- Extension configuration -------------------------------------------------
autodoc_mock_imports = [
    'tensorflow',
    'scipy',
    'numpy',
    'sklearn',
    'skimage',
    'pandas',
    'networkx',
    'nbformat',
    'cv2',
    'cython',
    'keras-preprocessing',
    'keras_retinanet',
    'deepcell_tracking',
    'deepcell_toolbox',
    'keras_applications',
    'matplotlib'
]

sys.modules['deepcell.utils.compute_overlap'] = mock.Mock()
sys.modules['tensorflow.python.keras.layers.convolutional_recurrent.ConvRNN2D'] = mock.Mock()

# Disable nbsphinx extension from running notebooks
nbsphinx_execute = 'never'
exclude_patterns = ['_build', '**.ipynb_checkpoints']

# TODO: fix relative URL for notebooks, using replace() is not perfect.
nbsphinx_prolog = (
r"""
{% if env.metadata[env.docname]['nbsphinx-link-target'] %}
{% set docpath = env.metadata[env.docname]['nbsphinx-link-target'].replace('../', '') %}
{% else %}
{% set docpath = env.doc2path(env.docname, base='docs/source') %}
{% endif %}
.. raw:: html

    <div class="admonition note">
        <p>This page was generated from <a href="https://github.com/vanvalenlab/deepcell-tf/blob/""" + git_rev + r"""{{ docpath }}">{{ docpath }}</a>
        </p>
    </div>
"""
)

# -- Options for intersphinx extension ---------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3.7', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'kiosk': ('https://deepcell-kiosk.readthedocs.io/en/{}/'.format(rtd_version), None),
    'kiosk-redis-consumer': (('https://deepcell-kiosk.readthedocs.io/'
                              'projects/kiosk-redis-consumer/en/{}/').format(rtd_version), None),
}

intersphinx_cache_limit = 0

# -- Custom Additions --------------------------------------------------------
nitpick_ignore = [
    ('py:class', 'function'),  # TODO: set type for "function" properly
    ('py:class', 'tensor'),  # TODO: set type for "tensor" properly
    ('py:class', 'numpy.array'),
    ('py:class', 'pandas.DataFrame'),
    ('py:class', 'tensorflow.keras.Model'),
    ('py:class', 'tensorflow.python.keras.Model'),
    ('py:class', 'tensorflow.keras.layers.Layer'),
    ('py:class', 'tensorflow.python.keras.layers.Layer'),
    ('py:class', 'tensorflow.python.keras.layers.ZeroPadding2D'),
    ('py:class', 'tensorflow.python.keras.layers.ZeroPadding3D'),
    ('py:class', 'tensorflow.python.keras.preprocessing.image.Iterator'),
    ('py:class', 'tensorflow.python.keras.preprocessing.image.ImageDataGenerator'),
]

StandaloneHTMLBuilder.supported_image_types = [
    'image/svg+xml',
    'image/gif',
    'image/png',
    'image/jpeg'
]
