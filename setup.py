#!/usr/bin/env python

import logging
import os

from codecs import open

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages


here = os.path.abspath(os.path.dirname(__file__))


def _parse_requirements(file_path):
    lineiter = (line.strip() for line in open(file_path))
    reqs = []
    for line in lineiter:
        # workaround to ignore keras_maskrcnn requirement
        # which is downloaded directly from github
        if line.startswith('#') or line.startswith('git+'):
            continue
        reqs.append(line)
    return reqs


try:
    install_reqs = _parse_requirements('requirements.txt')
except Exception:
    logging.warning('Failed to load requirements file, using default ones.')
    install_reqs = []


try:
    tests_reqs = _parse_requirements('requirements-test.txt')
except Exception:
    logging.warning('Failed to load test requirements file, using default ones.')
    tests_reqs = []


about = {}
with open(os.path.join(here, 'deepcell', '__version__.py'), 'r', 'utf-8') as f:
    exec(f.read(), about)


with open('README.md', 'r', 'utf-8') as f:
    readme = f.read()


setup(
    name=about['__title__'],
    version=about['__version__'],
    description=about['__description__'],
    long_description=readme,
    long_description_content_type='text/markdown',
    author=about['__author__'],
    author_email=about['__author_email__'],
    url=about['__url__'],
    license=about['__license__'],
    packages=find_packages(),
    install_requires=install_reqs,
    tests_require=tests_reqs,
    extras_require={
        'tests': ['pytest',
                  'pytest-cov'],
    }
)
