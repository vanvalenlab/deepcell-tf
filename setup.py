# Copyright 2016-2020 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-tf/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import logging

from codecs import open

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages


here = os.path.abspath(os.path.dirname(__file__))


with open(os.path.join(here, 'README.md'), 'r', 'utf-8') as f:
    README = f.read()


NAME = 'DeepCell'
VERSION = '0.8.0'
AUTHOR = 'Van Valen Lab'
AUTHOR_EMAIL = 'vanvalenlab@gmail.com'
URL = 'https://github.com/vanvalenlab/deepcell-tf'
DESCRIPTION = 'Deep learning for single cell image segmentation'


def _parse_requirements(file_path):
    try:
        with open(file_path, 'r', 'utf-8') as req_file:
            lineiter = (line.strip() for line in req_file)
            reqs = []
            for line in lineiter:
                # Ignore comments
                if line.startswith('#'):
                    continue
                reqs.append(line)
        return reqs
    except Exception:
        logging.warning('Failed to load %s, using default ones.', file_path)
        return []


setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    url=URL,
    download_url='{}/tarball/{}'.format(URL, VERSION),
    license='LICENSE',
    long_description=README,
    long_description_content_type='text/markdown',
    install_requires=_parse_requirements('requirements.txt'),
    extras_require={
        'tests': _parse_requirements('requirements-test.txt'),
    },
    packages=find_packages(),
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ]
)
