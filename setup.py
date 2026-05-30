import os

from codecs import open

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages


here = os.path.abspath(os.path.dirname(__file__))


with open(os.path.join(here, 'README.md'), 'r', 'utf-8') as f:
    readme = f.read()


about = {}
with open(os.path.join(here, 'deepcell', '_version.py'), 'r', 'utf-8') as f:
    exec(f.read(), about)


setup(
    name=about['__title__'],
    version=about['__version__'],
    author=about['__author__'],
    author_email=about['__author_email__'],
    description=about['__description__'],
    url=about['__url__'],
    download_url=about['__download_url__'],
    license=about['__license__'],
    long_description=readme,
    long_description_content_type='text/markdown',
    install_requires=[
        'tqdm',
        'numpy>=1.16.6',
        'pydot>=1.4.2,<2',  # for keras.utils.plot_model
        'scipy>=1.2.3,<2',
        'scikit-image>=0.19.3',
        'scikit-learn>=0.20.4',
        'tensorflow~=2.8.0',
        'tensorflow_addons~=0.16.1',
        'spektral~=1.0.4',
        'jupyter>=1.0.0,<2',
        'matplotlib',
        'opencv-python-headless<5',
        'deepcell-tracking~=0.6.5',
        'deepcell-toolbox>=0.12.1'
    ],
    extras_require={
        'tests': ['pytest', 'pytest-cov', 'ruff'],
    },
    packages=find_packages(),
    python_requires='>=3.7, <3.11',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ]
)
