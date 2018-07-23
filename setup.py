import logging
import pkg_resources
import pip

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages


def _parse_requirements(file_path):
    pip_ver = pkg_resources.get_distribution('pip').version
    pip_version = list(map(int, pip_ver.split('.')[:2]))
    if pip_version >= [6, 0]:
        raw = pip.req.parse_requirements(file_path,
                                         session=pip.download.PipSession())
    else:
        raw = pip.req.parse_requirements(file_path)
    return [str(i.req) for i in raw]


# parse_requirements() returns generator of pip.req.InstallRequirement objects
try:
    install_reqs = _parse_requirements('requirements.txt')
except Exception:
    logging.warning('Failed to load requirements file, using default ones.')
    install_reqs = []

setup(
    name='DeepCell',
    version='0.1',
    packages=find_packages(),
    install_requires=install_reqs,
    extras_require={
        'tests': ['pytest',
                  'pytest-cov'],
    },
    license='LICENSE.txt',
    author='David Van Valen',
    author_email='vanvalen@caltech.edu',
    description='Deep learning for single cell image segmentation',
)
