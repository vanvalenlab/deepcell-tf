import logging

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages


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

setup(
    name='DeepCell',
    version='0.5.0',
    packages=find_packages(),
    install_requires=install_reqs,
    extras_require={
        'tests': ['pytest',
                  'pytest-cov'],
    },
    license='LICENSE',
    author='David Van Valen',
    author_email='vanvalen@caltech.edu',
    description='Deep learning for single cell image segmentation',
)
