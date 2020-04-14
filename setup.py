import logging
import setuptools

from distutils.command.build_ext import build_ext as DistUtilsBuildExt

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages


class BuildExtension(setuptools.Command):
    description = DistUtilsBuildExt.description
    user_options = DistUtilsBuildExt.user_options
    boolean_options = DistUtilsBuildExt.boolean_options
    help_options = DistUtilsBuildExt.help_options

    def __init__(self, *args, **kwargs):
        from setuptools.command.build_ext import build_ext as SetupToolsBuildExt

        # Bypass __setatrr__ to avoid infinite recursion.
        self.__dict__['_command'] = SetupToolsBuildExt(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._command, name)

    def __setattr__(self, name, value):
        setattr(self._command, name, value)

    def initialize_options(self, *args, **kwargs):
        return self._command.initialize_options(*args, **kwargs)

    def finalize_options(self, *args, **kwargs):
        ret = self._command.finalize_options(*args, **kwargs)
        import numpy
        self.include_dirs.append(numpy.get_include())
        return ret

    def run(self, *args, **kwargs):
        return self._command.run(*args, **kwargs)


extensions = [
    setuptools.Extension(
        'deepcell.utils.compute_overlap',
        ['deepcell/utils/compute_overlap.pyx']
    ),
]


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

exec(open('deepcell/version.py').read())
setup(
    name='DeepCell',
    version=__version__,
    packages=find_packages(),
    install_requires=install_reqs,
    extras_require={
        'tests': ['pytest',
                  'pytest-cov'],
    },
    cmdclass={'build_ext': BuildExtension},
    license='LICENSE',
    author='David Van Valen',
    author_email='vanvalen@caltech.edu',
    description='Deep learning for single cell image segmentation',
    ext_modules=extensions,
)
