from setuptools import setup
from setuptools import find_packages

setup(
    name='DeepCell',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'scikit-image>=0.13.1,<1',
        'scikit-learn>=0.19.1,<1',
        'scipy>=1.1.0,<2',
        'tensorflow-gpu>=1.8.0,<2',
        'tifffile>=0.14.0,<1'
    ],
    license='LICENSE.txt',
    author='David Van Valen',
    author_email='vanvalen@caltech.edu',
    description='Deep learning for single cell image segmentation',
)
