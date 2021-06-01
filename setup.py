from setuptools import setup
from setuptools import find_packages

setup(
    name='torchsampler',
    version='1.0',
    description='up- or down- sampling for imbalanced datsaet.',
    install_requires=[
        'torch'
    ],
    packages=find_packages(),
    zip_safe=False,
)
