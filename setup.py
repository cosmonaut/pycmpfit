#!/usr/bin/env python

from distutils.core import setup
#from setuptools import find_packages, setup
from setuptools import find_packages
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy.distutils.misc_util import get_numpy_include_dirs


ext_modules = Extension("pycmpfit", ["pycmpfit.pyx", "cmpfit-1.4/mpfit.c"],
                        include_dirs = get_numpy_include_dirs() + ["cmpfit-1.4"]
)


with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name = 'pycmpfit',
    version = '0.2.0',
    packages=find_packages(),
    description = '',
    long_description = long_description,
    #long_description_content_type = 'text/markdown',
    author = 'Nicholas Nell',
    author_email = 'nicholas.nell@colorado.edu',
    url = 'https://github.com/cosmonaut/pycmpfit',
    #packages = ['pycmpfit'],
    provides = ['pycmpfit'],
    requires = ['numpy'],
    keywords = ['Scientific/Engineering'],
    cmdclass = {'build_ext': build_ext},
    ext_modules = [ext_modules]
)

