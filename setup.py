#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy.distutils.misc_util import get_numpy_include_dirs


ext_modules = Extension("pycmpfit", ["pycmpfit.pyx", "cmpfit-1.2/mpfit.c"],
                        include_dirs = get_numpy_include_dirs() + ["cmpfit-1.2"]
)


setup(
    name = 'pycmpfit',
    version = '0.1.0',
    description = '',
    author = 'Nicholas Nell',
    author_email = 'nicholas.nell@colorado.edu',
    url = 'https://github.com/cosmonaut/pycmpfit',
    packages = 'pycmpfit',
    provides = 'pycmpfit',
    requires = ['numpy'],
    keywords = ['Scientific/Engineering'],
    cmdclass = {'build_ext': build_ext},
    ext_modules = [ext_modules]
)

