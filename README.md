pycmpfit
========

A python interface to the MPFIT library written in C

The pycmpfit interface uses a library written in C that is maintained
[here](http://cow.physics.wisc.edu/~craigm/idl/cmpfit.html). The
license and disclaimer information for cmpfit is located in the cmpfit
subdirectory.


Dependencies:
=====

* python (3)
* numpy

Build Dependencies:
=====

* cython

Build Instructions:
=====

The python module can be built using:

        python setup.py build_ext

Documentation:
=====

See tests/unittests.py for example usage scenarios. More documentation
and examples may be added later.

