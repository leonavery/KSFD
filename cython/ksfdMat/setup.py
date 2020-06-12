#!/usr/bin/env python

#$ python setup.py build_ext --inplace

#from distutils.core import setup
from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy
import petsc4py

def configure():
    INCLUDE_DIRS = []
    LIBRARY_DIRS = []
    LIBRARIES    = []

    # PETSc
    import os
    try:
        PETSC_DIR  = os.environ['PETSC_DIR']
        PETSC_ARCH = os.environ.get('PETSC_ARCH', '')
    except KeyError as e:
        p4pyconfig = petsc4py.get_config()
        PETSC_DIR = p4pyconfig['PETSC_DIR']
        PETSC_ARCH = p4pyconfig['PETSC_ARCH']
    from os.path import join, isdir
    if PETSC_ARCH and isdir(join(PETSC_DIR, PETSC_ARCH)):
        INCLUDE_DIRS += [join(PETSC_DIR, PETSC_ARCH, 'include'),
                         join(PETSC_DIR, 'include')]
        LIBRARY_DIRS += [join(PETSC_DIR, PETSC_ARCH, 'lib')]
    else:
        if PETSC_ARCH: pass # XXX should warn ...
        INCLUDE_DIRS += [join(PETSC_DIR, 'include')]
        LIBRARY_DIRS += [join(PETSC_DIR, 'lib')]
    LIBRARIES += ['petsc']

    # PETSc for Python
    INCLUDE_DIRS = [petsc4py.get_include()] + INCLUDE_DIRS

    # NumPy
    INCLUDE_DIRS = [numpy.get_include()] + INCLUDE_DIRS

    return dict(
        include_dirs=INCLUDE_DIRS + [os.curdir],
        libraries=LIBRARIES,
        library_dirs=LIBRARY_DIRS,
        runtime_library_dirs=LIBRARY_DIRS,
    )

extensions = [
    Extension('ksfdMat',
              sources = ['ksfdMat.pyx'],
              **configure()),
]

setup(name = "ksfdMat",
      ext_modules = cythonize(
          extensions, 
          verbose=True,
          annotate=True,
          language_level=3,
          include_path=[petsc4py.get_include()]
      ),
)
