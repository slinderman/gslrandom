#!/usr/bin/env python

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(name='gslrandom',
      version='0.1',
      description='Cython wrappers for GSL random number generators',
      author='Scott Linderman and Aaron Schein',
      author_email='slinderman@seas.harvard.edu',
      url='http://www.github.com/slinderman/gslrandom',
      packages=['gslrandom'],
      ext_modules=cythonize('**/*.pyx'),
      include_dirs=[np.get_include(),],
     )
