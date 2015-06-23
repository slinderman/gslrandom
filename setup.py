#!/usr/bin/env python

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(name='gslrandom',
      version='0.1.1',
      description='Cython wrappers for GSL random number generators',
      author='Scott Linderman, Aaron Schein, and Matthew Johnson',
      author_email='slinderman@seas.harvard.edu',
      url='http://www.github.com/slinderman/gslrandom',
      license="MIT",
      packages=['gslrandom'],
      ext_modules=cythonize('**/*.pyx'),
      include_dirs=[np.get_include(),],
      install_requires=[
          'Cython >= 0.20.1',
          'numpy'
      ],
      classifiers=[
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
          'Programming Language :: C++',
      ],
      keywords=['monte-carlo'],
      platforms="ALL",
     )
