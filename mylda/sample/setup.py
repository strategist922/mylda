from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

#from Cython.Compiler.Options import directive_defaults
#directive_defaults['linetrace'] = True
#directive_defaults['binding'] = True

ext = Extension(
        name='_sample',
        sources=['_sample.pyx'],
        include_dirs=[np.get_include()],
#        define_macros=[('CYTHON_TRACE', '1')]
)

setup(
    ext_modules=cythonize(ext),
)
