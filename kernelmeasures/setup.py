from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("grail_kdtw", ["GRAIL_kdtw.pyx", "kdtw.cpp"], language='c++',)]

setup(name = 'kdtw', cmdclass = {'build_ext': build_ext}, ext_modules = ext_modules)
