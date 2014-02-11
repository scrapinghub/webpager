from setuptools import setup, find_packages
from distutils.extension import Extension
try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = {}
ext_modules = []

if use_cython:
    ext_modules.append(Extension("webpager.levenshtein_cython", ['webpager/levenshtein_cython.pyx']))
    cmdclass.update({'build_ext': build_ext})
else:
    ext_modules.append(Extension("webpager.levenshtein_cython", ['webpager/levenshtein_cython.c']))

setup(name='webpager',
      version='0.1',
      description="Paginating the web",
      long_description="",
      author="terry",
      author_email="terry@scrapinghub.com",
      install_requires=['scikit-learn', 'lxml'],
      packages=find_packages(),
      package_data = {
          'webpager.models': ['*.pkl'],
      },
      cmdclass=cmdclass,
      ext_modules=ext_modules
)