#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io
import os
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext
from setuptools import find_packages
from setuptools import setup


from numpy.distutils.core import setup, Extension


def read(*names, **kwargs):
    return io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ).read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name='py21cmmc_fg',
    version=find_version("src",'py21cmmc_fg', "__init__.py"),
    license='MIT license',
    description='A py21cmmc plugin which provides sky and instrumental foregrounds, along with a power-spectrum estimator.',
    long_description='%s\n%s' % (
        re.compile('^.. start-badges.*^.. end-badges', re.M | re.S).sub('', read('README.rst')),
        re.sub(':[a-z]+:`~?(.*?)`', r'``\1``', read('CHANGELOG.rst'))
    ),
    author='Bella Nasirudin',
    author_email='a.nasirudin@postgrad.curtin.edu.au',
    url='https://github.com/BellaNasirudin/py21cmmc_fg',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        "Natural Language :: English",
        'Operating System :: Unix',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: C',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
    install_requires=[
        'numpy',
        'powerbox==0.6.0',
        '21cmfast',
        'cosmoHammer',
        'h5py>=2.8.0',
        'emcee<3',
        'click',
        #'tqdm',
        'numpy',
        'pyyaml',
        'cosmoHammer',
        'cffi>=1.0',
        'scipy',
        'astropy>=2.0',
        'cached_property'
    ],
    package_data={"py21cmmc_fg":['data/*']},
    ext_modules=[
        Extension(
            'py21cmmc_fg.c_routines',
            ['src/py21cmmc_fg/routines.c'],
            extra_compile_args = ['-Ofast']
        ),

    ]
)
