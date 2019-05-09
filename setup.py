"""Irregular Machine Learning profiling toolkit.
"""

# NOTE: setup.py is based on the one from tensorflow.

from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b

import fnmatch
import os
import re
import subprocess
import pprint
import sys

from setuptools import find_packages
from setuptools import setup

# from distutils.command.build_py import build_py as _build_py
# from distutils.command.clean import clean as _clean
from distutils.spawn import find_executable

# Find the Protocol Compiler.
if 'PROTOC' in os.environ and os.path.exists(os.environ['PROTOC']):
  protoc = os.environ['PROTOC']
else:
  protoc = find_executable("protoc")

PYTHON_SRC_DIR = "python"

DOCLINES = __doc__.split('\n')

# https://the-hitchhikers-guide-to-packaging.readthedocs.io/en/latest/specification.html

# This version string is semver compatible, but incompatible with pip.
# For pip, we will remove all '-' characters from this string, and use the
# result for pip.
_VERSION = '1.0.0'

REQUIRED_PACKAGES = [
    # 'absl-py >= 0.1.6',
    # 'astor >= 0.6.0',
    # 'gast >= 0.2.0',
    # 'keras_applications >= 1.0.6',
    # 'keras_preprocessing >= 1.0.5',
    # 'numpy >= 1.13.3',
    # 'six >= 1.10.0',
    # 'protobuf >= 3.6.1',
    # 'tensorboard >= 1.13.0, < 1.14.0',
    # 'tensorflow_estimator >= 1.13.0, < 1.14.0rc0',
    # 'termcolor >= 1.1.0',
    'protobuf >= 3.6.1',
    'numpy >= 1.13.3',
    'tensorflow >= 1.3.1',
    'psutil >= 5.6.2',
    'GPUtil >= 1.4.0',
    'matplotlib >= 3.0.3',
    'pandas >= 0.24.2',
    'progressbar2>=3.39.2',
    'scipy >= 1.2.1',
    'seaborn >= 0.9.0',
    'tqdm >= 4.31.1',
]

# NOTE: dependencies for building docker images are defined in dockerfiles/requirements.txt
# DOCKER_PACKAGES = [
#     'PyYAML >= 5.1',
#     'absl-py >= 0.1.6',
#     'cerberus >= 1.3',
#     'docker >= 3.7.2',
# ]

project_name = 'iml_profiler'

# python3 requires wheel 0.26
if sys.version_info.major == 3:
  REQUIRED_PACKAGES.append('wheel >= 0.26')
else:
  REQUIRED_PACKAGES.append('wheel')

# pylint: disable=line-too-long
CONSOLE_SCRIPTS = [
    'iml-analyze = iml_profiler.scripts.analyze:main',
    'iml-util-sampler = iml_profiler.scripts.utilization_sampler:main',
    'iml-dump-proto = iml_profiler.scripts.dump_proto:main',
    'iml-generate-plot-index = iml_profiler.scripts.generate_plot_index:main',
]
# pylint: enable=line-too-long

TEST_PACKAGES = [
    'pytest >= 4.4.1',
    # 'scipy >= 0.15.1',
]

def find_files(pattern, root):
  for direc, dirs, files in os.walk(root):
      for filename in fnmatch.filter(files, pattern):
          yield os.path.join(direc, filename)

def generate_proto(source, require=True):
  """Invokes the Protocol Compiler to generate a _pb2.py from the given
  .proto file.  Does nothing if the output already exists and is newer than
  the input."""

  if not require and not os.path.exists(source):
      return

  output = source.replace(".proto", "_pb2.py").replace("../src/", "")

  if (not os.path.exists(output) or
          (os.path.exists(source) and
           os.path.getmtime(source) > os.path.getmtime(output))):
      print("Generating %s..." % output)

      if not os.path.exists(source):
          sys.stderr.write("Can't find required file: %s\n" % source)
          sys.exit(-1)

      if protoc is None:
          sys.stderr.write(
              "protoc is not installed nor found in ../src.  Please compile it "
              "or install the binary package.\n")
          sys.exit(-1)


      # protoc -I$PWD --python_out=. prof_protobuf/*.proto
      # protoc_command = [protoc, "-I.", "--python_out=.", "{dir}/*.proto".format(
      protoc_command = [protoc, "-I.", "--python_out=.", source]
      if subprocess.call(protoc_command) != 0:
          sys.exit(-1)

PROTOBUF_DIR = 'iml_profiler/protobuf'
proto_files = list(find_files('*.proto', PROTOBUF_DIR))

POSTGRES_SQL_DIR = 'postgres'

THIRD_PARTY_DIR = 'third_party'

# PYTHON_PACKAGE_DIRS = [_j(PYTHON_SRC_DIR, direc) \
#         for direc in find_packages(where=PYTHON_SRC_DIR)]

PYTHON_PACKAGE_DIRS = [
    'iml_profiler',
]
PACKAGE_DIRS = PYTHON_PACKAGE_DIRS
# PACKAGE_DIRS = PYTHON_PACKAGE_DIRS + \
#                [
#                    PROTOBUF_DIR,
#                    POSTGRES_SQL_DIR,
#                    THIRD_PARTY_DIR,
#                ]

def main():
    #
    # Parse command line arguments.
    #
    global DEBUG
    DEBUG = False
    if "--debug" in sys.argv:
        sys.argv.remove("--debug")
        DEBUG = True

    print("> Using protoc = {protoc}".format(protoc=protoc))

    if DEBUG:
        pprint.pprint({'proto_files':proto_files})
        pprint.pprint({'PACKAGE_DIRS':PACKAGE_DIRS})

    with open("README.md", "r") as fh:
        long_description = fh.read()

    def _proto(base):
        return _j(PROTOBUF_DIR, base)
    generate_proto(_proto('pyprof.proto'))
    generate_proto(_proto('unit_test.proto'))

    setup(
        name=project_name,
        version=_VERSION.replace('-', ''),
        description=DOCLINES[0],
        long_description=long_description,
        url='https://github.com/UofT-EcoSystem/iml',
        download_url='https://github.com/UofT-EcoSystem/iml/tags',
        author='James Gleeson',
        author_email='jagleeso@gmail.com',
        # Contained modules and scripts.
        packages=PACKAGE_DIRS,
        entry_points={
            'console_scripts': CONSOLE_SCRIPTS,
        },
        install_requires=REQUIRED_PACKAGES,
        tests_require=REQUIRED_PACKAGES + TEST_PACKAGES,
        # # https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras-optional-features-with-their-own-dependencies
        # # These requirements are only installed if these features are enabled when installing the pip package.
        # #
        # # $ pip install 'iml_profiler[docker]'
        # #
        # # Q: Does this work with .whl files?
        # extras_require={
        #     'docker': DOCKER_PACKAGES,
        # },
        package_data={
            'iml_profiler': ['**/*.py', '*.py'],
            # PROTOBUF_DIR: ['*.proto'],
            # POSTGRES_SQL_DIR: ['*.sql'],
            # THIRD_PARTY_DIR: [
            #     'FlameGraph/flamegraph.pl',
            # ],
            '': [
                _j(PROTOBUF_DIR, '*.proto'),
                _j(POSTGRES_SQL_DIR, '*.sql'),
                _j(THIRD_PARTY_DIR, 'FlameGraph/flamegraph.pl'),
            ],
        },
        classifiers=[
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        # TODO: Add license!
        keywords='iml ml profiling tensorflow machine learning',
    )

if __name__ == '__main__':
    main()

