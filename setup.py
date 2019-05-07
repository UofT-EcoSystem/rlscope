"""Irregular Machine Learning profiling toolkit.
"""

# NOTE: setup.py is based on the one from tensorflow.

from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b

import fnmatch
import os
import re
import pprint
import sys

from setuptools import find_packages
from setuptools import setup

#
# Parse command line arguments.
#
DEBUG = False
if "--debug" in sys.argv:
    sys.argv.remove("--debug")
    DEBUG = True

with open("README.md", "r") as fh:
    long_description = fh.read()

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
]

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
    # 'freeze_graph = tensorflow.python.tools.freeze_graph:run_main',
    # 'toco_from_protos = tensorflow.lite.toco.python.toco_from_protos:main',
    # 'tflite_convert = tensorflow.lite.python.tflite_convert:main',
    # 'toco = tensorflow.lite.python.tflite_convert:main',
    # 'saved_model_cli = tensorflow.python.tools.saved_model_cli:main',
    # # We need to keep the TensorBoard command, even though the console script
    # # is now declared by the tensorboard pip package. If we remove the
    # # TensorBoard command, pip will inappropriately remove it during install,
    # # even though the command is not removed, just moved to a different wheel.
    # 'tensorboard = tensorboard.main:run_main',
    # 'tf_upgrade_v2 = tensorflow.tools.compatibility.tf_upgrade_v2_main:main',
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

PROTOBUF_DIR = 'prof_protobuf'
proto_files = list(find_files('*.proto', PROTOBUF_DIR))
if DEBUG:
    pprint.pprint({'proto_files':proto_files})

POSTGRES_SQL_DIR = 'postgres'

THIRD_PARTY_DIR = 'third_party'

# PYTHON_PACKAGE_DIRS = [_j(PYTHON_SRC_DIR, direc) \
#         for direc in find_packages(where=PYTHON_SRC_DIR)]

PYTHON_PACKAGE_DIRS = [
    'iml_profiler',
]
PACKAGE_DIRS = PYTHON_PACKAGE_DIRS + \
               [
                   PROTOBUF_DIR,
                   POSTGRES_SQL_DIR,
                   THIRD_PARTY_DIR,
               ]
if DEBUG:
    pprint.pprint({'PACKAGE_DIRS':PACKAGE_DIRS})

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
    package_data={
        # 'protobuf': proto_files,
        # '': proto_files + ['*.proto'],
        PROTOBUF_DIR: ['*.proto'],
        POSTGRES_SQL_DIR: ['*.sql'],
        THIRD_PARTY_DIR: [
            'FlameGraph/flamegraph.pl',
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
