"""
RL-Scope: Cross-Stack Profiling for Deep Reinforcement Learning Workloads
"""

# NOTE: setup.py is based on the one from tensorflow.

from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b

import fnmatch
import argparse
from glob import glob
import shlex
import os
import re
import subprocess
import pprint
import textwrap
import sys

from setuptools import find_packages
from setuptools import setup

# from distutils.command.build_py import build_py as _build_py
# from distutils.command.clean import clean as _clean
from distutils.spawn import find_executable

ROOT = _d(os.path.realpath(__file__))

# HACK: Make it so we can import logging stuff.
sys.path.insert(0, ROOT)
from rlscope.profiler.rlscope_logging import logger
from rlscope.profiler import rlscope_logging
from rlscope import py_config

PROJECT_NAME = 'rlscope'

def pprint_msg(dic, prefix='  '):
    """
    Give logger.info a string for neatly printing a dictionary.

    Usage:
    logger.info(pprint_msg(arbitrary_object))
    """
    return "\n" + textwrap.indent(pprint.pformat(dic), prefix=prefix)

def get_files_by_ext(root, rm_prefix=None):
    files_by_ext = dict()
    for path in each_file_recursive(root):
        ext = file_extension(path)
        if ext not in files_by_ext:
            files_by_ext[ext] = []
        if rm_prefix is not None:
            path = re.sub(r'^{prefix}/'.format(prefix=rm_prefix), '', path)
        files_by_ext[ext].append(path)
    return files_by_ext

def file_extension(path):
    m = re.search(r'\.(?P<ext>[^.]+)$', path)
    if not m:
        return None
    return m.group('ext')

def each_file_recursive(root_dir):
    if not os.path.isdir(root_dir):
        raise ValueError("No such directory {root_dir}".format(root_dir=root_dir))
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for base in filenames:
            path = _j(dirpath, base)
            yield path

def cmd_debug_msg(cmd, env=None, dry_run=False):
    if type(cmd) == list:
        cmd_str = " ".join([shlex.quote(str(x)) for x in cmd])
    else:
        cmd_str = cmd

    lines = []
    if dry_run:
        lines.append("> CMD [setup.py] [dry-run]:")
    else:
        lines.append("> CMD [setup.py]:")
    lines.extend([
        "  $ {cmd}".format(cmd=cmd_str),
        "  PWD={pwd}".format(pwd=os.getcwd()),
    ])

    if env is not None and len(env) > 0:
        env_vars = sorted(env.keys())
        lines.append("  Environment:")
        for var in env_vars:
            lines.append("    {var}={val}".format(
                var=var,
                val=env[var]))
    string = '\n'.join(lines)

    return string

# def log_cmd(cmd, env=None, dry_run=False):
#     string = cmd_debug_msg(cmd, env=env, dry_run=dry_run)
#
#     logging.info(string)

def print_cmd(cmd, files=sys.stdout, env=None, dry_run=False):
    string = cmd_debug_msg(cmd, env=env, dry_run=dry_run)

    if type(files) not in [set, list]:
        if type(files) in [list]:
            files = set(files)
        else:
            files = set([files])

    for f in files:
        print(string, file=f)
        f.flush()

# Find the Protocol Compiler.
if 'PROTOC' in os.environ and os.path.exists(os.environ['PROTOC']):
  protoc = os.environ['PROTOC']
else:
  protoc = find_executable("protoc")

PYTHON_SRC_DIR = "python"

DOCLINES = __doc__.lstrip().rstrip().split('\n')

# https://the-hitchhikers-guide-to-packaging.readthedocs.io/en/latest/specification.html


REQUIREMENTS_TXT = _j(py_config.ROOT, "requirements.txt")

def read_requirements(requirements_txt):
    requires = []
    with open(requirements_txt) as f:
        for line in f:
            line = line.rstrip()
            line = re.sub(r'#.*', '', line)
            if re.search(r'^\s*$', line):
                continue
            requires.append(line)
    return requires

def read_version(version_txt):
    with open(version_txt) as f:
        version = f.read().rstrip()
        return version

#
# NOTE: this is identical to requirements found in requirements.txt.
# (please keep in sync)
#
REQUIRED_PACKAGES = read_requirements(REQUIREMENTS_TXT)
# This version string is semver compatible, but incompatible with pip.
# For pip, we will remove all '-' characters from this string, and use the
# result for pip.
#
# NOTE: We must change the RLSCOPE_VERSION every time we upload to pip./
RLSCOPE_VERSION = py_config.read_rlscope_version()

# NOTE: dependencies for building docker images are defined in dockerfiles/requirements.txt
# DOCKER_PACKAGES = [
#     'PyYAML >= 5.1',
#     'absl-py >= 0.1.6',
#     'cerberus >= 1.3',
#     'docker >= 3.7.2',
# ]

def get_cuda_version():
    """
    Determine the CUDA version we are building for.
    NOTE: C++ components have a hard CUDA version dependency.

    If CUDA environment variable is defined:
      Use $CUDA
    Elif /usr/local/cuda is a symlink to /usr/local/cuda-${CUDA_VERSION}:
      Use $CUDA_VERSION
    Else:
      Use 10.1 (default)
    """
    if 'CUDA' in os.environ:
        return os.environ['CUDA']
    elif os.path.islink('/usr/local/cuda'):
        cuda_path = os.path.realpath('/usr/local/cuda')
        m = re.search(r'^cuda-(?P<cuda_version>.*)', os.path.basename(cuda_path))
        cuda_version = m.group('cuda_version')
        return cuda_version
    # Default:
    return '10.1'

def get_pip_package_version():
    """
    Mimic how pytorch specifies cuda dependencies:
    e.g.
    torch==1.7.1+cu110
    For CUDA 11.0
    """
    # Mimic how pytorch specifies cuda dependencies:
    # e.g.
    # torch==1.7.1+cu110
    cuda_version = get_cuda_version()
    cu_version = re.sub(r'\.', '', cuda_version)

    pip_package_version = "{rlscope_version}+cu{cu_version}".format(
        rlscope_version=RLSCOPE_VERSION,
        cu_version=cu_version,
    )
    return pip_package_version

# python3 requires wheel 0.26
if sys.version_info.major == 3:
  REQUIRED_PACKAGES.append('wheel >= 0.26')
else:
  REQUIRED_PACKAGES.append('wheel')

# pylint: disable=line-too-long
CONSOLE_SCRIPTS = [
    'rls-prof = rlscope.scripts.cuda_api_prof:main',
    'rls-plot = rlscope.parser.calibration:main_plot',
    'rls-run = rlscope.scripts.analyze:main',

    'rls-util-sampler = rlscope.scripts.utilization_sampler:main',
    'rls-dump-proto = rlscope.scripts.dump_proto:main',
    'rls-calibrate = rlscope.parser.calibration:main_run',

    # Running various experiments for RL-Scope paper.
    # Used by artifact evaluation.
    'rls-run-expr = rlscope.scripts.run_expr:main',
    'rls-bench = rlscope.scripts.bench:main',
    'rls-quick-expr = rlscope.scripts.quick_expr:main',

    # Not yet ready for "prime time"...
    'rls-generate-plot-index = rlscope.scripts.generate_rlscope_plot_index:main',
]
# pylint: enable=line-too-long

# Only install these inside the docker development environment
# (or, when "python setup.py develop" is called).
DEVELOPMENT_SCRIPTS = [
    # NOTE: we don't install rls-analyze wrapper script; we instead depend on source_me.sh
    # (develop_rlscope in container) to add rls-test to PATH.

    # Python / C++ unit test runner.
    'rls-unit-tests = rlscope.scripts.rls_unit_tests:main',
    'rlscope-is-development-mode = rlscope.scripts.cpp.cpp_binary_wrapper:rlscope_is_development_mode'
]

# NOTE: the presence of this on "PATH" tells us whether rlscope was installed using a wheel file,
# or using "python setup.py develop" (i.e., don't install it, so it's not on PATH).
PRODUCTION_SCRIPTS = [
    'rls-analyze = rlscope.scripts.cpp.cpp_binary_wrapper:rls_analyze',

    # Wrapper around C++ unit tests.
    # Useful if we wish to run unit tests with the wheel file.
    # 'rls-test = rlscope.scripts.cpp.cpp_binary_wrapper:rls_test',

    # 'rlscope-pip-installed = rlscope.scripts.cpp.cpp_binary_wrapper:rlscope_pip_installed',
]

def find_files(pattern, root):
  for direc, dirs, files in os.walk(root):
      for filename in fnmatch.filter(files, pattern):
          yield os.path.join(direc, filename)

def generate_proto(source, require=True, regenerate=False):
  """Invokes the Protocol Compiler to generate a _pb2.py from the given
  .proto file.  Does nothing if the output already exists and is newer than
  the input."""

  if not require and not os.path.exists(source):
      return

  output = source.replace(".proto", "_pb2.py").replace("../src/", "")

  if regenerate or (
          not os.path.exists(output) or
          (os.path.exists(source) and
           os.path.getmtime(source) > os.path.getmtime(output))):
      logger.debug("Generating %s..." % output)

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
      print_cmd(protoc_command)
      if subprocess.call(protoc_command) != 0:
          sys.exit(-1)

PROTOBUF_DIR = 'rlscope/protobuf'
proto_files = list(find_files('*.proto', PROTOBUF_DIR))

POSTGRES_SQL_DIR = 'postgres'

THIRD_PARTY_DIR = 'third_party'

# PYTHON_PACKAGE_DIRS = [_j(PYTHON_SRC_DIR, direc) \
#         for direc in find_packages(where=PYTHON_SRC_DIR)]

PYTHON_PACKAGE_DIRS = [
    'rlscope',
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
    parser = argparse.ArgumentParser("Install RL-Scope python module", add_help=False)
    parser.add_argument('setup_cmd', nargs='?', default=None, help="setup.py command (e.g., develop, install, bdist_wheel, etc.)")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--help', '-h', action='store_true')
    parser.add_argument('--debug-skip-cpp', action='store_true',
                        help="(Internal debugging) Don't include librlscope.so")
    args, extra_argv = parser.parse_known_args()

    def is_development_mode():
        return args.setup_cmd == 'develop'

    def is_production_mode():
        return not is_development_mode()

    # Remove any arguments that were parsed using argparse.
    # e.g.,
    # ['setup.py', 'bdist_wheel', '--debug'] =>
    # ['setup.py', 'bdist_wheel']
    setup_argv = [sys.argv[0]]
    if args.setup_cmd is not None:
        setup_argv.append(args.setup_cmd)
    if args.help:
        setup_argv.append('--help')
    setup_argv.extend(extra_argv)
    sys.argv = setup_argv

    if args.debug:
        rlscope_logging.enable_debug_logging()
    else:
        rlscope_logging.disable_debug_logging()

    logger.debug("setup_argv = {argv}".format(argv=sys.argv))

    logger.debug("> Using protoc = {protoc}".format(protoc=protoc))
    logger.debug(pprint.pformat({
        'proto_files': proto_files,
        'PACKAGE_DIRS': PACKAGE_DIRS,
    }))

    with open("README.md", "r") as fh:
        long_description = fh.read()

    def _proto(base):
        return _j(PROTOBUF_DIR, base)
    if args.setup_cmd is not None:
        generate_proto(_proto('pyprof.proto'), regenerate=True)
        generate_proto(_proto('unit_test.proto'), regenerate=True)
        generate_proto(_proto('rlscope_prof.proto'), regenerate=True)

    rlscope_ext = get_files_by_ext('rlscope', rm_prefix='rlscope')

    logger.debug("rlscope_ext = {msg}".format(
        msg=pprint_msg(rlscope_ext),
    ))
    package_data = {
        'rlscope': [
            # NOTE: we avoid using glob(..) patterns like "**/*.py" here since
            # we need to make one for each directory level...
            # we really just want to glob for "all python files",
            # which we do using each_file_recursive(...).
        ],
    }
    keep_ext = {'cfg', 'ini', 'py', 'proto'}
    for ext in set(rlscope_ext.keys()).intersection(keep_ext):
        package_data['rlscope'].extend(rlscope_ext[ext])

    if is_production_mode() and not args.debug_skip_cpp:
        # If there exist files in rlscope/cpp/**/*
        # assume that we wish to package these into the wheel.
        cpp_files = glob(_j(ROOT, 'rlscope', 'cpp', '**', '*'))

        # Keep all rlscope/cpp/**/* files regardless of extension.
        cpp_ext = get_files_by_ext('rlscope/cpp', rm_prefix='rlscope')
        logger.debug("cpp_ext = \n{msg}".format(
            msg=pprint_msg(cpp_ext),
        ))

        if len(cpp_files) == 0:
            logger.error(textwrap.dedent("""\
                Looks like you're trying to build a python wheel for RL-Scope, but you haven't built the C++ components yet (i.e., librlscope.so, rls-analyze).
                To build a python wheel, run this:
                  $ cd {root}
                  $ BUILD_PIP=yes bash ./setup.sh
                """.format(
                root=py_config.ROOT,
            ).rstrip()))
            sys.exit(1)

        for ext, paths in cpp_ext.items():
            package_data['rlscope'].extend(paths)

    logger.debug("package_data = \n{msg}".format(
        msg=pprint_msg(package_data),
    ))

    console_scripts = []
    console_scripts.extend(CONSOLE_SCRIPTS)
    if is_production_mode():
        console_scripts.extend(PRODUCTION_SCRIPTS)
    else:
        console_scripts.extend(DEVELOPMENT_SCRIPTS)

    # logger.info("entry_points: {msg}".format(msg=pprint_msg(console_scripts)))

    if args.help:
        # Print both argparse usage AND setuptools setup.py usage info.
        parser.print_help()
    setup(
        name=PROJECT_NAME,
        version=get_pip_package_version(),
        description=DOCLINES[0],
        long_description=long_description,
        url='https://github.com/UofT-EcoSystem/rlscope',
        download_url='https://github.com/UofT-EcoSystem/rlscope/tags',
        author='James Gleeson',
        author_email='jgleeson@cs.toronto.edu',
        # Contained modules and scripts.
        packages=PACKAGE_DIRS,
        entry_points={
            'console_scripts': console_scripts,
        },
        install_requires=REQUIRED_PACKAGES,
        # tests_require=REQUIRED_PACKAGES + TEST_PACKAGES,
        # # https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras-optional-features-with-their-own-dependencies
        # # These requirements are only installed if these features are enabled when installing the pip package.
        # #
        # # $ pip install 'rlscope[docker]'
        # #
        # # Q: Does this work with .whl files?
        # extras_require={
        #     'docker': DOCKER_PACKAGES,
        # },
        package_data=package_data,
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
            'License :: OSI Approved :: Apache Software License',
        ],
        license='Apache 2.0',
        keywords='rlscope ml profiling tensorflow machine learning reinforcement learning',
    )

if __name__ == '__main__':
    main()

