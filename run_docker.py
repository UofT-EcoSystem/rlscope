# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Multipurpose TensorFlow Docker Helper.

- Assembles Dockerfiles
- Builds images (and optionally runs image tests)
- Pushes images to Docker Hub (provided with credentials)

Read README.md (in this directory) for instructions!
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b

import subprocess
import re
import os
import time
import argparse
import pwd
import collections
import pprint
import copy
import errno
import itertools
import multiprocessing
import os
import re
import shutil
import sys
import textwrap

import cerberus
import docker
from docker.models.images import Image
import yaml
# https://docker-py.readthedocs.io/en/stable/api.html#docker.api.build.BuildApiMixin.build
from io import BytesIO
from docker import APIClient
from pathlib import Path

# run_docker.py is at $REPO_ROOT/dockerfiles
REPO_ROOT = _d(_a(__file__))
sys.path.append(REPO_ROOT)

from rlscope import py_config
from rlscope.profiler.rlscope_logging import logger

from rlscope.profiler import nvidia_gpu_query

DOCKERFILES = _a(_j(REPO_ROOT, 'dockerfiles'))

NVIDIA_VISIBLE_DEVICES = [0]
assert len(NVIDIA_VISIBLE_DEVICES) > 0
PROJECT_NAME = 'rlscope'
RLSCOPE_BASH_SERVICE_NAME = 'bash'

HOME = str(Path.home())
DEFAULT_RLSCOPE_DRILL_PORT = 8129

# The tag used for a locally built "bash" RL-Scope dev environment
LOCAL_RLSCOPE_IMAGE_TAG = 'tensorflow:devel-rlscope-gpu-cuda'
DEFAULT_REMOTE_RLSCOPE_IMAGE_TAG = 'UofT-EcoSystem/rlscope:1.0.0'

RELEASE_TO_LOCAL_IMG_TAG = dict()
RELEASE_TO_LOCAL_IMG_TAG['rlscope'] = 'tensorflow:devel-rlscope-gpu-cuda'
RELEASE_TO_LOCAL_IMG_TAG['rlscope-ubuntu-20-04-cuda-11-0'] = 'tensorflow:ubuntu-20-04-devel-rlscope-cuda-11-0'

# How long should we wait for /bin/bash (rlscope_bash)
# to appear after running "docker stack deploy"?
DOCKER_DEPLOY_TIMEOUT_SEC = 10

# Schema to verify the contents of tag-spec.yml with Cerberus.
# Must be converted to a dict from yaml to work.
# Note: can add python references with e.g.
# !!python/name:builtins.str
# !!python/name:__main__.funcname
SCHEMA_TEXT = """
header:
    type: string

slice_sets:
    type: dict
    keyschema:
        type: string
    valueschema:
         type: list
         schema:
                type: dict
                schema:
                     add_to_name:
                         type: string
                     dockerfile_exclusive_name:
                         type: string
                     dockerfile_subdirectory:
                         type: string
                     partials:
                         type: list
                         schema:
                             type: string
                             ispartial: true
                     test_runtime:
                         type: string
                         required: false
                     tests:
                         type: list
                         default: []
                         schema:
                             type: string
                     args:
                         type: list
                         default: []
                         schema:
                             type: string
                             isfullarg: true
                             interpolate_arg: true
                     run_args:
                         type: list
                         default: []
                         schema:
                             type: string
                             isfullarg: true
                             interpolate_arg: true

releases:
    type: dict
    keyschema:
        type: string
    valueschema:
        type: dict
        schema:
            is_dockerfiles:
                type: boolean
                required: false
                default: false
            upload_images:
                type: boolean
                required: false
                default: true
            tag_specs:
                type: list
                required: true
                schema:
                    type: string
"""

# https://stackoverflow.com/questions/17215400/python-format-string-unused-named-arguments
class FormatDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'

class TfDockerTagValidator(cerberus.Validator):
    """Custom Cerberus validator for TF tag spec.

    Note: Each _validate_foo function's docstring must end with a segment
    describing its own validation schema, e.g. "The rule's arguments are...". If
    you add a new validator, you can copy/paste that section.
    """

    def __init__(self, *args, **kwargs):
        # See http://docs.python-cerberus.org/en/stable/customize.html
        if 'partials' in kwargs:
            self.partials = kwargs['partials']
        super(cerberus.Validator, self).__init__(*args, **kwargs)

    def _validate_ispartial(self, ispartial, field, value):
        """Validate that a partial references an existing partial spec.

        Args:
            ispartial: Value of the rule, a bool
            field: The field being validated
            value: The field's value
        The rule's arguments are validated against this schema:
        {'type': 'boolean'}
        """
        if ispartial and value not in self.partials:
            self._error(field,
                        '{} is not present in the partials directory.'.format(value))

    def _validate_isfullarg(self, isfullarg, field, value):
        """Validate that a string is either a FULL=arg or NOT.

        Args:
            isfullarg: Value of the rule, a bool
            field: The field being validated
            value: The field's value
        The rule's arguments are validated against this schema:
        {'type': 'boolean'}
        """
        if isfullarg and '=' not in value:
            self._error(field, '{} should be of the form ARG=VALUE.'.format(value))
        if not isfullarg and '=' in value:
            self._error(field, '{} should be of the form ARG (no =).'.format(value))

    def _validate_interpolate_arg(self, interpolate_arg, field, value):
        """Test the oddity of a value.
        The rule's arguments are validated against this schema:
        {'type': 'boolean'}
        """
        if interpolate_arg:
            validate_interpolate_arg(value, lambda message: self._error(field, message))

def default_report_error(message):
    raise RuntimeError(message)
def validate_interpolate_arg(value, report_error=default_report_error):
    bracket_regex = re.compile(r'\{([^}]+)\}')
    bracket_varnames = bracket_regex.findall(value)
    if len(bracket_varnames) == 0:
        return

    for varname in bracket_varnames:
        if varname not in ARG_VALUES:
            report_error((
                "Couldn't find {var} in --arg or in spec.yml for arg=\"{value}\"; available ARG_VALUES for use in "
                "--arg or spec.yml args are:\n"
                "{args}").format(
                value=value,
                var=varname,
                args=textwrap.indent(pprint.pformat(ARG_VALUES), prefix="    ")))


def aggregate_all_slice_combinations(spec, slice_set_names):
    """Figure out all of the possible slice groupings for a tag spec."""
    slice_sets = copy.deepcopy(spec['slice_sets'])

    for name in slice_set_names:
        for slice_set in slice_sets[name]:
            slice_set['set_name'] = name

    slices_grouped_but_not_keyed = [slice_sets[name] for name in slice_set_names]
    all_slice_combos = list(itertools.product(*slices_grouped_but_not_keyed))
    return all_slice_combos


def build_name_from_slices(format_string, slices, args, is_dockerfile=False):
    """Build the tag name (cpu-devel...) from a list of slices."""
    name_formatter = copy.deepcopy(args)
    name_formatter.update({s['set_name']: s['add_to_name'] for s in slices})
    name_formatter.update({
        s['set_name']: s['dockerfile_exclusive_name']
        for s in slices
        if is_dockerfile and 'dockerfile_exclusive_name' in s
    })
    name = format_string.format(**name_formatter)
    return name


def parse_build_arg(arg):
    key, sep, value = arg.partition('=')
    return key, value

def update_args_dict(args_dict, updater, keep_original=False):
    """Update a dict of arg values with more values from a list or dict."""
    def _get_env(env_var, default):
        if env_var in os.environ:
            return os.environ[env_var]
        return default

    if isinstance(updater, list):
        for arg in updater:
            key, sep, value = arg.partition('=')
            if sep == '=':
                # args_dict[key] = interpolate_arg(value)
                if key not in args_dict or not keep_original:
                    args_dict[key] = _get_env(key, str(value))
    if isinstance(updater, dict):
        for key, value in updater.items():
            # args_dict[key] = interpolate_arg(value)
            if key not in args_dict or not keep_original:
                args_dict[key] = _get_env(key, str(value))
    return args_dict


def get_slice_sets_and_required_args(slice_sets, tag_spec):
    """Extract used-slice-sets and required CLI arguments from a spec string.

    For example, {FOO}{bar}{bat} finds FOO, bar, and bat. Assuming bar and bat
    are both named slice sets, FOO must be specified on the command line.

    Args:
         slice_sets: Dict of named slice sets
         tag_spec: The tag spec string, e.g. {_FOO}{blep}

    Returns:
         (used_slice_sets, required_args), a tuple of lists
    """
    required_args = []
    used_slice_sets = []

    extract_bracketed_words = re.compile(r'\{([^}]+)\}')
    possible_args_or_slice_set_names = extract_bracketed_words.findall(tag_spec)
    for name in possible_args_or_slice_set_names:
        if name in slice_sets:
            used_slice_sets.append(name)
        else:
            required_args.append(name)

    return (used_slice_sets, required_args)


# IDEAL:
# BAZEL_BUILD_DIR can EITHER be provided via environment variable, or via --run_arg.
# If both are present, use env.BAZEL_BUILD_DIR.
def gather_tag_args(slices, cli_input_args, required_args=None, spec_field='args', cmd_opt=None):
    """Build a dictionary of all the CLI and slice-specified args for a tag."""
    if cmd_opt is None:
        cmd_opt = spec_field.rstrip('s')

    args = dict()

    for s in slices:
        if spec_field in s:
            args = update_args_dict(args, s[spec_field])

    # Only keep environment variables that have been "declared" in the spec.yml file.
    # e.g.
    # CHECKOUT_TF_SRC=
    for env_var, env_value in os.environ.items():
        if env_var in args:
            print("> Using environment variable {env}={val} for --{cmd_opt}".format(
                env=env_var,
                val=env_value,
                cmd_opt=cmd_opt))
            args[env_var] = env_value

    args = update_args_dict(args, cli_input_args)
    if required_args is not None:
        for arg in required_args:
            if arg not in args:
                logger.error(('> Error: {arg} is not a valid slice_set, and also isn\'t an arg '
                        'provided on the command line. If it is an arg, please specify '
                        'it with --{cmd_opt}. If not, check the slice_sets list.'.format(
                    arg=arg,
                    cmd_opt=cmd_opt,
                )))
                exit(1)

    return args


def gather_slice_list_items(slices, key):
    """For a list of slices, get the flattened list of all of a certain key."""
    return list(itertools.chain(*[s[key] for s in slices if key in s]))


def find_first_slice_value(slices, key):
    """For a list of slices, get the first value for a certain key."""
    for s in slices:
        if key in s and s[key] is not None:
            return s[key]
    return None


def assemble_tags(spec, cli_args, cli_run_args, enabled_release, all_partials):
    """Gather all the tags based on our spec.

    Args:
        spec: Nested dict containing full Tag spec
        cli_args: List of ARG=foo arguments to pass along to Docker build
        enabled_releases: List of releases to parse. Empty list = all
        all_partials: Dict of every partial, for reference

    Returns:
        Dict of tags and how to build them
    """
    tag_data = collections.defaultdict(list)

    valid_names = set(spec['releases'].keys())
    if enabled_release is not None and enabled_release not in valid_names:
        raise DeployError("No such --release \"{release}\" found in spec.yml; valid choices for --release are: {choices}".format(
            release=enabled_release,
            choices=valid_names,
        ))

    for name, release in spec['releases'].items():
        for tag_spec in release['tag_specs']:
            if enabled_release is not None and name != enabled_release:
                logger.info('> Skipping release {name}; (!= {enabled_release})'.format(
                    name=name,
                    enabled_release=enabled_release))
                continue

            used_slice_sets, required_cli_args = get_slice_sets_and_required_args(
                spec['slice_sets'], tag_spec)

            slice_combos = aggregate_all_slice_combinations(spec, used_slice_sets)
            for slices in slice_combos:

                tag_args = gather_tag_args(slices, cli_args, required_cli_args)
                tag_args.update(get_implicit_build_args())
                run_args = gather_tag_args(slices, cli_run_args, spec_field='run_args')
                tag_name = build_name_from_slices(tag_spec, slices, tag_args,
                                                  release['is_dockerfiles'])
                used_partials = gather_slice_list_items(slices, 'partials')
                used_tests = gather_slice_list_items(slices, 'tests')
                test_runtime = find_first_slice_value(slices, 'test_runtime')
                dockerfile_subdirectory = find_first_slice_value(
                    slices, 'dockerfile_subdirectory')
                dockerfile_contents = merge_partials(spec['header'], used_partials,
                                                     all_partials)

                tag_data[tag_name].append({
                    'release': name,
                    'tag_spec': tag_spec,
                    'is_dockerfiles': release['is_dockerfiles'],
                    'upload_images': release['upload_images'],
                    'cli_args': tag_args,
                    'run_args': run_args,
                    'dockerfile_subdirectory': dockerfile_subdirectory or '',
                    'partials': used_partials,
                    'tests': used_tests,
                    'test_runtime': test_runtime,
                    'dockerfile_contents': dockerfile_contents,
                })

    return tag_data


def merge_partials(header, used_partials, all_partials):
    """Merge all partial contents with their header."""
    used_partials = list(used_partials)
    ret = '\n'.join([header] + [all_partials[u] for u in used_partials])

    return ret


# def upload_in_background(hub_repository, dock, image, tag):
#     """Upload a docker image (to be used by multiprocessing)."""
#     image.tag(hub_repository, tag=tag)
#     logging.info(dock.images.push(hub_repository, tag=tag))


def mkdir_p(path):
    """Create a directory and its parents, even if it already exists."""
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def gather_existing_partials(partial_path):
    """Find and read all available partials.

    Args:
        partial_path (string): read partials from this directory.

    Returns:
        Dict[string, string] of partial short names (like "ubuntu/python" or
            "bazel") to the full contents of that partial.
    """
    partials = dict()
    for path, _, files in os.walk(partial_path):
        for name in files:
            fullpath = os.path.join(path, name)
            if not re.search(r'\.partial\.Dockerfile$', _b(fullpath)):
                logger.info(('> Probably not a problem: skipping {}, which is not a '
                        'partial.').format(fullpath))
                continue
            # partial_dir/foo/bar.partial.Dockerfile -> foo/bar
            simple_name = fullpath[len(partial_path) + 1:-len('.partial.dockerfile')]
            with open(fullpath, 'r') as f:
                partial_contents = f.read()
                check_null_byte(fullpath, partial_contents)
            partials[simple_name] = partial_contents
    return partials

def check_null_byte(name, string):
    null_idx = string.find('\x00')
    if null_idx != -1:
        logger.info("Found null byte in {name}:".format(name=name))
        logger.info((
            "> Before null byte:\n"
            "{str}"
        ).format(str=textwrap.indent(
            string[:null_idx],
            prefix='  ',
        )))
        logger.info((
            "> After null byte:\n"
            "{str}"
        ).format(str=textwrap.indent(
            string[null_idx:],
            prefix='  ')))

def get_build_logfile(repo_tag):
    return "{repo_tag}.build.log.txt".format(
        repo_tag=repo_tag)

"""
If you use any of these in your --arg or in arg specifications in spec.yml, 
they will get replaced with corresponding ARG_VALUES dict values.
"""
ARG_VALUES = {
    'RLSCOPE_DIR': py_config.RLSCOPE_DIR,
}
def interpolate_arg(arg):
    # bracket_regex = re.compile(r'\{([^}]+)\}')
    # bracket_varnames = bracket_regex.findall(arg)
    # if len(bracket_varnames) == 0:
    #     return arg
    #
    # for varname in bracket_varnames:
    #     if varname not in ARG_VALUES:
    #         raise RuntimeError((
    #             "Couldn't find ; available ARG_VALUES for use in "
    #             "--arg or spec.yml args are:\n"
    #             "{args}").format(
    #             args=textwrap.indent(pprint.pformat(ARG_VALUES), prefix="    ")))

    return arg.format(**ARG_VALUES)

def get_docker_run_env(tag_def, env_list):
    # Same as docker.from_env(); inherit current environment variables.
    # env = dict(os.environ)

    # if 'run_args' in tag_def:
    #     run_args = tag_def['run_args']
    # else:
    #     run_args = dict()
    # def _get_env(env_var, desc):
    #     if env_var not in os.environ and env_var in run_args:
    #         return run_args[env_var]
    #     elif env_var in os.environ:
    #         return os.environ[env_var]
    #     logger.info("> You must provide {env}=[ {desc} ]".format(
    #         env=env_var,
    #         desc=desc))
    #     sys.exit(1)

    env = dict()
    env.update(get_implicit_run_args())
    if 'run_args' in tag_def:
        env = update_args_dict(env, tag_def['run_args'], keep_original=True)

    for var, value in env.items():
        if value == '':
            logger.error(("> ERROR: you must provide a value for --run_arg {var}=<VALUE> "
                    "(or define an environment variable); see {spec} for documentation.").format(
                var=var,
                spec="spec.yml"))
            sys.exit(1)

    for env_str in env_list:
        assert '=' in env_str
        var, value = re.split(r'=', env_str)
        # Duplicate var?
        assert var not in env
        env[var] = value

    # env['RLSCOPE_USER'] = get_username()
    # env['RLSCOPE_UID'] = get_user_id()
    # env['RLSCOPE_GID'] = get_group_id()
    env['RLSCOPE_INSTALL_PREFIX'] = py_config.DOCKER_INSTALL_PREFIX
    env['RLSCOPE_BUILD_PREFIX'] = py_config.DOCKER_BUILD_PREFIX
    env['RLSCOPE_IS_DOCKER'] = 'yes'

    return env

def get_docker_runtime(tag_def):
    runtime = None
    if tag_def['test_runtime'] == 'nvidia':
        runtime = 'nvidia'
    else:
        # Use runtime=None for rocm.
        runtime = None

    return runtime


def get_docker_run_argv(argv):
    """
    absl will preserve "extra" arguments that were unparsed by the FLAGS specification.
    We pass all of these arguments directly to the "docker run" command.
    This makes it easy to:
    - Define additional environment variables (outside of required ones)
    - Define additional volume mounts (outside of required ones)

    :param argv:
        argv = ["run_docker.py" "extra_argument[0]", ...]
    :return:
    """
    return argv[1:]

def main():
    # Run all docker related commands from $ROOT/dockerfiles.
    os.chdir(DOCKERFILES)

    parser = argparse.ArgumentParser(description=__doc__)

    if shutil.which('docker-compose') is None:
        logger.error("Didn't find docker-compose on PATH; you must install it.")
        sys.exit(1)

    os.makedirs(_j(py_config.ROOT, 'dockerfiles/dockerfiles'), exist_ok=True)

    # Copy requirements.txt into dockerfiles so it can be accessed by Dockerfile during its build.
    shutil.copy(
        _j(py_config.ROOT, "requirements.txt"),
        _j(DOCKERFILES, "requirements.txt"),
    )

    # parser.add_argument('--hub_username',
    #                                         help='Dockerhub username, only used with --upload_to_hub')

    # parser.add_argument(
    #         '--hub_password',
    #         help=('Dockerhub password, only used with --upload_to_hub. Use from an env param'
    #            'so your password isn\'t in your history.'))

    parser.add_argument('--hub_timeout', default=3600,
                        type=int,
                        help='Abort Hub upload if it takes longer than this.')

    parser.add_argument('--pull', action='store_true',
                        help=textwrap.dedent("""
                        Pull pre-built RL-Scope dev environment image from DockerHub, 
                        instead of building locally.
                        
                        See --pull-image for specifying the image to pull.
                        """))

    parser.add_argument('--stop', action='store_true',
                        help=textwrap.dedent("""
                        Stop docker-compose containers (opposite of --deploy).
                        """))

    parser.add_argument('--reload', action='store_true',
                        help=textwrap.dedent("""
                        Restart running containers, even if the 
                        underlying docker file hasn't changed.
                        """))

    parser.add_argument('--mps', action='store_true',
                        help=textwrap.dedent("""
                        Use CUDA multi-process service (MPS) daemon to allow multiple GPU processes to 
                        share GPU execution simultaneously.
                        
                        This option will add mps related setup to the generated docker-compose file (stack.yml).
                        """))

    parser.add_argument('--pull-image', default=DEFAULT_REMOTE_RLSCOPE_IMAGE_TAG,
                        help=textwrap.dedent("""
                        RL-Scope dev environment image to pull from DockerHub
                        using "docker pull <pull_img>"
                        """))

    parser.add_argument('--deploy-rlscope-drill-port',
                        default=DEFAULT_RLSCOPE_DRILL_PORT,
                        type=int,
                        help=('What port to run rlscope-drill web server on '
                              '(when running "docker stack deploy -c stack.yml rlscope")'))

    parser.add_argument(
        '--repository', default='tensorflow',
        help='Tag local images as {repository}:tag (in addition to the '
             'hub_repository, if uploading to hub)')

    parser.add_argument(
        '--volume',
        action='append',
        default=[],
        help=textwrap.dedent("""\
        Translates into docker --volume option. 
        We mount the path at the same path as it is in the host.
        i.e. 
        # run_docker.py option:
        --volume /one/two
        #
        # becomes
        #
        # `docker run` option:
        --volume /one/two:/one/two
        """).rstrip())

    parser.add_argument(
        '--publish',
        action='append',
        default=[],
        help=textwrap.dedent("""\
        Translates into docker --publish option; e.g.
        --publish <HOST_PORT>:<CONTAINER_PORT> 
        """).rstrip())

    parser.add_argument(
        '--env',
        '-e',
        action='append',
        default=[],
        help=textwrap.dedent("""\
        Translates into docker --env option. 
        """).rstrip())

    parser.add_argument(
        '--hub_repository',
        help='Push tags to this Docker Hub repository, e.g. tensorflow/tensorflow')

    parser.add_argument(
        '--debug',
        action='store_true',
        help=textwrap.dedent("""
            In the generated dockerfiles, print start/end markers for the partial files its composed of; for e.g.:
                START: dockerfiles/partials/ubuntu/install_cuda_10_1.partial.Dockerfile
                RUN ...
                RUN ...
                ...
                END: dockerfiles/partials/ubuntu/install_cuda_10_1.partial.Dockerfile
            """))

    parser.add_argument(
        '--pdb',
        action='store_true',
        help=textwrap.dedent("""
            Debug: breakpoint on error.
            """))

    # parser.add_argument(
    #         '--upload_to_hub', '-u',
    #         help=('Push built images to Docker Hub (you must also provide --hub_username, '
    #            '--hub_password, and --hub_repository)'),
    # )

    parser.add_argument(
        '--construct_dockerfiles', '-d',
        action='store_true',
        help='Do not build images')

    parser.add_argument(
        '--keep_temp_dockerfiles', '-k',
        action='store_true',
        help='Retain .temp.Dockerfiles created while building images.')

    parser.add_argument(
        '--build_images', '-b',
        action='store_true',
        help='Do not build images')

    parser.add_argument(
        '--deploy',
        action='store_true',
        help=
        textwrap.dedent("""\
        Deploy the RL-Scope development environment using 
        "docker stack deploy -c stack.yml rlscope".
        """.format(USER=get_username())))

    parser.add_argument(
        '--run',
        action='store_true',
        help='Run built images; use --deploy if you want to deploy the whole RL-Scope development environment')

    # parser.add_argument(
    #     '--run_tests_path',
    #     help=('Execute test scripts on generated Dockerfiles before pushing them. '
    #           'Flag value must be a full path to the "tests" directory, which is usually'
    #           ' $(realpath ./tests). A failed tests counts the same as a failed build.'))

    parser.add_argument(
        '--stop_on_failure',
        action='store_true',
        help=('Stop processing tags if any one build fails. If False or not specified, '
              'failures are reported but do not affect the other images.'))

    parser.add_argument(
        '--dry_run', '-n',
        action='store_true',
        help='Do not build or deploy anything at all.',
    )

    parser.add_argument(
        '--exclude_tags_matching', '-x',
        help=('Regular expression that skips processing on any tag it matches. Must '
              'match entire string, e.g. ".*gpu.*" ignores all GPU tags.'),
    )

    parser.add_argument(
        '--only_tags_matching', '-i',
        help=('Regular expression that skips processing on any tag it does not match. '
              'Must match entire string, e.g. ".*gpu.*" includes only GPU tags.'),
    )

    parser.add_argument(
        '--dockerfile_dir', '-o',
        default='./dockerfiles',
        help='Path to an output directory for Dockerfiles.'
             ' Will be created if it doesn\'t exist.'
             ' Existing files in this directory will be deleted when new Dockerfiles'
             ' are made.',
    )

    parser.add_argument(
        '--partial_dir', '-p',
        default='./partials',
        help='Path to a directory containing foo.partial.Dockerfile partial files.'
             ' can have subdirectories, e.g. "bar/baz.partial.Dockerfile".',
    )

    parser.add_argument(
        '--release', '-r',
        default='rlscope',
        help='Set of releases to build and tag. Defaults to every release type.',
    )

    parser.add_argument(
        '--arg', '-a', default=[], action='append',
        help=('Extra build arguments. These are used for expanding tag names if needed '
              '(e.g. --arg _TAG_PREFIX=foo) and for using as build arguments (unused '
              'args will print a warning).'),
    )

    parser.add_argument(
        '--run_arg', default=[], action='append',
        help=('Extra container run arguments (NOT build).'))

    parser.add_argument(
        '--spec_file', '-s',
        default='./spec.yml',
        help='Path to the YAML specification file',
    )

    parser.add_argument(
        '--output_stack_yml',
        default='./stack.yml',
        help='Path to the generated YAML "Docker Compose" file for '
             'use with "docker stack deploy -c stack.yml rlscope"',
    )

    argv = list(sys.argv)
    args, extra_argv = parser.parse_known_args()

    # NOTE: If this fails, you need to enable nvidia-persistend daemon.
    nvidia_gpu_query.check_nvidia_smi(exit_if_fail=True)

    if not args.stop and not args.pull and not args.reload and not args.build_images and not args.run and not args.deploy:
        # Default options:
        args.construct_dockerfiles = True
        args.build_images = True
        args.deploy = True

    if args.deploy and args.run:
        parser.error(
            "Provide either --deploy or --run.  "
            "Use --deploy to deploy the RL-Scope development environment (probably what you want)")

    try:
        assembler = Assembler(parser, argv, args, extra_argv)
        assembler.run()
    except DeployError as e:
        if args.debug:
            raise e
        print("ERROR: {e}".format(e=e), file=sys.stderr)
        sys.exit(1)


class Assembler:
    def __init__(self, parser, argv, args, extra_argv):
        self.parser = parser
        self.argv = argv
        self.args = args
        self.extra_argv = extra_argv

    def read_spec_yml(self):
        args = self.args
        # Read the full spec file, used for everything
        with open(args.spec_file, 'r') as spec_file:
            tag_spec = yaml.load(spec_file, Loader=yaml.FullLoader)

        for arg in args.arg:
            key, value = parse_build_arg(arg)
            validate_interpolate_arg(value)

        for run_arg in args.run_arg:
            key, value = parse_build_arg(run_arg)
            validate_interpolate_arg(value)

        # Get existing partial contents
        partials = gather_existing_partials(args.partial_dir)

        # Abort if spec.yaml is invalid
        schema = yaml.load(SCHEMA_TEXT, Loader=yaml.FullLoader)
        v = TfDockerTagValidator(schema, partials=partials)
        if not v.validate(tag_spec):
            logger.error('> Error: {} is an invalid spec! The errors are:'.format(
                args.spec_file))
            logger.error(yaml.dump(v.errors, indent=2))
            exit(1)
        tag_spec = v.normalized(tag_spec)

        # Assemble tags and images used to build them
        run_arg_required = []
        for run_arg in args.run_arg:
            var, value = parse_build_arg(run_arg)
            if is_required_run_arg(var):
                run_arg_required.append(run_arg)
        all_tags = assemble_tags(tag_spec, args.arg, run_arg_required, args.release, partials)

        return all_tags

    def docker_build(self, dockerfile, repo_tag, tag_def, debug):
        args = self.args
        build_kwargs = dict(
            timeout=args.hub_timeout,
            path=DOCKERFILES,
            # path='dockerfiles',
            dockerfile=_a(dockerfile),
            buildargs=tag_def['cli_args'],
            tag=repo_tag,
        )
        if args.debug:
            print("> dock.images.build")
            print(textwrap.indent(pprint.pformat(build_kwargs), prefix="    "))

        build_cmd = get_docker_cmdline('build', **build_kwargs)
        logger.info(get_cmd_string(build_cmd, show_cwd=True))
        build_output_generator = self.dock_cli.build(decode=True, **build_kwargs)
        response = tee_docker(
            build_output_generator,
            file=get_build_logfile(repo_tag),
            debug=debug)
        # I've seen "docker build" fail WITHOUT any error indication
        # (e.g. return code, raising dockers.errors.APIError),
        # so just grep for "error" in the response['message'].
        check_docker_response(response, dockerfile, repo_tag, cmd=build_cmd)

        image = self.dock.images.get(repo_tag)

        return image

    def docker_run(self, image, tag_def, extra_argv):
        # Run the container.
        args = self.args
        docker_run_env = get_docker_run_env(tag_def, args.env)
        rlscope_volumes = get_rlscope_volumes(args, docker_run_env, args.volume)
        runtime = get_docker_runtime(tag_def)
        run_kwargs = dict(
            image=image,
            # command='/tests/' + test,
            working_dir='/',
            log_config={'type': 'journald'},
            detach=True,
            stderr=True,
            stdout=True,
            environment=docker_run_env,
            volumes=rlscope_volumes,
            # volumes={
            #     args.run_tests_path: {
            #         'bind': '/tests',
            #         'mode': 'ro'
            #     }
            # },
            remove=True,
            cap_add=['SYS_ADMIN', 'SYS_PTRACE'],
            security_opt=['seccomp=unconfined'],
            runtime=runtime,
            name="rlscope",
        )
        if tag_def['test_runtime'] == 'rocm':
            def device_opt(path):
                return '{path}:{path}:rwm'.format(path=path)
            # --device=/dev/kfd --device=/dev/dri
            run_kwargs['devices'] = [
                device_opt('/dev/kfd'),
                device_opt('/dev/dri'),
            ]
            # --group-add video
            run_kwargs['group_add'] = [
                'video',
            ]
        docker_run_argv = extra_argv
        cmd = get_docker_cmdline('run', docker_run_argv, **run_kwargs)
        logger.info(get_cmd_string(cmd))
        subprocess.run(cmd)
        # Q: Save output?

    # def _is_rlscope_deployed(self):
    #     self.docker_ps_rlscope_bash() is not

    def docker_ps_rlscope_bash(self):
        # logging.info("(1) run docker ps")
        p = subprocess.run("docker ps | grep rlscope_bash",
                           shell=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           # check=True,
                           )
        # I've seen this fail with returncode=1, but NO error message...
        # I feel like its a bug in docker...
        if p.returncode != 0:
            return None
        # logging.info("(2) run docker ps")
        # p = subprocess.run("docker ps | grep rlscope_bash",
        #                    shell=True,
        #                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        #                    check=True,
        #                    )
        lines = p.stdout.decode('utf-8').splitlines()
        for line in lines:
            if re.search(r'rlscope_bash', line):
                fields = re.split(r'\s+', line)
                container_id = fields[0]
                return container_id
        return None

    def _wait_for_bash(self, container_filter):
        timeout_sec = time.time() + DOCKER_DEPLOY_TIMEOUT_SEC
        now_sec = time.time()
        while now_sec < timeout_sec:

            # services = self.dock_cli.services(filters={'name':'rlscope_bash'})
            # if len(services) > 0:
            #     info = self.dock_cli.inspect_service('rlscope_bash')
            #     return info

            # containers = self.dock.containers.list(filters={'name': 'rlscope_bash'})
            containers = self.dock.containers.list(filters={'name': container_filter})
            if len(containers) > 0:
                # containers.name
                assert len(containers) == 1
                return containers[0]

            # container_id = self.docker_ps_rlscope_bash()
            # if container_id is not None:
            #     return container_id

            time.sleep(0.5)
            now_sec = time.time()

        print(textwrap.dedent("""
        > FAILURE: Waited for /bin/bash (rlscope_bash) container to appear, but it didn't after {sec} seconds.
          To figure out what went wrong, try the following:

            # Look at docker logs:
            $ sudo journalctl -u docker.service

            # Look at service logs:
            $ docker service logs --raw rlscope_bash
            
            # See what's actually running
            $ docker ps
        """.format(
            sec=DOCKER_DEPLOY_TIMEOUT_SEC,
        )))
        sys.exit(1)

    def _wait_for_rlscope_to_stop(self):
        """
        Wait until all services/processes belonging to the rlscope dev environment have stopped.

        Currently 'docker stack rm rlscope' returns immediately even though the container hasn't terminated yet,
        leading to race conditions when re-creating rlscope (causes errors when creating rlscope_default network).

        :param self:
        :return:
        """
        while True:
            rlscope_services = self.dock_cli.services(filters={'name': 'rlscope'})
            pprint.pprint({'rlscope_services': rlscope_services})
            if len(rlscope_services) == 0:
                return
            time.sleep(0.1)

    def docker_stack_rm(self):
        # NOTE: DON'T remove the container...just stop it (unless --rm)
        # This will make starting in mps/non-mps mode less annoying.
        services = self.dock.services.list(filters={'name': 'rlscope'})
        if len(services) == 0:
            # RL-Scope dev environment not running yet.
            return
        print("> Detected old RL-Scope dev environment; removing it first")
        for srv in services:
            srv.remove()
        after = self.dock.services.list(filters={'name': 'rlscope'})
        assert len(after) == 0

        # IMPORTANT: apparently, removing the services above does NOT immediately remove the containers.
        # So, instead we busy wait until they dissapear.
        # We need this to avoid errors when calling 'docker stack deploy'.
        while True:
            containers = self.dock.containers.list(filters={'name': 'rlscope'})
            if len(containers) == 0:
                break
            time.sleep(0.1)

    def project_name(self):
        args = self.args
        if args.mps:
            return "{PROJECT_NAME}_mps".format(PROJECT_NAME=PROJECT_NAME)
        return PROJECT_NAME

    def docker_stop(self, extra_argv):
        args = self.args
        # Terminate/remove an existing deployment if it exists first.
        # self.docker_stack_rm()

        # cmd = ['docker', 'stack', 'deploy']
        # cmd.extend(extra_argv)
        # cmd.extend([
        #     '--compose-file', 'stack.yml',
        #     # Name of the created stack.
        #     'rlscope',
        # ])
        # container_filter = "rlscope_bash"

        cmd = ['docker-compose']
        cmd.extend([
            '--file', 'stack.yml',
            '--project-name', self.project_name(),
        ])
        cmd.extend([
            'stop',
        ])
        cmd.extend(extra_argv)
        container_filter = "rlscope"

        logger.info(get_cmd_string(cmd))
        subprocess.check_call(cmd)

        if args.mps:
            # Reset the "compute mode" of the GPUs from "Exclsuive" back to "Default".
            # We use exclusive mode just to make sure GPU apps are actually using MPS and not bypassing it.
            nvidia_set_compute_mode('DEFAULT', NVIDIA_VISIBLE_DEVICES)
            # nvidia_set_compute_mode('EXCLUSIVE_PROCESS', NVIDIA_VISIBLE_DEVICES)

    def docker_deploy(self, extra_argv, reload):
        """
        $ docker stack deploy {extra_args} --compose-file stack.yml rlscope
        $ docker attach <rlscope_bash /bin/bash container>

        :param extra_argv:
            Extra arguments to pass to "stack deploy"
        :return:
        """
        args = self.args

        # Terminate/remove an existing deployment if it exists first.
        # self.docker_stack_rm()

        # cmd = ['docker', 'stack', 'deploy']
        # cmd.extend(extra_argv)
        # cmd.extend([
        #     '--compose-file', 'stack.yml',
        #     # Name of the created stack.
        #     'rlscope',
        # ])
        # container_filter = "rlscope_bash"

        if args.mps:
            # Set "compute mode" of the GPUs to "exclusive" to make sure GPU apps
            # are actually using MPS and not bypassing it.
            nvidia_set_compute_mode('EXCLUSIVE_PROCESS', NVIDIA_VISIBLE_DEVICES)

        cmd = ['docker-compose']
        cmd.extend([
            '--file', 'stack.yml',
            '--project-name', self.project_name(),
        ])
        cmd.extend([
            'up',
            # Run containers in background (similar to docker stack deploy)
            '--detach',

            # NOTE: Don't do this since we change stack.yml when using --mps
            # (don't want to delete mps container).
            #
            # When container_name changes, deleted old container_name.
            # '--remove-orphans'

            # Name of the created stack.
            # 'rlscope',
        ])
        if reload:
            cmd.append('--force-recreate')
        cmd.extend(extra_argv)
        container_filter = "{project}_{RLSCOPE_BASH_SERVICE_NAME}".format(
            project=self.project_name(),
            RLSCOPE_BASH_SERVICE_NAME=RLSCOPE_BASH_SERVICE_NAME)

        logger.info(get_cmd_string(cmd))
        subprocess.check_call(cmd)
        rlscope_bash_container = self._wait_for_bash(container_filter)

        ps_cmd = ['docker', 'ps']
        logger.info(get_cmd_string(ps_cmd))
        subprocess.check_call(ps_cmd)
        print("> Deployed RL-Scope development environment")
        print("> Attaching to /bin/bash in the dev environment:")
        # Login to existing container using a new /bin/bash shell so we are greeted with the _rlscope_banner
        exec_cmd = ['docker', 'exec', '-i', '-t', rlscope_bash_container.name, '/bin/bash']
        logger.info(get_cmd_string(exec_cmd))
        subprocess.run(exec_cmd)

    def docker_pull(self, pull_img):
        """
        $ docker pull <pull_img>
        """
        cmd = ['docker', 'pull', pull_img]
        logger.info(get_cmd_string(cmd))
        subprocess.check_call(cmd)
        image = self.dock.images.get(pull_img)
        return image

    def run_tests(self, image, repo_tag, tag, tag_def):
        tag_failed = False
        args = self.args
        docker_run_env = get_docker_run_env(tag_def, args.env)
        if not tag_def['tests']:
            logger.info('>>> No tests to run.')
        for test in tag_def['tests']:
            logger.info('>> Testing {}...'.format(test))

            runtime = get_docker_runtime(tag_def)

            test_kwargs = dict(
                image=image,
                command=_j(docker_run_env['RLSCOPE_TEST_SH'], test),
                # command='/tests/' + test,
                working_dir='/',
                log_config={'type': 'journald'},
                detach=True,
                stderr=True,
                stdout=True,
                environment=docker_run_env,
                volumes={
                    # args.run_tests_path: {
                    docker_run_env['RLSCOPE_TEST_SH']: {
                        # 'bind': '/tests',
                        'bind': docker_run_env['RLSCOPE_TEST_SH'],
                        'mode': 'ro'
                    }
                },
                runtime=runtime,
            )

            if args.debug:
                print("> TEST: dock.containers.run")
                print(textwrap.indent(pprint.pformat(test_kwargs), prefix="    "))

            container = self.dock.containers.run(**test_kwargs)
            ret = container.wait()
            code = ret['StatusCode']
            out = container.logs(stdout=True, stderr=False)
            err = container.logs(stdout=False, stderr=True)
            container.remove()
            if out:
                logger.info('>>> Output stdout:')
                logger.info(out.decode('utf-8'))
            else:
                logger.info('>>> No test standard out.')
            if err:
                logger.info('>>> Output stderr:')
                logger.info(out.decode('utf-8'))
            else:
                logger.info('>>> No test standard err.')
            if code != 0:
                logger.error('>> {} failed tests with status: "{}"'.format(
                    repo_tag, code))
                self.failed_tags.append(tag)
                tag_failed = True
                if args.stop_on_failure:
                    logger.error('>> ABORTING due to --stop_on_failure!')
                    exit(1)
            else:
                logger.info('>> Tests look good!')

        return tag_failed

    def generate_stack_yml(self, tag_def):
        args = self.args
        generator = StackYMLGenerator(self.project_name())

        docker_run_env = get_docker_run_env(tag_def, args.env)
        rlscope_volumes = get_rlscope_volumes(args, docker_run_env, args.volume)
        rlscope_ports = get_rlscope_ports(docker_run_env, args.publish)

        if not args.pull:
            if args.release not in RELEASE_TO_LOCAL_IMG_TAG:
                logger.error("ERROR: Not sure what image tag to use for --release={release}; please modify run_docker.py by setting RELEASE_TO_LOCAL_IMG_TAG['{release}']".format(release=args.release))
                sys.exit(1)

        if args.pull:
            rlscope_image = args.pull_image
        else:
            rlscope_image = RELEASE_TO_LOCAL_IMG_TAG[args.release]

        yml = generator.generate(
            assembler_cmd=self.argv,
            env=docker_run_env,
            volumes=rlscope_volumes,
            ports=rlscope_ports,
            rlscope_drill_port=args.deploy_rlscope_drill_port,
            rlscope_image=rlscope_image,
            use_mps=args.mps,
        )
        print("> Write 'docker stack deploy' stack.yml file to {path}".format(path=args.output_stack_yml))
        with open(args.output_stack_yml, 'w') as f:
            f.write(yml)

    def run(self):
        parser = self.parser
        argv = self.argv
        args = self.args
        extra_argv = self.extra_argv

        # Sanity check:
        if not _e('spec.yml'):
            print("ERROR: run_docker.py must run within {ROOT}/dockerfiles".format(ROOT=py_config.ROOT))
            sys.exit(1)

        # In order to copy $ROOT/requirements.txt into the container,
        # it can't be in an upper directory (i.e. ../.....).
        shutil.copy(
            src=_j(py_config.ROOT, 'dockerfiles/requirements.txt'),
            dst='dockerfiles/requirements.txt')

        all_tags = self.read_spec_yml()

        # Empty Dockerfile directory if building new Dockerfiles
        if args.construct_dockerfiles:
            logger.info('> Emptying Dockerfile dir "{}"'.format(args.dockerfile_dir))
            shutil.rmtree(args.dockerfile_dir, ignore_errors=True)
            mkdir_p(args.dockerfile_dir)

        # Set up Docker helper
        self.dock = docker.from_env()
        self.dock_cli = APIClient()

        # # Login to Docker if uploading images
        # if args.upload_to_hub:
        #     if not args.hub_username:
        #         logger.info('> Error: please set --hub_username when uploading to Dockerhub.')
        #         exit(1)
        #     if not args.hub_repository:
        #         logger.info(
        #                 '> Error: please set --hub_repository when uploading to Dockerhub.')
        #         exit(1)
        #     if not args.hub_password:
        #         logger.info('> Error: please set --hub_password when uploading to Dockerhub.')
        #         exit(1)
        #     dock.login(
        #             username=args.hub_username,
        #             password=args.hub_password,
        #     )

        # Each tag has a name ('tag') and a definition consisting of the contents
        # of its Dockerfile, its build arg list, etc.
        self.failed_tags = []
        for tag, tag_defs in all_tags.items():
            for tag_def in tag_defs:
                logger.info('> Working on {}'.format(tag))

                if args.exclude_tags_matching and re.match(args.exclude_tags_matching, tag):
                    logger.info('>> Excluded due to match against "{}".'.format(
                        args.exclude_tags_matching))
                    continue

                if args.only_tags_matching and not re.match(args.only_tags_matching, tag):
                    logger.info('>> Excluded due to failure to match against "{}".'.format(
                        args.only_tags_matching))
                    continue

                # Write releases marked "is_dockerfiles" into the Dockerfile directory
                if args.construct_dockerfiles and tag_def['is_dockerfiles']:
                    path = os.path.join(args.dockerfile_dir,
                                        tag_def['dockerfile_subdirectory'],
                                        tag + '.Dockerfile')
                    logger.info('>> Writing {}...'.format(path))
                    if not args.dry_run:
                        mkdir_p(os.path.dirname(path))
                        with open(path, 'w') as f:
                            f.write(tag_def['dockerfile_contents'])

                # Don't build any images for dockerfile-only releases
                if not args.build_images:
                    continue

                # Generate a temporary Dockerfile to use to build, since docker-py
                # needs a filepath relative to the build context (i.e. the current
                # directory)
                dockerfile = os.path.join(args.dockerfile_dir, tag + '.temp.Dockerfile')
                if not args.dry_run:
                    with open(dockerfile, 'w') as f:
                        f.write(tag_def['dockerfile_contents'])
                logger.info('>> (Temporary) writing {}...'.format(dockerfile))

                repo_tag = '{}:{}'.format(args.repository, tag)
                logger.info('>> Building {} using build args:'.format(repo_tag))
                for arg, value in tag_def['cli_args'].items():
                    logger.info('>>> {}={}'.format(arg, value))

                # Note that we are NOT using cache_from, which appears to limit
                # available cache layers to those from explicitly specified layers. Many
                # of our layers are similar between local builds, so we want to use the
                # implied local build cache.
                tag_failed = False
                image, logs = None, []
                if not args.dry_run:
                    try:

                        if args.pull:
                            image = self.docker_pull(args.pull_image)
                        else:
                            image = self.docker_build(dockerfile, repo_tag, tag_def, args.debug)

                        # Run tests if requested, and dump output
                        # Could be improved by backgrounding, but would need better
                        # multiprocessing support to track failures properly.

                        # ROCM_EXTRA_PARAMS="--device=/dev/kfd --device=/dev/dri --group-add video"

                        if args.run and not args.stop:
                            self.docker_run(image, tag_def, extra_argv)

                        if args.deploy and not args.stop:
                            self.generate_stack_yml(tag_def)
                            self.docker_deploy(extra_argv, reload=args.reload)

                        if args.stop:
                            self.generate_stack_yml(tag_def)
                            self.docker_stop(extra_argv)

                        # if args.run_tests_path:
                        #     tag_failed = self.run_tests(image, repo_tag, tag, tag_def)

                    except docker.errors.BuildError as e:
                        logger.error('>> {} failed to build with message: "{}"'.format(
                            repo_tag, e.msg))
                        logger.error('>> Build logs follow:')
                        log_lines = [l.get('stream', '') for l in e.build_log]
                        logger.error(''.join(log_lines))
                        self.failed_tags.append(tag)
                        tag_failed = True
                        if args.stop_on_failure:
                            logger.error('>> ABORTING due to --stop_on_failure!')
                            exit(1)

                    except DockerError as e:
                        if hasattr(e, 'message'):
                            logger.error(e.message)
                        else:
                            logger.error(e)
                        sys.exit(1)

                    # Clean temporary dockerfiles if they were created earlier
                    if not args.keep_temp_dockerfiles:
                        if _e(dockerfile):
                            os.remove(dockerfile)

                # # Upload new images to DockerHub as long as they built + passed tests
                # if args.upload_to_hub:
                #     if not tag_def['upload_images']:
                #         continue
                #     if tag_failed:
                #         continue
                #
                #     logger.info('>> Uploading to {}:{}'.format(args.hub_repository, tag))
                #     if not args.dry_run:
                #         p = multiprocessing.Process(
                #                 target=upload_in_background,
                #                 args=(args.hub_repository, dock, image, tag))
                #         p.start()

        if self.failed_tags:
            logger.error(
                '> Some tags failed to build or failed testing, check scrollback for '
                'errors: {}'.format(','.join(self.failed_tags)))
            exit(1)

def tee_docker(generator, file=None, to_stdout=True, append=False, flush=True, debug=False):
    def output_line(line):
        if to_stdout:
            sys.stdout.write(line)
        f.write(line)
        if flush:
            f.flush()
    with ScopedLogFile(file=file, append=append) as f:
        last_dic = None
        for dic in generator:
            if debug:
                pprint.pprint(dic)
            if 'stream' in dic:
                line = dic['stream']
                output_line(line)
            else:
                # line = pprint.pformat(dic)
                # output_line(line)
                pass
            last_dic = dic
        return last_dic

def check_docker_response(response, path, repo_tag, cmd=None):
    if is_docker_error_response(response):
        raise DockerError(path, repo_tag, response, cmd=cmd)

def is_docker_error_response(response):
    return 'error' in response or (
            'message' in response and
            re.search(r'\berror\b', response['message'], flags=re.IGNORECASE)
    )

class ScopedLogFile:
    def __init__(self, file, append=False):
        self.file = file
        self.append = append

    def __enter__(self):
        if self._is_path:
            # if self.append:
            #         self.mode = 'ab'
            # else:
            #         self.mode = 'wb'

            if self.append:
                self.mode = 'a'
            else:
                self.mode = 'w'
            self.f = open(self.file, self.mode)
            return self.f
        else:
            # We can only append to a stream.
            self.f = self.file
            return self.f

    @property
    def _is_path(self):
        return type(self.file) == str

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.flush()
        if self._is_path:
            self.f.close()

def get_implicit_build_args():
    """
    --build-arg USER_ID=$(id -u ${USER})
    --build-arg GROUP_ID=$(id -u ${USER})
    --build-arg USER_NAME=${USER}
    :return:
    """
    # build_args = {
    #     "USER_ID": get_user_id(),
    #     "GROUP_ID": get_user_id(),
    #     "USER_NAME": get_username(),
    # }
    build_args = dict()

    build_args['RLSCOPE_USER'] = get_username()
    build_args['RLSCOPE_UID'] = get_user_id()
    build_args['RLSCOPE_GID'] = get_group_id()
    build_args['RLSCOPE_DIR'] = py_config.RLSCOPE_DIR
    # virtualenv directory within container.
    build_args['VIRTUALENV'] = "/home/{user}/venv".format(user=build_args['RLSCOPE_USER'])

    for k in build_args.keys():
        # Otherwise, when we do dock_cli.build we get:
        #     docker.errors.APIError: 400 Client Error: Bad Request ("error reading build args: json: cannot unmarshal number into Go value of type string")
        build_args[k] = str(build_args[k])

    return build_args

def get_implicit_run_args():
    run_args = {
        "RLSCOPE_DIR": py_config.RLSCOPE_DIR,
    }
    return run_args

RUN_ARGS_REQUIRED = [
    'RLSCOPE_DIR',
    # 'RLSCOPE_DRILL_DIR',
    # The root directory of a 'patched' TensorFlow checkout
    # 'TENSORFLOW_DIR',
    # The local path where we should output bazel objects (overrides $HOME/.cache/bazel)
    # 'BAZEL_BUILD_DIR',
]

def is_required_run_arg(var):
    return var in RUN_ARGS_REQUIRED

def setup_volume_dir(direc):
    os.makedirs(direc, exist_ok=True)
    subprocess.check_call("chown -R {user}:{user} {path}".format(
        user=get_username(),
        path=direc,
    ), shell=True)

def get_rlscope_volumes(args, run_args, extra_volumes):
    """
    host_dir -> container_dir

    --build-arg USER_ID=$(id -u ${USER})
    --build-arg GROUP_ID=$(id -u ${USER})
    --build-arg USER_NAME=${USER}
    :return:
    """
    volumes = dict()
    for arg in RUN_ARGS_REQUIRED:
        direc = run_args[arg]
        volumes[direc] = direc
    # Store bazel compilation files in bazel folder in repo root directory:
    #   $HOME/clone/rlscope/bazel -> $HOME/.cache/bazel
    host_bazel_dir = _j(py_config.RLSCOPE_DIR, 'bazel')
    cont_bazel_dir = _j('/home', get_username(), '.cache', 'bazel')
    assert host_bazel_dir not in volumes
    volumes[host_bazel_dir] = cont_bazel_dir
    setup_volume_dir(host_bazel_dir)

    shutil.chown(host_bazel_dir, get_username(), get_username())

    host_directories = set()
    if extra_volumes is None:
        import pdb; pdb.set_trace()
    host_directories.update(extra_volumes)
    host_directories.update(volumes.keys())
    for host_path in host_directories:
        if not os.path.exists(host_path):
            os.makedirs(host_path, exist_ok=True)
        subprocess.check_call("chown -R {user}:{user} {path}".format(
            user=get_username(),
            path=host_path,
        ), shell=True)

    for i, direc in enumerate(extra_volumes):
        env_name = 'CMDLINE_VOLUME_{i}'.format(i=i)
        assert env_name not in volumes
        # If this fails you're probably using --volume wrong; see usage info.
        assert ':' not in direc
        # volumes[env_name] = direc
        volumes[direc] = direc

    return volumes

def get_rlscope_ports(run_args, extra_ports):
    return extra_ports

class DockerError(Exception):
    """
    :param path
        Path to Dockerfile
    :param repo_tag
        Identifier for build
    """
    def __init__(self, path, repo_tag, response, cmd=None):
        self.path = path
        self.repo_tag = repo_tag
        self.response = response
        self.cmd = cmd
        message = self.construct_message()
        super().__init__(message)

    def construct_message(self):
        # This format of response is very undocumented...
        # Print it out in case we miss something.
        pprint.pprint({
            'response': self.response,
        })

        lines = []
        lines.append(
            "Failed to build dockerfile {path} with repo_tag={repo_tag}.".format(
                path=_a(self.path),
                repo_tag=self.repo_tag))
        if 'errorDetail' in self.response and 'code' in self.response['errorDetail']:
            lines.append(
                ind("Exit status: {code}".format(
                    code=self.response['errorDetail']['code'],
                ), indent=1)
            )

        if self.cmd is not None:
            lines.append(get_cmd_string(self.cmd))

        lines.append("> Failed with:")

        if 'message' in self.response:
            lines.append(ind(self.response['message'], indent=1))

        if 'errorDetail' in self.response and 'message' in self.response['errorDetail']:
            # lines.append("    Error message:")
            lines.append(
                "{err}".format(
                    err=ind(
                        self.response['errorDetail']['message'],
                        indent=2),
                )
            )

        message = ''.join("{line}\n".format(line=line) for line in lines)
        return message

def get_docker_cmdline(docker_command='run', extra_argv=[], **kwargs):
    """
    kwargs = kwargs accepted by docker-py
    See: https://docker-py.readthedocs.io/en/stable/containers.html

    :param docker_command:
    :param kwargs:
    :param extra_argv:
        Append these to the end of the generated docker command-line.
    :return:
    """
    print("> docker.cmdline.kwargs")
    pprint.pprint({'docker.cmdline.kwargs':kwargs})

    if docker_command == 'run':
        cmd = ["docker", docker_command, "-i", "-t"]
    else:
        # e.g. build
        cmd = ["docker", docker_command]

    def as_cmd_opt(name):
        cmd_opt = name
        if len(name) == 1:
            # Short option
            cmd_opt = '-{cmd_opt}'.format(cmd_opt=cmd_opt)
            return cmd_opt
        cmd_opt = re.sub('_', '-', cmd_opt)
        cmd_opt = "--{cmd_opt}".format(cmd_opt=cmd_opt)
        return cmd_opt

    def add_opt(name):
        if name not in kwargs or kwargs[name] is None:
            return
        value = str(kwargs[name])
        cmd_opt = as_cmd_opt(name)
        cmd.extend([cmd_opt, value])

    def add_opt_value(name, value, opt_type=str):
        cmd_opt = as_cmd_opt(name)
        if opt_type == bool and value:
            cmd.extend([cmd_opt])
        else:
            cmd.extend([cmd_opt, str(value)])

    def add_opt_from(opt_name, from_name, opt_type=str):
        if from_name not in kwargs:
            return
        add_opt_value(opt_name, kwargs[from_name], opt_type)

    def add_pos_opt(name):
        value = kwargs[name]
        cmd.append(value)

    def add_pos_opt_value(value):
        cmd.append(value)

    add_opt('runtime')
    add_opt('name')

    add_opt_from('workdir', 'working_dir')

    add_opt_from('rm', 'remove', opt_type=bool)

    def add_opt_list(opt_name, from_name):
        if from_name in kwargs:
            for value in kwargs[from_name]:
                add_opt_value(opt_name, value)

    def add_opt_dict(opt_name, from_name):
        if from_name in kwargs:
            for key, value in kwargs[from_name].items():
                opt_value = "{key}={value}".format(key=key, value=value)
                add_opt_value(opt_name, opt_value)

    # if docker_command == 'build':
    add_opt_dict('build-arg', 'buildargs')

    add_opt_from('file', 'dockerfile')

    add_opt_from('tag', 'tag')

    add_opt_list('group-add', 'group_add')

    add_opt_list('device', 'devices')

    add_opt_list('cap-add', 'cap_add')

    add_opt_list('security-opt', 'security_opt')

    if 'volumes' in kwargs:
        for host_dir, container_dir in kwargs['volumes'].items():
            add_opt_value('volume', "{host_dir}:{container_dir}".format(
                host_dir=host_dir, container_dir=container_dir,
            ))

    if 'environment' in kwargs:
        environment = kwargs['environment']
        for env_var, env_value in environment.items():
            value = "{var}={val}".format(var=env_var, val=env_value)
            # e.g.
            # -e NVIDIA_VISIBLE_DEVICES="0"
            add_opt_value('e', value)

    if 'log_config' in kwargs:
        log_config = kwargs['log_config']
        if 'type' in log_config:
            add_opt_value('log-driver', log_config['type'])

    # NOTE: We MUST add "docker run" options BEFORE the image/tag.
    cmd.extend(extra_argv)

    if 'image' in kwargs:
        image = kwargs['image']
        if isinstance(image, Image) and len(image.tags) > 0:
            tag = image.tags[0]
        else:
            tag = image

        add_pos_opt_value(tag)

    if docker_command == 'build' and 'path' in kwargs:
        add_pos_opt_value(kwargs['path'])

    return cmd

class StackYMLGenerator:
    """
    Generate stack.yml.
    """
    def __init__(self, project_name):
        """
        :param cmd
            run_docker.py command.
        """
        # Extra volumes added programmatically (e.g., tmpfs volume for nvidia mps IPC)
        self._extra_volumes = dict()
        self.indent_str = 4*' '
        self.project_name = project_name

    def _mps_daemon_container_name(self):
        return "{project}_mps-daemon_1".format(project=self.project_name)

    def get_template(self, use_mps):
        if use_mps:
            # NOTE: I think sample code uses this to run 3 instances of the nbody process.
            # scale: 3

            # volumes:
            # - nvidia_mps:/tmp/nvidia-mps
            mps_lines = textwrap.dedent("""\
            
            #
            # MPS: launch container AFTER mps daemon is running to ensure all GPU apps use the daemon.
            #
            depends_on:
              # - rlscope-mps-daemon
              - mps-daemon
            # Share the IPC namespace with the MPS control daemon.
            # ipc: container:rlscope-mps-daemon
            # HACK: as of docker-compose version 3.7, IPC doesn't (yet) support services.
            # Instead, we must provide a container name and use "ipc: container:<container_name>".
            # https://docs.docker.com/engine/reference/run/#ipc-settings---ipc
            # Eventually this feature will be upstreamed though (it's been accepted):
            # https://github.com/docker/compose/pull/7417
            # ipc: service:mps-daemon
            ipc: container:{RLSCOPE_MPS_DAEMON_CONTAINER_NAME}
            """)
            self._extra_volumes['nvidia_mps'] = '/tmp/nvidia-mps'
        else:
            mps_lines = ""
        template = textwrap.dedent("""\
        # DO NOT MODIFY!
        # Automatically generated using run_docker.py with the following command/working-directory:
        #     > CMD: {assembler_cmd}
        #       PWD: {PWD}

        version: '3.7'

        services:

            {RLSCOPE_BASH_SERVICE_NAME}:
                {bash_template}
                {mps_lines}
                
            {mps_template}
            
        {mps_volumes_template}
        """).rstrip()
        # template.format(
        template = template.format_map(FormatDict(
            bash_template=self._indent(self._template_bash_body(), indent=2),
            mps_template=self._indent(self._template_mps_services(), indent=1) if use_mps else "",
            mps_volumes_template=self._indent(self._template_mps_volumes(), indent=0) if use_mps else "",
            mps_lines=self._indent(mps_lines, indent=2),
        ))
        # )
        return template

    def _template_mps_services(self):
        """
        Based on nvidia docker sample:
        https://gitlab.com/nvidia/container-images/samples/-/blob/7a94cb4b1784ede192c18d6268205f4639f13176/mps/docker-compose.yml
        Documentation:
        https://github.com/NVIDIA/nvidia-docker/wiki/MPS-(EXPERIMENTAL)

        """
        template = textwrap.dedent("""\
        # From the MPS documentation:
        # When using MPS it is recommended to use EXCLUSIVE_PROCESS mode to ensure
        # that only a single MPS server is using the GPU, which provides additional
        # insurance that the MPS server is the single point of arbitration between
        # all CUDA processes for that GPU.
        # rlscope-exclusive-mode:
        # exclusive-mode:
        #     image: debian:stretch-slim
        #     command: nvidia-smi -c EXCLUSIVE_PROCESS
        #     # https://github.com/nvidia/nvidia-container-runtime#environment-variables-oci-spec
        #     # NVIDIA_VISIBLE_DEVICES will default to "all" (from file .env), unless
        #     # the variable is exported on the command-line.
        #     environment:
        #       - "NVIDIA_VISIBLE_DEVICES={NVIDIA_VISIBLE_DEVICES}"
        #       - "NVIDIA_DRIVER_CAPABILITIES=utility"
        #     # runtime: nvidia
        #     network_mode: none
        #     # CAP_SYS_ADMIN is required to modify the compute mode of the GPUs.
        #     # This capability is granted only to this ephemeral container, not to
        #     # the MPS daemon.
        #     cap_add:
        #       - SYS_ADMIN

        # rlscope-mps-daemon:
        mps-daemon:
            image: nvidia/mps
            container_name: {RLSCOPE_MPS_DAEMON_CONTAINER_NAME}
            restart: on-failure
            # The "depends_on" only guarantees an ordering for container *start*:
            # https://docs.docker.com/compose/startup-order/
            # There is a potential race condition: the MPS or CUDA containers might
            # start before the exclusive compute mode is set. If this happens, one
            # of the CUDA application will fail to initialize since MPS will not be
            # the single point of arbitration for GPU access.
            # depends_on:
            #   # - rlscope-exclusive-mode
            #   - exclusive-mode
            environment:
              - "NVIDIA_VISIBLE_DEVICES={NVIDIA_VISIBLE_DEVICES}"
              - "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
            # runtime: nvidia
            init: true
            network_mode: none
            ulimits:
              memlock:
                soft: -1
                hard: -1
            # The MPS control daemon, the MPS server, and the associated MPS
            # clients communicate with each other via named pipes and UNIX domain
            # sockets. The default directory for these pipes and sockets is
            # /tmp/nvidia-mps.
            # Here we share a tmpfs between the applications and the MPS daemon.
            volumes:
              - nvidia_mps:/tmp/nvidia-mps
                
        """).rstrip()
        return template

    def _indent(self, txt, indent):
        return textwrap.indent(txt, indent*self.indent_str).lstrip()

    def _template_mps_volumes(self):
        """
        Based on nvidia docker sample:
        https://gitlab.com/nvidia/container-images/samples/-/blob/7a94cb4b1784ede192c18d6268205f4639f13176/mps/docker-compose.yml
        Documentation:
        https://github.com/NVIDIA/nvidia-docker/wiki/MPS-(EXPERIMENTAL)
        """
        template = textwrap.dedent("""\
        volumes:
            nvidia_mps:
                driver_opts:
                    type: tmpfs
                    device: tmpfs
        """).rstrip()
        return template

    def _template_bash_body(self):
        template = textwrap.dedent("""\
        #
        # "Bash" development environment.
        #
        image: {rlscope_image}
        
        user: "{RLSCOPE_USER}"
        
        ports:
            # Expose port that the rlscope-drill web server runs on.
            - {rlscope_drill_port}:{DEFAULT_RLSCOPE_DRILL_PORT}
            {port_list}
        
        volumes:
        {volume_list}
        
        environment:
        {env_list}
        
        # docker run --cap-add=SYS_ADMIN
        # We need this, otherwise libcupti PC sampling fails with CUPTI_ERROR_INSUFFICIENT_PRIVILEGES during cuptiActivityConfigurePCSampling.
        cap_add:
          - SYS_ADMIN
          - SYS_PTRACE
        security_opt:
          # Allow us to debug with gdb inside container:
          # https://stackoverflow.com/questions/35860527/warning-error-disabling-address-space-randomization-operation-not-permitted
          - seccomp:unconfined
        
        logging:
            driver: journald
        stdin_open: true
        tty: true
        entrypoint: /bin/bash
        """).rstrip()
        return template

    def doesnt_work(self):
        # I tried added this to allow doing "gdb -p",
        # but it doesn't work with "docker stack deploy",
        # since it ignores these options.
        # But, these options DO work with "docker run".
        textwrap.dedent("""
        bash:
            cap_add:
            # Allow attaching to processes using "gdb -p <PID>" inside the container.
            - SYS_PTRACE

            security_opt:
            - seccomp = unconfined
        """).rstrip()

    def generate(self,
                 assembler_cmd, env,
                 volumes, ports,
                 rlscope_drill_port=DEFAULT_RLSCOPE_DRILL_PORT,
                 rlscope_image=LOCAL_RLSCOPE_IMAGE_TAG,
                 use_mps=False):

        # +1 for "service: ..."
        # +1 for "bash: ..."
        bash_indent = 2

        template = self.get_template(use_mps=use_mps)

        return template.format(
            env_list=self.env_list(env, indent=bash_indent),
            # build
            #   args:
            #     - ...
            # yml_build_args_list=self.build_args_list(build_args, indent=bash_indent + 2),
            volume_list=self.volume_list(volumes, indent=bash_indent),
            port_list=self.port_list(ports, indent=bash_indent + 1),
            USER=get_username(),
            assembler_cmd=' '.join(assembler_cmd),
            PWD=os.getcwd(),
            DEFAULT_RLSCOPE_DRILL_PORT=DEFAULT_RLSCOPE_DRILL_PORT,
            rlscope_drill_port=rlscope_drill_port,
            rlscope_image=rlscope_image,
            RLSCOPE_MPS_DAEMON_CONTAINER_NAME=self._mps_daemon_container_name(),
            RLSCOPE_BASH_SERVICE_NAME=RLSCOPE_BASH_SERVICE_NAME,
            RLSCOPE_UID=get_user_id(),
            RLSCOPE_USER=get_username(),
            NVIDIA_VISIBLE_DEVICES=','.join([str(dev) for dev in NVIDIA_VISIBLE_DEVICES]),
        )

    def _yml_list(self, values, indent):
        yml_lines = []
        if values is None:
            values = []
        for value in values:
            yml_lines.append("- {value}".format(
                value=value))
        # NOTE: lstrip() to remove indent from first line, since it's already in the template.
        return textwrap.indent('\n'.join(yml_lines), indent*self.indent_str).lstrip()

    def _yml_dict_as_list(self, values : dict, sep, indent):
        envs = dict_as_env_list(values, sep)
        return self._yml_list(envs, indent)

    def volume_list(self, volumes : dict, indent):
        all_volumes = dict(volumes)
        for host_volume in self._extra_volumes:
            # Don't overwrite volumes
            assert host_volume not in all_volumes
        all_volumes.update(self._extra_volumes)
        return self._yml_dict_as_list(all_volumes, sep=':', indent=indent)

    def port_list(self, ports : list, indent):
        return self._yml_list(ports, indent=indent)

    def env_list(self, envs : dict, indent):
        return self._yml_dict_as_list(envs, sep='=', indent=indent)

    def build_args_list(self, build_args : dict, indent):
        return self._yml_dict_as_list(build_args, sep='=', indent=indent)

def dict_as_env_list(values : dict, sep='='):
    envs = ["{var}{sep}{value}".format(var=var, value=values[var], sep=sep)
            for var in sorted(values.keys())]
    return envs

def get_cmd_string(cmd, show_cwd=False):
    if type(cmd) == list:
        cmd_str = ' '.join(cmd)
    else:
        cmd_str = cmd
    if show_cwd:
        return textwrap.dedent("""\
        > CMD:
          $ {cmd}
          PWD={pwd}
        """.format(
            cmd=cmd_str,
            pwd=os.getcwd())).rstrip()
    else:
        return textwrap.dedent("""\
        > CMD:
          $ {cmd}
        """.format(
            cmd=cmd_str)).rstrip()
    # return ("> CMD:\n"
    #         "  $ {cmd}").format(
    #     cmd=cmd_str)

def ind(string, indent=1):
    return textwrap.indent(string, prefix='  '*indent)

def get_username():
    return pwd.getpwuid(os.getuid())[0]

def get_user_id():
    return os.getuid()

def get_group_id():
    return os.getgid()

def nvidia_set_compute_mode(compute_mode, device_ids):
    assert compute_mode in {'EXCLUSIVE_PROCESS', 'DEFAULT', 'PROHIBITED'}
    assert len(device_ids) > 0
    device_id_str = ','.join([str(dev) for dev in device_ids])
    cmd = [
        'sudo',
        'nvidia-smi',
        f"--compute-mode={compute_mode}",
        f"--id={device_id_str}",
    ]
    logger.info(get_cmd_string(cmd))
    subprocess.check_call(cmd)

class DeployError(RuntimeError):
    pass

if __name__ == '__main__':
    main()
