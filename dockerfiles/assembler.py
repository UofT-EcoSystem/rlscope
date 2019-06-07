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

# from absl import app
# from absl import flags
import cerberus
import docker
from docker.models.images import Image
import yaml
# https://docker-py.readthedocs.io/en/stable/api.html#docker.api.build.BuildApiMixin.build
from io import BytesIO
from docker import APIClient
from pathlib import Path

from iml_profiler import py_config

def get_username():
    return pwd.getpwuid(os.getuid())[0]

def get_user_id():
    return os.getuid()

HOME = str(Path.home())
DEFAULT_POSTGRES_PORT = 5432
DEFAULT_IML_DRILL_PORT = 8129
# Default storage location for postgres database
# (postgres is used for loading raw trace-data and analyzing it).
DEFAULT_POSTGRES_PGDATA_DIR = _j(HOME, 'iml', 'pgdata')
# The tag used for a locally built "bash" IML dev environment
LOCAL_IML_IMAGE_TAG = 'tensorflow:devel-iml-gpu-cuda'
DEFAULT_REMOTE_IML_IMAGE_TAG = 'jagleeso/iml:1.0.0'

# How long should we wait for /bin/bash (iml_bash)
# to appear after running "docker stack deploy"?
DOCKER_DEPLOY_TIMEOUT_SEC = 60

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

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, flush=True, **kwargs)


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
                eprint(('> Error: {arg} is not a valid slice_set, and also isn\'t an arg '
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


def assemble_tags(spec, cli_args, cli_run_args, enabled_releases, all_partials):
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

    for name, release in spec['releases'].items():
        for tag_spec in release['tag_specs']:
            if enabled_releases and name not in enabled_releases:
                eprint('> Skipping release {}'.format(name))
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
    return '\n'.join([header] + [all_partials[u] for u in used_partials])


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
            if '.partial.Dockerfile' not in fullpath:
                eprint(('> Probably not a problem: skipping {}, which is not a '
                        'partial.').format(fullpath))
                continue
            # partial_dir/foo/bar.partial.Dockerfile -> foo/bar
            simple_name = fullpath[len(partial_path) + 1:-len('.partial.dockerfile')]
            with open(fullpath, 'r') as f:
                partial_contents = f.read()
            partials[simple_name] = partial_contents
    return partials

def get_build_logfile(repo_tag):
    return "{repo_tag}.build.log.txt".format(
        repo_tag=repo_tag)

"""
If you use any of these in your --arg or in arg specifications in spec.yml, 
they will get replaced with corresponding ARG_VALUES dict values.
"""
ARG_VALUES = {
    'IML_DIR': py_config.IML_DIR,
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
    #     eprint("> You must provide {env}=[ {desc} ]".format(
    #         env=env_var,
    #         desc=desc))
    #     sys.exit(1)

    env = dict()
    env.update(get_implicit_run_args())
    if 'run_args' in tag_def:
        env = update_args_dict(env, tag_def['run_args'], keep_original=True)

    for var, value in env.items():
        if value == '':
            eprint(("> ERROR: you must provide a value for --run_arg {var}=<VALUE> "
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
        argv = ["assembler.py" "extra_argument[0]", ...]
    :return:
    """
    return argv[1:]

def main():
    parser = argparse.ArgumentParser(description=__doc__)

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
                        Pull pre-built IML dev environment image from DockerHub, 
                        instead of building locally.
                        
                        See --pull-image for specifying the image to pull.
                        """))

    parser.add_argument('--pull-image', default=DEFAULT_REMOTE_IML_IMAGE_TAG,
                        help=textwrap.dedent("""
                        IML dev environment image to pull from DockerHub
                        using "docker pull <pull_img>"
                        """))

    parser.add_argument('--deploy-iml-drill-port',
                        default=DEFAULT_IML_DRILL_PORT,
                        type=int,
                        help=('What port to run iml-drill web server on '
                              '(when running "docker stack deploy -c stack.yml iml")'))

    parser.add_argument('--deploy-postgres-port',
                        default=DEFAULT_POSTGRES_PORT,
                        type=int,
                        help=('What port to run postgres on '
                              '(when running "docker stack deploy -c stack.yml iml")'))

    parser.add_argument('--deploy-postgres-pgdata-dir',
                        default=DEFAULT_POSTGRES_PGDATA_DIR,
                        help=('Default storage location for postgres database '
                              '(postgres is used for loading raw trace-data and analyzing it).'))

    parser.add_argument(
        '--repository', default='tensorflow',
        help='Tag local images as {repository}:tag (in addition to the '
             'hub_repository, if uploading to hub)')

    parser.add_argument(
        '--volume',
        action='append',
        help=textwrap.dedent("""\
        Translates into docker --volume option. 
        We mount the path at the same path as it is in the host.
        i.e. 
        # assembler.py option:
        --volume /one/two
        #
        # becomes
        #
        # `docker run` option:
        --volume /one/two:/one/two
        """).rstrip())

    parser.add_argument(
        '--env',
        '-e',
        action='append',
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
                START: dockerfiles/partials/ubuntu/devel-nvidia.partial.Dockerfile
                RUN ...
                RUN ...
                ...
                END: dockerfiles/partials/ubuntu/devel-nvidia.partial.Dockerfile
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
        Deploy the IML development environment using 
        "docker stack deploy -c stack.yml iml".
        
        In particular:
        - iml: 
          The python library that collects profiling info.
          
        - tensorflow.patched: 
          tensorflow patched with C++ modifications to support iml tracing.
          
        - iml-drill: 
          The web server for visualizing collected data.
          
        - postgres: 
          Used by iml for storing/analyzing trace-data.
          
        The development environment takes care of installing dependencies needed 
        for building tensorflow.patched.
        
        Scripts are provided on your $PATH; some common ones:
        - make_tflow.sh:
          Build tensorflow.patched from source.
          
        - run_baselines.sh:
          Run Atari pong benchmark from baselines repo.
          
        For more scripts:
        - See <REPO>/dockerfiles/sh in the repo for all the scripts, OR
        - See /home/{USER}/dockerfiles/sh in the deployed container
        """.format(USER=get_username())))

    parser.add_argument(
        '--run',
        action='store_true',
        help='Run built images; use --deploy if you want to deploy the whole IML development environment')

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
        '--release', '-r', default=[], action='append',
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
             'use with "docker stack deploy -c stack.yml iml"',
    )

    argv = list(sys.argv)
    args, extra_argv = parser.parse_known_args()

    if args.deploy and args.run:
        parser.error(
            "Provide either --deploy or --run.  "
            "Use --deploy to deploy the IML development environment (probably what you want)")

    assembler = Assembler(parser, argv, args, extra_argv)
    assembler.run()

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
            tag_spec = yaml.load(spec_file)

        for arg in args.arg:
            key, value = parse_build_arg(arg)
            validate_interpolate_arg(value)

        for run_arg in args.run_arg:
            key, value = parse_build_arg(run_arg)
            validate_interpolate_arg(value)

        # Get existing partial contents
        partials = gather_existing_partials(args.partial_dir)

        # Abort if spec.yaml is invalid
        schema = yaml.load(SCHEMA_TEXT)
        v = TfDockerTagValidator(schema, partials=partials)
        if not v.validate(tag_spec):
            eprint('> Error: {} is an invalid spec! The errors are:'.format(
                args.spec_file))
            eprint(yaml.dump(v.errors, indent=2))
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

    def docker_build(self, dockerfile, repo_tag, tag_def):
        args = self.args
        build_kwargs = dict(
            timeout=args.hub_timeout,
            path='.',
            dockerfile=dockerfile,
            buildargs=tag_def['cli_args'],
            tag=repo_tag,
        )
        if args.debug:
            print("> dock.images.build")
            print(textwrap.indent(pprint.pformat(build_kwargs), prefix="    "))

        build_output_generator = self.dock_cli.build(decode=True, **build_kwargs)
        response = tee_docker(
            build_output_generator,
            file=get_build_logfile(repo_tag))
        check_docker_response(response, dockerfile, repo_tag)

        image = self.dock.images.get(repo_tag)

        return image

    def docker_run(self, image, tag_def, extra_argv):
        # Run the container.
        args = self.args
        docker_run_env = get_docker_run_env(tag_def, args.env)
        iml_volumes = get_iml_volumes(docker_run_env, args.volume)
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
            volumes=iml_volumes,
            # volumes={
            #     args.run_tests_path: {
            #         'bind': '/tests',
            #         'mode': 'ro'
            #     }
            # },
            remove=True,
            cap_add=['SYS_PTRACE'],
            security_opt=['seccomp=unconfined'],
            runtime=runtime,
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
        eprint(get_cmd_string(cmd))
        subprocess.run(cmd, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)
        # Q: Save output?

    # def _is_iml_deployed(self):
    #     self.docker_ps_iml_bash() is not

    def docker_ps_iml_bash(self):
        # logging.info("(1) run docker ps")
        p = subprocess.run("docker ps | grep iml_bash",
                           shell=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           # check=True,
                           )
        # I've seen this fail with returncode=1, but NO error message...
        # I feel like its a bug in docker...
        if p.returncode != 0:
            return None
        # logging.info("(2) run docker ps")
        # p = subprocess.run("docker ps | grep iml_bash",
        #                    shell=True,
        #                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        #                    check=True,
        #                    )
        lines = p.stdout.decode('utf-8').splitlines()
        for line in lines:
            if re.search(r'iml_bash', line):
                fields = re.split(r'\s+', line)
                container_id = fields[0]
                return container_id
        return None

    def _wait_for_bash(self):
        timeout_sec = time.time() + DOCKER_DEPLOY_TIMEOUT_SEC
        now_sec = time.time()
        while now_sec < timeout_sec:

            # services = self.dock_cli.services(filters={'name':'iml_bash'})
            # if len(services) > 0:
            #     info = self.dock_cli.inspect_service('iml_bash')
            #     return info

            containers = self.dock.containers.list(filters={'name': 'iml_bash'})
            if len(containers) > 0:
                # containers.name
                assert len(containers) == 1
                return containers[0]

            # container_id = self.docker_ps_iml_bash()
            # if container_id is not None:
            #     return container_id

            time.sleep(0.5)
            now_sec = time.time()

        print(("> FAILURE: Waited for /bin/bash (iml_bash) container to appear, but it didn't after {sec} seconds"
               "  run 'docker ps' to see what's actually running").format(
            sec=DOCKER_DEPLOY_TIMEOUT_SEC))

        sys.exit(1)

    def _wait_for_iml_to_stop(self):
        """
        Wait until all services/processes belonging to the iml dev environment have stopped.

        Currently 'docker stack rm iml' returns immediately even though the container hasn't terminated yet,
        leading to race conditions when re-creating iml (causes errors when creating iml_default network).

        :param self:
        :return:
        """
        while True:
            iml_services = self.dock_cli.services(filters={'name': 'iml'})
            pprint.pprint({'iml_services': iml_services})
            if len(iml_services) == 0:
                return
            time.sleep(0.1)

    def docker_stack_rm(self):
        services = self.dock.services.list(filters={'name': 'iml'})
        if len(services) == 0:
            # IML dev environment not running yet.
            return
        print("> Detected old IML dev environment; removing it first")
        for srv in services:
            srv.remove()
        after = self.dock.services.list(filters={'name': 'iml'})
        assert len(after) == 0

        # IMPORTANT: apparently, removing the services above does NOT immediately remove the containers.
        # So, instead we busy wait until they dissapear.
        # We need this to avoid errors when calling 'docker stack deploy'.
        while True:
            containers = self.dock.containers.list(filters={'name': 'iml'})
            if len(containers) == 0:
                break
            time.sleep(0.1)

    def docker_deploy(self, extra_argv):
        """
        $ docker stack deploy {extra_args} --compose-file stack.yml iml
        $ docker attach <iml_bash /bin/bash container>

        :param extra_argv:
            Extra arguments to pass to "stack deploy"
        :return:
        """

        # Terminate/remove an existing deployment if it exists first.
        self.docker_stack_rm()

        cmd = ['docker', 'stack', 'deploy']
        cmd.extend(extra_argv)
        cmd.extend([
            '--compose-file', 'stack.yml',
            # Name of the created stack.
            'iml',
        ])
        eprint(get_cmd_string(cmd))
        subprocess.check_call(cmd, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)
        iml_bash_container = self._wait_for_bash()

        ps_cmd = ['docker', 'ps']
        eprint(get_cmd_string(ps_cmd))
        subprocess.check_call(ps_cmd, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)
        print("> Deployed IML development environment")
        print("> Attaching to /bin/bash in the dev environment:")
        # Login to existing container using a new /bin/bash shell so we are greeted with the _iml_banner
        exec_cmd = ['docker', 'exec', '-i', '-t', iml_bash_container.name, '/bin/bash']
        eprint(get_cmd_string(exec_cmd))
        subprocess.run(exec_cmd, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)

    def docker_pull(self, pull_img):
        """
        $ docker pull <pull_img>
        """
        cmd = ['docker', 'pull', pull_img]
        eprint(get_cmd_string(cmd))
        subprocess.check_call(cmd, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)
        image = self.dock.images.get(pull_img)
        return image

    def run_tests(self, image, repo_tag, tag, tag_def):
        tag_failed = False
        args = self.args
        docker_run_env = get_docker_run_env(tag_def, args.env)
        if not tag_def['tests']:
            eprint('>>> No tests to run.')
        for test in tag_def['tests']:
            eprint('>> Testing {}...'.format(test))

            runtime = get_docker_runtime(tag_def)

            test_kwargs = dict(
                image=image,
                command=_j(docker_run_env['IML_TEST_SH'], test),
                # command='/tests/' + test,
                working_dir='/',
                log_config={'type': 'journald'},
                detach=True,
                stderr=True,
                stdout=True,
                environment=docker_run_env,
                volumes={
                    # args.run_tests_path: {
                    docker_run_env['IML_TEST_SH']: {
                        # 'bind': '/tests',
                        'bind': docker_run_env['IML_TEST_SH'],
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
                eprint('>>> Output stdout:')
                eprint(out.decode('utf-8'))
            else:
                eprint('>>> No test standard out.')
            if err:
                eprint('>>> Output stderr:')
                eprint(out.decode('utf-8'))
            else:
                eprint('>>> No test standard err.')
            if code != 0:
                eprint('>> {} failed tests with status: "{}"'.format(
                    repo_tag, code))
                self.failed_tags.append(tag)
                tag_failed = True
                if args.stop_on_failure:
                    eprint('>> ABORTING due to --stop_on_failure!')
                    exit(1)
            else:
                eprint('>> Tests look good!')

        return tag_failed

    def generate_stack_yml(self, tag_def):
        args = self.args
        generator = StackYMLGenerator()

        docker_run_env = get_docker_run_env(tag_def, args.env)
        iml_volumes = get_iml_volumes(docker_run_env, args.volume)

        if args.pull:
            iml_image = args.pull_image
        else:
            iml_image = LOCAL_IML_IMAGE_TAG

        yml = generator.generate(
            assembler_cmd=self.argv,
            env=docker_run_env,
            volumes=iml_volumes,
            iml_drill_port=args.deploy_iml_drill_port,
            postgres_pgdata_dir=args.deploy_postgres_pgdata_dir,
            postgres_port=args.deploy_postgres_port,
            iml_image=iml_image,
        )
        print("> Write 'docker stack deploy' stack.yml file to {path}".format(path=args.output_stack_yml))
        with open(args.output_stack_yml, 'w') as f:
            f.write(yml)

    def run(self):
        parser = self.parser
        argv = self.argv
        args = self.args
        extra_argv = self.extra_argv

        if not _e('./spec.yml'):
            print("ERROR: You must execute assembly.py from dockerfiles/assembler.py")
            sys.exit(1)

        # In order to copy $ROOT/requirements.txt into the container,
        # it can't be in an upper directory (i.e. ../.....).
        shutil.copy(
            src=_j(py_config.ROOT, 'requirements.txt'),
            dst='./requirements.txt')

        all_tags = self.read_spec_yml()

        # Empty Dockerfile directory if building new Dockerfiles
        if args.construct_dockerfiles:
            eprint('> Emptying Dockerfile dir "{}"'.format(args.dockerfile_dir))
            shutil.rmtree(args.dockerfile_dir, ignore_errors=True)
            mkdir_p(args.dockerfile_dir)

        # Set up Docker helper
        self.dock = docker.from_env()
        self.dock_cli = APIClient()

        # # Login to Docker if uploading images
        # if args.upload_to_hub:
        #     if not args.hub_username:
        #         eprint('> Error: please set --hub_username when uploading to Dockerhub.')
        #         exit(1)
        #     if not args.hub_repository:
        #         eprint(
        #                 '> Error: please set --hub_repository when uploading to Dockerhub.')
        #         exit(1)
        #     if not args.hub_password:
        #         eprint('> Error: please set --hub_password when uploading to Dockerhub.')
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
                eprint('> Working on {}'.format(tag))

                if args.exclude_tags_matching and re.match(args.exclude_tags_matching, tag):
                    eprint('>> Excluded due to match against "{}".'.format(
                        args.exclude_tags_matching))
                    continue

                if args.only_tags_matching and not re.match(args.only_tags_matching, tag):
                    eprint('>> Excluded due to failure to match against "{}".'.format(
                        args.only_tags_matching))
                    continue

                # Write releases marked "is_dockerfiles" into the Dockerfile directory
                if args.construct_dockerfiles and tag_def['is_dockerfiles']:
                    path = os.path.join(args.dockerfile_dir,
                                        tag_def['dockerfile_subdirectory'],
                                        tag + '.Dockerfile')
                    eprint('>> Writing {}...'.format(path))
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
                eprint('>> (Temporary) writing {}...'.format(dockerfile))

                repo_tag = '{}:{}'.format(args.repository, tag)
                eprint('>> Building {} using build args:'.format(repo_tag))
                for arg, value in tag_def['cli_args'].items():
                    eprint('>>> {}={}'.format(arg, value))

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
                            image = self.docker_build(dockerfile, repo_tag, tag_def)

                        # Run tests if requested, and dump output
                        # Could be improved by backgrounding, but would need better
                        # multiprocessing support to track failures properly.

                        # ROCM_EXTRA_PARAMS="--device=/dev/kfd --device=/dev/dri --group-add video"

                        if args.run:
                            self.docker_run(image, tag_def, extra_argv)

                        if args.deploy:
                            self.generate_stack_yml(tag_def)
                            self.docker_deploy(extra_argv)

                        # if args.run_tests_path:
                        #     tag_failed = self.run_tests(image, repo_tag, tag, tag_def)

                    except docker.errors.BuildError as e:
                        eprint('>> {} failed to build with message: "{}"'.format(
                            repo_tag, e.msg))
                        eprint('>> Build logs follow:')
                        log_lines = [l.get('stream', '') for l in e.build_log]
                        eprint(''.join(log_lines))
                        self.failed_tags.append(tag)
                        tag_failed = True
                        if args.stop_on_failure:
                            eprint('>> ABORTING due to --stop_on_failure!')
                            exit(1)

                    # Clean temporary dockerfiles if they were created earlier
                    if not args.keep_temp_dockerfiles:
                        os.remove(dockerfile)

                # # Upload new images to DockerHub as long as they built + passed tests
                # if args.upload_to_hub:
                #     if not tag_def['upload_images']:
                #         continue
                #     if tag_failed:
                #         continue
                #
                #     eprint('>> Uploading to {}:{}'.format(args.hub_repository, tag))
                #     if not args.dry_run:
                #         p = multiprocessing.Process(
                #                 target=upload_in_background,
                #                 args=(args.hub_repository, dock, image, tag))
                #         p.start()

        if self.failed_tags:
            eprint(
                '> Some tags failed to build or failed testing, check scrollback for '
                'errors: {}'.format(','.join(self.failed_tags)))
            exit(1)

def tee_docker(generator, file=None, to_stdout=True, append=False, flush=True):
    def output_line(line):
        if to_stdout:
            sys.stdout.write(line)
        f.write(line)
        if flush:
            f.flush()
    with ScopedLogFile(file=file, append=append) as f:
        last_dic = None
        for dic in generator:
            if 'stream' in dic:
                line = dic['stream']
                output_line(line)
            else:
                # line = pprint.pformat(dic)
                # output_line(line)
                pass
            last_dic = dic
        return last_dic

def check_docker_response(response, path, repo_tag):
    if is_docker_error_reponse(response):
        raise DockerError(path, repo_tag, response)

def is_docker_error_reponse(reponse):
    return 'error' in reponse

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
    build_args = {
        "USER_ID": get_user_id(),
        "GROUP_ID": get_user_id(),
        "USER_NAME": get_username(),
    }
    for k in build_args.keys():
        # Otherwise, when we do dock_cli.build we get:
        #     docker.errors.APIError: 400 Client Error: Bad Request ("error reading build args: json: cannot unmarshal number into Go value of type string")
        build_args[k] = str(build_args[k])
    return build_args

def get_implicit_run_args():
    run_args = {
        "IML_DIR": py_config.IML_DIR,
    }
    return run_args

RUN_ARGS_REQUIRED = [
    'IML_DIR',
    'IML_DRILL_DIR',
    # The root directory of a 'patched' TensorFlow checkout
    'TENSORFLOW_DIR',
    # The local path where we should output bazel objects (overrides $HOME/.cache/bazel)
    'BAZEL_BUILD_DIR',
]

def is_required_run_arg(var):
    return var in RUN_ARGS_REQUIRED

def get_iml_volumes(run_args, extra_volumes):
    """
    --build-arg USER_ID=$(id -u ${USER})
    --build-arg GROUP_ID=$(id -u ${USER})
    --build-arg USER_NAME=${USER}
    :return:
    """
    volumes = dict()
    for arg in RUN_ARGS_REQUIRED:
        direc = run_args[arg]
        volumes[direc] = direc
    for i, direc in enumerate(extra_volumes):
        env_name = 'CMDLINE_VOLUME_{i}'.format(i=i)
        assert env_name not in volumes
        # If this fails you're probably using --volume wrong; see usage info.
        assert ':' not in direc
        # volumes[env_name] = direc
        volumes[direc] = direc
    return volumes

class DockerError(Exception):
    """
    :param path
        Path to Dockerfile
    :param repo_tag
        Identifier for build
    """
    def __init__(self, path, repo_tag, response):
        self.path = path
        self.repo_tag = repo_tag
        self.response = response
        message = self.construct_message()
        super().__init__(message)

    def construct_message(self):
        pprint.pprint({
            'response': self.response,
        })

        # code = self.response['errorDetail']['code']
        # err = self.response['errorDetail']['message']
        lines = []
        lines.append(
            "Failed to build dockerfile {path} with repo_tag={repo_tag}.".format(
                path=_a(self.path),
                repo_tag=self.repo_tag))
        if 'code' in self.response['errorDetail']:
            lines.append("    Exit status: {code}".format(
                code=self.response['errorDetail']['code'],
            ))

        lines.append("    Error message:")
        lines.append(
            "{err}".format(
                err=textwrap.indent(
                    self.response['errorDetail']['message'],
                    prefix="    "*2),
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

    add_opt_from('workdir', 'working_dir')

    add_opt_from('rm', 'remove', opt_type=bool)

    def add_opt_list(opt_name, from_name):
        if from_name in kwargs:
            for value in kwargs[from_name]:
                add_opt_value(opt_name, value)

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

    return cmd

class StackYMLGenerator:
    """
    Generate stack.yml.
    """
    def __init__(self):
        """
        :param cmd
            assembler.py command.
        """
        self.template = textwrap.dedent("""\
        # DO NOT MODIFY!
        # Automatically generated using assembler.py with the following command/working-directory:
        #     > CMD: {assembler_cmd}
        #       PWD: {PWD}

        version: '3.1'

        services:

            # Postgres instance.
            #
            # iml uses this for storing/analyzing trace data.
            db:
                image: postgres
                restart: always
                environment:
                    # Use the current user as postgres user (default when using psql).
                    # `psql` at the command-line defaults to this user, as do API's 
                    # for making postgres connections.
                    - POSTGRES_USER={USER}
                ports:
                    - {postgres_port}:{DEFAULT_POSTGRES_PORT}
                volumes:
                    # Persist processed trace-logs between deployments on the host inside {postgres_pgdata_dir}.
                    #
                    # NOTE: {postgres_pgdata_dir} will have different permissions than the host user; 
                    # We don't bother to fix this since: 
                    # (1) Fixes are host OS dependent [See "Arbitrary --user Notes" @ https://hub.docker.com/_/postgres]
                    #     and hence sacrifice reproducibility
                    # (2) Database access isn't sacrificed since it's still all done through 
                    #     "psql -h localhost".
                    #
                    - {postgres_pgdata_dir}:/var/lib/postgresql/data

            # "Bash" development environment.
            #
            # Original docker cmd:
            # $ docker run -it --runtime nvidia ... tensorflow:devel-iml-gpu-cuda
            bash:
                #
                # NOTE:
                # - "docker stack deploy" does NOT support 'build:', so this container must be
                #     built separately via "docker build ..."
                #
                # build: ./dockerfiles/devel-iml-gpu-cuda.Dockerfile
                image: {iml_image}
                
                depends_on:
                    - db
                restart: always
                
                # Fails, not supported yet: instead for now just manually edit /etc/docker/daemon.json
                # as described below:
                #
                #   https://github.com/NVIDIA/k8s-device-plugin#preparing-your-gpu-nodes
                #
                # In particular, add the line show below:
                #   {{
                #           "default-runtime": "nvidia",    // <-- add this ABOVE "runtimes"
                #           "runtimes": {{
                #              ...
                #            }}
                #   }}
                #
                # runtime: nvidia
                
                ports:
                    # Expose port that the iml-drill web server runs on.
                    - {iml_drill_port}:{DEFAULT_IML_DRILL_PORT}
                
                volumes:
                {volume_list}
                
                environment:
                - IML_POSTGRES_HOST=db
                {env_list}
                
                logging:
                    driver: journald
                stdin_open: true
                tty: true
                entrypoint: /bin/bash
        """).rstrip()

        self.indent_str = 4*' '

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
                 assembler_cmd, env, volumes,
                 iml_drill_port=DEFAULT_IML_DRILL_PORT,
                 postgres_port=DEFAULT_POSTGRES_PORT,
                 postgres_pgdata_dir=DEFAULT_POSTGRES_PGDATA_DIR,
                 iml_image=LOCAL_IML_IMAGE_TAG):

        # +1 for "service: ..."
        # +1 for "bash: ..."
        bash_indent = 2

        print("> Create postgres PGDATA directory @ {path} for storing databases".format(
            path=postgres_pgdata_dir))
        os.makedirs(postgres_pgdata_dir, exist_ok=True)

        # return textwrap.dedent(self.template.format(
        return self.template.format(
            env_list=self.env_list(env, indent=bash_indent),
            volume_list=self.volume_list(volumes, indent=bash_indent),
            DEFAULT_POSTGRES_PORT=DEFAULT_POSTGRES_PORT,
            postgres_port=postgres_port,
            postgres_pgdata_dir=postgres_pgdata_dir,
            USER=get_username(),
            assembler_cmd=' '.join(assembler_cmd),
            PWD=os.getcwd(),
            DEFAULT_IML_DRILL_PORT=DEFAULT_IML_DRILL_PORT,
            iml_drill_port=iml_drill_port,
            iml_image=iml_image,
        )

    def _yml_list(self, values, indent):
        yml_lines = []
        for value in values:
            yml_lines.append("- {value}".format(
                value=value))
        # NOTE: lstrip() to remove indent from first line, since it's already in the template.
        return textwrap.indent('\n'.join(yml_lines), indent*self.indent_str).lstrip()

    def _yml_dict_as_list(self, values : dict, sep, indent):
        values = ["{var}{sep}{value}".format(var=var, value=values[var], sep=sep)
                  for var in sorted(values.keys())]
        return self._yml_list(values, indent)

    def volume_list(self, volumes : dict, indent):
        return self._yml_dict_as_list(volumes, sep=':', indent=indent)

    def env_list(self, envs : dict, indent):
        return self._yml_dict_as_list(envs, sep='=', indent=indent)

def get_cmd_string(cmd):
    return ("> CMD:\n"
            "  $ {cmd}").format(
        cmd=' '.join(cmd))

if __name__ == '__main__':
    main()
