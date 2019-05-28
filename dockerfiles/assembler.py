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

from iml_profiler import py_config

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
    '--run',
    action='store_true',
    help='Run built images')

parser.add_argument(
    '--run_tests_path',
    help=('Execute test scripts on generated Dockerfiles before pushing them. '
     'Flag value must be a full path to the "tests" directory, which is usually'
     ' $(realpath ./tests). A failed tests counts the same as a failed build.'))

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


# FLAGS = flags.FLAGS
#
# # flags.DEFINE_string('hub_username', None,
# #                                         'Dockerhub username, only used with --upload_to_hub')
#
# # flags.DEFINE_string(
# #         'hub_password', None,
# #         ('Dockerhub password, only used with --upload_to_hub. Use from an env param'
# #            'so your password isn\'t in your history.'))
#
# flags.DEFINE_integer('hub_timeout', 3600,
#                                            'Abort Hub upload if it takes longer than this.')
#
# flags.DEFINE_string(
#         'repository', 'tensorflow',
#         'Tag local images as {repository}:tag (in addition to the '
#         'hub_repository, if uploading to hub)')
#
# flags.DEFINE_string(
#         'hub_repository', None,
#         'Push tags to this Docker Hub repository, e.g. tensorflow/tensorflow')
#
# flags.DEFINE_boolean(
#         'debug', False,
#         textwrap.dedent("""
#         In the generated dockerfiles, print start/end markers for the partial files its composed of; for e.g.:
#             START: dockerfiles/partials/ubuntu/devel-nvidia.partial.Dockerfile
#             RUN ...
#             RUN ...
#             ...
#             END: dockerfiles/partials/ubuntu/devel-nvidia.partial.Dockerfile
#         """))
#
# # flags.DEFINE_boolean(
# #         'upload_to_hub',
# #         False,
# #         ('Push built images to Docker Hub (you must also provide --hub_username, '
# #            '--hub_password, and --hub_repository)'),
# #         short_name='u',
# # )
#
# flags.DEFINE_boolean(
#         'construct_dockerfiles', False, 'Do not build images', short_name='d')
#
# flags.DEFINE_boolean(
#         'keep_temp_dockerfiles',
#         False,
#         'Retain .temp.Dockerfiles created while building images.',
#         short_name='k')
#
# flags.DEFINE_boolean(
#         'build_images', False, 'Do not build images', short_name='b')
#
# flags.DEFINE_boolean(
#     'run', False, 'Run built images')
#
# flags.DEFINE_string(
#         'run_tests_path', None,
#         ('Execute test scripts on generated Dockerfiles before pushing them. '
#            'Flag value must be a full path to the "tests" directory, which is usually'
#            ' $(realpath ./tests). A failed tests counts the same as a failed build.'))
#
# flags.DEFINE_boolean(
#         'stop_on_failure', False,
#         ('Stop processing tags if any one build fails. If False or not specified, '
#            'failures are reported but do not affect the other images.'))
#
# flags.DEFINE_boolean(
#         'dry_run',
#         False,
#         'Do not build or deploy anything at all.',
#         short_name='n',
# )
#
# flags.DEFINE_string(
#         'exclude_tags_matching',
#         None,
#         ('Regular expression that skips processing on any tag it matches. Must '
#            'match entire string, e.g. ".*gpu.*" ignores all GPU tags.'),
#         short_name='x')
#
# flags.DEFINE_string(
#         'only_tags_matching',
#         None,
#         ('Regular expression that skips processing on any tag it does not match. '
#            'Must match entire string, e.g. ".*gpu.*" includes only GPU tags.'),
#         short_name='i')
#
# flags.DEFINE_string(
#         'dockerfile_dir',
#         './dockerfiles', 'Path to an output directory for Dockerfiles.'
#         ' Will be created if it doesn\'t exist.'
#         ' Existing files in this directory will be deleted when new Dockerfiles'
#         ' are made.',
#         short_name='o')
#
# flags.DEFINE_string(
#         'partial_dir',
#         './partials',
#         'Path to a directory containing foo.partial.Dockerfile partial files.'
#         ' can have subdirectories, e.g. "bar/baz.partial.Dockerfile".',
#         short_name='p')
#
# flags.DEFINE_multi_string(
#         'release', [],
#         'Set of releases to build and tag. Defaults to every release type.',
#         short_name='r')
#
# flags.DEFINE_multi_string(
#         'arg', [],
#         ('Extra build arguments. These are used for expanding tag names if needed '
#            '(e.g. --arg _TAG_PREFIX=foo) and for using as build arguments (unused '
#            'args will print a warning).'),
#         short_name='a')
#
# flags.DEFINE_multi_string(
#     'run_arg', [],
#     ('Extra container run arguments (NOT build).'))
#
# flags.DEFINE_string(
#         'spec_file',
#         './spec.yml',
#         'Path to the YAML specification file',
#         short_name='s')

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
                     run_args_optional:
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


def assemble_tags(spec, cli_args, cli_run_args, cli_run_args_optional, enabled_releases, all_partials):
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
                run_args_optional = gather_tag_args(slices, cli_run_args_optional, spec_field='run_args_optional', cmd_opt='run_arg')
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
                        'run_args_optional': run_args_optional,
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
#     print(dock.images.push(hub_repository, tag=tag))


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

def get_docker_run_env(tag_def):
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

    if 'run_args_optional' in tag_def:
        # We DON'T check that run_args_optional is set (default is '' from spec.yml is allowed)
        env = update_args_dict(env, tag_def['run_args_optional'], keep_original=True)

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

# def main(argv):
def main():
    args, extra_argv = parser.parse_known_args()

    pprint.pprint({'extra_argv':extra_argv})
    # sys.exit(0)

    if not _e('./spec.yml'):
        print("ERROR: You must execute assembly.py from dockerfiles/assembler.py")
        sys.exit(1)

    # if len(argv) > 1:
    #     raise parser.error('Too many command-line arguments.')

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
    run_arg_optional = []
    for run_arg in args.run_arg:
        var, value = parse_build_arg(run_arg)
        if is_required_run_arg(var):
            run_arg_required.append(run_arg)
        else:
            run_arg_optional.append(run_arg)
    all_tags = assemble_tags(tag_spec, args.arg, run_arg_required, run_arg_optional, args.release, partials)

    # Empty Dockerfile directory if building new Dockerfiles
    if args.construct_dockerfiles:
        eprint('> Emptying Dockerfile dir "{}"'.format(args.dockerfile_dir))
        shutil.rmtree(args.dockerfile_dir, ignore_errors=True)
        mkdir_p(args.dockerfile_dir)

    # pprint.pprint({
    #     'FLAGS': FLAGS,
    #     'args.__dict__': args.__dict__,
    #     'args.flag_values_dict()': args.flag_values_dict(),
    # })

    # Set up Docker helper
    dock = docker.from_env()
    dock_cli = APIClient()
    # dock_cli = APIClient(base_url='tcp://127.0.0.1:2375')
    # Q: Do we need to pass environment variables (like docker.from_env())?

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
    failed_tags = []
    for tag, tag_defs in all_tags.items():
        for tag_def in tag_defs:
            docker_run_env = get_docker_run_env(tag_def)
            eprint('> Working on {}'.format(tag))

            if args.exclude_tags_matching and re.match(args.exclude_tags_matching,
                                                                                                    tag):
                eprint('>> Excluded due to match against "{}".'.format(
                        args.exclude_tags_matching))
                continue

            if args.only_tags_matching and not re.match(args.only_tags_matching,
                                                                                                     tag):
                eprint('>> Excluded due to failure to match against "{}".'.format(
                        args.only_tags_matching))
                continue

            if tag == "devel-iml-gpu-py3":
                    import ipdb; ipdb.set_trace()

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

                    build_output_generator = dock_cli.build(decode=True, **build_kwargs)
                    response = tee_docker(
                        build_output_generator,
                        file=get_build_logfile(repo_tag),
                        to_stdout=args.debug)
                    check_docker_response(response, dockerfile, repo_tag)

                    image = dock.images.get(repo_tag)

                    # Re-run dock.images.build just to get the "Image" instance for running the container.
                    # image, logs = dock.images.build(**build_kwargs)

                    # Print logs after finishing
                    # log_lines = [l.get('stream', '') for l in logs]
                    # eprint(''.join(log_lines))

                    # Run tests if requested, and dump output
                    # Could be improved by backgrounding, but would need better
                    # multiprocessing support to track failures properly.


                    # ROCM_EXTRA_PARAMS="--device=/dev/kfd --device=/dev/dri --group-add video"

                    if args.run:
                        # Run the container.
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
                        # docker_run_argv = get_docker_run_argv(argv)
                        docker_run_argv = extra_argv
                        run_cmd = get_docker_cmdline('run', docker_run_argv, **run_kwargs)
                        # pprint.pprint({'run_cmd': run_cmd})
                        eprint("> CMD:\n"
                                     "    {cmd}".format(
                            cmd=' '.join(run_cmd)))
                        subprocess.check_call(run_cmd, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)
                        # Q: Save output?

                    if args.run_tests_path:
                        if not tag_def['tests']:
                            eprint('>>> No tests to run.')
                        for test in tag_def['tests']:
                            eprint('>> Testing {}...'.format(test))

                            os.makedirs(docker_run_env['IML_TEST_DIR'], exist_ok=True)

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

                            container = dock.containers.run(**test_kwargs)
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
                                failed_tags.append(tag)
                                tag_failed = True
                                if args.stop_on_failure:
                                    eprint('>> ABORTING due to --stop_on_failure!')
                                    exit(1)
                            else:
                                eprint('>> Tests look good!')

                except docker.errors.BuildError as e:
                    eprint('>> {} failed to build with message: "{}"'.format(
                            repo_tag, e.msg))
                    eprint('>> Build logs follow:')
                    log_lines = [l.get('stream', '') for l in e.build_log]
                    eprint(''.join(log_lines))
                    failed_tags.append(tag)
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

    if failed_tags:
        eprint(
                '> Some tags failed to build or failed testing, check scrollback for '
                'errors: {}'.format(','.join(failed_tags)))
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

def get_username():
    return pwd.getpwuid(os.getuid())[0]

def get_user_id():
    return os.getuid()

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
        "IML_TEST_DIR": py_config.IML_TEST_DIR,
    }
    return run_args

RUN_ARGS_REQUIRED = [
    'IML_DIR',
    'IML_DRILL_DIR',
    # The --iml-directory argument to training scripts, which is where trace-data files are stored.
    'IML_TEST_DIR',
    # The root directory of a 'patched' TensorFlow checkout
    'TENSORFLOW_DIR',
    # The local path where we should output bazel objects (overrides $HOME/.cache/bazel)
    'BAZEL_BUILD_DIR',
]
RUN_ARGS_OPTIONAL = [
    # The root directory of a checkout of TensorFlow benchmarks repo (https://github.com/tensorflow/benchmarks)
    'TENSORFLOW_BENCHMARKS_DIR',
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
    for arg in RUN_ARGS_OPTIONAL:
        if arg not in run_args:
            continue
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
    :param reponse
        {'error': 'The command \'/bin/sh -c if [ "${IML_DIR}" = "" ] ; then                 '
                                    'pip install git+https://github.com/UofT-EcoSystem/iml.git;         '
                                    'else;                 (                 cd $IML_DIR;                 python setup.py '
                                    "develop;                 );         fi' returned a non-zero code: 2",
         'errorDetail': {'code': 2,
                                         'message': 'The command \'/bin/sh -c if [ "${IML_DIR}" = "" ] '
                                                                '; then                 pip install '
                                                                'git+https://github.com/UofT-EcoSystem/iml.git;         '
                                                                'else;                 (                 cd $IML_DIR;                 '
                                                                "python setup.py develop;                 );         fi' "
                                                                'returned a non-zero code: 2'}}
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

if __name__ == '__main__':
    main()
    # app.run(main)
