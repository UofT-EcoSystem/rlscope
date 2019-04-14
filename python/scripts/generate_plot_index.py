import signal
import time
import subprocess
import shutil
import argparse
import textwrap
import psutil
import platform
import cpuinfo
import concurrent.futures

from os.path import join as _j, abspath as _a, dirname as _d, exists as _e, basename as _b

# import GPUtil

# from proto.protobuf.pyprof_pb2 import Pyprof, MachineUtilization, DeviceUtilization, UtilizationSample

import py_config

from parser.common import *

# from profiler.profilers import trace_suffix, get_util_sampler_parser, MIN_UTIL_SAMPLE_FREQUENCY_SEC

class GeneratePlotIndex:
    def __init__(self, directory,
                 basename=None,
                 debug=False,
                 replace=False,
                 dry_run=False):
        self.directory = directory
        self.basename = basename
        self.debug = debug
        self.replace = replace
        self.dry_run = dry_run

        self.index = dict()

    def each_file(self):
        for dirpath, dirnames, filenames in os.walk(self.directory):
            for base in filenames:
                path = _j(dirpath, base)
                if self.is_venn_js_path(path):
                    yield path

    def is_venn_js_path(self, path):
        base = _b(path)
        m = re.search(r'\.venn_js\.json$', base)
        return m

    def is_overlap_js_path(self, path):
        base = _b(path)
        m = re.search(r'\.overlap_js\.json$', base)
        return m

    def read_metadata(self, path):
        with open(path, 'r') as f:
            js = json.load(f)
        if 'metadata' not in js:
            return None
        md = js['metadata']
        return md

    def lookup_entry(self, md, path=None):
        if md['overlap_type'] == 'CategoryOverlap':

            # self.index['process'][md['process']] \
            #     ['phase'][md['phase']] \
            # ['resource_overlap'][tuple(md['resource_overlap'])] \
            #     ['operation'][tuple(md['operation'])] \
            #     ['venn_js_path'] = venn_js_path
            return mkds(self.index,
                 md['overlap_type'],
                 'process', md['process'],
                 'phase', md['phase'],
                 'resource_overlap', tuple(md['resource_overlap']),
                 'operation', md['operation'],
                 )

        elif md['overlap_type'] in ['ResourceOverlap', 'ResourceSubplot']:
            return mkds(self.index,
                 md['overlap_type'],
                 'process', md['process'],
                 'phase', md['phase'],
                 )

        elif md['overlap_type'] == 'OperationOverlap':
            return mkds(self.index,
                 md['overlap_type'],
                 'process', md['process'],
                 'phase', md['phase'],
                 'resource_overlap', tuple(md['resource_overlap']),
                 )

        else:
            raise NotImplementedError(
                ("Not sure how to insert OverlapType={OverlapType} for path={path} into index; "
                 "metadata = {md}").format(
                    OverlapType=md['overlap_type'],
                    path=path,
                    md=md))

    def run(self):
        for path in self.each_file():

            md = self.read_metadata(path)
            if md is None:
                print("WARNING: didn't find any metadata in {path}; SKIP.".format(path=path))
                continue

            if self.debug:
                print("> index: {path}".format(path=path))

            entry = self.lookup_entry(md, path)
            relpath = os.path.relpath(path, self.directory)
            if self.is_venn_js_path(path):
                entry['venn_js_path'] = relpath
            elif self.is_overlap_js_path(path):
                entry['overlap_js_path'] = relpath
            else:
                raise NotImplementedError

        self.dump_plot_index_py()

    @property
    def plot_index_path(self):
        return _j(self.directory, self.basename)

    def dump_plot_index_py(self):
        cmd = sys.argv

        os.makedirs(_d(self.plot_index_path), exist_ok=True)
        if _e(self.plot_index_path) and not self.replace:
            print("WARNING: {path} exists; skipping".format(path=self.plot_index_path))
            return

        with open(self.plot_index_path, 'w') as f:

            contents = textwrap.dedent("""
                #!/usr/bin/env python3
                
                ### GENERATED FILE; do NOT modify!
                ### generated using: 
                ### CMD:
                ###   {cmd}
                ### PWD:
                ###   {pwd}
                
                DIRECTORY = "{dir}"
                INDEX = \\
                {index}
                """.format(
                dir=os.path.realpath(self.directory),
                index=textwrap.indent(
                    pprint.pformat(self.index),
                    prefix="    "),
                cmd=" ".join(cmd),
                pwd=os.getcwd(),
            ))
            if self.debug:
                print()
                print("> Generated file: {path}".format(path=self.plot_index_path))
                print(contents)

            if not self.dry_run:
                f.write(contents)

        if not self.dry_run:
            shutil.copyfile(
                _j(py_config.ROOT, 'python/scripts/plot_index.py'),
                _j(self.directory, 'plot_index.py'),
            )

def mkd(dic, key):
    if key not in dic:
        dic[key] = dict()
    return dic[key]

def mkds(dic, *keys):
    for key in keys:
        dic = mkd(dic, key)
    return dic

def main():
    parser = argparse.ArgumentParser("Generate index of *.venn_js.json files.")
    parser.add_argument('--directory',
                        required=True,
                        help=textwrap.dedent("""
    Look for *.venn_js.json rooted at this directory.
    The output file will be <directory>/plot_index_data.py.
    All the venn_js_path's in the index will be relative to --directory.
    """))
    parser.add_argument('--debug',
                        action='store_true',
                        help=textwrap.dedent("""
    Debug
    """))
    parser.add_argument('--dry-run',
                        action='store_true',
                        help=textwrap.dedent("""
    Don't write file.
    """))
    parser.add_argument('--basename',
                        default='plot_index_data.py',
                        help=textwrap.dedent("""
    Name of python file to generate.
    """))
    parser.add_argument('--replace',
                        action='store_true',
                        help=textwrap.dedent("""
    Replace if exists.
    """))
    args = parser.parse_args()

    obj = GeneratePlotIndex(
        directory=args.directory,
        basename=args.basename,
        debug=args.debug,
        replace=args.replace,
        dry_run=args.dry_run,
    )
    obj.run()

if __name__ == '__main__':
    main()
