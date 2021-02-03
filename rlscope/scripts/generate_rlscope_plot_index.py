"""
Generate index of ``*.venn_js.json`` files (for multi-process visualization).
"""
from rlscope.profiler.rlscope_logging import logger
import signal
import time
import subprocess
import shutil
import argparse
import sys
import textwrap
import psutil
import platform
import cpuinfo
import concurrent.futures

from rlscope.parser.plot_index import SEL_ORDER, TITLE_ORDER

from os.path import join as _j, abspath as _a, dirname as _d, exists as _e, basename as _b

from rlscope import py_config

from rlscope.parser import check_host
from rlscope.parser.exceptions import RLScopeConfigurationError

from rlscope.parser.common import *

class GeneratePlotIndex:
    def __init__(self, directory, out_dir,
                 basename=None,
                 debug=False,
                 replace=False,
                 dry_run=False):
        self.directory = directory
        self.out_dir = out_dir
        self.basename = basename
        self.debug = debug
        self.replace = replace
        self.dry_run = dry_run

        self.index = dict()

    def each_file(self):
        for dirpath, dirnames, filenames in os.walk(self.directory):
            for base in filenames:
                path = _j(dirpath, base)
                if self.is_venn_js_path(path) or self.is_js_path(path):
                    yield path

    def is_venn_js_path(self, path):
        base = _b(path)
        m = re.search(r'\.venn_js\.json$', base)
        return m

    def is_overlap_js_path(self, path):
        base = _b(path)
        m = re.search(r'\.overlap_js\.json$', base)
        return m

    def is_js_path(self, path):
        base = _b(path)
        m = re.search(r'\.js_path\.json$', base)
        return m

    def read_metadata(self, path):
        with open(path, 'r') as f:
            js = json.load(f)
        if 'metadata' not in js:
            return None
        md = js['metadata']
        return md

    def lookup_entry(self, md, path=None):

        def map_resource_overlap(field, value):

            # if self.debug:
            #     logger.info("field={field}, value={value}".format(
            #         field=field,
            #         value=value,
            #     ))

            if field == 'resource_overlap':
                return tuple(value)

            if field == 'operation' and type(value) == list:
                assert len(value) == 1
                # Q: just use single item..?
                return tuple(value)

            return value

        def mk_keys(md, plot_type_field, value_map=None):
            plot_type = md[plot_type_field]
            keys = [md[plot_type_field]]
            for field in SEL_ORDER[plot_type]:
                keys.append(field)
                value = md[field]
                if value_map is not None:
                    value = value_map(field, value)
                keys.append(value)
            return keys

        if 'overlap_type' in md:
            if md['overlap_type'] not in SEL_ORDER:
                raise NotImplementedError(
                    ("Not sure how to insert OverlapType={OverlapType} for path={path} into index; "
                     "metadata = {md}").format(
                        OverlapType=md['overlap_type'],
                        path=path,
                        md=md))
            keys = mk_keys(md, 'overlap_type', value_map=map_resource_overlap)
            dic = mkds(self.index, *keys)
            return dic
        elif 'plot_type' in md:
            if md['plot_type'] not in SEL_ORDER:
                raise NotImplementedError(
                    ("Not sure how to insert plot_type={PlotType} for path={path} into index; "
                     "metadata = {md}").format(
                        OverlapType=md['plot_type'],
                        path=path,
                        md=md))
            keys = mk_keys(md, 'plot_type', value_map=map_resource_overlap)
            dic = mkds(self.index, *keys)
            return dic
        else:
            raise NotImplementedError("Not sure what plot-type this metadata contains:\n{md}".format(
                md=textwrap.indent(pprint.pformat(md), prefix='  ')
            ))

        #     if md['overlap_type'] == 'CategoryOverlap':
        #
        #         # self.index['process'][md['process']] \
        #         #     ['phase'][md['phase']] \
        #         # ['resource_overlap'][tuple(md['resource_overlap'])] \
        #         #     ['operation'][tuple(md['operation'])] \
        #         #     ['venn_js_path'] = venn_js_path
        #         return mkds(self.index,
        #              md['overlap_type'],
        #              'process', md['process'],
        #              'phase', md['phase'],
        #              'resource_overlap', tuple(md['resource_overlap']),
        #              'operation', md['operation'],
        #              )
        #
        #     elif md['overlap_type'] in ['ResourceOverlap', 'ResourceSubplot']:
        #         return mkds(self.index,
        #              md['overlap_type'],
        #              'process', md['process'],
        #              'phase', md['phase'],
        #              )
        #
        #     elif md['overlap_type'] == 'OperationOverlap':
        #         return mkds(self.index,
        #              md['overlap_type'],
        #              'process', md['process'],
        #              'phase', md['phase'],
        #              'resource_overlap', tuple(md['resource_overlap']),
        #              )
        #
        #     else:
        # elif 'plot_type' in md:
        #     if md['plot_type'] == 'HeatScale':
        #         return mkds(self.index,
        #                     md['plot_type'],
        #                     'device_name', md['device_name'])
        #     else:

    def run(self):
        i = 0
        for path in self.each_file():

            if self.debug:
                logger.info("path[{i}] = {path}".format(
                    i=i,
                    path=path))

            md = self.read_metadata(path)
            if md is None:
                logger.info("WARNING: didn't find any metadata in {path}; SKIP.".format(path=path))
                continue

            if self.debug:
                logger.info("> index: {path}".format(path=path))

            entry = self.lookup_entry(md, path)
            relpath = os.path.relpath(path, self.directory)
            if self.is_venn_js_path(path):
                entry['venn_js_path'] = relpath
            elif self.is_overlap_js_path(path):
                entry['overlap_js_path'] = relpath
            elif self.is_js_path(path):
                entry['js_path'] = relpath
            else:
                raise NotImplementedError

        self.dump_plot_index_py()

    @property
    def plot_index_path(self):
        return _j(self.out_dir, self.basename)

    def dump_plot_index_py(self):
        cmd = sys.argv

        if not self.dry_run:
            src = _j(py_config.ROOT, 'rlscope/scripts/rlscope_plot_index.py')
            dst = _j(self.out_dir, 'rlscope_plot_index.py')
            logger.info("cp {src} -> {dst}".format(src=src, dst=dst))
            os.makedirs(_d(dst), exist_ok=True)
            shutil.copyfile(src, dst)

        os.makedirs(_d(self.plot_index_path), exist_ok=True)
        if _e(self.plot_index_path) and not self.replace:
            logger.warning("{path} exists; skipping".format(path=self.plot_index_path))
            return

        with open(self.plot_index_path, 'w') as f:

            contents = textwrap.dedent("""\
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
                """).format(
                dir=os.path.realpath(self.directory),
                index=textwrap.indent(
                    pprint.pformat(self.index),
                    prefix="    "),
                cmd=" ".join(cmd),
                pwd=os.getcwd(),
            )
            if self.debug:
                logger.info("> Generated file: {path}".format(path=self.plot_index_path))
                logger.info(contents)

            if not self.dry_run:
                f.write(contents)

def mkd(dic, key):
    if key not in dic:
        dic[key] = dict()
    return dic[key]

def mkds(dic, *keys):
    for key in keys:
        dic = mkd(dic, key)
    return dic

from rlscope.profiler.rlscope_logging import logger
def main():

    try:
        check_host.check_config()
    except RLScopeConfigurationError as e:
        logger.error(e)
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__.lstrip().rstrip()),
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--rlscope-directory',
                        required=True,
                        help=textwrap.dedent("""\
    Look for *.venn_js.json rooted at this directory.
    The output file will be <directory>/rlscope_plot_index_data.py.
    All the venn_js_path's in the index will be relative to --directory.
    """))
    parser.add_argument('--out-dir',
                        help=textwrap.dedent("""\
    The output file will be <out-dir>/rlscope_plot_index_data.py.
    Default: --directory
    """))
    parser.add_argument('--debug',
                        action='store_true',
                        help=textwrap.dedent("""\
    Debug
    """))
    parser.add_argument('--dry-run',
                        action='store_true',
                        help=textwrap.dedent("""\
    Don't write file.
    """))
    parser.add_argument('--basename',
                        default='rlscope_plot_index_data.py',
                        help=textwrap.dedent("""\
    Name of python file to generate.
    """))
    parser.add_argument('--replace',
                        action='store_true',
                        help=textwrap.dedent("""\
    Replace if exists.
    """))
    parser.add_argument('--pdb',
                        action='store_true',
                        help=textwrap.dedent("""\
    Python debugger on unhandled exception.
    """))
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = args.rlscope_directory


    try:
        obj = GeneratePlotIndex(
            directory=args.rlscope_directory,
            out_dir=args.out_dir,
            basename=args.basename,
            debug=args.debug,
            replace=args.replace,
            dry_run=args.dry_run,
        )
        obj.run()
    except Exception as e:
        if not args.pdb:
            raise
        print("> RL-Scope: Detected exception:")
        print(e)
        print("> Entering pdb:")
        import pdb
        pdb.post_mortem()
        raise

if __name__ == '__main__':
    main()
