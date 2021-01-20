#!/usr/bin/env python
# read -r -d '' BASH_SCRIPT << EOM
# cd \$RLSCOPE_DIR;
# source source_me.sh;
# "$@";
# EOM
#
# docker exec --user $USER -i -t dockerfiles_bash_1 /bin/bash -c "$BASH_SCRIPT"

import subprocess
import argparse
import textwrap
import sys
import shutil
import shlex
import getpass
import os

import logging
from rlscope.profiler import rlscope_logging

from rlscope.parser.common import *
from rlscope.profiler.util import cmd_debug_msg

def main():
    rlscope_logging.setup_logging()
    parser = argparse.ArgumentParser(textwrap.dedent("""
    Test profiling scripts to make sure we correctly measure time spent in Python/C++/GPU.
    """))
    args, argv = parser.parse_known_args()

    shell = ' '.join([shlex.quote(opt) for opt in argv])
    bash_script = textwrap.dedent("""
    set -e;
    cd $RLSCOPE_DIR;
    source source_me.sh;
    {shell};
    """).format(shell=shell)

    # docker exec --user $USER -i -t dockerfiles_bash_1 /bin/bash -c "$BASH_SCRIPT" -c "$@"
    cmd = [
        'docker',
        'exec',
        '--user',
        getpass.getuser(),
        '-i',
        '-t',
        'rlscope_bash_1',
        '/bin/bash',
        '-c',
        bash_script,
    ]

    # proc = subprocess.run(
    #     cmd,
    #     # stdout=subprocess.STD_OUTPUT_HANDLE,
    #     # stderr=subprocess.STD_ERROR_HANDLE,
    #     # stdin=subprocess.STD_INPUT_HANDLE,
    # )
    # sys.exit(proc.returncode)
    path = shutil.which(cmd[0])
    env = os.environ
    logging.info(cmd_debug_msg(cmd, env=None))
    os.execve(path, cmd, env)
    assert False

if __name__ == '__main__':
    main()
