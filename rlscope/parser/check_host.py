"""
Check RL-Scope host configuration to make things are configured properly.
e.g.
- apt-get dependencies are installed
"""

import shutil
import textwrap

from rlscope.profiler.rlscope_logging import logger
from rlscope.parser.exceptions import RLScopeConfigurationError

from rlscope import py_config

def check_apt_get():
    """
    Check that apt-get dependencies have been installed.
    """
    error_messages = []
    # if True:
    if not shutil.which('pdfcrop'):
        error_messages.append(
            textwrap.dedent("""\
            Couldn't find command 'pdfcrop' on PATH; to install on Ubuntu, run:
              $ sudo apt install texlive-extra-utils
            """))
    if len(error_messages) > 0:
        raise RLScopeConfigurationError("\n".join(error_messages).rstrip())

def check_config():
    """
    Check host configuration
    """
    check_apt_get()