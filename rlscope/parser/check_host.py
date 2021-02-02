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

    def _check_bin(binary, apt_package):
        if not shutil.which(binary):
            error_messages.append(
                textwrap.dedent("""\
            Couldn't find command '{binary}' on PATH; to install on Ubuntu, run:
              $ sudo apt install {apt}
            """).format(
                    binary=binary,
                    apt=apt_package,
                ))

    _check_bin('pdfcrop', 'texlive-extra-utils')

    # PDF -> PNG converted.
    # PNG's display better in jupyter notebooks.
    # PDF's are better for papers.
    _check_bin('pdftoppm', 'poppler-utils')

    if len(error_messages) > 0:
        raise RLScopeConfigurationError("\n".join(error_messages).rstrip())

def check_config():
    """
    Check host configuration
    """
    check_apt_get()