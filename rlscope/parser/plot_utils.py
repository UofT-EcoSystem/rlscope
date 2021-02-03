import shutil
import subprocess
import re
import textwrap
import os
from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b

from rlscope.parser.exceptions import RLScopeConfigurationError
from rlscope.profiler.rlscope_logging import logger

# NOTE: before using matplotlib you should do this:
# from rlscope.parser.plot_utils import setup_matplotlib
# setup_matplotlib()
SETUP_DONE = False
def setup_matplotlib():
    global SETUP_DONE
    if SETUP_DONE:
        return
    import matplotlib
    matplotlib.use('agg')

def is_pdf(path):
    return re.search(r'.pdf$', _b(path))

def crop_pdf(path, output=None):
    if output is None:
        # output (in-place)
        output = path
    # pdfcrop comes from texlive-extra-utils apt package on ubuntu.
    if not shutil.which('pdfcrop'):
        raise RLScopeConfigurationError(
            textwrap.dedent("""\
            Couldn't find command 'pdfcrop' on PATH; to install on Ubuntu, run:
              $ sudo apt install texlive-extra-utils
            """))
    subprocess.check_call(["pdfcrop",
                           # input
                           path,
                           output,
                           ])

def pdf2svg(path, output=None, can_skip=False):
    if output is None:
        output = re.sub(r'\.pdf$', '.svg', path)
        # If this fails, then "path" doesn't end with .pdf.
        assert path != output
    if not shutil.which('pdf2svg'):
        if can_skip:
            logger.warning(f"pdf2svg shell command not found; SKIP: pdf2svg {path} {output}")
            return
        else:
            raise RuntimeError(f"pdf2svg shell command not found for: \"pdf2svg {path} {output}\".  Install with \"sudo apt install pdf2svg\"")
    subprocess.check_call(['pdf2svg',
                           # input
                           path,
                           output,
                           ])

def pdf2png(path, output=None, can_skip=True, silent=True):
    if not is_pdf(path) and can_skip:
        return
    if output is None:
        output = re.sub(r'\.pdf$', '.png', path)
        # If this fails, then "path" doesn't end with .pdf.
        assert path != output
    if not shutil.which('pdftoppm'):
        if can_skip:
            if not silent:
                logger.warning(f"pdftoppm shell command not found; SKIP: \"pdftoppm {path} {output}\"")
            return
        raise RLScopeConfigurationError(f"pdftoppm shell command not found for: \"pdftoppm {path} {output}\".  Install with \"sudo apt install poppler-utils\"")
    with open(output, 'wb') as f:
        subprocess.check_call([
            'pdftoppm',
            # input
            path,
            '-png',
            # first page
            '-f', '1',
            # single page pdf
            '-singlefile',
        ], stdout=f)

