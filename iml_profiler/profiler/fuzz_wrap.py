import logging
import traceback
import textwrap

from iml_profiler.parser.common import *

class FuzzFuncWrapper:
    """
    We want to print out all the stack-trace locations where there is a
    call to session.run().

    We just intercept the TF_SessionRun_wrapper function to do this.
    """
    def __init__(self, func):
        # NOTE: to be as compatible as possible with intercepting existing code,
        # we forward setattr/getattr on this object back to the func we are wrapping
        # (e.g. func might be some weird SWIG object).
        super().__setattr__('func', func)

    def __call__(self, *args, **kwargs):
        logging.info(textwrap.dedent("""
        > IML: Call to session.run(...):
        {stack}
        """.format(stack=get_stacktrace())))
        ret = self.func(*args, **kwargs)
        return ret

    def __setattr__(self, name, value):
        return setattr(self.func, name, value)

    def __getattr__(self, name):
        return getattr(self.func, name)
