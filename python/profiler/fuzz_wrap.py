import traceback
import textwrap

from parser.common import *

def get_stacktrace(n_skip):
    """
    Return a stacktrace ready for printing; usage:

    # Dump the stack of the current location of this line.
    '\n'.join(get_stacktrace(0))

    :param n_skip:
        Number of stack-levels to skip in the caller.

    :return:
    """
    # We want to skip "get_stacktrace", plus whatever the caller wishes to skip.
    n_skip_total = n_skip + 1
    stack = traceback.extract_stack()
    keep_stack = stack[n_skip_total:]
    return traceback.format_list(keep_stack)

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
        stacktrace = get_stacktrace()
        print(textwrap.dedent("""
        > IML: Call to session.run(...):
        {stack}
        """.format(stack=textwrap.indent(
            '\n'.join(stacktrace),
            prefix="  "))))
        ret = self.func(*args, **kwargs)
        return ret

    def __setattr__(self, name, value):
        return setattr(self.func, name, value)

    def __getattr__(self, name):
        return getattr(self.func, name)
