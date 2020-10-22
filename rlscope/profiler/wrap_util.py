"""
Functions for iterating over symbols in Python native library objects,
and wrapping/unwrapping them with a callable class.

See also
---------
rlscope.clib_wrap.CFuncWrapper : callable wrapper class around native functions for recording start/end timestamps.
rlscope.clib_wrap : uses wrap_util functions to wrap TensorFlow, PyTorch, and simulator native libraries.
"""
import re
import inspect
import types

from rlscope.profiler.rlscope_logging import logger

from rlscope import py_config

def wrap_lib(FuncWrapperKlass, import_libname, wrapper_args=tuple(), func_regex=None, wrap_libname=None):
    # wrapper_args = (category, prefix)
    if wrap_libname is None:
        wrap_libname = import_libname
    lib = None
    try:

        exec("import {import_lib}".format(import_lib=import_libname))
        lib = eval("{wrap_lib}".format(wrap_lib=wrap_libname))
        assert lib is not None

        # import tensorflow.pywrap_tensorflow
        if py_config.DEBUG_WRAP_CLIB:
            logger.info('  ... success')
    except (ImportError, NameError) as e:
        # Failed to import library; skip wrapping the library.
        logger.info('  ... FAILED: cannot wrap module {lib}; stacktrace:'.format(lib=wrap_libname))
        logger.info(e)
        return False
    wrap_module(FuncWrapperKlass, lib, wrapper_args, func_regex=func_regex)
    return True

def unwrap_lib(FuncWrapperKlass, import_libname, wrap_libname):
    try:
        if py_config.DEBUG_WRAP_CLIB:
            logger.info('> lookup {libname}...'.format(libname=wrap_libname))

        exec("import {import_lib}".format(import_lib=import_libname))
        # May throw NameError
        lib = eval("{wrap_lib}".format(wrap_lib=wrap_libname))

        if py_config.DEBUG_WRAP_CLIB:
            logger.info('  ... success')
    except NameError as e:
        logger.info('  ... FAILED: cannot unwrap module {lib}; stacktrace:'.format(lib=wrap_libname))
        logger.info(e)
        return
    unwrap_module(FuncWrapperKlass, lib)

def is_builtin(func):
    return isinstance(func, types.BuiltinFunctionType)

def wrap_module(FuncWrapperKlass, module,
                wrapper_args=None,
                func_regex=None, ignore_func_regex="^_", should_wrap=None):
    for name in dir(module):
        wrap_func(FuncWrapperKlass, module, name, wrapper_args,
                  func_regex=func_regex, ignore_func_regex=ignore_func_regex, should_wrap=should_wrap)

_EMPTY_ARGS = tuple()
def wrap_func(FuncWrapperKlass, module, name,
              wrapper_args=None,
              func_regex=None, ignore_func_regex="^_", should_wrap=None):
    # for name in dir(module):
    if wrapper_args is None:
        wrapper_args = _EMPTY_ARGS
    if re.search(ignore_func_regex, name):
        if py_config.DEBUG_WRAP_CLIB:
            logger.info("  Skip func={name}".format(name=name))
        return
    func = getattr(module, name)
    if type(func) == FuncWrapperKlass or not callable(func):
        return
    if func_regex is not None and not re.search(func_regex, name):
        return
    if should_wrap is not None and not should_wrap(name, func):
        return
    if inspect.isclass(func) or inspect.ismodule(func):
        logger.warning("Cannot wrap {module}.{name} since it's not a function: {value}".format(
            module=module.__name__,
            name=name,
            value=func,
        ))
        return

    func_wrapper = FuncWrapperKlass(func, *wrapper_args)
    setattr(module, name, func_wrapper)

def unwrap_module(FuncWrapperKlass, module):
    for name in dir(module):
        unwrap_func(FuncWrapperKlass, module, name)

def unwrap_func(FuncWrapperKlass, module, name):
    # if re.search(func_regex, name):
    func_wrapper = getattr(module, name)
    if type(func_wrapper) != FuncWrapperKlass:
        return
    func = func_wrapper.func
    setattr(module, name, func)
