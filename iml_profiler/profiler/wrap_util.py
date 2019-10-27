import re
import logging

from iml_profiler import py_config

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
            logging.info('  ... success')
    except (ImportError, NameError) as e:
        # Failed to import library; skip wrapping the library.
        logging.info('  ... FAILED: cannot wrap module {lib}; stacktrace:'.format(lib=wrap_libname))
        logging.info(e)
        return False
    wrap_module(FuncWrapperKlass, lib, wrapper_args, func_regex=func_regex)
    return True

def unwrap_lib(FuncWrapperKlass, import_libname, wrap_libname):
    try:
        if py_config.DEBUG_WRAP_CLIB:
            logging.info('> lookup {libname}...'.format(libname=wrap_libname))

        exec("import {import_lib}".format(import_lib=import_libname))
        # May throw NameError
        lib = eval("{wrap_lib}".format(wrap_lib=wrap_libname))

        if py_config.DEBUG_WRAP_CLIB:
            logging.info('  ... success')
    except NameError as e:
        logging.info('  ... FAILED: cannot unwrap module {lib}; stacktrace:'.format(lib=wrap_libname))
        logging.info(e)
        return
    unwrap_module(FuncWrapperKlass, lib)

def wrap_module(FuncWrapperKlass, module, wrapper_args,
                func_regex=None, ignore_func_regex="^_", should_wrap=None):
    for name in dir(module):
        if re.search(ignore_func_regex, name):
            if py_config.DEBUG_WRAP_CLIB:
                logging.info("  Skip func={name}".format(name=name))
            continue
        func = getattr(module, name)
        if not re.search(ignore_func_regex, name) and (
                (
                        func_regex is None or re.search(func_regex, name)
                ) or (
                        should_wrap is not None and should_wrap(name, func)
                )
        ):
            if type(func) == FuncWrapperKlass or not callable(func):
                continue
            func_wrapper = FuncWrapperKlass(func, *wrapper_args)
            setattr(module, name, func_wrapper)

def unwrap_module(FuncWrapperKlass, module):
    for name in dir(module):
        # if re.search(func_regex, name):
        func_wrapper = getattr(module, name)
        if type(func_wrapper) != FuncWrapperKlass:
            continue
        func = func_wrapper.func
        setattr(module, name, func)

