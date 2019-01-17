import re

DEBUG = True

def wrap_lib(FuncWrapperKlass, import_libname, wrapper_args=tuple(), func_regex=None, wrap_libname=None):
    # wrapper_args = (category, prefix)
    if wrap_libname is None:
        wrap_libname = import_libname
    lib = None
    try:

        # if DEBUG:
        #     print('> import {libname}...'.format(libname=import_libname))
        # importlib.import_module(import_libname)

        # May throw NameError
        # stmt = "import {import_lib}; lib = {wrap_lib}".format(import_lib=import_libname, wrap_lib=wrap_libname)
        # lib = eval(wrap_libname)
        # lib = eval(stmt)
        exec("import {import_lib}".format(import_lib=import_libname))
        # exec("lib = {wrap_lib}".format(wrap_lib=wrap_libname))
        lib = eval("{wrap_lib}".format(wrap_lib=wrap_libname))
        # exec(stmt)
        assert lib is not None

        # import tensorflow.pywrap_tensorflow
        if DEBUG:
            print('  ... success')
    except (ImportError, NameError) as e:
        # Failed to import library; skip wrapping the library.
        if DEBUG:
            print('  ... FAILED: cannot wrap module {lib}; stacktrace:'.format(lib=wrap_libname))
            print(e)
        return False
    wrap_module(FuncWrapperKlass, lib, wrapper_args, func_regex=func_regex)
    return True

def unwrap_lib(FuncWrapperKlass, wrap_libname):
    try:
        if DEBUG:
            print('> lookup {libname}...'.format(libname=wrap_libname))
        # May throw NameError
        lib = eval(wrap_libname)
        if DEBUG:
            print('  ... success')
    except NameError as e:
        if DEBUG:
            print('  ... FAILED: cannot unwrap module {lib}; stacktrace:'.format(lib=wrap_libname))
            print(e)
        return
    unwrap_module(FuncWrapperKlass, lib)

def wrap_module(FuncWrapperKlass, module, wrapper_args,
                func_regex=None, ignore_func_regex="^_"):
    for name in dir(module):
        if not re.search(ignore_func_regex, name) and (
            func_regex is None or re.search(func_regex, name)
        ):
            func = getattr(module, name)
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

