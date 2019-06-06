# ale_python_interface.py
# Author: Ben Goodrich
# This directly implements a python version of the arcade learning
# environment interface.
__all__ = ['PythonInterface']

import logging
from ctypes import *
import numpy as np
from numpy.ctypeslib import as_ctypes
import os

from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b

from iml_profiler import py_config

# https://stackoverflow.com/questions/9386636/profiling-a-system-with-extensively-reused-decorators
import types

# NOTE: Looks like all the C calls are now being grouped into a single function name.
#
#  ncalls  tottime  percall  cumtime  percall filename:lineno(function)
#       2 3.354368489 1.6771842445 3.354368489 1.6771842445 /mnt/data/james/clone/dnn_tensorflow_cpp/python/test/py_interface.py:43(__call__)
#       1 0.793662559 0.793662559 1.00004555 1.00004555 /mnt/data/james/clone/dnn_tensorflow_cpp/python/test/test_call_c.py:76(run_python)
# 3162558 0.20635535400000002 6.524950815131297e-08 0.20635535400000002 6.524950815131297e-08 {built-in method time.time}
#       2 4.3299e-05 2.16495e-05 4.3299e-05 2.16495e-05 {built-in method builtins.print}
#       1 2.0474e-05 2.0474e-05 2.3543471250000003 2.3543471250000003 /mnt/data/james/clone/dnn_tensorflow_cpp/python/test/py_interface.py:202(gpu_sleep)
#       2 1.2456e-05 6.228e-06 1.2456e-05 6.228e-06 {method 'format' of 'str' objects}
#       1 1.2454000000000001e-05 1.2454000000000001e-05 2.354387697 2.354387697 /mnt/data/james/clone/dnn_tensorflow_cpp/python/test/test_call_c.py:67(run_gpu)
#       1 1.0382e-05 1.0382e-05 1.0588000000000001e-05 1.0588000000000001e-05 /home/james/clone/clone/baselines/baselines/deepq/simple_refactor.py:3155(stop)
#       1 7.06e-06 7.06e-06 1.000048898 1.000048898 /mnt/data/james/clone/dnn_tensorflow_cpp/python/test/py_interface.py:205(run_cpp)
#       1 4.26e-06 4.26e-06 1.000053158 1.000053158 /mnt/data/james/clone/dnn_tensorflow_cpp/python/test/test_call_c.py:72(run_cpp)
#       1 4.002e-06 4.002e-06 1.4590000000000001e-05 1.4590000000000001e-05 /mnt/data/james/clone/dnn_tensorflow_cpp/python/test/test_call_c.py:110(disable_profiling)
#       1 2.0600000000000002e-07 2.0600000000000002e-07 2.0600000000000002e-07 2.0600000000000002e-07 {method 'disable' of '_lsprof.Profiler' objects}

# What we really want (separate __call__ into CLIB__gpu_sleep and CLIB__run_cpp):
#
# ncalls  tottime  percall  cumtime  percall filename:lineno(function)
# 1 2.334516991 2.334516991 2.334516991 2.334516991 /mnt/data/james/clone/dnn_tensorflow_cpp/python/test/py_interface.py:55(CLIB__gpu_sleep)
# 1 1.0000441310000001 1.0000441310000001 1.0000441310000001 1.0000441310000001 /mnt/data/james/clone/dnn_tensorflow_cpp/python/test/py_interface.py:55(CLIB__run_cpp)

class CLibWrapper:
  def __init__(self, name, LibraryLoader=cdll, prefix="CLIB__"):
    self.LibraryLoader = LibraryLoader
    self.lib = self.LibraryLoader.LoadLibrary(name)
    self.prefix = prefix

    class MyFuncWrapper:
      def __init__(self, c_func, prefix):
        # self.c_func = c_func
        super().__setattr__('c_func', c_func)
        super().__setattr__('prefix', prefix)

        # # wrapper_02
        # def call(*args, **kwargs):
        #   return self.c_func(*args, **kwargs)
        # logging.info("> call.name = {name}".format(name=c_func.__name__))
        # call.__name__ = c_func.__name__
        # super().__setattr__('call', call)

        # wrapper_03
        def call(*args, **kwargs):
          # NOTE: for some reason, if we don't use a local variable here,
          # it will return None!  Bug in python3...?
          ret = self.c_func(*args, **kwargs)
          return ret

        #
        # If we comment this out, then "call" returns the gpu_sleep time...
        # Not sure WHY.
        #

        name = self.wrapper_name(c_func.__name__)
        logging.info("> call.name = {name}".format(name=name))
        # c = call.func_code
        # Python3
        c = call.__code__

        # https://docs.python.org/3.0/whatsnew/3.0.html
        #
        # """
        # The function attributes named func_X have been renamed to use the __X__ form,
        # freeing up these names in the function attribute namespace for user-defined attributes.
        # To wit, func_closure, func_code, func_defaults, func_dict, func_doc, func_globals,
        # func_name were renamed to __closure__, __code__, __defaults__, __dict__, __doc__, __globals__,
        # __name__, respectively.
        # """

        new_code = types.CodeType(
          c.co_argcount, c.co_kwonlyargcount, c.co_nlocals, c.co_stacksize,
          c.co_flags, c.co_code, c.co_consts,
          c.co_names, c.co_varnames, c.co_filename,
          name, c.co_firstlineno, c.co_lnotab,
          c.co_freevars, c.co_cellvars
        )
        call = types.FunctionType(
          new_code, call.__globals__, name, call.__defaults__,
          call.__closure__)

        super().__setattr__('call', call)

      def wrapper_name(self, name):
        return "{prefix}{name}".format(
          prefix=self.prefix,
          name=name)

      def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

      def __setattr__(self, name, value):
        return setattr(self.c_func, name, value)

      def __getattr__(self, name):
        return getattr(self.c_func, name)

    self.MyFuncWrapper = MyFuncWrapper

  def __getattr__(self, name):
    # NOTE: __getattr__ only ever gets called if the object doesn't have the attribute set yet.
    c_func = getattr(self.lib, name)
    c_func_wrapper = self.MyFuncWrapper(c_func, self.prefix)
    # https://stackoverflow.com/questions/13184281/python-dynamic-function-creation-with-custom-names
    # c_func_wrapper.__call__.__name__ = "MyFuncWrapper__{name}".format(name=c_func.__name__)
    # wrapper_01
    # c_func_wrapper.__name__ = "MyFuncWrapper__{name}".format(name=c_func.__name__)
    setattr(self, name, c_func_wrapper)
    return c_func_wrapper

  def __getitem__(self, name):
    return getattr(self, name)

  # def wrap_c_lib(self):


# py_lib = cdll.LoadLibrary(_j(py_config.BUILD_DIR, 'libpy_interface.so'))
py_lib = CLibWrapper(_j(py_config.BUILD_DIR, 'libpy_interface.so'))

py_lib.NewLibHandle.argtypes = None
py_lib.NewLibHandle.restype = c_void_p

py_lib.call_c.argtypes = [c_void_p]
py_lib.call_c.restype = None

py_lib.guess_gpu_freq_mhz.argtypes = [c_void_p]
py_lib.guess_gpu_freq_mhz.restype = c_double

py_lib.gpu_sleep.argtypes = [c_void_p, c_double]
py_lib.gpu_sleep.restype = c_double

py_lib.run_cpp.argtypes = [c_void_p, c_double]
py_lib.run_cpp.restype = c_double

py_lib.set_gpu_freq_mhz.argtypes = [c_void_p, c_double]
py_lib.set_gpu_freq_mhz.restype = None

# py_lib.ALE_new.argtypes = None
# py_lib.ALE_new.restype = c_void_p
# py_lib.ALE_del.argtypes = [c_void_p]
# py_lib.ALE_del.restype = None
# py_lib.getString.argtypes = [c_void_p, c_char_p]
# py_lib.getString.restype = c_char_p
# py_lib.getInt.argtypes = [c_void_p, c_char_p]
# py_lib.getInt.restype = c_int
# py_lib.getBool.argtypes = [c_void_p, c_char_p]
# py_lib.getBool.restype = c_bool
# py_lib.getFloat.argtypes = [c_void_p, c_char_p]
# py_lib.getFloat.restype = c_float
# py_lib.setString.argtypes = [c_void_p, c_char_p, c_char_p]
# py_lib.setString.restype = None
# py_lib.setInt.argtypes = [c_void_p, c_char_p, c_int]
# py_lib.setInt.restype = None
# py_lib.setBool.argtypes = [c_void_p, c_char_p, c_bool]
# py_lib.setBool.restype = None
# py_lib.setFloat.argtypes = [c_void_p, c_char_p, c_float]
# py_lib.setFloat.restype = None
# py_lib.loadROM.argtypes = [c_void_p, c_char_p]
# py_lib.loadROM.restype = None
# py_lib.act.argtypes = [c_void_p, c_int]
# py_lib.act.restype = c_int
# py_lib.game_over.argtypes = [c_void_p]
# py_lib.game_over.restype = c_bool
# py_lib.reset_game.argtypes = [c_void_p]
# py_lib.reset_game.restype = None
# py_lib.getAvailableModes.argtypes = [c_void_p, c_void_p]
# py_lib.getAvailableModes.restype = None
# py_lib.getAvailableModesSize.argtypes = [c_void_p]
# py_lib.getAvailableModesSize.restype = c_int
# py_lib.setMode.argtypes = [c_void_p, c_int]
# py_lib.setMode.restype = None
# py_lib.getAvailableDifficulties.argtypes = [c_void_p, c_void_p]
# py_lib.getAvailableDifficulties.restype = None
# py_lib.getAvailableDifficultiesSize.argtypes = [c_void_p]
# py_lib.getAvailableDifficultiesSize.restype = c_int
# py_lib.setDifficulty.argtypes = [c_void_p, c_int]
# py_lib.setDifficulty.restype = None
# py_lib.getLegalActionSet.argtypes = [c_void_p, c_void_p]
# py_lib.getLegalActionSet.restype = None
# py_lib.getLegalActionSize.argtypes = [c_void_p]
# py_lib.getLegalActionSize.restype = c_int
# py_lib.getMinimalActionSet.argtypes = [c_void_p, c_void_p]
# py_lib.getMinimalActionSet.restype = None
# py_lib.getMinimalActionSize.argtypes = [c_void_p]
# py_lib.getMinimalActionSize.restype = c_int
# py_lib.getFrameNumber.argtypes = [c_void_p]
# py_lib.getFrameNumber.restype = c_int
# py_lib.lives.argtypes = [c_void_p]
# py_lib.lives.restype = c_int
# py_lib.getEpisodeFrameNumber.argtypes = [c_void_p]
# py_lib.getEpisodeFrameNumber.restype = c_int
# py_lib.getScreen.argtypes = [c_void_p, c_void_p]
# py_lib.getScreen.restype = None
# py_lib.getRAM.argtypes = [c_void_p, c_void_p]
# py_lib.getRAM.restype = None
# py_lib.getRAMSize.argtypes = [c_void_p]
# py_lib.getRAMSize.restype = c_int
# py_lib.getScreenWidth.argtypes = [c_void_p]
# py_lib.getScreenWidth.restype = c_int
# py_lib.getScreenHeight.argtypes = [c_void_p]
# py_lib.getScreenHeight.restype = c_int
# py_lib.getScreenRGB.argtypes = [c_void_p, c_void_p]
# py_lib.getScreenRGB.restype = None
# py_lib.getScreenGrayscale.argtypes = [c_void_p, c_void_p]
# py_lib.getScreenGrayscale.restype = None
# py_lib.saveState.argtypes = [c_void_p]
# py_lib.saveState.restype = None
# py_lib.loadState.argtypes = [c_void_p]
# py_lib.loadState.restype = None
# py_lib.cloneState.argtypes = [c_void_p]
# py_lib.cloneState.restype = c_void_p
# py_lib.restoreState.argtypes = [c_void_p, c_void_p]
# py_lib.restoreState.restype = None
# py_lib.cloneSystemState.argtypes = [c_void_p]
# py_lib.cloneSystemState.restype = c_void_p
# py_lib.restoreSystemState.argtypes = [c_void_p, c_void_p]
# py_lib.restoreSystemState.restype = None
# py_lib.deleteState.argtypes = [c_void_p]
# py_lib.deleteState.restype = None
# py_lib.saveScreenPNG.argtypes = [c_void_p, c_char_p]
# py_lib.saveScreenPNG.restype = None
# py_lib.encodeState.argtypes = [c_void_p, c_void_p, c_int]
# py_lib.encodeState.restype = None
# py_lib.encodeStateLen.argtypes = [c_void_p]
# py_lib.encodeStateLen.restype = c_int
# py_lib.decodeState.argtypes = [c_void_p, c_int]
# py_lib.decodeState.restype = c_void_p
# py_lib.setLoggerMode.argtypes = [c_int]
# py_lib.setLoggerMode.restype = None

class PythonInterface(object):
  # # Logger enum
  # class Logger:
  #   Info = 0
  #   Warning = 1
  #   Error = 2

  def __init__(self):
    self.obj = py_lib.NewLibHandle()

  def call_c(self):
    return py_lib.call_c(self.obj)

  def guess_gpu_freq_mhz(self):
    return py_lib.guess_gpu_freq_mhz(self.obj)

  def gpu_sleep(self, seconds):
    return py_lib.gpu_sleep(self.obj, seconds)

  def run_cpp(self, seconds):
    return py_lib.run_cpp(self.obj, seconds)

  def set_gpu_freq_mhz(self, mhz):
    return py_lib.set_gpu_freq_mhz(self.obj, mhz)

  # def getString(self, key):
  #   return py_lib.getString(self.obj, key)
  # def getInt(self, key):
  #   return py_lib.getInt(self.obj, key)
  # def getBool(self, key):
  #   return py_lib.getBool(self.obj, key)
  # def getFloat(self, key):
  #   return py_lib.getFloat(self.obj, key)
  #
  # def setString(self, key, value):
  #   py_lib.setString(self.obj, key, value)
  # def setInt(self, key, value):
  #   py_lib.setInt(self.obj, key, value)
  # def setBool(self, key, value):
  #   py_lib.setBool(self.obj, key, value)
  # def setFloat(self, key, value):
  #   py_lib.setFloat(self.obj, key, value)
  #
  # def loadROM(self, rom_file):
  #   py_lib.loadROM(self.obj, rom_file)
  #
  # def act(self, action):
  #   return py_lib.act(self.obj, int(action))
  #
  # def game_over(self):
  #   return py_lib.game_over(self.obj)
  #
  # def reset_game(self):
  #   py_lib.reset_game(self.obj)
  #
  # def getLegalActionSet(self):
  #   act_size = py_lib.getLegalActionSize(self.obj)
  #   act = np.zeros((act_size), dtype=np.intc)
  #   py_lib.getLegalActionSet(self.obj, as_ctypes(act))
  #   return act
  #
  # def getMinimalActionSet(self):
  #   act_size = py_lib.getMinimalActionSize(self.obj)
  #   act = np.zeros((act_size), dtype=np.intc)
  #   py_lib.getMinimalActionSet(self.obj, as_ctypes(act))
  #   return act
  #
  # def getAvailableModes(self):
  #   modes_size = py_lib.getAvailableModesSize(self.obj)
  #   modes = np.zeros((modes_size), dtype=np.intc)
  #   py_lib.getAvailableModes(self.obj, as_ctypes(modes))
  #   return modes
  #
  # def setMode(self, mode):
  #   py_lib.setMode(self.obj, mode)
  #
  # def getAvailableDifficulties(self):
  #   difficulties_size = py_lib.getAvailableDifficultiesSize(self.obj)
  #   difficulties = np.zeros((difficulties_size), dtype=np.intc)
  #   py_lib.getAvailableDifficulties(self.obj, as_ctypes(difficulties))
  #   return difficulties
  #
  # def setDifficulty(self, difficulty):
  #   py_lib.setDifficulty(self.obj, difficulty)
  #
  # def getLegalActionSet(self):
  #   act_size = py_lib.getLegalActionSize(self.obj)
  #   act = np.zeros((act_size), dtype=np.intc)
  #   py_lib.getLegalActionSet(self.obj, as_ctypes(act))
  #   return act
  #
  # def getMinimalActionSet(self):
  #   act_size = py_lib.getMinimalActionSize(self.obj)
  #   act = np.zeros((act_size), dtype=np.intc)
  #   py_lib.getMinimalActionSet(self.obj, as_ctypes(act))
  #   return act
  #
  # def getFrameNumber(self):
  #   return py_lib.getFrameNumber(self.obj)
  #
  # def lives(self):
  #   return py_lib.lives(self.obj)
  #
  # def getEpisodeFrameNumber(self):
  #   return py_lib.getEpisodeFrameNumber(self.obj)
  #
  # def getScreenDims(self):
  #   """returns a tuple that contains (screen_width, screen_height)
  #   """
  #   width = py_lib.getScreenWidth(self.obj)
  #   height = py_lib.getScreenHeight(self.obj)
  #   return (width, height)
  #
  # def getScreen(self, screen_data=None):
  #   """This function fills screen_data with the RAW Pixel data
  #   screen_data MUST be a numpy array of uint8/int8. This could be initialized like so:
  #   screen_data = np.empty(w*h, dtype=np.uint8)
  #   Notice,  it must be width*height in size also
  #   If it is None,  then this function will initialize it
  #   Note: This is the raw pixel values from the atari,  before any RGB palette transformation takes place
  #   """
  #   if(screen_data is None):
  #     width = py_lib.getScreenWidth(self.obj)
  #     height = py_lib.getScreenHeight(self.obj)
  #     screen_data = np.zeros(width*height, dtype=np.uint8)
  #   py_lib.getScreen(self.obj, as_ctypes(screen_data))
  #   return screen_data
  #
  # def getScreenRGB(self, screen_data=None):
  #   """This function fills screen_data with the data in RGB format
  #   screen_data MUST be a numpy array of uint8. This can be initialized like so:
  #   screen_data = np.empty((height,width,3), dtype=np.uint8)
  #   If it is None,  then this function will initialize it.
  #   """
  #   if(screen_data is None):
  #     width = py_lib.getScreenWidth(self.obj)
  #     height = py_lib.getScreenHeight(self.obj)
  #     screen_data = np.empty((height, width,3), dtype=np.uint8)
  #   py_lib.getScreenRGB(self.obj, as_ctypes(screen_data[:]))
  #   return screen_data
  #
  # def getScreenGrayscale(self, screen_data=None):
  #   """This function fills screen_data with the data in grayscale
  #   screen_data MUST be a numpy array of uint8. This can be initialized like so:
  #   screen_data = np.empty((height,width,1), dtype=np.uint8)
  #   If it is None,  then this function will initialize it.
  #   """
  #   if(screen_data is None):
  #     width = py_lib.getScreenWidth(self.obj)
  #     height = py_lib.getScreenHeight(self.obj)
  #     screen_data = np.empty((height, width,1), dtype=np.uint8)
  #   py_lib.getScreenGrayscale(self.obj, as_ctypes(screen_data[:]))
  #   return screen_data
  #
  # def getRAMSize(self):
  #   return py_lib.getRAMSize(self.obj)
  #
  # def getRAM(self, ram=None):
  #   """This function grabs the atari RAM.
  #   ram MUST be a numpy array of uint8/int8. This can be initialized like so:
  #   ram = np.array(ram_size, dtype=uint8)
  #   Notice: It must be ram_size where ram_size can be retrieved via the getRAMSize function.
  #   If it is None,  then this function will initialize it.
  #   """
  #   if(ram is None):
  #     ram_size = py_lib.getRAMSize(self.obj)
  #     ram = np.zeros(ram_size, dtype=np.uint8)
  #   py_lib.getRAM(self.obj, as_ctypes(ram))
  #   return ram
  #
  # def saveScreenPNG(self, filename):
  #   """Save the current screen as a png file"""
  #   return py_lib.saveScreenPNG(self.obj, filename)
  #
  # def saveState(self):
  #   """Saves the state of the system"""
  #   return py_lib.saveState(self.obj)
  #
  # def loadState(self):
  #   """Loads the state of the system"""
  #   return py_lib.loadState(self.obj)
  #
  # def cloneState(self):
  #   """This makes a copy of the environment state. This copy does *not*
  #   include pseudorandomness, making it suitable for planning
  #   purposes. By contrast, see cloneSystemState.
  #   """
  #   return py_lib.cloneState(self.obj)
  #
  # def restoreState(self, state):
  #   """Reverse operation of cloneState(). This does not restore
  #   pseudorandomness, so that repeated calls to restoreState() in
  #   the stochastic controls setting will not lead to the same
  #   outcomes.  By contrast, see restoreSystemState.
  #   """
  #   py_lib.restoreState(self.obj, state)
  #
  # def cloneSystemState(self):
  #   """This makes a copy of the system & environment state, suitable for
  #   serialization. This includes pseudorandomness and so is *not*
  #   suitable for planning purposes.
  #   """
  #   return py_lib.cloneSystemState(self.obj)
  #
  # def restoreSystemState(self, state):
  #   """Reverse operation of cloneSystemState."""
  #   py_lib.restoreSystemState(self.obj, state)
  #
  # def deleteState(self, state):
  #   """ Deallocates the ALEState """
  #   py_lib.deleteState(state)
  #
  # def encodeStateLen(self, state):
  #   return py_lib.encodeStateLen(state)
  #
  # def encodeState(self, state, buf=None):
  #   if buf == None:
  #     length = py_lib.encodeStateLen(state)
  #     buf = np.zeros(length, dtype=np.uint8)
  #   py_lib.encodeState(state, as_ctypes(buf), c_int(len(buf)))
  #   return buf
  #
  # def decodeState(self, serialized):
  #   return py_lib.decodeState(as_ctypes(serialized), len(serialized))
  #
  # def __del__(self):
  #   py_lib.ALE_del(self.obj)
  #
  # @staticmethod
  # def setLoggerMode(mode):
  #   dic = {'info': 0, 'warning': 1, 'error': 2}
  #   mode = dic.get(mode, mode)
  #   assert mode in [0, 1, 2], "Invalid Mode! Mode must be one of 0: info, 1: warning, 2: error"
  #   py_lib.setLoggerMode(mode)
