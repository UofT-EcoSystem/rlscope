"""
Determine if CUDA version and TensorFlow version are compatiable.
e.g., GPU hardware metrics require CUDA 10.1+, which requires TensorFlow v2.1.0+.
"""
#!/usr/bin/env python3
from rlscope.profiler.rlscope_logging import logger
import re
import argparse
import textwrap
import sys
import pprint
from io import StringIO

class TF_VersionInfo:
    def __init__(self, tf, py, compiler, bazel, cudnn, cuda):
        self.tf = tf
        self.py = py
        self.compiler = compiler
        self.bazel = bazel
        self.cudnn = cudnn
        self.cuda = cuda
        self.tf_version_number = self._get_tf_version_number()

    def _get_tf_version_number(self):
        m = re.search(
            r'(?P<prefix>tensorflow(?:_gpu)?)-(?P<version>(?:\d+)(?:\.\d+)+)',
            self.tf)
        assert m
        return m.group('version')

    @property
    def expect_string(self):
        return textwrap.dedent(f"""\
        Python version       = {self.py}
        Compiler version     = {self.compiler}
        Bazel version        = {self.bazel}
        cuDNN version        = {self.cudnn}
        CUDA toolkit version = {self.cuda}
        """).rstrip()

    def __repr__(self):
        ss = StringIO()

        ss.write('TF_VersionInfo(')

        ss.write(f"tf={self.tf}")

        ss.write(', ')
        ss.write(f"py={self.py}")

        ss.write(', ')
        ss.write(f"compiler={self.compiler}")

        ss.write(', ')
        ss.write(f"bazel={self.bazel}")

        ss.write(', ')
        ss.write(f"cudnn={self.cudnn}")

        ss.write(', ')
        ss.write(f"cuda={self.cuda}")

        ss.write(')')
        return ss.getvalue()

TF_VERSION_INFOS = []
TF_VERSION_TO_INFO = dict()
def _mk_version(*args):
    info = TF_VersionInfo(*args)
    assert info.tf_version_number not in TF_VERSION_TO_INFO
    TF_VERSION_TO_INFO[info.tf_version_number] = info
    TF_VERSION_INFOS.append(info)

# https://www.tensorflow.org/install/source#gpu
_mk_version('tensorflow-2.2.0', '3.5-3.8', 'GCC 7.3.1', 'Bazel 2.0.0', '7.6', '10.1')
_mk_version('tensorflow-2.1.0', '2.7, 3.5-3.7', 'GCC 7.3.1', 'Bazel 0.27.1', '7.6', '10.1')
_mk_version('tensorflow-2.0.0', '2.7, 3.3-3.7', 'GCC 7.3.1', 'Bazel 0.26.1', '7.4', '10.0')
_mk_version('tensorflow_gpu-1.14.0', '2.7, 3.3-3.7', 'GCC 4.8', 'Bazel 0.24.1', '7.4', '10.0')
_mk_version('tensorflow_gpu-1.13.1', '2.7, 3.3-3.7', 'GCC 4.8', 'Bazel 0.19.2', '7.4', '10.0')
_mk_version('tensorflow_gpu-1.12.0', '2.7, 3.3-3.6', 'GCC 4.8', 'Bazel 0.15.0', '7', '9')
_mk_version('tensorflow_gpu-1.11.0', '2.7, 3.3-3.6', 'GCC 4.8', 'Bazel 0.15.0', '7', '9')
_mk_version('tensorflow_gpu-1.10.0', '2.7, 3.3-3.6', 'GCC 4.8', 'Bazel 0.15.0', '7', '9')
_mk_version('tensorflow_gpu-1.9.0', '2.7, 3.3-3.6', 'GCC 4.8', 'Bazel 0.11.0', '7', '9')
_mk_version('tensorflow_gpu-1.8.0', '2.7, 3.3-3.6', 'GCC 4.8', 'Bazel 0.10.0', '7', '9')
_mk_version('tensorflow_gpu-1.7.0', '2.7, 3.3-3.6', 'GCC 4.8', 'Bazel 0.9.0', '7', '9')
_mk_version('tensorflow_gpu-1.6.0', '2.7, 3.3-3.6', 'GCC 4.8', 'Bazel 0.9.0', '7', '9')
_mk_version('tensorflow_gpu-1.5.0', '2.7, 3.3-3.6', 'GCC 4.8', 'Bazel 0.8.0', '7', '9')
_mk_version('tensorflow_gpu-1.4.0', '2.7, 3.3-3.6', 'GCC 4.8', 'Bazel 0.5.4', '6', '8')
_mk_version('tensorflow_gpu-1.3.0', '2.7, 3.3-3.6', 'GCC 4.8', 'Bazel 0.4.5', '6', '8')
_mk_version('tensorflow_gpu-1.2.0', '2.7, 3.3-3.6', 'GCC 4.8', 'Bazel 0.4.5', '5.1', '8')
_mk_version('tensorflow_gpu-1.1.0', '2.7, 3.3-3.6', 'GCC 4.8', 'Bazel 0.4.2', '5.1', '8')
_mk_version('tensorflow_gpu-1.0.0', '2.7, 3.3-3.6', 'GCC 4.8', 'Bazel 0.4.2', '5.1', '8')

# pprint.pprint(TF_VERSION_TO_INFO)

def main():
    parser = argparse.ArgumentParser("Check TensorFlow is installed with GPU support enabled")
    args = parser.parse_args()

    try:
        import tensorflow as tf
    except ImportError as e:
        print(textwrap.dedent("""\
        ERROR: TensorFlow isn't installed (\"import tensorflow\" failed)
        """).rstrip(), file=sys.stderr)
        sys.exit(1)

    gpu_supported = tf.test.is_gpu_available(cuda_only=True)
    if not gpu_supported:
        print(textwrap.dedent("""\
        
        ERROR: CUDA GPU support is NOT enabled for TensorFlow.
          This probably means you have the WRONG CUDA version installed for this version of TensorFlow.
          Each TensorFlow version only works with a SPECIFIC CUDA version.
          Look at the above error message to determine which CUDA version you should be using.
          
          To view which CUDA toolkit and cuDNN TensorFlow supports see:
              https://www.tensorflow.org/install/source#gpu
          You're currently running TensorFlow version {tf_version}.
        """).rstrip().format(
            tf_version=tf.__version__,
        ), file=sys.stderr)

        if tf.__version__ in TF_VERSION_TO_INFO:
            info = TF_VERSION_TO_INFO[tf.__version__]
            print(textwrap.dedent("""\
              TensorFlow {tf_version} expects the following:
              {tf_expect}
            """).rstrip().format(
                tf_version=tf.__version__,
                tf_expect=textwrap.indent(info.expect_string, prefix='  '),
            ), file=sys.stderr)

        sys.exit(1)

    # Everything looks fine.
    print("> Success :: TensorFlow installation supports GPU execution.")

if __name__ == '__main__':
    main()
