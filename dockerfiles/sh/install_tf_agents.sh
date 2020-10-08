#!/usr/bin/env bash
# NOTE: This should run inside a docker container.
#
# NOTE: It's up to you to call ./configure from inside the TENSORFLOW_DIR of the container!
# This script is just meant for doing iterative rebuilds (i.e. "make")
set -e
DEBUG=${DEBUG:-no}
if [ "$DEBUG" == 'yes' ]; then
    set -x
fi
SH_DIR="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")"
cd "$SH_DIR"

source $SH_DIR/make_utils.sh

_check_env
_upgrade_pip

_check_TF_AGENTS_DIR

_check_tensorflow
_check_iml

set -u

#if [ "${STABLE_BASELINES_DIR}" = "" ]; then
#    # Install directly from git repo.
#    _do pip install git+https://github.com/jagleeso/stable-baselines.git@iml
#else
# Install from local checkout of repo.
_do cd "${TF_AGENTS_DIR}"
_do python setup.py develop
#fi

# For some reason, v0.5 "requires" tensorflow probability that's designed for TensorFlow v2.3+
#_do pip install tensorflow-probability==0.10.0
# ... actually I got an error with tensorflow-probability==0.10.0 ...
#   Traceback (most recent call last):
#     File "/home/jgleeson/clone/agents/tf_agents/agents/ddpg/examples/v2/train_eval.rlscope.py", line 49, in <module>
#       from tf_agents.agents.ddpg import actor_network
#     File "/home/jgleeson/clone/agents/tf_agents/agents/__init__.py", line 17, in <module>
#       from tf_agents.agents import tf_agent
#     File "/home/jgleeson/clone/agents/tf_agents/agents/tf_agent.py", line 26, in <module>
#       from tf_agents.specs import tensor_spec
#     File "/home/jgleeson/clone/agents/tf_agents/specs/__init__.py", line 20, in <module>
#       from tf_agents.specs.distribution_spec import DistributionSpec
#     File "/home/jgleeson/clone/agents/tf_agents/specs/distribution_spec.py", line 22, in <module>
#       import tensorflow_probability as tfp
#     File "/root/venv/lib/python3.6/site-packages/tensorflow_probability-0.10.0-py3.6.egg/tensorflow_probability/__init__.py", line 76, in <module>
#       from tensorflow_probability.python import *  # pylint: disable=wildcard-import
#     File "/root/venv/lib/python3.6/site-packages/tensorflow_probability-0.10.0-py3.6.egg/tensorflow_probability/python/__init__.py", line 23, in <module>
#       from tensorflow_probability.python import distributions
#     File "/root/venv/lib/python3.6/site-packages/tensorflow_probability-0.10.0-py3.6.egg/tensorflow_probability/python/distributions/__init__.py", line 88, in <module>
#       from tensorflow_probability.python.distributions.pixel_cnn import PixelCNN
#     File "/root/venv/lib/python3.6/site-packages/tensorflow_probability-0.10.0-py3.6.egg/tensorflow_probability/python/distributions/pixel_cnn.py", line 37, in <module>
#       from tensorflow_probability.python.layers import weight_norm
#     File "/root/venv/lib/python3.6/site-packages/tensorflow_probability-0.10.0-py3.6.egg/tensorflow_probability/python/layers/__init__.py", line 31, in <module>
#       from tensorflow_probability.python.layers.distribution_layer import CategoricalMixtureOfOneHotCategorical
#     File "/root/venv/lib/python3.6/site-packages/tensorflow_probability-0.10.0-py3.6.egg/tensorflow_probability/python/layers/distribution_layer.py", line 28, in <module>
#       from cloudpickle.cloudpickle import CloudPickler
#   ImportError: cannot import name 'CloudPickler'

_do pip install tensorflow-probability==0.9.0
