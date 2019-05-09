#!/usr/bin/env bash
# NOTE: This should run OUTSIDE a docker container.
# This script is used to build AND start the docker container.
set -e
set -x
SH_DIR="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")"
cd "$SH_DIR"

source $SH_DIR/make_utils.sh

_check_env

cd $TENSORFLOW_DIR
install_dir=$HOME/iml_tensorflow
if [ -d $install_dir ]; then
    echo "SKIP install_tensorflow: $install_dir already exists."
    exit 0
fi

# Q: Should we use a container-local directory instead...?
mkdir -p $install_dir
cd $install_dir
ln -s $TENSORFLOW_DIR/bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/* .
ln -s $TENSORFLOW_DIR/tensorflow/tools/pip_package/* .
python setup.py develop
