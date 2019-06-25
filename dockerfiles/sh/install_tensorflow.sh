#!/usr/bin/env bash
# NOTE: This should run OUTSIDE a docker container.
# This script is used to build AND start the docker container.
set -e
if [ "$DEBUG" == 'yes' ]; then
    set -x
fi
SH_DIR="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")"
cd "$SH_DIR"

source $SH_DIR/make_utils.sh

check_tensorflow_build

cd $TENSORFLOW_DIR
install_dir=$HOME/iml_tensorflow
if [ ! -d $install_dir ]; then
    mkdir -p $install_dir
    cd $install_dir
    ln -s $TENSORFLOW_DIR/bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/* .
    ln -s $TENSORFLOW_DIR/tensorflow/tools/pip_package/* .
fi
#echo "SKIP install_tensorflow: $install_dir already exists."
#exit 0

# Uninstall existing tensorflow install if there is on.
_do pip uninstall -y tensorflow || true

# Q: Should we use a container-local directory instead...?
_do cd $install_dir
_do python setup.py develop
