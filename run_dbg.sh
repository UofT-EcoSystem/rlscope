#!/usr/bin/env bash
# read -r -d '' BASH_SCRIPT << EOM
# cd \$IML_DIR;
# $@;
# EOM
#
# docker exec --user $USER -i -t dockerfiles_bash_1 /bin/bash -c "$BASH_SCRIPT"
set -e
if [ "$DEBUG" = 'yes' ]; then
    set -x
fi
cd $(dirname $0)
#python ./run.py bash -c "cd Debug && make -j1 VERBOSE=1"
#python ./run.py bash -c "cd Debug && make -j$(nproc)"
#python ./run.py bash -c "cd Debug && make -j1"
python ./run.py bash -c "cd Debug && make -j8"
python ./run.py "$@"
# Debug/cpp_dump_proto --mode overlap --iml_directory output/perf_debug
