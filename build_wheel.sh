#!/usr/bin/env bash
#
# Run this script to build a .whl file in dist/*.whl.
# You can (re)install it using:
# $ pip install --upgrade dist/*.whl
#
set -e
set -x

IML_DIR="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")"
cd "$IML_DIR"

python setup.py sdist bdist_wheel --verbose --debug
