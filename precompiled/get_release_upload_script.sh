#!/usr/bin/env bash
set -e
if [ "$DEBUG" == 'yes' ]; then
    set -x
fi
PRECOMPILED_DIR="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")"
cd "$PRECOMPILED_DIR"

if [ ! -e linux-amd64-github-release.tar.bz2 ]; then
    wget https://github.com/aktau/github-release/releases/download/v0.7.2/linux-amd64-github-release.tar.bz2
fi
if [ ! -e github-release ]; then
    mkdir -p github_release 
    cd github_release 
    tar xf ../linux-amd64-github-release.tar.bz2
    mv bin/linux/amd64/github-release ../
fi
echo " > Extracted github-release executable to $PWD/github-release"
echo "   You can use this to upload precompiled tensorflow wheel builds private github repos; "
echo "   in particular, https://github.com/UofT-EcoSystem/iml/releases"
echo
echo "   Example invocation:"
echo
echo "   # Upload wheel file for tag 1.0.0"
echo "   # It will appear under https://github.com/UofT-EcoSystem/iml/releases/tag/1.0.0"
echo "   $ GITHUB_TOKEN=\"...\" ./github-release upload \\"
echo "       --user UofT-EcoSystem \\"
echo "       --repo iml \\"
echo "       --tag 1.0.0 \\"
echo "       --name tensorflow-1.13.1-cp35-cp35m-linux_x86_64.whl \\"
echo "       --file tensorflow-1.13.1-cp35-cp35m-linux_x86_64.whl \\"
echo
echo "   NOTE: You need to setup a github API token by visiting https://github.com/settings/tokens"
echo "   For help with github-release, see https://github.com/aktau/github-release#how-to-use"
