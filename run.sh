#!/usr/bin/env bash
read -r -d '' BASH_SCRIPT << EOM
cd \$IML_DIR;
$@;
EOM

docker exec --user $USER -i -t dockerfiles_bash_1 /bin/bash -c "$BASH_SCRIPT"
