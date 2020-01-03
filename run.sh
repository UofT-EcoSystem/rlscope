#!/usr/bin/env bash
#read -r -d '' BASH_SCRIPT << EOM
#cd \$IML_DIR;
#source source_me.sh;
#"$@";
#EOM
#
#docker exec --user $USER -i -t dockerfiles_bash_1 /bin/bash -c "$BASH_SCRIPT"

read -r -d '' BASH_SCRIPT << EOM
cd \$IML_DIR;
source source_me.sh;
EOM

docker exec --user $USER -i -t dockerfiles_bash_1 /bin/bash -c "$BASH_SCRIPT" -c "$@"
