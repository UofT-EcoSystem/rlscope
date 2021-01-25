#!/usr/bin/env bash
# Test that docker host is configured properly.
# In particular, we need --runtime=nvidia so we can launch the RL-Scope container using docker-compose.
#
# NOTE: This should run inside a docker container.
set -e
if [ "$DEBUG" == 'yes' ]; then
    set -x
fi

IS_ZSH="$(ps -ef | grep $$ | grep -v --perl-regexp 'grep|ps ' | grep zsh --quiet && echo yes || echo no)"
if [ "$IS_ZSH" = 'yes' ]; then
  SH_DIR="$(readlink -f "$(dirname "${0:A}")")"
else
  SH_DIR="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")"
fi
source $SH_DIR/docker_build_common.sh

main() {
  has_nvidia_smi=no
  errmsg=
  if which nvidia-smi > /dev/null 2>&1; then
    has_nvidia_smi=yes
  else
    has_nvidia_smi=no
    errmsg="Couldn't locate 'nvidia-smi' on PATH."
  fi

  nvidia_smi_size_bytes=
  if [ "$has_nvidia_smi" = 'yes' ]; then
    nvidia_smi_size_bytes="$(stat --printf="%s" $(which nvidia-smi))"
    if [ "$nvidia_smi_size_bytes" = 0 ]; then
      errmsg="Found 'nvidia-smi' on PATH, but it is an empty file."
    fi
  fi
#  if true; then
  if [ "$has_nvidia_smi" = 'no' ] || [ "$nvidia_smi_size_bytes" = 0 ]; then
    (
      TXT_UNDERLINE=yes
      TXT_BOLD=yes
      log_error "> RL-Scope :: docker host configuration error:"
    )
    log_error "$errmsg"
    log_error "This likely means you have not properly configured the host docker to use --runtime=nvidia as the default docker runtime."
    log_error "To fix this, do the following:"
    log_error "  (1) Stop docker:"
    log_error "      $ sudo service docker stop"
    log_error
    log_error "  (2) Paste the following contents into /etc/docker/daemon.json:"
    read -r -d '' DOCKER_JSON << EOF || true
      {
        "default-runtime": "nvidia",
        "runtimes": {
          "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
          }
        }
      }
EOF
    log_error "      $DOCKER_JSON"
    log_error
    log_error "  (3) Start docker:"
    log_error "      $ sudo service docker start"
    exit 1
  else
      log_info "> RL-Scope :: docker host configuration looks correct."
  fi

}

main "$@"

