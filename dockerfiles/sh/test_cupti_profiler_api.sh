#!/usr/bin/env bash
# Test that docker host is configured properly.
# In particular, we need --runtime=nvidia so we can launch the RL-Scope container using docker-compose.
#
# NOTE: This should run inside a docker container.
set -e
if [ "$DEBUG" == 'yes' ]; then
    set -x
fi

SH_DIR="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")"
source $SH_DIR/make_utils.sh

main() {

  # Test GPU HW profiling (required by RL-Scope).
  cd /usr/local/cuda/extras/CUPTI/samples/extensions/src/profilerhost_util
  make
  cd /usr/local/cuda/extras/CUPTI/samples/userrange_profiling
  make
  cp -r /usr/local/cuda/extras/CUPTI/samples /home/${IML_USER}/CUPTI-samples
  chown -R ${IML_USER}:${IML_USER} /home/${IML_USER}/CUPTI-samples
  # Fails with error 35 during docker build...
  # This happens when we haven't set "NVreg_RestrictProfilingToAdminUsers=0"
  set +e
  (
    set -e
    cd /home/${IML_USER}/CUPTI-samples/userrange_profiling
    local cmd="LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64 ./userRangeSample"
    echo "> CMD:"
    echo "  $ $cmd"
    echo "  PWD=$PWD"
    echo "  USER=${IML_USER}"
    runuser ${IML_USER} -c "$cmd"
  )
  local ret=$?
  set -e

#  if true; then
  if [ "$ret" != 0 ]; then
    (
      TXT_UNDERLINE=yes
      TXT_BOLD=yes
      log_error "> RL-Scope nvidia driver host configuration error:"
    )
    log_error "You must configure the nvidia driver kernel module to allow GPU hardware profiler counters by non-admin users."
    log_error "To fix this, do the following:"
    log_error "  (1) Paste the following contents into /etc/modprobe.d/nvidia-profiler.conf:"
    read -r -d '' MODPROBE_NVIDIA << EOF || true
      options nvidia NVreg_RestrictProfilingToAdminUsers=0
EOF
    log_error "      $MODPROBE_NVIDIA"
    log_error
    log_error "  (2) Reboot the machine for the changes to take effect:"
    log_error "      $ sudo reboot now"
    exit 1
  else
      log_info "> RL-Scope nvidia driver host configuration looks correct."
  fi

}

main "$@"

