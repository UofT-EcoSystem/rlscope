#!/usr/bin/env bash
#
# Install apt-get dependencies used by RL-Scope.
# NOTE: These package names are based on ubuntu 20.04 and 18.04.
#

main() {
  APT_DEPENDENCIES=(
    # Needed for 'pdfcrop' command when generating plots.
    texlive-extra-utils
  )
  sudo apt-get install -y "${APT_DEPENDENCIES[@]}"
}

main "$@"