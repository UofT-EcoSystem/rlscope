#!/usr/bin/env bash
# This file gets automatically source'd by interactive bash shells.
# Useful for adding custom commands, and adding stuff to the default
# Ubuntu bashrc (see _bash_rc).

# $ man terminfo
# Color       #define       Value       RGB
# black     COLOR_BLACK       0     0, 0, 0
# red       COLOR_RED         1     max,0,0
# green     COLOR_GREEN       2     0,max,0
# yellow    COLOR_YELLOW      3     max,max,0
# blue      COLOR_BLUE        4     0,0,max
# magenta   COLOR_MAGENTA     5     max,0,max
# cyan      COLOR_CYAN        6     0,max,max
# white     COLOR_WHITE       7     max,max,max

# Text style:
# tput bold    # Select bold mode
# tput dim     # Select dim (half-bright) mode
# tput smul    # Enable underline mode
# tput rmul    # Disable underline mode
# tput rev     # Turn on reverse video mode
# tput smso    # Enter standout (bold) mode
# tput rmso    # Exit standout mode

# Other:
# tput sgr0    # Reset text format to the terminal's default
# tput bel     # Play a bell

TXT_UNDERLINE=no
TXT_BOLD=no
TXT_COLOR=
TXT_STYLE=
TXT_CLEAR="$(tput sgr0)"
set_TXT_STYLE() {
  if [ "$TXT_UNDERLINE" = 'yes' ]; then
    TXT_STYLE="${TXT_STYLE}$(tput smul)"
  fi
  if [ "$TXT_BOLD" = 'yes' ]; then
    TXT_STYLE="${TXT_STYLE}$(tput bold)"
  fi
  if [ "$TXT_COLOR" != '' ]; then
    TXT_STYLE="${TXT_STYLE}$(tput setaf $TXT_COLOR)"
  fi
}
log_info() {
  TXT_COLOR=2
  set_TXT_STYLE
  echo -e "${TXT_STYLE}$@${TXT_CLEAR}"
}
log_error() {
  # red
  TXT_COLOR=1
  set_TXT_STYLE
  echo -e "${TXT_STYLE}$@${TXT_CLEAR}"
}
log_warning() {
  # yellow
  TXT_COLOR=3
  set_TXT_STYLE
  echo -e "${TXT_STYLE}$@${TXT_CLEAR}"
}

rlscope_help() {
(
  set -eu

  (
  TXT_BOLD=yes
  log_info "$ build_rlscope"
  )
  log_info "  Build RL-Scope C++ components (i.e., librlscope.so, rls-analyze)."
  log_info

  (
  TXT_BOLD=yes
  log_info "$ install_experiments"
  )
  log_info "  Clone RL repos with RL-Scope annotations in them into $RLSCOPE_DIR/third_party, and install them."
  log_info "  Only needed if reproducing figures from RL-Scope paper."
  log_info

  (
  TXT_BOLD=yes
  log_info "$ build_wheel"
  )
  log_info "  Build RL-Scope into a python wheel file at $RLSCOPE_DIR/dist/rlscope*.whl"
  log_info "  Useful if you wish to run RL-Scope outside of this container."
  log_info

  (
  TXT_BOLD=yes
  log_info "$ build_docs"
  )
  log_info "  Build RL-Scope html documentation at $RLSCOPE_DIR/build.docs/index.html"
  log_info

  (
  TXT_BOLD=yes
  log_info "$ rls-unit-tests"
  )
  log_info "  Run RL-Scope unit tests."
  log_info "  NOTE: you must run build_rlscope.sh first to do this."
  log_info

  (
  TXT_BOLD=yes
  log_info "$ develop_rlscope"
  )
  log_info "  Enter development mode: i.e., place compiled C++ binaries and python source files on PATH for easier development."
  log_info "  NOTE: This happens automatically when you initially enter the container"
  log_info "        If you install the wheel file produced by build_rlscope.sh, and"
  log_info "        subsequently make source code changes you will need to run this to see their effect."
  log_info

)
}

RLSCOPE_QUIET=no
_rls_do() {
    if [ "$RLSCOPE_QUIET" = 'yes' ]; then
      "$@" > /dev/null
    else
      echo "> CMD [rlscope]:"
      echo "  $ $@"
      echo "  PWD=$PWD"
      "$@"
    fi
}

_source_rlscope() {
  local curdir="$PWD"
  local ret=
  cd $RLSCOPE_DIR
  RLSCOPE_QUIET=yes
  _rls_do source source_me.sh
  ret=$?
  RLSCOPE_QUIET=no
  if [ "$ret" != "0" ]; then
    cd "$curdir"
    log_error "ERROR: failed to source $RLSCOPE_DIR/source_me.sh"
    return $ret
  fi
  cd "$curdir"
}

develop_rlscope() {
  local ret=
  _source_rlscope
  ret=$?
  if [ "$ret" != "0" ]; then
    return $ret
  fi
  log_info "> Setting up rlscope (python setup.py develop)..."
  _develop_rlscope
  ret=$?
  if [ "$ret" != "0" ]; then
    return $ret
  fi
  log_info "  Done"
  log_info "> You have entered development mode."
  log_info "  Compiled RL-Scope binaries and python sources"
  log_info "  from $RLSCOPE_DIR are now on your PATH."
}

_develop_rlscope() {
  local curdir="$PWD"
  local ret=
  cd $RLSCOPE_DIR
  RLSCOPE_QUIET=yes
  _rls_do develop_rlscope.sh
  ret=$?
  RLSCOPE_QUIET=no
  if [ "$ret" != "0" ]; then
    cd "$curdir"
    log_error "ERROR: failed to run \"python setup.py\" from $RLSCOPE_DIR"
    return $ret
  fi
  cd "$curdir"
}

# Aliases for consistency with other commands.
build_rlscope() {
  build_rlscope.sh
}
install_experiments() {
  install_experiments.sh
}
build_wheel() {
  build_wheel.sh
}
build_docs() {
  build_docs.sh
}

_bash_rc() {

  local ret=

  # Make bash append to the history file after every command invocation.
  #
  # https://askubuntu.com/questions/67283/is-it-possible-to-make-writing-to-bash-history-immediate
  # append to history, don't overwrite it
  shopt -s histappend
  export PROMPT_COMMAND="history -a; history -c; history -r; $PROMPT_COMMAND"

  # Make ctrl-w in bash delete up to a separator-character, instead of space-character
  # This behaves like ctrl-w in zsh.
  #
  # https://superuser.com/questions/212446/binding-backward-kill-word-to-ctrlw
  stty werase undef
  bind '"\C-w": backward-kill-word'

  # Make ctrl-u in bash "undo" the last change (typically word deletion using ctrl-w)
  stty kill undef
  bind '"\C-u": undo'

  # NOTE: Installing rlscope during login fails with "apt busy, resource not available" error.
  # Just delay installation/building until they run "install_experiments.sh" or "experiment_...sh".
  _source_rlscope
  if ! rlscope_installed; then
    log_info "> Setting up rlscope (python setup.py develop)..."
    _develop_rlscope && log_info "  Done"
  fi

}

install_experiments() {
  local ret=
  if ! rlscope_installed; then
    develop_rlscope
    ret=$?
    if [ "$ret" != "0" ]; then
      return $ret
    fi
  fi
(
  set -eu
  install_experiments.sh
)
}

rlscope_installed() {
(
  set -eu
  # DON'T run from rlscope repo directory otherwise it will show up in pip freeze.
  cd $HOME
  pip freeze | grep -q rlscope
)
}

rlscope_built() {
(
  set -eu
  which rls-analyze > /dev/null
)
}

#_setup_venv() {
#  set +u
#  source $VIRTUALENV/bin/activate
#  set -u
#}
#
#_rlscope_setup_shell() {
#  _setup_venv
#  set -eux
#}

_bash_rc
