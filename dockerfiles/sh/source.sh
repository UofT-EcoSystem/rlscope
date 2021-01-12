#!/usr/bin/env bash
# This file gets automatically source'd by interactive bash shells.
# Useful for adding custom commands, and adding stuff to the default
# Ubuntu bashrc (see _bash_rc).

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

rlscope_help() {
  (
  set -eux
  echo "$ install_rlscope.sh"
  echo "  Build RL-Scope into a python wheel file (at $IML_DIR/dist/rlscope*.whl) and install it."
  echo
  echo "$ install_experiments.sh"
  echo "  Clone RL repos with RL-Scope annotations in them into $IML_DIR/third_party, and install them."
  echo
  echo "$ develop_rlscope.sh"
  echo "  Development mode: install RL-Scope from repo files at $IML_DIR."
  echo
  echo "$ build_rlscope.sh"
  echo "  Development mode: build C++ RL-Scope library."
  echo
  )
}

_bash_rc() {

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

}

_bash_rc