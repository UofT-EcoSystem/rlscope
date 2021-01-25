#!/usr/bin/env bash
# Shell script that is available at Docker BUILD time.
# Prefer to add shell commands to docker_runtime_common.sh to avoid spurious container rebuilds.
#
# NOTE: This should run inside a docker container.

IS_ZSH="$(ps -ef | grep $$ | grep -v --perl-regexp 'grep|ps ' | grep zsh --quiet && echo yes || echo no)"
if [ "$IS_ZSH" = 'yes' ]; then
  SH_DIR="$(readlink -f "$(dirname "${0:A}")")"
else
  SH_DIR="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")"
fi
_ROOT_DIR="$(readlink -f "$SH_DIR/../..")"
source $SH_DIR/exports.sh

_do() {
    echo "> CMD:"
    echo "  $ $@"
    "$@"
}

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

_link() {
  # _link $target $destination
  #
  # Create symlink at $destination that links to $target.
  # Replace existing symlink if it exists (but only if it's also a symlink).
  local target="$1"
  local destination="$2"
  shift 2

  if [ -L "$destination" ]; then
    rm "$destination"
  fi
  _do ln -s -T "$target" "$destination" "$@"
}

PY_MODULE_INSTALLED_SILENT=no
py_module_installed() {
    local py_module="$1"
    shift 1
    # Returns 1 if ImportError is thrown.
    # Returns 0 if import succeeds.
    if [ "$PY_MODULE_INSTALLED_SILENT" == 'yes' ]; then
        python -c "import ${py_module}" > /dev/null 2>&1
    else
        echo "> Make sure we can import ${py_module}"
        _do python -c "import ${py_module}"
    fi
}

py_maybe_install() {
    local py_module="$1"
    local sh_install="$2"
    shift 2

    if ! py_module_installed $py_module; then
        ${sh_install}
        echo "> Installed python module $py_module: $sh_install"
        if ! py_module_installed $py_module; then
            echo "ERROR: $py_module still failed even after running $sh_install"
            exit 1
        fi
        echo "> $py_module installed"
    fi

}

_yes_or_no() {
    if "$@" > /dev/null 2>&1; then
        echo yes
    else
        echo no
    fi
}
_has_sudo() {
    if [ "$HAS_SUDO" = '' ]; then
        HAS_SUDO="$(_yes_or_no /usr/bin/sudo -v)"
    fi
    echo $HAS_SUDO
}
_has_exec() {
    _yes_or_no which "$@"
}
_has_lib() {
    local lib="$1"
    shift 1
    on_ld_path() {
        ldconfig -p \
            | grep --quiet "$lib"
    }
    in_local_path() {
        ls $INSTALL_DIR/lib \
            | grep --quiet "$lib"
    }
    __has_lib() {
        on_ld_path || in_local_path
    }
    _yes_or_no __has_lib
}