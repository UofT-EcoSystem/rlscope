#
# Preinstall things, or setup your development environment.
# You may wish to tailor this file to your own configuration to speed up deployment,
# however try not to commit those changes to the repo.
#

RUN pip install ipdb

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    htop \
    tree \
    python3-dbg \
    gdb \
    strace
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash-completion \
    silversearcher-ag \
    vim
USER ${RLSCOPE_USER}