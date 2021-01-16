# Grant members of 'sudo' group passwordless privileges
# Comment out to require sudo
COPY sudo-nopasswd /etc/sudoers.d/sudo-nopasswd
# According to /etc/sudoers.d/README: "all files in this directory should be mode 0440."
RUN chmod 0440 /etc/sudoers.d/sudo-nopasswd
RUN apt-get update && apt-get install -y --no-install-recommends \
        sudo

ARG RLSCOPE_USER
ARG RLSCOPE_UID
ARG RLSCOPE_GID
RUN addgroup ${RLSCOPE_USER} --gid ${RLSCOPE_GID}
# Default for all the user entries: --gecos ""
RUN adduser ${RLSCOPE_USER} --disabled-password --gecos "" --uid ${RLSCOPE_UID} --gid ${RLSCOPE_GID}
RUN usermod -a -G sudo ${RLSCOPE_USER}

USER ${RLSCOPE_USER}
