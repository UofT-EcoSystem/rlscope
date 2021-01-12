# Grant members of 'sudo' group passwordless privileges
# Comment out to require sudo
COPY sudo-nopasswd /etc/sudoers.d/sudo-nopasswd
# According to /etc/sudoers.d/README: "all files in this directory should be mode 0440."
RUN chmod 0440 /etc/sudoers.d/sudo-nopasswd
RUN apt-get update && apt-get install -y --no-install-recommends \
        sudo

ARG IML_USER
ARG IML_UID
ARG IML_GID
RUN addgroup ${IML_USER} --gid ${IML_GID}
# Default for all the user entries: --gecos ""
RUN adduser ${IML_USER} --disabled-password --gecos "" --uid ${IML_UID} --gid ${IML_GID}
RUN usermod -a -G sudo ${IML_USER}

USER ${IML_USER}
ENV HOME /home/${IML_USER}
