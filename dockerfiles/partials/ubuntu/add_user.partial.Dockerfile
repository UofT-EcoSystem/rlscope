# https://jtreminio.com/blog/running-docker-containers-as-current-host-user/#ok-so-what-actually-works
ARG USER_ID
ARG GROUP_ID

# REQUIRED:
# --build-arg USER_ID=(id -u ${USER})
ARG USER_ID
RUN test -n "$USER_ID"
ENV USER_ID $USER_ID

# REQUIRED:
# --build-arg GROUP_ID=(id -u ${USER})
ARG GROUP_ID
RUN test -n "$GROUP_ID"
ENV GROUP_ID $GROUP_ID

# REQUIRED:
# --build-arg USER_NAME=${USER}
ARG USER_NAME
RUN test -n "$USER_NAME"
ENV USER_NAME $USER_NAME

RUN groupadd -g ${GROUP_ID} ${USER_NAME}
RUN useradd -l -u ${USER_ID} -g ${USER_NAME} ${USER_NAME}
RUN install -d -m 0755 -o ${USER_NAME} -g ${USER_NAME} /home/${USER_NAME}
RUN usermod -a -G sudo ${USER_NAME}

ENV HOME /home/$USER_NAME
RUN mkdir -p $HOME
RUN chown -R $USER_NAME:$USER_NAME $HOME

# Grant members of 'sudo' group passwordless privileges
# Comment out to require sudo
COPY sudo-nopasswd /etc/sudoers.d/sudo-nopasswd
RUN apt-get install -y --no-install-recommends \
        sudo

# This is meant to be used as an interactive developer container
# Create user rocm-user as member of sudo group
# Append /opt/rocm/bin to the system PATH variable
#RUN useradd --create-home -G sudo --shell /bin/bash rocm-user
