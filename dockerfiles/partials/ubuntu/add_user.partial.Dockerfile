# Grant members of 'sudo' group passwordless privileges
# Comment out to require sudo
COPY sudo-nopasswd /etc/sudoers.d/sudo-nopasswd
RUN apt-get update && apt-get install -y --no-install-recommends \
        sudo
