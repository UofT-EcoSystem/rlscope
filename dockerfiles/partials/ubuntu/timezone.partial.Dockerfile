##
## Setup timezone.
## Required for Ubuntu 20.04 otherwise interactive timezone setup happens.
##

# Use timezone matching host machine.
USER root
ARG TZ="America/New_York"
# FAILS:
#RUN timedatectl set-timezone ${TZ}
## NOTE: this WORKS with Ubuntu 18.04, but FAILS with 20.04 since it STILL asks for timezone info...
RUN ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    tzdata
RUN dpkg-reconfigure --frontend noninteractive tzdata
USER ${RLSCOPE_USER}
