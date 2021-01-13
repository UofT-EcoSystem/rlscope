# Download GPU install of tensorflow.
# May be useful for running stuff.
#WORKDIR $HOME/pip_whl

# WARNING: "WORKDIR" always creates directories as root, it does NOT consider "USER"
# => To get desired behaviour, we must instead do:
#
#    # create directory using current "USER"
#    RUN mkdir -p <dir>
#    # cd into existing directory for subsequent "RUN" commands (cd doesn't work)
#    WORKDIR <dir>
RUN mkdir -p /root/pip_whl && cd /root/pip_whl
WORKDIR /root/pip_whl
# Defined in run_docker.py
ARG TENSORFLOW_VERSION
RUN pip download tensorflow-gpu==${TENSORFLOW_VERSION}
