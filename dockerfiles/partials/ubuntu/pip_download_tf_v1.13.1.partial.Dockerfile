# Download v1.13.1 GPU install of tensorflow.
# May be useful for running stuff.
#WORKDIR $HOME/pip_whl

# WARNING: "WORKDIR" always creates directories as root, it does NOT consider "USER"
# => To get desired behaviour, we must instead do:
#
#    # create directory using current "USER"
#    RUN mkdir -p <dir>
#    # cd into existing directory for subsequent "RUN" commands (cd doesn't work)
#    WORKDIR <dir>
RUN mkdir -p $HOME/pip_whl && cd $HOME/pip_whl
WORKDIR $HOME/pip_whl
RUN pip download tensorflow-gpu==1.13.1
