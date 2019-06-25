# Create a virtualenv, and override

# Ubuntu 18.04 uses python3.6
# Ubuntu 16.04 uses python3.5
ARG PYTHON_BIN_BASENAME=python3.6
# https://pythonspeed.com/articles/activate-virtualenv-dockerfile/
ENV VIRTUAL_ENV=/root/venv
RUN python -m virtualenv -p ${PYTHON_BIN_BASENAME} $VIRTUAL_ENV
# We can't do "actiate venv" from Dockerfile.
# So instead, just overwrite PATH so it finds /root/bin/python instead of
# /usr/bin/python.
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
