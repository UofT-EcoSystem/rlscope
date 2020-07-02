# Create a virtualenv, and override

# Ubuntu 18.04 uses python3.6
# Ubuntu 16.04 uses python3.5
ARG PYTHON_BIN_BASENAME=python3
# https://pythonspeed.com/articles/activate-virtualenv-dockerfile/
ENV VIRTUAL_ENV=/root/venv
# NOTE: We use --system-site-packages since some python packages we need get installed through deb packages.
# e.g.
#  python3-libnvinfer:
#     # Allows us to use this package:
#     import tensorrt
#     # Sadly, cannot pip install this separately (unavailable in pip).
RUN python -m virtualenv -p /usr/bin/${PYTHON_BIN_BASENAME} $VIRTUAL_ENV --system-site-packages
# We can't do "actiate venv" from Dockerfile.
# So instead, just overwrite PATH so it finds /root/bin/python instead of
# /usr/bin/python.
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
