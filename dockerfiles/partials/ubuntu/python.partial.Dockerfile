USER root
ARG USE_PYTHON_3_NOT_2
ARG _PY_SUFFIX=${USE_PYTHON_3_NOT_2:+3}
ARG PYTHON=python${_PY_SUFFIX}
ARG PIP=pip${_PY_SUFFIX}

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    ${PYTHON} \
    ${PYTHON}-pip

# For some reason, installing all at once can cause errors (unsure why).
#RUN ${PIP} --no-cache-dir install --upgrade \
#    pip \
#    setuptools \
#    virtualenv

RUN ${PIP} install --no-cache-dir --upgrade pip
RUN ${PIP} install --no-cache-dir --upgrade virtualenv
RUN ${PIP} install --no-cache-dir --upgrade setuptools

# Different way of invoking pip...
#RUN ${PYTHON} -m pip install --no-cache-dir --upgrade pip
#RUN ${PYTHON} -m pip install --no-cache-dir --upgrade virtualenv
#RUN ${PYTHON} -m pip install --no-cache-dir --upgrade setuptools

# NOTE: We install virtualenv above so we can install python source-repos as $USER using:
#   python setup.py develop
# We do this for the TensorFlow repo as well as IML.

# Some TF tools expect a "python" binary
# RUN ln -s $(which ${PYTHON}) /usr/local/bin/python
USER ${RLSCOPE_USER}
