# TensorFlow dependencies; let tensorflow installation install these.
#RUN ${PIP} --no-cache-dir install \
#    Pillow \
#    h5py \
#    keras_applications \
#    keras_preprocessing \
#    matplotlib \
#    mock \
#    numpy \
#    scipy \
#    sklearn \
#    pandas \
#    && test "${USE_PYTHON_3_NOT_2}" -eq 1 && true || ${PIP} --no-cache-dir install \
#    enum34
