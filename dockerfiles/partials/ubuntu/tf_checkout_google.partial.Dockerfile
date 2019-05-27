# Check out TensorFlow source code if --build_arg CHECKOUT_TENSORFLOW=1
ARG CHECKOUT_TF_SRC=0
RUN test "${CHECKOUT_TF_SRC}" -eq 1 && \
  git clone https://github.com/tensorflow/tensorflow.git /tensorflow_src || true
