# Check out TensorFlow source code if --build_arg CHECKOUT_TENSORFLOW=1
ARG CHECKOUT_TF_SRC=0
# NOTE: b1.13-rocm is a branch, NOT a tag.
# For reference, when I ran this, it corresponded to commit: f0924385f0f93ef0a45053cc4511ccffcb1c92b4
# This built AND worked on tf_cnn_benchmarks.py
RUN test "${CHECKOUT_TF_SRC}" -eq 1 && \
  git clone -b r1.13-rocm https://github.com/ROCmSoftwarePlatform/tensorflow-upstream.git /tensorflow_src || true
