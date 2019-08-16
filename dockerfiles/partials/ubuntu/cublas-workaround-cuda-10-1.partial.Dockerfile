# CUDA 10.1 moved the libcublas libraries from /usr/local/cuda to /usr/lib/x86_64-linux-gnu,
# which screws up the tensorflow build system.
#
# Add symlink to libcublas from /usr/local/cuda to make tensorflow happy.
#
# https://github.com/tensorflow/tensorflow/issues/26150
# https://github.com/tensorflow/tensorflow/issues/26150#issuecomment-499934800

# This removes:
# cuda-libraries-10-1 libcublas10 libnvinfer5
RUN apt-get remove -y libcublas10
# libcublas10-dev

RUN apt-get install -y 'libcublas10=10.1*'
RUN apt-mark hold libcublas10
RUN apt-get install -y 'libcublas-dev=10.1*'
RUN apt-mark hold libcublas-dev
RUN echo "> WORK-AROUND: until CUDA fixes their repos and to ensure only CUDA 10.1 packages are installed and NOT CUDA 10.2 packages, we will use apt to 'hold' the libcublas package. " && \
    echo "Held packages (these packages cannot be upgraded/removed; lookup "apt-mark hold" for details):" && \
    dpkg -l | grep '^hi'
# NOTE: If this command FAILS, then libcublas 10.2 is installed and
# will cause the TensorFlow build system to complain.
RUN dpkg-query -Wf '${Version}' libcublas10 | grep -v '10\.2'

RUN apt-get install -y 'cuda-libraries-10-1=10.1.105-1'
RUN apt-get install -y 'libnvinfer5=*cuda10.1'
# This will fail unless we install the libcublas-dev=10.1* package (above).
RUN apt-get install -y 'libnvinfer-dev=*cuda10.1'

# Add libcublas symlinks as described at top of this file.
RUN cp --no-dereference --no-clobber /usr/lib/x86_64-linux-gnu/libcublas* /usr/local/cuda/lib64

# These CUDA include files are inside system-paths, tensorflow expects them to be
# inside /usr/local/cuda/include in its #include statements.
# RUN cp --no-dereference --no-clobber /usr/include/cublas*.h /usr/local/cuda/include
RUN mv --no-clobber /usr/include/cublas*.h /usr/local/cuda/include
# RUN cp --no-dereference --no-clobber /usr/include/cudnn.h /usr/local/cuda/include

# libcublas.10.1 symlink is missing.
# Same for all the other libraries.
RUN for lib in cublas cublasLt cuart cufft cufftw cuinj64 curand cusolver cusparse; do \
        if [ ! -e /usr/local/cuda/lib64/lib${lib}.so.10.1 ]; then \
            ln -s /usr/local/cuda/lib64/lib${lib}.so.10.1.* /usr/local/cuda/lib64/lib${lib}.so.10.1; \
        fi; \
    done;

# NOTE: belongs in dockerfiles/partials/ubuntu/nvidia_rtx_2070_env.partial.Dockerfile
ENV TF_CUDA_COMPUTE_CAPABILITIES=7.5,6.1

# For some reason, TensorFlow is unable to load libcublas.so.10.1 unless we stick
# /usr/local/cuda/lib64 on LD_LIBRARY_PATH, so we do:
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
