#
# Test the host configuration:
# (1) Check that /etc/docker/daemon.json uses --runtime=nvidia by default
# (2) Check that "NVreg_RestrictProfilingToAdminUsers=0" is set in modprobe nvidia kernel module
#
USER root

ADD sh/docker_build_common.sh ${ADD_DIR}/sh/docker_build_common.sh
ADD sh/exports.sh ${ADD_DIR}/sh/exports.sh

# (1) Check that /etc/docker/daemon.json uses --runtime=nvidia by default
ADD sh/test_runtime_nvidia.sh ${ADD_DIR}/sh/test_runtime_nvidia.sh
RUN bash ${ADD_DIR}/sh/test_runtime_nvidia.sh

# (2) Check that "NVreg_RestrictProfilingToAdminUsers=0" is set in modprobe nvidia kernel module
ADD sh/test_cupti_profiler_api.sh ${ADD_DIR}/sh/test_cupti_profiler_api.sh
RUN bash ${ADD_DIR}/sh/test_cupti_profiler_api.sh

USER ${RLSCOPE_USER}
