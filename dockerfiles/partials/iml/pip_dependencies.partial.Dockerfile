#
# Add repo src files to container.
#
#ENV IML_ROOT /home/${USER_NAME}/bin/iml
#WORKDIR ${IML_ROOT}
#RUN mkdir -p ${IML_ROOT}/dockerfiles/sh

# ADD's files with root as owner; make it $USER_NAME
ADD requirements.txt ${IML_ROOT}/requirements.txt

COPY bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

#RUN apt-get update && apt-get install -y --no-install-recommends
#     llvm-8
#     llvm-8-dev
#     llvm-8-tools
#ENV LLVM_CONFIG=/usr/bin/llvm-config-8
#RUN pip install "numba==0.46.0"

RUN ls -l ${IML_ROOT}
RUN pip install --no-cache-dir -r ${IML_ROOT}/requirements.txt

#ENV PATH="${IML_ROOT}/dockerfiles/sh:${PATH}"
