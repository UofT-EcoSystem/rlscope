#
# Add repo src files to container.
#
#ENV ADD_DIR /home/${USER_NAME}/bin/rlscope
#WORKDIR ${ADD_DIR}
#RUN mkdir -p ${ADD_DIR}/dockerfiles/sh

# ADD's files with root as owner; make it $USER_NAME
ADD requirements.txt ${ADD_DIR}/requirements.txt
ADD requirements.txt ${ADD_DIR}/requirements.docs.txt
ADD requirements.txt ${ADD_DIR}/requirements.develop.txt
USER root
RUN chmod ugo+r ${ADD_DIR}/requirements.txt
RUN chmod ugo+r ${ADD_DIR}/requirements.docs.txt
RUN chmod ugo+r ${ADD_DIR}/requirements.develop.txt
USER ${RLSCOPE_USER}

#RUN apt-get update && apt-get install -y --no-install-recommends
#     llvm-8
#     llvm-8-dev
#     llvm-8-tools
#ENV LLVM_CONFIG=/usr/bin/llvm-config-8
#RUN pip install "numba==0.46.0"

RUN ls -l ${ADD_DIR}
RUN pip install --no-cache-dir -r ${ADD_DIR}/requirements.txt
RUN pip install --no-cache-dir -r ${ADD_DIR}/requirements.docs.txt
RUN pip install --no-cache-dir -r ${ADD_DIR}/requirements.develop.txt

#ENV PATH="${ADD_DIR}/dockerfiles/sh:${PATH}"
