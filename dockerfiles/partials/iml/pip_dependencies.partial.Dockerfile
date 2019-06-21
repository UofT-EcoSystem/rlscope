#
# Add repo src files to container.
#
#ENV IML_ROOT /home/${USER_NAME}/bin/iml
#WORKDIR ${IML_ROOT}
#RUN mkdir -p ${IML_ROOT}/dockerfiles/sh

# ADD's files with root as owner; make it $USER_NAME
ADD requirements.txt ${IML_ROOT}/requirements.txt

RUN ls -l ${IML_ROOT}
RUN pip install -r ${IML_ROOT}/requirements.txt

#ENV PATH="${IML_ROOT}/dockerfiles/sh:${PATH}"
