#
# Add repo src files to container.
#
#ENV IML_ROOT /home/${USER_NAME}/bin/iml
#WORKDIR ${IML_ROOT}
#RUN mkdir -p ${IML_ROOT}/dockerfiles/sh

# ADD's files with root as owner; make it $USER_NAME
#ADD sh ${IML_ROOT}/dockerfiles/sh/
ADD requirements.txt ${IML_ROOT}/requirements.txt
USER root
RUN chown -R ${USER_NAME}:${USER_NAME} ${IML_ROOT}
RUN chmod -R +x ${IML_ROOT}
USER ${USER_NAME}

RUN ls -l ${IML_ROOT}
RUN pip install -r ${IML_ROOT}/requirements.txt
#RUN chmod +x ${IML_ROOT}/dockerfiles/sh/*

#ENV PATH="${IML_ROOT}/dockerfiles/sh:${PATH}"
