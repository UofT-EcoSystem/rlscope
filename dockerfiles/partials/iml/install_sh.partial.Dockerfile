#
# Add repo src files to container.
#
ENV IML_ROOT /iml
WORKDIR ${IML_ROOT}
RUN mkdir -p ${IML_ROOT}/dockerfiles/sh
ADD sh ${IML_ROOT}/dockerfiles/sh/
RUN chmod +x ${IML_ROOT}/dockerfiles/sh/*
ENV PATH="${IML_ROOT}/dockerfiles/sh:${PATH}"
