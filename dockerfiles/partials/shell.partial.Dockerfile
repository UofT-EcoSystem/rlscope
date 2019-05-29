USER root
COPY bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc
USER ${USER_NAME}
