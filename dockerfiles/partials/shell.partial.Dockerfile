#USER root

# Make everything created in /root (e.g. virtualenv stuff in /root/venv) accessible by anyone.
# This is to avoid having to do this at container startup, since it takes a long time
# (especially for /root/venv).
#RUN chmod -R ugo+rwx /root

USER root
# Install custom /etc/bash.bashrc to display rlscope-banner
COPY bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

WORKDIR ${HOME}
RUN rm -rf /root/tar_files/*
RUN rm -rf /root/pip_whl
USER ${RLSCOPE_USER}
