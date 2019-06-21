#USER root

# Make everything created in /root (e.g. virtualenv stuff in /root/venv) accessible by anyone.
# This is to avoid having to do this at container startup, since it takes a long time
# (especially for /root/venv).
RUN chmod -R ugo+rwx /root

COPY bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc
