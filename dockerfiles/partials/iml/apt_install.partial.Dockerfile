#
# Install apt dependencies for iml-drill.
# In particular, postgres client (psql).
#
USER root
RUN apt-get install -y --no-install-recommends \
    postgresql-client \
    figlet
USER ${USER_NAME}

