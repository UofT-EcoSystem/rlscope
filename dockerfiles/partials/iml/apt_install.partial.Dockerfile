#
# Install apt dependencies for iml-drill.
# In particular, postgres client (psql).
#
USER root
RUN apt-get install -y --no-install-recommends \
    postgresql-client
USER ${USER_NAME}

