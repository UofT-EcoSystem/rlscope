#
# Install apt dependencies for iml-drill.
# In particular, postgres client (psql).
#
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    figlet

