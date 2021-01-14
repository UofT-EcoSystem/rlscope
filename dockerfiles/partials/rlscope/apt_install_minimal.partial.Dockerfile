USER root

# rlscope-banner requirements.
RUN apt-get update && apt-get install -y --no-install-recommends \
    figlet

USER ${IML_USER}
