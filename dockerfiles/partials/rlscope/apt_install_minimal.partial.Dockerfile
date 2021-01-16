USER root

# rlscope-banner requirements.
RUN apt-get update && apt-get install -y --no-install-recommends \
    figlet

USER ${RLSCOPE_USER}
