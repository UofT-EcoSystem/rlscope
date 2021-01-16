# bash autocomplete and terminal colours
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash-completion
USER ${RLSCOPE_USER}
# Define this in env.partial.Dockerfile instead
#ENV TERM=xterm-256color
