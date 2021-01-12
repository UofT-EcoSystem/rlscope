# bash autocomplete and terminal colours
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash-completion
USER ${IML_USER}
ENV TERM=xterm-256color
