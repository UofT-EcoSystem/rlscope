ARG NODEJS_VERSION=10

WORKDIR /root/nodejs

# NodeJS Installation instructions for debian systems.
# The ubuntu 16.04 nodejs is out-of-date.
# We use nodejs 10 for rlscope-drill.
# https://github.com/nodesource/distributions/blob/master/README.md
RUN curl -sL https://deb.nodesource.com/setup_${NODEJS_VERSION}.x > setup_nodejs_${NODEJS_VERSION}.sh
RUN sudo -E bash ./setup_nodejs_${NODEJS_VERSION}.sh
RUN sudo apt-get install -y nodejs
