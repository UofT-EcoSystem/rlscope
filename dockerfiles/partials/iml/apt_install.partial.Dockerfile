#
# Install apt dependencies for iml-drill.
# In particular, postgres client (psql).
#
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    figlet \
    mercurial

# Needed for pdfcrop shell command, which removes whitespace in a graph pdf
# (used by plotting scripts).
# Sadly, pdfcrop is bundled with a bunch of latex stuff we don't really need (~ 300 MB).
# Oh well.
RUN apt-get update && apt-get install -y --no-install-recommends \
    texlive-extra-utils \
    ghostscript
