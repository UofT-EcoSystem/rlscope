USER root

##
## OpenMPI in Ubuntu 16.04 repo is pretty out of date; just compile it ourselves.
##
## We do this since otherwise, ML scripts that use mpi4py print warnings.
## For details, see:
##   https://github.com/UofT-EcoSystem/rlscope/wiki/Issues-and-TODOs
##
#ARG OPENMPI_VERSION=3.1.4
#WORKDIR /root/tar_files
#RUN wget --quiet https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-${OPENMPI_VERSION}.tar.bz2
#WORKDIR /root/openmpi
#RUN tar -xf /root/tar_files/openmpi-${OPENMPI_VERSION}.tar.bz2
#WORKDIR /root/openmpi/openmpi-${OPENMPI_VERSION}
#RUN ./configure
#RUN make -j$(nproc)
#RUN make install

# Ubuntu 18.04 openmpi is new enough.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenmpi3 libopenmpi-dev openmpi-bin
# OpenMPI really wants ssh to be installed.
# Otherwise, ML scripts that import mpi4py will crash with an ugly error:
# """
# The value of the MCA parameter "plm_rsh_agent" was set to a path
# that could not be found:
# ...
# """
RUN apt-get update && apt-get install -y --no-install-recommends \
    ssh
WORKDIR ${HOME}
USER ${RLSCOPE_USER}
