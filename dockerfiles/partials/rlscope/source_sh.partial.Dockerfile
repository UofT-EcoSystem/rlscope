ARG IML_DIR

# Add dockerfiles/sh to PATH
RUN \
    echo >> $HOME/.bashrc; \
    echo "if [ -e ${IML_DIR}/dockerfiles/sh ]; then export PATH=\$PATH:${IML_DIR}/dockerfiles/sh ; fi" >> $HOME/.bashrc;

# Load exports.sh
RUN \
    echo >> $HOME/.bashrc; \
    echo "# Location of third party experiment repo directories (e.g., BASELINES_DIR)" >> $HOME/.bashrc; \
    echo "if [ -e ${IML_DIR}/dockerfiles/sh/exports.sh ]; then source ${IML_DIR}/dockerfiles/sh/exports.sh; fi" >> $HOME/.bashrc;

# Run source.sh (basically like a .bashrc file).
RUN \
    echo >> $HOME/.bashrc; \
    echo "# Automatically include source.sh when bash starts" >> $HOME/.bashrc; \
    echo "if [ -e ${IML_DIR}/dockerfiles/sh/source.sh ]; then source ${IML_DIR}/dockerfiles/sh/source.sh; fi" >> $HOME/.bashrc;
