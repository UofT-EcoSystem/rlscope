ARG IML_DIR

# RUN pip install git+https://github.com/UofT-EcoSystem/iml.git
# NOTE: This actually works!  It does of course prompt for an ssh
# username/password, but that's good enough for now!

RUN if [ "${IML_DIR}" = "" ] ; then \
        pip install git+https://github.com/UofT-EcoSystem/iml.git; \
    else \
        ( \
        cd ${IML_DIR}; \
        python setup.py develop; \
        ); \
    fi
