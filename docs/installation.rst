Installation
============

The following page describes the steps to install rlscope using the standard ``pip`` python tool so you can use
it in your own RL code base.
In particular, to install RL-Scope you must enable GPU hardware counter profiling, and install an RL-Scope version that
matches the CUDA version used by your DL framework.

.. note::
    *Don't* follow these steps if you are trying to reproduce RL-Scope paper artifacts; instead,
    follow the instructions for running RL-Scope inside a reproducible docker environment: :doc:`artifacts`.

1. NVIDIA driver
----------------
By default, the ``nvidia`` kernel module doesn't allow non-root users to access GPU hardware counters.
To allow non-root user access, do the following:

1. Paste the following contents into :file:`/etc/modprobe.d/nvidia-profiler.conf`:

    .. code-block:: text

        options nvidia NVreg_RestrictProfilingToAdminUsers=0

2. Reboot the machine for the changes to take effect:

    .. code-block:: console

        [host]$ sudo reboot now

.. warning::
    If you forget to do this, RL-Scope will fail during profiling with an ``CUPTI_ERROR_INSUFFICIENT_PRIVILEGES`` error
    when attempting to read GPU hardware counters.

2. Determine the CUDA version used by your DL framework
-------------------------------------------------------

RL-Scope does not have dependencies on DL frameworks,
but it does have dependencies on different CUDA versions.

In order to host multiple CUDA versions, we provide
`our own wheel file index  <https://uoft-ecosystem.github.io/rlscope/whl>`_
*instead* of hosting packages on PyPi (NOTE: this is the same approach taken by PyTorch).

DL frameworks like TensorFlow and PyTorch have their own CUDA version dependencies.
So, depending on which DL framework version you are using, you must choose to install
RL-Scope with a matching CUDA version.

TensorFlow
^^^^^^^^^^

For TensorFlow, the CUDA version it uses is determined by your TensorFlow version.
For example TensorFlow v2.4.0 uses CUDA 11.0.
You can find a full list `here <https://www.tensorflow.org/install/source#gpu>`_.

PyTorch
^^^^^^^

For PyTorch, multiple CUDA versions are available, but your specific PyTorch installation will
only support one CUDA version.
You can determine the CUDA version by looking at the version of the installed PyTorch by doing

.. code-block:: console

    $ pip freeze | grep torch
    torch==1.7.1+cu101

In this case the installed CUDA version is "101" which corresponds to 10.1.

3. pip installation
-------------------

Once you've determined your CUDA version, you can use pip to install rlscope.
To install RL-Scope version 0.0.1, CUDA 10.1 you can run:

.. code-block:: console

    $ pip install rlscope==0.0.1+cu101 -f https://uoft-ecosystem.github.io/rlscope/whl

More generally, the syntax is:

.. code-block:: console

    $ pip install rlscope==${RLSCOPE_VERSION}+cu${CUDA_VERSION}

Where ``RLSCOPE_VERSION`` corresponds to a tag on github, and ``CUDA_VERSION`` corresponds
to a CUDA version with "." removed (e.g., 10.1 :math:`\rightarrow` 101).

For a full list of available releases and CUDA versions, visit the
`RL-Scope github releases page <https://github.com/UofT-EcoSystem/rlscope/releases>`_.

4. requirements.txt
-------------------
To add RL-Scope to your requirements.txt file, make sure to add **two** lines to the file:

.. code-block:: console

    $ cat requirements.txt
    -f https://uoft-ecosystem.github.io/rlscope/whl
    rlscope==0.0.1+cu101

The ``-f ...`` line ensures that the rlscope package is fetched using our custom wheel
index (otherwise, ``pip`` will fail when it attempts to install from the default PyPi index).

.. warning::
    ``pip freeze`` will *not* remember to add ``-f https://uoft-ecosystem.github.io/rlscope/whl``, so
    avoid generating requirements.txt using its raw output alone.
