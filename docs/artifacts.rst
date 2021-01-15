RL-Scope Artifact Evaluation
============================

This is a tutorial for reproducing figures in the RL-Scope paper.
To ease reproducibility, all experiments will run within a docker development environment.

1. Running the docker development environment
---------------------------------------------
In order to run the docker development environment,
you must first perform a one-time configuration of your host system,
then use ``run_docker.py`` to build/run the RL-Scope container.
To do this, follow **all** the instructions at :doc:`Host Configuration <host_config>`.
Afterwards, you should be running inside the RL-Scope container, which looks like this:

.. image:: images/rlscope_banner.png

All remaining instructions will run commands inside this container, which we will
emphasize with ``[container]$``.

2. Building RL-Scope
--------------------
RL-Scope uses a C++ library to collect CUDA profiling information (``librlscope.so``),
and offline analysis of collected traces is performed using a C++ binary (``rls-analyze``)

To build the C++ components, run the following:

.. code-block:: console

    [container]$ build_rlscope.sh

3. Installing experiments
-------------------------
The experiments in RL-Scope consist of taking an existing RL repository and adding RL-Scope annotations to it.
In order to clone these repositories and install them using ``pip``, run the following:

.. code-block:: console

    [container]$ install_experiments.sh

4. Running experiments
----------------------
The RL-Scope paper consists of several case studies.
Each case study has its own shell script for reproducing figures from that section.
The shell script will collect traces from each relevant algorithm/simulator/framework,
then generate a figure seen in the paper in a corresponding subfolder :file:`output/artifacts/*`
of the RL-Scope repository.

RL Framework Comparison
^^^^^^^^^^^^^^^^^^^^^^^
This will reproduce results from the "Case Study: Selecting an RL Framework" section from the RL-Scope paper;
In particular, the "RL framework comparison" figures, shown below for reference:

.. image:: images/rlscope_figures/fig_RL_framework_comparison.png

To run the experiment and generate the figures, run:

.. code-block:: console

    [container]$ experiment_RL_framework_comparison.sh

Figures will be output to :file:`output/artifacts/experiment_RL_framework_comparison/*.pdf`.

RL Algorithm Comparison
^^^^^^^^^^^^^^^^^^^^^^^
This will reproduce results from the "Case Study: RL Algorithm and Simulator Survey" section from the RL-Scope paper;
In particular, the "Simulator choice" figures, shown below for reference:

.. image:: images/rlscope_figures/fig_algorithm_choice.png

To run the experiment and generate the figures, run:

.. code-block:: console

    [container]$ experiment_algorithm_choice.sh

Figures will be output to :file:`output/artifacts/experiment_algorithm_choice/*.pdf`.

Simulator Comparison
^^^^^^^^^^^^^^^^^^^^
This will reproduce results from the "Case Study: Simulator Survey" section from the RL-Scope paper;
In particular, the "Simulator choice" figures, shown below for reference:

.. image:: images/rlscope_figures/fig_simulator_choice.png

To run the experiment and generate the figures, run:

.. code-block:: console

    [container]$ experiment_simulator_choice.sh

Figures will be output to :file:`output/artifacts/experiment_simulator_choice/*.pdf`.
