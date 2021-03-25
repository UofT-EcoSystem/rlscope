# RL-Scope: Cross-Stack Profiling for Deep Reinforcement Learning Workloads

RL-Scope collects cross-stack profiling information (CUDA API time, GPU kernel time, ML backend time, etc.), and provides a breakdown of CPU/GPU execution time.

RL-Scope's complete documentation can be found here: <https://rl-scope.readthedocs.io/en/latest/index.html>

Here are some convenient links to common parts of the documentation:
- [Installation](https://rl-scope.readthedocs.io/en/latest/installation.html)
- [RL-Scope artifact evaluation](https://rl-scope.readthedocs.io/en/latest/artifacts.html)
- [Interactive "Getting Started" notebook on Google Colab](https://colab.research.google.com/github/UofT-Ecosystem/rlscope/blob/master/jupyter/01_rlscope_getting_started.ipynb)
- [Docker development environment](https://rl-scope.readthedocs.io/en/latest/host_config.html)

# Paper

For convenience, you can find our paper on arxiv:
https://arxiv.org/abs/2102.04285

When citing RL-Scope, please cite our MLSys 2021 publication:

{% raw %}

    @inproceedings{gleeson2021rlscope,
     author = {Gleeson, James and Krishnan, Srivatsan and Gabel, Moshe and Janapa Reddi, Vijay and de Lara, Eyal and Pekhimenko, Gennady},
     booktitle = {Proceedings of Machine Learning and Systems},
     title = {{RL-Scope:} Cross-Stack Profiling for Deep Reinforcement Learning Workloads},
     year = {2021}
    }
    
{% endraw %}
