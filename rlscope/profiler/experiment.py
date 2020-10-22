"""
Helper code for plotting GPU utilization over time.

See also
---------
rlscope.profiler.cpu_gpu_util
"""
import os
import shutil
from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b
from rlscope.parser.common import *

def experiment_config_path(directory):
    path = _j(directory, 'experiment_config.json')
    return path

def dump_experiment_config(expr_type, directory, config):
    path = experiment_config_path(directory)
    data = dict(config)
    data['expr_type'] = expr_type
    do_dump_json(data, path)

def load_experiment_config(directory):
    path = experiment_config_path(directory)
    data = load_json(path)
    return data

class OverallMachineUtilization:
    """
    <root-directory>/algo/env
    ├── num_workers_1
    │   ├── machine_util.trace_0.proto
    │   ├── ...
    │   └── experiment_config.json
    ├── num_workers_2
    ├── ...
    └── num_workers_N
    """
    expr_type = 'OverallMachineUtilization'

    def __init__(self, root_directory, replace=False):
        self.root_directory = root_directory
        self.replace = replace

    def results_exist(self, algo, env, num_workers):
        workers_dir = self.workers_dir(algo, env, num_workers)
        return _e(workers_dir)

    def remove_results(self, algo, env, num_workers):
        workers_dir = self.workers_dir(algo, env, num_workers)
        if not os.path.isdir(workers_dir):
            return
        logger.info("Removing old {expr} results @ {dir} (replace=True)".format(
            expr=self.__class__.__name__,
            dir=workers_dir,
        ))
        shutil.rmtree(workers_dir)

    def should_run(self, algo, env, num_workers):
        return self.replace or not self.results_exist(algo, env, num_workers)

    def _maybe_remove(self, algo, env, num_workers):
        if self.replace:
            self.remove_results(algo, env, num_workers)
            return

    def workers_dir(self, algo, env, num_workers):
        return _j(self.root_directory, algo, env, "num_workers_{n}".format(
            n=num_workers))

    def dump_expr(self, algo, env, num_workers):
        assert self.should_run(algo, env, num_workers)
        workers_dir = self.workers_dir(algo, env, num_workers)
        self._maybe_remove(workers_dir)
        os.makedirs(workers_dir, exist_ok=True)
        config = {
            'algo': algo,
            'env': env,
            'num_workers': num_workers,
        }
        dump_experiment_config(self.expr_type, workers_dir, config)
        assert self.results_exist(algo, env, num_workers)
