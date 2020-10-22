"""
Read RL-Scope trace files into SQL database (PostgreSQL).

.. deprecated:: 1.0.0
    We now read from files directly since it's faster.
"""
from rlscope.profiler.rlscope_logging import logger
import copy
import itertools
import argparse

from rlscope.protobuf.pyprof_pb2 import CategoryEventsProto, MachineUtilization, DeviceUtilization, UtilizationSample
from rlscope.parser.common import *
from rlscope.profiler.util import pprint_msg
from rlscope.profiler import experiment
from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

from rlscope.parser.dataframe import TrainingProgressDataframeReader

from rlscope.parser import stacked_bar_plots
from rlscope.parser.db import SQLCategoryTimesReader, sql_input_path

from rlscope.profiler.rlscope_logging import logger

class ExtrapolatedTrainingTimeParser:
    def __init__(self,
                 directory,
                 host=None,
                 user=None,
                 password=None,
                 dependency=None,
                 # ignore_phase=False,
                 algo_env_from_dir=False,
                 debug=False,
                 # Swallow any excess arguments
                 **kwargs):
        """
        :param directories:
        :param debug:
        """
        self.directory = directory
        self.host = host
        self.user = user
        self.password = password
        self.dependency = dependency
        if self.dependency is not None:
            assert _e(self.dependency)
            self.dependency_js = load_json(self.dependency)
        # self.ignore_phase = ignore_phase
        self.algo_env_from_dir = algo_env_from_dir
        self.debug = debug

        self.added_fields = set()

    @property
    def _csv_path(self):
        return _j(self.directory, "extrapolated_training_time.csv")

    @property
    def db_path(self):
        return sql_input_path(self.directory)

    def run(self):
        """
        # PSEUDOCODE for computing total training time without profiling overhead

        End_time = latest event.end_time_us recorded in Event's

        Tps = Select all TrainingProgress up to end_training_time_us = end_time
        Events = Select all Event's from start of trace up to end_time_us = end_time

        Def Compute_training_time(events):
          Use split_op_stacks on events
          Instead of computing overlap, just compute total event time that doesn't include "profiling overhead"
          op_stack = []
          # NOTE: CPU events may overlap with GPU events...
          # We just want time covered by all CPU/GPU events, MINUS time covered by "profiling overhead" events.
          # for event in sorted(events, by=event.start_time_usec):
          total_time_us = 0
          while len(events) > 0:
              go to next next start or end of an event
              if start of event:
                  if is_op_event(event):
                      op_event = event
                  elif is_cpu_event(event):
                      start_t = event.start_time_usec
                  elif is_gpu_event():
              elif end of event:
                  if is_op_event(event):
                      total_time_us += event.end_time_us - start_t
                      start_t = None

        # total training time
        trace_time_sec = compute_training_time(events)
        Last_tps = tps[-1]
        timesteps_per_sec = ( last_tps['end_num_timesteps'] - last_tps['start_num_timesteps'] ) / trace_time_sec
        Total_trace_time_sec = ( 1/timesteps_per_sec ) * tps['total_timesteps']

        - Q: How do we compute total training time; particularly, how do we do it for minigo?
            - We need to be able to compute the "critical path", then take the "total training time" summed across that path
            - How do we get the critical path?
                - Follow all paths from leaf-node to root node, and determine which among those paths was the longest.
                - If there are multiple leaf-nodes, collect paths from all starting leaf-nodes.
                - NOTE: the length of a node should be the extrapolated total training time of that node.
            - For visualizing minigo figure, to find start time of a particular (process, phase), we must increment its start time by the increased extrapolation of its parent nodes.

        PSEUDOCODE:
        Type: Path = ListOf[Phase]

        def path_length(path):
            return sum(phase.extrapolated_total_training_time for phase in path)

        # PROBLEM: the topology of phases gathered from the minigo script isn't
        # reflective of the fork-join pattern of the scripts...
        # In particular, sgd_updates is NOT the child of selfplay_worker_1 and selfplay_worker_2.
        # In reality, there's a shell-script that coordinates launching these phases in a serialized order:
        # - loop_main.sh
        #   - loop_selfplay.py
        #     - loop_selfplay_worker.py [1]
        #     - loop_selfplay_worker.py [2]
        # - loop_train_eval.py
        #   - sgd_updates
        #   - evaluate
        #
        # However, conceptually it makes more sense to think of as a dependency graph.
        # Currently, the dependency graph structure cannot be recovered from this fork-join pattern,
        # and it must be hard-coded.
        # So, we cannot use to determine "paths" needed for computing total training time.
        # HACK: If there is more than one phase, require the user to specify the dependencies
        # in a dependency.json file:
        dependency.json
        {
        'directed_edges': [
            [
              # A -> B
              [A.machine_name, A.process_name, A.phase_name],
              [B.machine_name, B.process_name, B.phase_name],
            ]
        ]
        }

        def find_all_paths(leaf):
            def _find_all_paths(phase):
                if phase.has_parents:
                    for parent in phase.parents:
                        for path in _find_all_paths(parent)
                            path = list(path)
                            path.append(parent)
                            yield path
                else:
                    # Base case:
                    yield [phase]

        # Return all the Phase's that have NO children.
        leaves = sql_reader.leaf_nodes()
        paths = []
        for leaf in leaves:
            leaf_paths = find_all_paths(leaf)
            paths.extend(leaf_paths)
        critical_path = min(paths, key=path_length)
        total_training_time = path_length(critical_path)
        """
        self.sql_reader = SQLCategoryTimesReader(self.db_path, host=self.host, user=self.user, password=self.password)

        machines = self.sql_reader.machines()
        for machine in machines:
            processes = self.sql_reader.processes(machine_name=machine.machine_name)
            for process in processes:
                phases = self.sql_reader.phases(machine_name=machine.machine_name, process_name=process.process_name)
                logger.info(pprint_msg({
                    'machines': machines,
                    'processes': processes,
                    'phases': phases}))
                if len(phases) == 1:
                    self.csv_single_phase(machine, process, phases[0])
                else:
                    raise NotImplementedError("Haven't implemented total training time extrapolation for multi-phase apps.")

        self.sql_reader.close()

    def csv_single_phase(self, machine, process, phase):
        """
        # Refactor stacked_bar_plots.py code to re-use this.
        Read the OperationOverlap file into a dataframe.
        # Specific to instrumented code.
        df['trace_time_sec'] = sum(df[col] for col in df.keys() if col != OPERATION_PYTHON_PROFILING_OVERHEAD) / constants.USEC_IN_SEC
        # extrap_total_training_time(df['trace_time_sec'], md['percent_complete'])
        df['total_training_time_sec'] = df['trace_time_sec'] / md['percent_complete']


        :param machine:
        :param process:
        :param phase:
        :return:
        """
        pass

