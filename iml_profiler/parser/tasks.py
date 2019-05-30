"""
Define IML DAG execution graph using "luigi" DAG execution framework.

For luigi documetation, see:
https://luigi.readthedocs.io/en/stable/index.html
"""
import luigi

import textwrap
import datetime
import pprint
import sys
import os

from iml_profiler.parser.tfprof import TotalTimeParser, TraceEventsParser
from iml_profiler.parser.pyprof import PythonProfileParser, PythonFlameGraphParser, PythonProfileTotalParser
from iml_profiler.parser.plot import TimeBreakdownPlot, PlotSummary, CombinedProfileParser, CategoryOverlapPlot, UtilizationPlot, HeatScalePlot
from iml_profiler.parser.db import SQLParser

PARSER_KLASSES = [PythonProfileParser, PythonFlameGraphParser, PlotSummary, TimeBreakdownPlot, CategoryOverlapPlot, UtilizationPlot, HeatScalePlot, TotalTimeParser, TraceEventsParser, SQLParser]
PARSER_NAME_TO_KLASS = dict((ParserKlass.__name__, ParserKlass) \
                            for ParserKlass in PARSER_KLASSES)

# Make this pipeline:
# - iml-analyze --directories ~/clone/baselines/output/PongNoFrameskip-v4/docker --rules SQLParser
# |-> iml-analyze --directories ~/clone/baselines/output/PongNoFrameskip-v4/docker --rules UtilizationPlot --overlap-type CategoryOverlap
# |-> iml-analyze --directories ~/clone/baselines/output/PongNoFrameskip-v4/docker --rules UtilizationPlot --overlap-type ResourceOverlap
# |-> iml-analyze --directories ~/clone/baselines/output/PongNoFrameskip-v4/docker --rules UtilizationPlot --overlap-type OperationOverlap
# |-> iml-analyze --directories ~/clone/baselines/output/PongNoFrameskip-v4/docker --rules UtilizationPlot --overlap-type ResourceSubPlot
#
# Wrap running all the existing classes with a class that outputs a "done" marker file.
# That way, we don't need to modify the classes to use luigi's "with self.output().open('w') as outfile:".

# List of runnable tasks for iml-analyze.
# Used for generating command-line usage help.

# IML_TASKS = ...
# NOT_RUNNABLE_TASKS = ...
def get_NOT_RUNNABLE_TASKS():
    return [IMLTask, _UtilizationPlotTask]

def get_IML_TASKS():
    global NOT_RUNNABLE_TASKS
    iml_tasks = []
    for name, cls in globals().items():
        if isinstance(cls, type) and issubclass(cls, IMLTask) and cls not in NOT_RUNNABLE_TASKS:
            iml_tasks.append(cls)
    return iml_tasks

class IMLTask(luigi.Task):
    iml_directory = luigi.Parameter(description="Location of trace-files")
    debug = luigi.BoolParameter(description="debug")
    debug_single_thread = luigi.BoolParameter(description=textwrap.dedent("""
        Run any multiprocessing stuff using a single thread for debugging.
        """))

    skip_output = False

    def output(self):
        return luigi.LocalTarget(self._done_file)

    @property
    def _done_file(self):
        """
        e.g.

        <--iml-directory>/SQLParserTask.task
        """
        return "{dir}/{name}.task".format(
            dir=self.iml_directory, name=self._task_name)

    @property
    def _task_name(self):
        return self.__class__.__name__

    def mark_done(self, start_t, end_t):
        if self.skip_output:
            print("> Skipping output={path} for task {name}".format(
                path=self._done_file,
                name=self._task_name))
            return
        with self.output().open('w') as f:
            print_cmd(cmd=sys.argv, file=f)
            delta = end_t - start_t
            minutes, seconds = divmod(delta.total_seconds(), 60)
            print(textwrap.dedent("""\
            > Started running at {start}
            > Ended running at {end}
            > Took {min} minutes and {sec} seconds.
            """.format(
                start=start_t,
                end=end_t,
                min=minutes,
                sec=seconds,
            )), file=f)

    def iml_run(self):
        raise NotImplementedError("{klass} must override iml_run()".format(
            klass=self.__class__.__name__))

    def run(self):
        start_t = datetime.datetime.now()
        self.iml_run()
        end_t = datetime.datetime.now()

        self.mark_done(start_t, end_t)

class SQLParserTask(IMLTask):
    def requires(self):
        return []

    def iml_run(self):
        self.sql_parser = SQLParser(
            directory=self.iml_directory,
            debug=self.debug,
            debug_single_thread=self.debug_single_thread,
        )
        self.sql_parser.run()

class _UtilizationPlotTask(IMLTask):
    def requires(self):
        return [
            SQLParserTask(iml_directory=self.iml_directory, debug=self.debug, debug_single_thread=self.debug_single_thread),
        ]

    def iml_run(self):
        self.sql_parser = UtilizationPlot(
            overlap_type=self.overlap_type,
            directory=self.iml_directory,
            debug=self.debug,
            debug_single_thread=self.debug_single_thread,
        )
        self.sql_parser.run()

class ResourceOverlapTask(_UtilizationPlotTask):
    overlap_type = 'ResourceOverlap'

class CategoryOverlapTask(_UtilizationPlotTask):
    overlap_type = 'CategoryOverlap'

class ResourceSubplotTask(_UtilizationPlotTask):
    overlap_type = 'ResourceSubplot'

class OperationOverlapTask(_UtilizationPlotTask):
    overlap_type = 'OperationOverlap'

class All(IMLTask):
    # Don't output All.task, so that if (for example)
    # ResourceOverlapTask.task is deleted, we will still re-run it.
    skip_output = True

    def requires(self):
        kwargs = self.param_kwargs
        return [
            ResourceOverlapTask(**kwargs),
            CategoryOverlapTask(**kwargs),
            ResourceSubplotTask(**kwargs),
            OperationOverlapTask(**kwargs),
            HeatScaleTask(**kwargs),
        ]

    def iml_run(self):
        pass

class HeatScaleTask(IMLTask):
    # step_sec=1.,
    # pixels_per_square=10,
    # decay=0.99,

    def requires(self):
        return [
            SQLParserTask(iml_directory=self.iml_directory, debug=self.debug, debug_single_thread=self.debug_single_thread),
        ]

    def iml_run(self):
        self.heat_scale = HeatScalePlot(
            directory=self.iml_directory,
            debug=self.debug,
        )
        self.heat_scale.run()

def print_cmd(cmd, file=sys.stdout):
    if type(cmd) == list:
        cmd_str = " ".join([str(x) for x in cmd])
    else:
        cmd_str = cmd
    print(("> CMD:\n"
           "  $ {cmd}\n"
           "  PWD={pwd}"
           ).format(
        cmd=cmd_str,
        pwd=os.getcwd(),
    ), file=file)

def main(argv=None):
    if argv is None:
        argv = list(sys.argv[1:])
    luigi.run(cmdline_args=argv)

NOT_RUNNABLE_TASKS = get_NOT_RUNNABLE_TASKS()
IML_TASKS = get_IML_TASKS()

if __name__ == "__main__":
    main()
