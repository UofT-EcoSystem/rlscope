"""
Define IML DAG execution graph using "luigi" DAG execution framework.

For luigi documetation, see:
https://luigi.readthedocs.io/en/stable/index.html
"""
import logging
import luigi

import pwd
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
    iml_tasks = set()
    for name, cls in globals().items():
        if isinstance(cls, type) and issubclass(cls, IMLTask) and cls not in NOT_RUNNABLE_TASKS:
            iml_tasks.add(cls)
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
            logging.info("> Skipping output={path} for task {name}".format(
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

def get_username():
    return pwd.getpwuid(os.getuid())[0]

param_postgres_password = luigi.Parameter(description="Postgres password; default: env.PGPASSWORD", default=os.environ.get('PGPASSWORD', None))
param_postgres_user = luigi.Parameter(description="Postgres user", default=get_username())
param_postgres_host = luigi.Parameter(description="Postgres host", default='localhost')

class SQLParserTask(IMLTask):
    postgres_password = param_postgres_password
    postgres_user = param_postgres_user
    postgres_host = param_postgres_host

    def requires(self):
        return []

    def iml_run(self):
        self.sql_parser = SQLParser(
            directory=self.iml_directory,
            host=self.postgres_host,
            user=self.postgres_user,
            password=self.postgres_password,
            debug=self.debug,
            debug_single_thread=self.debug_single_thread,
        )
        self.sql_parser.run()

class _UtilizationPlotTask(IMLTask):
    postgres_password = param_postgres_password
    postgres_user = param_postgres_user
    postgres_host = param_postgres_host

    def requires(self):
        return [
            SQLParserTask(iml_directory=self.iml_directory, debug=self.debug, debug_single_thread=self.debug_single_thread,
                          postgres_host=self.postgres_host, postgres_user=self.postgres_user, postgres_password=self.postgres_password),
        ]

    def iml_run(self):
        self.sql_parser = UtilizationPlot(
            overlap_type=self.overlap_type,
            directory=self.iml_directory,
            host=self.postgres_host,
            user=self.postgres_user,
            password=self.postgres_password,
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
    postgres_password = param_postgres_password
    postgres_user = param_postgres_user
    postgres_host = param_postgres_host

    def requires(self):
        return [
            SQLParserTask(iml_directory=self.iml_directory, debug=self.debug, debug_single_thread=self.debug_single_thread,
                          postgres_host=self.postgres_host, postgres_user=self.postgres_user, postgres_password=self.postgres_password),
        ]

    def iml_run(self):
        self.heat_scale = HeatScalePlot(
            directory=self.iml_directory,
            host=self.postgres_host,
            user=self.postgres_user,
            password=self.postgres_password,
            debug=self.debug,
        )
        self.heat_scale.run()

class TraceEventsTask(luigi.Task):
    iml_directory = luigi.Parameter(description="Location of trace-files")
    debug = luigi.BoolParameter(description="debug")
    debug_single_thread = luigi.BoolParameter(description=textwrap.dedent("""
        Run any multiprocessing stuff using a single thread for debugging.
        """))

    postgres_password = param_postgres_password
    postgres_user = param_postgres_user
    postgres_host = param_postgres_host

    overlaps_event_id = luigi.IntParameter(description="show events that overlap with this event (identified by its event_id)", default=None)
    op_name = luigi.Parameter(description="operation name (e.g. q_forward)", default=None)
    process_name = luigi.Parameter(description="show events belonging to this process", default=None)
    start_usec = luigi.FloatParameter(description="Show events whose start-time is >= start_usec", default=None)
    end_usec = luigi.FloatParameter(description="Show events whose end-time is <= end_usec", default=None)

    skip_output = False

    def requires(self):
        return [
            SQLParserTask(iml_directory=self.iml_directory, debug=self.debug, debug_single_thread=self.debug_single_thread,
                          postgres_host=self.postgres_host, postgres_user=self.postgres_user, postgres_password=self.postgres_password),
        ]

    def output(self):
        # return luigi.LocalTarget(self._done_file)
        return []

    # @property
    # def _done_file(self):
    #     """
    #     e.g.
    #
    #     <--iml-directory>/SQLParserTask.task
    #     """
    #     return "{dir}/{name}.task".format(
    #         dir=self.iml_directory, name=self._task_name)

    # @property
    # def _task_name(self):
    #     return self.__class__.__name__

    # def mark_done(self, start_t, end_t):
    #     if self.skip_output:
    #         logging.info("> Skipping output={path} for task {name}".format(
    #             path=self._done_file,
    #             name=self._task_name))
    #         return
    #     with self.output().open('w') as f:
    #         print_cmd(cmd=sys.argv, file=f)
    #         delta = end_t - start_t
    #         minutes, seconds = divmod(delta.total_seconds(), 60)
    #         print(textwrap.dedent("""\
    #         > Started running at {start}
    #         > Ended running at {end}
    #         > Took {min} minutes and {sec} seconds.
    #         """.format(
    #             start=start_t,
    #             end=end_t,
    #             min=minutes,
    #             sec=seconds,
    #         )), file=f)

    # def iml_run(self):
    #     raise NotImplementedError("{klass} must override iml_run()".format(
    #         klass=self.__class__.__name__))

    def run(self):
        self.dumper = TraceEventsParser(
            directory=self.iml_directory,
            host=self.postgres_host,
            user=self.postgres_user,
            password=self.postgres_password,
            debug=self.debug,
            overlaps_event_id=self.overlaps_event_id,
            op_name=self.op_name,
            process_name=self.process_name,
            # process_op_nest=False,
            start_usec=self.start_usec,
            end_usec=self.end_usec,
        )
        self.dumper.run()

        # start_t = datetime.datetime.now()
        # self.iml_run()
        # end_t = datetime.datetime.now()
        #
        # self.mark_done(start_t, end_t)


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

from iml_profiler.profiler import glbl
def main(argv=None):
    glbl.setup_logging()
    if argv is None:
        argv = list(sys.argv[1:])
    luigi.run(cmdline_args=argv)

NOT_RUNNABLE_TASKS = get_NOT_RUNNABLE_TASKS()
IML_TASKS = get_IML_TASKS()
IML_TASKS.add(TraceEventsTask)

if __name__ == "__main__":
    main()
