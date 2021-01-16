"""
Read RL-Scope trace files into SQL database (PostgreSQL).

.. deprecated:: 1.0.0
    We now read from files directly since it's faster.
"""
from rlscope.profiler.rlscope_logging import logger
import psutil
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import subprocess
import progressbar
import sys
import tempfile
from decimal import Decimal
import getpass
psycopg2 = None
# import psycopg2
# import psycopg2.extras
# import psycopg2.pool
import random
import string
import itertools

import sqlite3

from os.path import join as _j, dirname as _d, exists as _e

from rlscope.profiler.util import pprint_msg
from rlscope.protobuf.pyprof_pb2 import CategoryEventsProto, MachineUtilization

from rlscope import py_config

from rlscope.parser.trace_events import dump_category_times

from rlscope.parser.readers import TFProfCategoryTimesReader, CUDAAPIStatsReader, CategoryEventsReader, CUDADeviceEventsReader, \
    DEFAULT_group_by_device, \
    DEFAULT_ignore_categories, \
    DEFAULT_debug \

import contextlib

from rlscope.parser.common import *
from rlscope.parser import constants

from rlscope.parser.stats import category_times_add_time

from rlscope.parser.stats import KernelTime

from rlscope.protobuf.pyprof_pb2 import CategoryEventsProto, MachineUtilization, ProcessMetadata, TP_HAS_PROGRESS, TP_NO_PROGRESS

from rlscope.parser.dataframe import TrainingProgressDataframeReader

SQLITE_TABLE_SQL = _j(py_config.ROOT, "sqlite", "tables.sql")
SQLITE_INDICES_SQL = _j(py_config.ROOT, "sqlite", "indices.sql")

PSQL_TABLE_SQL = _j(py_config.ROOT, "postgres", "tables.sql")
PSQL_INDICES_SQL = _j(py_config.ROOT, "postgres", "indices.sql")
PSQL_CONSTRAINTS_SQL = _j(py_config.ROOT, "postgres", "constraints.sql")

# If true, then Tasks run with rls-run will be forcefully limited to a maximum of
# 1 Postgres SQL connection per python-process.
#
# Ideally, we'd use this to prevent connection leak bugs.
# Haven't tested this so much, so best to leave it False until then.
USE_CONNECTION_POOLING = False
# USE_CONNECTION_POOLING = True

def Worker_get_device_names(kwargs):
    if kwargs['debug']:
        logger.info("> Start: Worker_get_device_names cuda_device_events_path={path}".format(path=kwargs['cuda_device_events_path']))
    reader = CUDADeviceEventsReader(kwargs['cuda_device_events_path'])
    device_names = reader.get_device_names()
    if kwargs['debug']:
        pprint.pprint({
            'device_names':device_names,
            'cuda_device_events_path':kwargs['cuda_device_events_path']})
        logger.info("> Stop: Worker_get_device_names cuda_device_events_path={path}".format(path=kwargs['cuda_device_events_path']))

    return reader.machine_name, device_names

class SQLParser:
    """
    Given a bunch of different trace files, insert them into a single SQLite database.

    The SQLite database represents a single "profiled run" of an RL algorithm.

    A "profiled run" can consist of:
    - Multiple samplings
    - Spread over multiple processes

    A single trace file only represents part of the profiled run, and for a single process.
    The SQLite database collect ALL trace files over all processes.

    <direc>/procceses/<process_name>:
        # The first "sampling period"
        CategoryEventsProto.trace_1.prof
        tfprof.trace_1.prof
    """

    # NOTE: By default postgres only allows 100 connections.  Also, we may be running multiple insert jobs.
    # So, keep this to a reasonable value.
    # Otherwise, we will trigger confusing errors like this:
    #
    # psycopg2.OperationalError: server closed the connection unexpectedly
    # This probably means the server terminated abnormally
    # before or while processing the request.
    MAX_CSV_WORKERS = 8

    def __init__(self, directory,
                 host=None,
                 user=None,
                 password=None,
                 # Swallow any excess arguments
                 debug=False,
                 debug_single_thread=False,
                 **kwargs):
        self.directory = directory
        self.host = host
        self.user = user
        self.password = password

        self.conn = get_sql_connection(db_path=self.db_path, host=self.host, user=self.user, password=self.password, debug=debug)
        self.debug = debug
        self.debug_single_thread = debug_single_thread
        self.block_size = 50000

    def get_source_files(self, should_keep, allow_empty=False, description='tfprof/pyprof'):
        """
        Visit every file in rlscope trace directory, and see whether we
        should insert in to the SQL database.

        :param allow_empty:
        :param should_keep:
            func: path -> bool

            Return true if we should keep/return this file.
        """
        src_files = []
        for dirpath, dirnames, filenames in os.walk(self.directory):
            for base in filenames:
                path = _j(dirpath, base)
                if should_keep(path):
                    src_files.append(path)
        if not allow_empty and len(src_files) == 0:
            raise MissingInputFiles(textwrap.dedent("""
            {klass}: Couldn't find any {desc} files root at {dir}.
            """.format(
                klass=self.__class__.__name__,
                desc=description,
                dir=self.directory,
            )))
        return src_files

    def _remove_buggy_data(self):
        """
        TODO: BUG: sometimes, this results in a negative duration_us for Event.duration_us for Python event's.
        HACK: just filter out these rare events when building SQL database.

        For details; see CFuncWrapper.__call__
        """
        logger.info("> BUG: remove Python Events with negative Event.duration_us")
        c = self.cursor

        negative_duration_query = textwrap.dedent("""
        SELECT e.event_name, e.duration_us, c.category_name
        FROM
          Event AS e 
          NATURAL JOIN Category c
        WHERE 
            e.duration_us < 0
        """)
        sql_exec_query(c, negative_duration_query, klass=self.__class__, debug=self.debug)
        rows = c.fetchall()

        if len(rows) == 0:
            return

        pprint.pprint({
            'num_rows': len(rows),
            'Events with negative Event.duration_us':[dict(row) for row in rows],
        })

        del_negative_duration_query = textwrap.dedent("""
        DELETE
        FROM
            Event AS e
        WHERE
            e.duration_us < 0
        """)
        sql_exec_query(c, del_negative_duration_query, klass=self.__class__, debug=self.debug)

    def _check_no_partial_or_complete_op_overlap(self):
        """
        User adds profiling annotations in stack-oriented fashion:

            e.g.

            # with op1:
            prof.set_operation('op1')
            # with op2:
            prof.set_operation('op2')
            prof.end_operation('op2')
            prof.end_operation('op1')

        We don't allow non-stack-oriented annotations:

            e.g.

            prof.set_operation('op1')
            prof.set_operation('op2')
            prof.end_operation('op1')
            prof.end_operation('op2')

        The reason we don't allow this is so when we have overlapping operations types
        in a super-operation/sub-operation arrangement,
        we can unambiguously tell which operation to attribute execution time to
        (i.e. op2 "absorbs" inner operation time of op1 in the first example)

        To check for this, we assert that are no 2 op_events in the database that have partial-overlap:

        e.g.

        [    op1    ]
                [    op2    ]

                ----
                partial overlap

        Partial overlap condition:
          op1.start < op2.start < op1.end < op2.end

                [    op1    ]
                [    op2    ]

                ----
                complete overlap

        Partial overlap condition:
          op1.start == op2.start and
          op1.end   == op2.end

        We only allow "full overlap", or "no overlap"
        """
        c = self.cursor

        # [    op1    ]
        #         [    op2    ]
        #         -----
        #         partial overlap
        #
        # op1.start         < op2.start         < op1.end                                 < op2.end
        # op1.start_time_us < op2.start_time_us < ( op1.start_time_us + op1.duration_us ) < ( op2.start_time_us + op2.duration_us )
        partial_overlap_query = """
        SELECT * 
        FROM Event AS op1 NATURAL JOIN Category c1
        WHERE 
        c1.category_name = '{CATEGORY_OPERATION}' AND
        EXISTS (
            SELECT * 
            FROM Event AS op2 NATURAL JOIN Category c2
            WHERE 
                op1.event_id != op2.event_id AND
                op1.process_id = op2.process_id AND
                c2.category_name = '{CATEGORY_OPERATION}' AND
                (
                    -- Partial overlap
                    op1.start_time_us < op2.start_time_us AND
                                        op2.start_time_us < op1.end_time_us AND
                                                             op1.end_time_us < op2.end_time_us 
                )
        )
        """.format(CATEGORY_OPERATION=constants.CATEGORY_OPERATION)
        sql_exec_query(c, partial_overlap_query, klass=self.__class__, debug=self.debug)
        rows = c.fetchall()
        if len(rows) > 0:
            pprint.pprint({'Events with partial overlap':[dict(row) for row in rows]})
            raise RuntimeError("ERROR: Detected partial-overlap between operation-type events")

        # op1.start == op2.start and
        # op1.end   == op2.end
        full_overlap_query = """
        SELECT * FROM Event AS op1 NATURAL JOIN Category c1
        WHERE 
        c1.category_name = '{CATEGORY_OPERATION}' AND
        EXISTS (
            SELECT * FROM Event AS op2 NATURAL JOIN Category c2
            WHERE 
            op1.event_id != op2.event_id AND
            op1.process_id = op2.process_id AND
            c2.category_name = '{CATEGORY_OPERATION}' AND
            (
                -- Complete overlap
                op1.start_time_us == op2.start_time_us AND
                op1.end_time_us == op2.end_time_us
            )
        )
        """.format(CATEGORY_OPERATION=constants.CATEGORY_OPERATION)
        sql_exec_query(c, full_overlap_query, klass=self.__class__, debug=self.debug)
        rows = c.fetchall()
        if len(rows) > 0:
            pprint.pprint({'Events with complete overlap':[dict(row) for row in rows]})
            raise RuntimeError("ERROR: Detected complete-overlap between operation-type events")

    def single_thread_iter(self, worker, worker_kwargs):
        for kwargs in worker_kwargs:
            ret = worker(kwargs)
            yield ret

    def run(self):
        # 1. Create SQLite db file
        # for each input file:
        #   if is pyprof file:
        #     ...
        #     insert into db
        #   elif if tfprof file:
        #     ...
        #     insert into db

        # SQLite: create a new trace.db file; delete if already exists.
        # Postgres: allocate/recreate database.
        self.create_db(recreate=True)

        self.conn.create_connection()

        self.process_to_id = dict()
        self.phase_to_id = dict()
        self.category_to_id = dict()
        self.device_to_id = dict()
        self.machine_to_id = dict()

        src_files = self.get_source_files(should_keep=is_insertable_file)
        pprint.pprint({
            'rule':self.__class__.__name__,
            'src_files':src_files,
        })

        if self.debug:
            logger.info("> Read metadata.")

        # Read metadata from CPU/GPU utilization files
        # e.g. machine_util.trace_0.proto
        util_metas = []
        for path in src_files:
            if not is_machine_util_file(path):
                continue
            if self.debug:
                logger.info("> get_util_metadata path={path}".format(path=path))
            md = get_util_metadata(path)
            util_metas.append(md)
        machine_name_to_device_names = dict()
        for util_meta in util_metas:
            if util_meta['machine_name'] not in machine_name_to_device_names:
                machine_name_to_device_names[util_meta['machine_name']] = set()
            machine_name_to_device_names[util_meta['machine_name']].update(util_meta['device_names'])

        if self.debug:
            logger.info("> Insert machines.")
            logger.info(pprint_msg({'machine_name_to_device_names':machine_name_to_device_names}))
        for machine_name in machine_name_to_device_names.keys():
            self.insert_machine_name(machine_name, debug=self.debug)

        if self.debug:
            logger.info("> Insert devices.")
        for machine_name, device_names in machine_name_to_device_names.items():
            machine_id = self.machine_to_id[machine_name]
            for device_name in device_names:
                self.insert_device_name(device_name, fields={
                    'machine_id': machine_id,
                })

        process_trace_metas = []
        for path in src_files:
            if not is_process_trace_file(path):
                continue
            if self.debug:
                logger.info("> get_process_trace_metadata path={path}".format(path=path))
            md = get_process_trace_metadata(path)
            process_trace_metas.append(md)

        # ProcessMetadata
        process_metadata_paths = self.get_source_files(
            should_keep=is_process_metadata_file,
            allow_empty=False,
            description='ProcessMetadata protobuf')
        process_metadatas = [read_process_metadata_file(path) for path in process_metadata_paths
                             if is_process_metadata_file(path)]

        if self.debug:
            logger.info("> Insert processes.")

        # Options
        # 1. Insert each process in reference order
        # 2. Sort process_name's, assign process_id's in name-order, then bulk-insert processes using back-reference to assign parent_id's
        #    PRO: bulk-insert friendly

        # process_names = sorted(meta['process_name'] for meta in process_trace_metas)
        process_names = set([md.process_name for md in process_metadatas])
        process_to_assigned_id = dict((process_name, ident) for ident, process_name in enumerate(process_names, start=1))

        if self.debug:
            logger.info(pprint_msg({'process_to_assigned_id': process_to_assigned_id}))

        for md in process_metadatas:
            fields = {
                'process_id': process_to_assigned_id[md.process_name],
                'machine_id': self.machine_to_id[md.machine_name],
            }
            # TODO: We should use better defaults for unset values in protobuf (e.g. -1 for ints/floats)
            if md.parent_process_name:
                fields['parent_process_id'] = process_to_assigned_id[md.parent_process_name]
            # if md.training_progress.content_code == TP_HAS_PROGRESS:
            #     fields['percent_complete'] = md.training_progress.percent_complete
            #     if md.training_progress.num_timesteps != 0:
            #         fields['num_timesteps'] = md.training_progress.num_timesteps
            #     if md.training_progress.total_timesteps != 0:
            #         fields['total_timesteps'] = md.training_progress.total_timesteps
            self.insert_process_name(md.process_name, fields=fields, debug=True)

        if self.debug:
            logger.info("> Insert phases.")

        phase_names = sorted(meta['phase_name'] for meta in process_trace_metas)
        for phase_name in phase_names:
            self.insert_phase_name(phase_name)

        if self.debug:
            logger.info("> Read util metadata.")


        if self.debug:
            logger.info("> Insert categories.")
        categories = sorted(set(constants.CATEGORIES_ALL))
        for category in categories:
            self.insert_category_name(category)

        if self.debug:
            logger.info("> Insert tfprof device names.")

        def get_Worker_get_device_names_kwargs(cuda_device_events_path):
            return {'cuda_device_events_path':cuda_device_events_path, 'debug':self.debug}

        device_names_kwargs = [get_Worker_get_device_names_kwargs(path)
                               for path in src_files if is_cuda_device_events_file(path)]

        if not self.debug_single_thread:
            device_name_pool = multiprocessing.Pool()
            imap_iter = device_name_pool.imap_unordered(Worker_get_device_names, device_names_kwargs)
        else:
            imap_iter = self.single_thread_iter(Worker_get_device_names, device_names_kwargs)

        machine_to_device_names = dict()
        for machine_name, dev_names in tqdm_progress(imap_iter, desc='Device names', total=len(device_names_kwargs)):
            if machine_name not in machine_to_device_names:
                machine_to_device_names[machine_name] = set()
            machine_to_device_names[machine_name].update(dev_names)
            # if is_tfprof_file(path):
            #     reader = TFProfCategoryTimesReader(path)
            #     device_names.update(reader.get_device_names())
        if self.debug:
            logger.info(pprint_msg({'machine_to_device_names':machine_to_device_names}))

        if not self.debug_single_thread:
            device_name_pool.close()
            device_name_pool.join()

        for machine_name, device_names in machine_to_device_names.items():
            machine_id = self.machine_to_id[machine_name]
            for device_name in device_names:
                self.insert_device_name(device_name, fields={
                    'machine_id': machine_id,
                })

        if self.debug:
            logger.info("> Commit.")
        self.conn.commit()

        logger.info("Entity id maps:\n{maps}".format(
            maps=textwrap.indent(pprint.pformat({
                'process_to_id':self.process_to_id,
                'category_to_id':self.category_to_id,
                'device_to_id':self.device_to_id,
                'machine_to_id':self.machine_to_id,
            }), prefix='  ')))

        if not self.debug_single_thread:
            pool = multiprocessing.Pool(processes=SQLParser.MAX_CSV_WORKERS)
        # table = 'Event'
        # id_field = 'event_id'
        id_field = None

        if self.debug:
            logger.info("> Insert table files.")

        def get_worker_kwargs(path):
            if is_machine_util_file(path):
                table = 'DeviceUtilization'
            elif is_training_progress_file(path):
                table = 'TrainingProgress'
            else:
                table = 'Event'

            return {
                'path':path,
                'db_path':self.db_path,
                'table':table,
                'block_size':self.block_size,
                'id_field':id_field,
                'directory':self.directory,
                'host':self.host,
                'user':self.user,
                'password':self.password,
                'debug':self.debug,
            }
        worker_kwargs = [get_worker_kwargs(path) for path in src_files]

        if not self.debug_single_thread:
            if self.debug:
                logger.info("> Insert table files using thread pool.")
            imap_iter = pool.imap_unordered(mk_TraceFileInserter, worker_kwargs)
        else:
            if self.debug:
                logger.info("> Insert table files using single thread.")
            imap_iter = self.single_thread_iter(mk_TraceFileInserter, worker_kwargs)

        with progressbar.ProgressBar(max_value=len(src_files), prefix="SQL insert") as bar:
            for i, result in enumerate(imap_iter):
                # logger.info("> i={i}, result={result}".format(
                #     i=i, result=result))
                bar.update(i)

        if not self.debug_single_thread:
            pool.close()
            pool.join()

        # Create indices at the end to reduce per-insert overhead.
        self.create_indices()
        self.create_constraints()

        # This takes REALLY long with psql...
        # Not sure why.
        # self._check()

        self._HACK_clean_data()

        self.conn.commit()
        self.conn.close()

    def _check(self):
        """
        Run any post-insert checks not captured by database constraints.
        """
        logger.info("> Check event data...")
        start_t = time.time()
        self._check_no_partial_or_complete_op_overlap()
        end_t = time.time()
        time_sec = end_t - start_t
        logger.info("  Took {sec} seconds".format(sec=time_sec))

    def _HACK_clean_data(self):
        self._remove_buggy_data()

    def maybe_commit(self, i):
        if (i + 1) % self.block_size == 0:
            self.conn.commit()

    def insert_process_name(self, process_name, fields=None, debug=False):
        return self._insert_name(
            'Process',
            'process_id', 'process_name',
            self.process_to_id,
            process_name,
            fields=fields,
            debug=debug,
            # TODO: add fields: parent_process_id, machine_id
            # TODO: parent process must be inserted BEFORE child process; do we generate id's locally to avoid this?
            # TODO: make sure machine_id is inserted before process that references it
        )

    def insert_phase_name(self, phase_name, fields=None, debug=False):
        return self._insert_name(
            'Phase',
            'phase_id', 'phase_name',
            self.phase_to_id,
            phase_name,
            fields=fields,
            debug=debug,
        )

    def insert_device_name(self, device_name, fields=None, debug=False):
        return self._insert_name(
            'Device',
            'device_id', 'device_name',
            self.device_to_id,
            device_name,
            fields=fields,
            debug=debug,
        )

    def insert_machine_name(self, machine_name, fields=None, debug=False):
        return self._insert_name(
            'Machine',
            'machine_id', 'machine_name',
            self.machine_to_id,
            machine_name,
            fields=fields,
            debug=debug,
        )

    def insert_category_name(self, category_name, fields=None, debug=False):
        return self._insert_name(
            'Category',
            'category_id', 'category_name',
            self.category_to_id,
            category_name,
            fields=fields,
            debug=debug,
        )

    @property
    def cursor(self):
        return self.conn.cursor

    def _insert_name(self, table, id_field, name_field, name_to_id, name, fields=None, debug=False):
        """
        Insert into a table like this, if a row does not exist already.

        CREATE TABLE Table (
          table_id SERIAL NOT NULL PRIMARY KEY,
          table_name TEXT,
          UNIQUE (table_name)
        );

        :param table:
            e.g. 'Table' in the above schema.
        :param id_field:
            e.g. 'table_id' in the above schema.
        :param name_field:
            e.g. 'table_name' in the above schema.
        :param name_to_id:
            dict: primary-key -> string
            A cached mapping primary-key to table_name from table-row.
        :param name:
            Name to store inside table_name row field.
        :param fields:
            A dict mapping from Table.attr to row-value.
            Extra fields to insert for this row.
        :return:
        """

        # See if the row is cached.
        if name in name_to_id:
            return name_to_id[name]

        # See if the row already exists.
        c = self.cursor
        query = textwrap.dedent("""
        SELECT {id_field} from {table} WHERE {name_field} = {p}
        """.format(
            id_field=id_field,
            table=table,
            name_field=name_field,
            p=sql_placeholder(),
        ))
        params = (name,)
        sql_exec_query(c, query, params, debug=debug)
        rows = c.fetchall()

        if len(rows) == 0:
            # Row doesn't exist; insert it and obtain the insert id (if autoincrement).
            dic = {
                name_field: name,
            }
            if fields is not None:
                dic.update(fields)
            if debug:
                logger.info(pprint_msg({'insert_dict':dic}))
            ident = self.conn.insert_dict(table, dic, id_field=id_field, debug=debug)
        else:
            # Row exists; obtain primary-key.
            ident = rows[0][id_field]
            if debug:
                logger.info("Row exists but primary-key wasn't cached: {table}['{name}'] = {id}".format(
                    table=table,
                    name=name,
                    id=ident,
                ))

        if debug:
            logger.info("Cache entity: {table}['{name}'] = {id}".format(
                table=table,
                name=name,
                id=ident,
            ))
        name_to_id[name] = ident

        return ident

    # def insert_pyprof_file(self, path):
    #     with open(path, 'rb') as f:
    #         proto = CategoryEventsProto()
    #         proto.ParseFromString(f.read())
    #
    #     logger.info("> Insert pyprof file: {p}".format(p=path))
    #     if self.debug:
    #         logger.info(proto)
    #
    #     c = self.cursor
    #     # Insert Process
    #     process_id = self.insert_process_name(proto.process_name)
    #
    #     # categories = set()
    #     def insert_category_events(event_conn, eventattr_conn, category, events):
    #         # Insert Category
    #         # categories.add(category)
    #         category_id = self.insert_category_name(category)
    #         # category_id = self.category_to_id[category]
    #         for event in events:
    #             # Insert Event
    #             is_debug_event = bool(match_debug_event_name(event.name))
    #             event_id = event_conn.insert_dict('Event', {
    #                 'thread_id':event.thread_id,
    #                 'start_time_us':event.start_time_us,
    #                 'end_time_us':event.start_time_us + event.duration_us,
    #                 'duration_us':event.duration_us,
    #                 'event_name':event.name,
    #                 'category_id':category_id,
    #                 'process_id':process_id,
    #                 'is_debug_event':is_debug_event,
    #             })
    #             # Insert EventAttr
    #             for attr_name, attr_value in event.attrs.items():
    #                 attr_id = eventattr_conn.insert_dict('EventAttr', {
    #                     'event_id':event_id,
    #                     'attr_name':attr_name,
    #                     'attr_value':attr_value,
    #                 })
    #
    #     def each_category_events():
    #         for step, python_events in proto.python_events.items():
    #             yield constants.CATEGORY_PYTHON, python_events.events
    #
    #         for step, clibs in proto.clibs.items():
    #             for category, clib_events in clibs.clibs.items():
    #                 yield category, clib_events.events
    #
    #     num_all_events = sum(len(events) for category, events in each_category_events())
    #
    #     with progressbar.ProgressBar(max_value=num_all_events) as bar, \
    #         bulk_inserter(self.conn, 'EventAttr', self.block_size, bar, directory=self.directory) as event_attr_bulk, \
    #         bulk_inserter(self.conn, 'Event', self.block_size, progress_bar=None, id_field='event_id', directory=self.directory) as event_bulk:
    #         for category, events in each_category_events():
    #             insert_category_events(event_bulk, event_attr_bulk, category, events)
    #
    #     self.conn.commit()

    def create_db(self, recreate):
        self.conn.create_db(recreate)

        # NOTE: This doesn't provide line numbers when things fail.
        #
        # with open(SQLITE_TABLE_SQL) as f:
        #     script = f.read()
        # self.cursor.executescript(script)

    def create_indices(self):
        logger.info("> Create indices...")
        start_t = time.time()
        self.conn.create_indices()
        end_t = time.time()
        time_sec = end_t - start_t
        logger.info("  Took {sec} seconds".format(sec=time_sec))

    def create_constraints(self):
        logger.info("> Create constraints...")
        start_t = time.time()
        self.conn.create_constraints()
        end_t = time.time()
        time_sec = end_t - start_t
        logger.info("  Took {sec} seconds".format(sec=time_sec))

    @property
    def db_path(self):
        return sql_input_path(self.directory)

@contextlib.contextmanager
def transaction(conn):
    conn.cursor.execute('BEGIN TRANSACTION')
    try:
        yield
    except Exception as e:
        raise
    conn.cursor.execute('COMMIT')

@contextlib.contextmanager
def connection(conn):
    try:
        yield
    finally:
        conn.close()
    # Q: Should we commit?
    # except Exception as e:
    #     conn.close()
    #     raise
    # conn.commit()
    # conn.close()

class BulkInserter:
    def __init__(self, conn, table, block_size=50000, progress_bar=None, id_field=None, directory=None):
        """
        Bulk insert rows into an SQLite table efficiently by doing batching
        and starting each batch with a BEGIN/END TRANSACTION.

        :param conn:
        :param table:
        :param block_size:
        :param progress_bar:
        :param id_field:
            The autoincrement primary key of the table.
            If this is set, query the last insert id, and inject the next id into each add_insert call.

            Useful when bulk inserting multiple tables that reference each other.
        """
        self.conn = conn
        self.table = table
        # Ignored.
        # self.directory = directory
        self.inserts = []
        self.block_size = block_size
        self.total_inserts = 0
        self.progress_bar = progress_bar
        self.id_field = id_field
        self.next_id = None

        if self.id_field is not None:
            self.next_id = self._last_insert_id() + 1

    def _last_insert_id(self):
        assert self.id_field is not None
        c = self.conn.cursor
        c.execute("""
        SELECT MAX({id}) as id FROM {table}
        """.format(
            id=self.id_field,
            table=self.table))
        row = c.fetchone()
        ident = row['id']
        if ident is None:
            # There are no rows in the table!
            # logger.info("> IDENT is None, use 0")
            ident = 0
        return ident

    def _commit_inserts(self):
        with transaction(self.conn):
            self.conn.insert_dicts(self.table, self.inserts)
        self.conn.commit()
        self.total_inserts += len(self.inserts)
        self.inserts.clear()
        if self.progress_bar is not None:
            self.progress_bar.update(self.total_inserts)

    # Match TracesSQLiteConnection interface
    def insert_dict(self, table, insert):
        assert self.table == table
        return self.add_insert(insert)

    def add_insert(self, insert):
        ret = None

        if self.id_field is not None:
            insert[self.id_field] = self.next_id
            ret = self.next_id
            self.next_id += 1

        self.inserts.append(insert)
        if len(self.inserts) >= self.block_size:
            self._commit_inserts()

        return ret

    def finish(self):
        if len(self.inserts) > 0:
            self._commit_inserts()

class AutoincrementID:
    def __init__(self, conn, table, id_field, counter=None, initial_id=None):
        self.conn = conn
        self.table = table
        self.id_field = id_field
        self.counter = counter
        self.initial_id = initial_id
        if self.counter is None:
            self._init_counter()

    def _init_counter(self):
        self.counter = multiprocessing.Value('i', 0)
        self.counter.value = self._last_insert_id() + 1
        self.initial_id = self.counter.value

    def num_inserts(self, next_id):
        return next_id - self.initial_id

    def next_id(self):
        with self.counter.get_lock():
            ident = self.counter.value
            self.counter.value += 1
        return ident

    def _last_insert_id(self):
        assert self.id_field is not None
        c = self.conn.cursor
        c.execute("""
        SELECT MAX({id}) as id FROM {table}
        """.format(
            id=self.id_field,
            table=self.table))
        row = c.fetchone()
        ident = row['id']
        if ident is None:
            # There are no rows in the table!
            # logger.info("> IDENT is None, use 0")
            ident = 0
        return ident

def mk_TraceFileInserter(kwargs):
    # Limit trace file insertion worker to 1 postgres connection.
    with GetConnectionPool(conn_kwargs=dict(
        db_path=kwargs['db_path'],
        host=kwargs['host'],
        user=kwargs['user'],
        password=kwargs['password'],
    ), maxconn=1, debug=kwargs['debug']) as pool:
        worker = TraceFileInserter(**kwargs)
        worker.run()


class CSVInserter:
    def __init__(self, db_path, table, block_size=50000, id_field=None, directory=None,
                 host=None,
                 user=None,
                 password=None,
                 cursor=None,
                 csv_path=None,
                 debug=False):
        self.host = host
        self.user = user
        self.password = password
        self.cursor = cursor
        self.db_path = db_path
        self.table = table
        self.block_size = block_size
        self.id_field = id_field
        self.directory = directory
        self.debug = debug
        self.conn = None

        if directory is None:
            self.directory = os.getcwd()
        else:
            self.directory = directory

        self.progress_bar = None

        self.csv_path = csv_path

    def __enter__(self):
        self._init_csv()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()

    def _init_csv(self):
        """
        If this is being run inside of a multiprocessing.Process,
        need to make sure database connections get created AFTER entering Process.
        Init after new Process.
        """
        self.fields = None
        self.header_written = False
        self.total_inserts = 0
        self.is_empty = True

        if self.csv_path is not None:
            self.tmp_path = self.csv_path
            self.tmp_f = open(self.tmp_path, 'w')
        else:
            self.tmp_f, self.tmp_path = tempfile.mkstemp(
                dir=self.directory,
                prefix="{table}_".format(
                    table=self.table,
                ),
                suffix=".csv",
                text=True)
            os.chmod(self.tmp_path, 0o664)
            self.tmp_f = os.fdopen(self.tmp_f, mode='w')

        if self.cursor is None:
            self.conn = get_sql_connection(db_path=self.db_path, host=self.host, user=self.user, password=self.password, debug=self.debug)
            self.cursor = self.conn.cursor

        self.process_to_id = self.build_name_to_id('Process', 'process_id', 'process_name')
        self.phase_to_id = self.build_name_to_id('Phase', 'phase_id', 'phase_name')
        self.category_to_id = self.build_name_to_id('Category', 'category_id', 'category_name')
        self.device_to_id = self.build_name_to_id('Device', 'device_id', 'device_name')
        self.machine_to_id = self.build_name_to_id('Machine', 'machine_id', 'machine_name')

    # @property
    # def cursor(self):
    #     return self.conn.cursor

    def build_name_to_id(self, table, id_field, name_field):
        c = self.cursor
        query = """
        SELECT {name_field} AS name, {id_field} AS ident 
        FROM {table} 
        """.format(
            id_field=id_field,
            table=table,
            name_field=name_field,
            p=sql_placeholder(),
        )
        sql_exec_query(
            c, query,
            # klass=self.__class__, debug=self.debug)
            klass=self.__class__, debug=False)
        rows = c.fetchall()
        name_to_id = dict((row['name'], row['ident']) for row in rows)
        return name_to_id

    def insert_dict(self, table, insert):
        assert self.table == table
        return self.add_insert(insert)

    def insert_event(self, device_id, process_id, phase_id,
                     category, start_time_us, duration_us, name,
                     thread_id=None):
        """
        Insert a row into Event.
        """
        category_id = self.category_to_id[category]
        end_time_us = Decimal(start_time_us) + Decimal(duration_us)
        is_debug_event = bool(match_debug_event_name(name))
        insert = {
            'start_time_us':start_time_us,
            'end_time_us':end_time_us,
            'duration_us':duration_us,
            'event_name':name,
            'category_id':category_id,
            'process_id':process_id,
            'phase_id':phase_id,
            'device_id':device_id,
            'is_debug_event':is_debug_event,
        }
        if thread_id is not None:
            insert['thread_id'] = thread_id
        self.add_insert(insert)

    def add_insert(self, insert):
        ret = None

        if self.id_field is not None:
            next_id = self.autoincrement.next_id()
            if type(insert) == dict:
                insert[self.id_field] = next_id
            else:
                insert.append(next_id)
            ret = next_id

        if not self.header_written:
            if self.fields is None:
                # Already contains id_field if required.
                self.fields = sorted(insert.keys())
            elif self.id_field is not None:
                self.fields.append(self.id_field)
            self._write_csv_header()
            self.header_written = True

        self._write_csv_insert(insert)

        self.total_inserts += 1
        if self.progress_bar is not None:
            self.progress_bar.update(self.total_inserts)

        return ret

    def finish(self):
        self.tmp_f.flush()

        if not self.is_empty:
            start_t = time.time()
            if self.conn is not None:
                self.conn.insert_csv(self.tmp_path, self.table)
            else:
                self.cursor.conn.insert_csv(self.tmp_path, self.table)
            end_t = time.time()
            if self.debug:
                logger.info("> Loading CSV into {table} took {sec} seconds".format(
                    table=self.table,
                    sec=end_t - start_t))

        self.tmp_f.close()
        os.remove(self.tmp_path)
        self.close()

    def close(self):
        # Q: Should we do this...? If someone passes a cursor to us, we'd like THEM to manage its lifetime...
        # If someone doesn't pass a cursor to us, then self.conn.close() will take care of closing it.
        # if self.cursor is not None:
        #     self.cursor.close()
        if self.conn is not None:
            self.conn.close()

    def commit(self):
        if self.conn is not None:
            self.conn.commit()

    def _csv_val(self, val):
        if val is None:
            return ''
        if type(val) == str:
            if ',' in val:
                raise NotImplementedError("Need to escape comma's, or use a quote-char.")
        return str(val)

    def _write_csv_insert(self, insert):
        if type(insert) == dict:
            line = ','.join(self._csv_val(insert[k]) for k in self.fields)
        else:
            line = ','.join(self._csv_val(val) for val in insert)
        self.tmp_f.write(line)
        self.tmp_f.write("\n")
        self.is_empty = False

    def _write_csv_header(self):
        line = ','.join(self.fields)
        self.tmp_f.write(line)
        self.tmp_f.write("\n")

class TraceFileInserter:
    def __init__(self,
                 # RL-Scope trace file proto.
                 path,
                 # CSVInserter fields.
                 db_path, table, block_size=50000, id_field=None, directory=None,
                 host=None,
                 user=None,
                 password=None,
                 debug=False,
                 csv_path=None):
        # RL-Scope trace file proto.
        self.path = path
        self.csv_inserter = CSVInserter(
            db_path, table,
            block_size=block_size,
            id_field=id_field,
            directory=directory,
            host=host,
            user=user,
            password=password,
            debug=debug,
            csv_path=csv_path)
        self.debug = debug

    def run(self):
        with self.csv_inserter:
            self.insert_file(self.path)

    def insert_file(self, path):
        if is_category_events_file(path):
            self.insert_category_events_file(path)
        elif is_training_progress_file(path):
            self.insert_training_progress_file(path)
        elif is_machine_util_file(path):
            self.insert_machine_util_file(path)
        elif is_cuda_device_events_file(path):
            self.insert_cuda_device_events_file(path)
        elif is_cuda_api_stats_file(path):
            self.insert_cuda_api_stats_file(path)
        else:
            raise NotImplementedError("Not sure how to insert into path={path} into database".format(path=path))

    def get_cpu_device(self):
        # We don't bother to record the device-name CUDA API call belongs to;
        # During analysis we count this as CPU-side events, so we associate
        # with the single CPU device on the machine.
        cpu_devices = [dev for dev, ident in self.csv_inserter.device_to_id.items() if is_cpu_device(dev)]
        assert len(cpu_devices) == 1
        device = cpu_devices[0]
        return device

    def lookup_device_id(self, path, device):
        if device not in self.csv_inserter.device_to_id:
            logger.info("> ERROR: Couldn't find device={dev} in path={path}".format(
                dev=device,
                path=path))
            pprint.pprint({
                'device_to_id': self.csv_inserter.device_to_id})
        device_id = self.csv_inserter.device_to_id[device]
        return device_id

    def insert_cuda_api_stats_file(self, path):
        reader = CUDAAPIStatsReader(path)

        logger.info("> Insert CUDA API Stats file: {p}".format(p=path))

        process_id = self.csv_inserter.process_to_id[reader.process_name]
        phase_id = self.csv_inserter.phase_to_id[reader.phase_name]

        device = self.get_cpu_device()

        for i, event in enumerate(reader.all_events(debug=True)):
            category, start_time_us, duration_us, name = event
            device_id = self.lookup_device_id(path, device)
            self.csv_inserter.insert_event(
                device_id, process_id, phase_id,
                category, start_time_us, duration_us, name)

    def insert_cuda_device_events_file(self, path):
        reader = CUDADeviceEventsReader(path)

        logger.info("> Insert CUDA Device Events file: {p}".format(p=path))

        process_id = self.csv_inserter.process_to_id[reader.process_name]
        phase_id = self.csv_inserter.phase_to_id[reader.phase_name]

        for i, event in enumerate(reader.all_events(debug=True)):
            device, category, start_time_us, duration_us, name = event
            device_id = self.lookup_device_id(path, device)
            self.csv_inserter.insert_event(
                device_id, process_id, phase_id,
                category, start_time_us, duration_us, name)

    def _pyprof_each_category_events(self, pyprof_proto):
        for step, python_events in pyprof_proto.python_events.items():
            yield constants.CATEGORY_PYTHON, python_events.events

        for step, clibs in pyprof_proto.clibs.items():
            for category, clib_events in clibs.clibs.items():
                yield category, clib_events.events

    def insert_training_progress_file(self, path):
        proto = read_training_progress_file(path)

        logger.info("> Insert training_progress file: {p}".format(p=path))

        process_id = self.csv_inserter.process_to_id[proto.process_name]
        phase_id = self.csv_inserter.phase_to_id[proto.phase]

        fields = {
            'process_id': process_id,
            'phase_id': phase_id,
            'total_timesteps': proto.total_timesteps,
            'start_trace_time_us': proto.start_trace_time_us,
            'start_percent_complete': proto.start_percent_complete,
            'start_num_timesteps': proto.start_num_timesteps,
            'start_training_time_us': proto.start_training_time_us,
            'end_percent_complete': proto.end_percent_complete,
            'end_training_time_us': proto.end_training_time_us,
            'end_num_timesteps': proto.end_num_timesteps,
        }
        self.csv_inserter.insert_dict('TrainingProgress', fields)

        self.csv_inserter.commit()

    def insert_category_events_file(self, path):
        reader = CategoryEventsReader(path)

        logger.info("> Insert Category Events file: {p}".format(p=path))

        process_id = self.csv_inserter.process_to_id[reader.process_name]
        phase_id = self.csv_inserter.phase_to_id[reader.phase_name]

        device = self.get_cpu_device()

        for i, event in enumerate(reader.all_events(debug=True)):
            category, start_time_us, duration_us, name = event
            device_id = self.lookup_device_id(path, device)
            self.csv_inserter.insert_event(
                device_id, process_id, phase_id,
                category, start_time_us, duration_us, name)

    def insert_machine_util_file(self, path):
        proto = read_machine_util_file(path)

        logger.info("> Insert machine CPU/GPU utilization file: {p}".format(p=path))

        machine_id = self.csv_inserter.machine_to_id[proto.machine_name]

        def each_sample(machine_util):
            for device_name, device_util in machine_util.device_util.items():
                for sample in device_util.samples:
                    device_id = self.csv_inserter.device_to_id[device_name]
                    yield device_id, sample.start_time_us, sample.util, sample.total_resident_memory_bytes

        def count_samples(machine_util):
            n = 0
            for device_name, device_util in machine_util.device_util.items():
                n += len(device_util.samples)
            return n

        num_all_samples = count_samples(proto)

        with progressbar.ProgressBar(max_value=num_all_samples) as bar:
            for i, (device_id, start_time_us, util, total_resident_memory_bytes) in enumerate(each_sample(proto)):
                ins = {
                    'device_id':device_id,
                    'machine_id':machine_id,
                    'start_time_us':start_time_us,
                    'util':util,
                    'total_resident_memory_bytes':total_resident_memory_bytes,
                }
                device_util_id = self.csv_inserter.insert_dict('DeviceUtilization', ins)
                bar.update(i)

        self.csv_inserter.commit()

class Cursor:
    def __init__(self, cursor, conn):
        self.cursor = cursor
        self.conn = conn

    def execute(self, query, vars=None):
        return self.cursor.execute(query, vars=vars)

    def executemany(self, query, vars_list):
        return self.cursor.executemany(query, vars_list)

    def fetchall(self):
        return self.cursor.fetchall()

    def fetchone(self):
        return self.cursor.fetchone()

    def fetchmany(self, size=None):
        return self.cursor.fetchall(size)

    @property
    def rowcount(self):
        return self.cursor.rowcount

    def close(self):
        return self.cursor.close()

    def __iter__(self):
        self._iter = iter(self.cursor)
        return self

    def __next__(self):
        row = next(self._iter)
        return row

class TracesPostgresConnection:
    def __init__(self, db_config_path, db_basename='traces', host=None, user=None, password=None, keep_alive=True, pool=None):
        self.host = host
        self.user = user
        self.password = password
        self.rand_suffix_len = 4
        self.db_config_path = db_config_path
        self.db_config = None
        self.db_basename = db_basename
        self.db_name = None
        self._cursor = None
        self._conn = None
        self.pg_conn = None
        self.pg_cursor = None
        self.keep_alive = keep_alive
        self.pool = pool
        self._cursors = []

    @property
    def _pool_closed(self):
        if self.pool is None:
            return False
        return self.pool.closed

    def _check_alive(self, allow_reconnect=False):
        try:
            self._cursor.execute('SELECT 1')
            rows = self._cursor.fetchall()
            assert len(rows) == 1
            assert len(rows[0]) == 1
            assert rows[0][0] == 1
        except psycopg2.OperationalError as e:
            if not allow_reconnect:
                logger.info("Detected dead connection to {user}@{host} db={db}; failed reconnect".format(
                    user=self.user,
                    host=self.host,
                    db=self.db_name,
                ))
                raise
            logger.info("Detected closed connection to {user}@{host} db={db}; attempting reconnect".format(
                user=self.user,
                host=self.host,
                db=self.db_name,
            ))
            self._conn.close()
            self._cursor.close()
            self._conn = None
            self._cursor = None
            self._maybe_create_db_connection()
            self._check_alive(allow_reconnect=False)

    @property
    def conn(self):
        assert not self._pool_closed
        if self.keep_alive and self._cursor is not None:
            self._check_alive(allow_reconnect=True)
        self._maybe_create_db_connection()
        return self._conn

    def insert_csv(self, csv_path, table):
        return psql_insert_csv(
            csv_path, table, self.db_name,
            host=self.host,
            user=self.user,
            password=self.password)

    def commit(self):
        if self._conn is None:
            # self.create_connection()
            return
        self.conn.commit()

    def close(self, from_pool=False):
        if not from_pool and self.pool is not None:
            # Make it convenient to grab a connection from a pool then call conn.close() on it to return it to the pool
            # (instead of having to call pool.putconn(conn)).
            self.pool.putconn(conn=self, from_conn=True)
        else:
            # Actually close the SQL connection.
            #
            # The ConnectionPool wants this connection, or the connection
            # isn't being managed by a ConnectionPool.
            self._maybe_close_db_connection()
            self._maybe_close_postgres_connection()

    def get_cursor(self):
        if self._conn is None:
            self.create_connection()
        c = self._conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor = Cursor(cursor=c, conn=self)
        # Q: Will we still check if the connection is "still alive?"
        # A: every time we fetch a cursor with "c = self.cursor" it calls self.create_connection;
        # that's the point at which we will check if the connection is still alive.
        # So, this is "just as good" as when we use "c = self.cursor".
        self._cursors.append(cursor)
        return cursor

    def put_cursor(self, cursor):
        cursor.close()
        self._cursors.remove(cursor)

    @property
    def cursor(self):
        assert not self._pool_closed
        if self.keep_alive and self._cursor is not None:
            assert self._conn is not None
            self._check_alive(allow_reconnect=True)
        if self._conn is None:
            self.create_connection()
        return self._cursor

    def insert_dict(self, table, dic, id_field=None, debug=False):
        c = self._cursor
        cols, placeholders, colnames = self._get_insert_info(dic)
        values = [dic[col] for col in cols]
        if debug:
            logger.info(pprint_msg({
                'cols':cols, 'colnames':colnames, 'values':values,
            }))
        query_lines = [
            "INSERT INTO {table} ({colnames})".format(
                table=table,
                colnames=colnames,
            ),
            "VALUES ({placeholders})".format(
                placeholders=placeholders,
            )
        ]
        if id_field is not None:
            query_lines.append(
                "RETURNING {id_field}".format(
                    id_field=id_field))
        query = "\n".join(query_lines)
        params = values
        sql_exec_query(c, query, params, debug=debug)
        if id_field is not None:
            id_rows = c.fetchall()
            assert len(id_rows) == 1
            id_row = id_rows[0]
            assert len(id_row) == 1
            primary_key = id_row[0]
        else:
            primary_key = c.lastrowid
        if debug:
            logger.info("Autoincrement id = {id}".format(id=primary_key))
        return primary_key

    def _get_insert_info(self, dic):
        cols = sorted(dic.keys())
        placeholders = ','.join([sql_placeholder()] * len(cols))
        colnames = ','.join(cols)
        return cols, placeholders, colnames

    def _get_vals(self, cols, dic):
        return [dic[col] for col in cols]

    def insert_dicts(self, table, dics):
        c = self._cursor
        cols, placeholders, colnames = self._get_insert_info(dics[0])

        all_values = [self._get_vals(cols, dic) for dic in dics]

        c.executemany("INSERT INTO {table} ({colnames}) VALUES ({placeholders})".format(
            placeholders=placeholders,
            table=table,
            colnames=colnames,
        ), all_values)

    def create_connection(self):
        if self._conn is not None:
            return

        self._maybe_read_db_config()

        db_exists, self._conn, self._cursor = self._create_connection(self.db_name)
        return db_exists

    def _create_connection(self, db_name):
        """ create a database connection to a SQLite database """
        assert not self._pool_closed

        self._maybe_close_db_connection()

        # conn = psycopg2.connect( dbname=db_name, user=self.user, host=self.host, isolation_level=None)
        # conn = psycopg2.connect("dbname={db} user={user}".format(
        #     db=db_name,
        #     user=self.user,
        # ), isolation_level=None)
        try:
            kwargs = dict()
            if self.password is not None:
                kwargs['password'] = self.password
            conn = psycopg2.connect(
                dbname=db_name,
                user=self.user,
                host=self.host,
                isolation_level=None,
                **kwargs)
            # Q: Will this prevent connections from randomly dying?
            #
            # http://initd.org/psycopg/docs/connection.html#connection.autocommit
            # " default, any query execution, including a simple SELECT will start a transaction:
            # for long-running programs, if no further action is taken, the session will remain
            # “idle in transaction”, an undesirable condition for several reasons (locks are
            # held by the session, tables bloat…). For long lived scripts, either ensure to terminate a
            # transaction as soon as possible or use an autocommit connection."
            # logger.info("autocommit before: {auto}".format(auto=conn.autocommit))
            conn.autocommit = True
            # logger.info("autocommit after: {auto}".format(auto=conn.autocommit))
            assert conn.autocommit
        except psycopg2.OperationalError as e:
            if not re.search(r'database.*does not exist', e.args[0]):
                raise
            # Database does not exist; just make a new one.
            return False, None, None
        conn.set_session(autocommit=True, isolation_level='READ UNCOMMITTED')

        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        return True, conn, cursor

    def _maybe_create_postgres_connection(self):
        if self.pg_conn is not None:
            return

        db_exists, self.pg_conn, self.pg_cursor = self._create_connection('postgres')
        return db_exists

    def _maybe_close_postgres_connection(self):
        if self.pg_cursor is not None:
            self.pg_cursor.close()
            self.pg_cursor = None

        if self.pg_conn is not None:
            self.pg_conn.close()
            self.pg_conn = None

    def _maybe_read_db_config(self):
        if self.db_config is None:
            self._read_db_config()

    def _read_db_config(self):
        assert _e(self.db_config_path)
        self.db_config = load_json(self.db_config_path)
        self.db_name = self.db_config['db_name']
        assert self.db_basename == self.db_config['db_basename']

    def _dump_db_config(self):
        do_dump_json(self.db_config, self.db_config_path)

    def _db_config_exists(self):
        return self.db_config is not None

    def _drop_connections(self):
        logger.info("> Drop existing connections to {db}".format(db=self.db_name))
        # psutil.
        # pid = 1234 # The pid whose info you're looking for
        # p = psutil.Process(pid)
        # print p.cmdline
        c = self.cursor

        c.execute("""
        SELECT pid, pg_terminate_backend(pid) as terminated
        FROM pg_stat_activity
        WHERE datname = current_database() AND pid <> pg_backend_pid();
        """.format(db=self.db_name))
        active_pids = c.fetchall()

        proc_info = []
        for row in active_pids:
            try:
                cmdline = psutil.Process(row['pid']).cmdline()
            except psutil.NoSuchProcess:
                continue
            info = {
                'pid':row['pid'],
                'cmdline':cmdline,
            }
            proc_info.append(info)

        c.execute("""
        SELECT pid, pg_terminate_backend(pid) as terminated
        FROM pg_stat_activity
        WHERE datname = current_database() AND pid <> pg_backend_pid();
        """.format(db=self.db_name))
        dropped_pids = c.fetchall()

        if len(proc_info) > 0 or len(dropped_pids) > 0:
            pprint.pprint({
                'proc_info':proc_info,
                'dropped_pids':dropped_pids,
            })

    def create_db(self, recreate):
        if _e(self.db_config_path):
            self._read_db_config()
            if self._db_config_exists():
                db_exists = self._maybe_create_db_connection()
                if db_exists:
                    self._drop_connections()
            if recreate:
                self._drop_database()
                self._create_database(self.db_name)
                self._load_tables()
        else:
            self.db_name = self._alloc_db_name()
            self.db_config = {
                'db_name': self.db_name,
                'db_basename': self.db_basename,
            }
            self._create_database(self.db_name)
            self._dump_db_config()
            self._load_tables()

        # Database may already exist (e.g. we are just plotting something)
        self._maybe_create_db_connection()

    def _maybe_create_db_connection(self):
        if self._conn is not None:
            return

        db_exists, self._conn, self._cursor = self._create_connection(self.db_name)
        return db_exists

    def random_string(self, N):
        """
        Random string made of [a-z0-9], N characters wide.
        """
        rand_str = ''.join(
            random.choice(
                string.ascii_lowercase + string.digits
            ) for _ in range(N))
        return rand_str

    def _alloc_db_name(self):
        databases = set(self.databases)
        db_name = None
        while True:
            suffix = self.random_string(self.rand_suffix_len)
            db_name = "{base}_{suffix}".format(
                base=self.db_basename,
                suffix=suffix)
            if db_name not in databases:
                return db_name

    def _create_database(self, db_name):
        self._maybe_create_postgres_connection()
        c = self.pg_cursor
        c.execute("""
        CREATE DATABASE {db};
        """.format(db=db_name))

    @property
    def tables(self):
        self._maybe_create_db_connection()
        c = self.cursor
        c.execute("""
        SELECT tablename 
        FROM pg_catalog.pg_tables 
        WHERE 
            schemaname != 'pg_catalog' AND 
            schemaname != 'information_schema';
        """)
        rows = c.fetchall()
        tables = [row['tablename'] for row in rows]
        return tables

    def _close_cursors(self):
        for cursor in self._cursors:
            cursor.close()
        self._cursors.clear()

    def with_cursors(self):
        """
        # Usage:
        with conn.with_cursors():
            c1 = conn.get_cursor()
            c2 = conn.get_cursor()
            ...
        # c1 and c2 are automatically closed.
        """
        return CursorContext(conn=self)

    def _maybe_close_db_connection(self):
        self._close_cursors()

        if self._cursor is not None:
            self._cursor.close()
            self._cursor = None

        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _drop_database(self):
        logger.info("> Drop database = {db}".format(db=self.db_name))
        self._maybe_create_postgres_connection()
        self._maybe_close_db_connection()
        c = self.pg_cursor
        c.execute("""
        DROP DATABASE IF EXISTS {db};
        """.format(db=self.db_name))

    def _drop_tables(self):
        tables = self.tables
        logger.info("> Drop database tables: db={db}, tables={tables}".format(
            db=self.db_name,
            tables=tables))

        self._maybe_create_db_connection()
        c = self.cursor
        for table in tables:
            c.execute("""
            DROP TABLE {table};
            """.format(table=table))

    @property
    def databases(self):
        self._maybe_create_postgres_connection()

        c = self.pg_cursor
        c.execute("""
        SELECT datname 
        FROM pg_database 
        WHERE 
            datistemplate = false;
        """)
        databases = [row['datname'] for row in c.fetchall()]
        return databases

    def create_constraints(self):
        self.run_sql_file(self.db_name, PSQL_CONSTRAINTS_SQL)

    def create_indices(self):
        self.run_sql_file(self.db_name, PSQL_INDICES_SQL)

    def _load_tables(self):
        logger.info("> Load table schema into {db} from {path}".format(
            db=self.db_name,
            path=PSQL_TABLE_SQL,
        ))
        self.run_sql_file(self.db_name, PSQL_TABLE_SQL)

    def run_sql_file(self, db_name, sql_path):
        """
        Use subprocess.run(...) to run a .sql file;
        This way, so we get actual line numbers when something fails

        e.g.

        sqlite3 blaz.db -bail -echo -cmd '.read sqlite/tables.sql' -cmd '.quit'
        """
        with open(sql_path, 'r') as sql_f:
            # cmd_kwargs = self._psql_cmd_args(db_name)
            logger.info("run sql file: {msg}".format(
                msg=pprint_msg({
                    'host':self.host,
                    'user':self.user,
                    'password':self.password,
                })
            ))
            cmd_kwargs = psql_cmd_args(
                db_name,
                host=self.host,
                user=self.user,
                password=self.password,
            )
            proc = subprocess.run(
                stdin=sql_f,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **cmd_kwargs)
        if proc.returncode != 0:
            if proc.stdout is not None:
                print(proc.stdout.decode('utf-8'))
            if proc.stderr is not None:
                print(proc.stderr.decode('utf-8'))
            # raise RuntimeError to be compatiable with ForkedProcessPool
            raise RuntimeError("ERROR: failed to run sql file @ {path}; ret={ret}".format(
                ret=proc.returncode, path=db_name))

class CursorContext:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn._close_cursors()


class TracesSQLiteConnection:
    def __init__(self, db_path, host=None):
        self.db_path = db_path
        self.host = host
        self._cursor = None
        self.conn = None

    def create_db(self, recreate):

        if recreate and _e(self.db_path):
            os.remove(self.db_path)
        elif not recreate and _e(self.db_path):
            return

        self.conn.run_sql_file(self.db_path, SQLITE_TABLE_SQL)
        assert _e(self.db_path)

    def create_indices(self):
        self.conn.run_sql_file(self.db_path, SQLITE_INDICES_SQL)

    def create_constraints(self):
        # self.conn.run_sql_file(self.db_name, PSQL_CONSTRAINTS_SQL)
        # SQLite tables.sql still contains constraints.
        pass

    def enable_fast_inserts(self):
        c = self.cursor
        c.execute("""
        PRAGMA synchronous = OFF
        """)

    def disable_fast_inserts(self):
        c = self.cursor
        c.execute("""
        PRAGMA synchronous = NORMAL
        """)

    def commit(self):
        if self.conn is None:
            self.create_connection()
        self.conn.commit()

    def close(self):
        self.cursor.close()
        self.conn.close()

    @property
    def cursor(self):
        if self.conn is None:
            self.create_connection()
        return self._cursor

    def insert_dict(self, table, dic):
        c = self._cursor
        cols, placeholders, colnames = self._get_insert_info(dic)
        values = [dic[col] for col in cols]
        c.execute("INSERT INTO {table} ({colnames}) VALUES ({placeholders})".format(
            placeholders=placeholders,
            table=table,
            colnames=colnames,
        ), values)
        return c.lastrowid

    def _get_insert_info(self, dic):
        cols = sorted(dic.keys())
        placeholders = ','.join([sql_placeholder()] * len(cols))
        colnames = ','.join(cols)
        return cols, placeholders, colnames

    def _get_vals(self, cols, dic):
        return [dic[col] for col in cols]

    def insert_dicts(self, table, dics):
        c = self._cursor
        cols, placeholders, colnames = self._get_insert_info(dics[0])

        all_values = [self._get_vals(cols, dic) for dic in dics]

        # all_values = [None]*(len(cols) * len(dics))
        # i = 0
        # for dic in dics:
        #     vals = self._get_vals(cols, dic)
        #     for j in range(len(vals)):
        #         all_values[i] = vals[j]
        #     i += len(vals)

        c.executemany("INSERT INTO {table} ({colnames}) VALUES ({placeholders})".format(
            placeholders=placeholders,
            table=table,
            colnames=colnames,
        ), all_values)

        # def bracket(x):
        #     return "({x})".format(x=x)
        # all_placeholders = ', '.join(bracket(placeholders) for i in range(len(dics)))
        # c.execute("INSERT INTO {table} ({colnames}) VALUES {all_placeholders}".format(
        #     all_placeholders=all_placeholders,
        #     table=table,
        #     colnames=colnames,
        # ), all_values)

    def create_connection(self):
        """ create a database connection to a SQLite database """
        if self.conn is not None:
            return

        # NOTE: isolation_level = None means no transaction (we're the only user of this database).
        # This is required for BulkInserter to work,
        # otherwise you'll get an error on 'COMMIT':
        #   sqlite3.OperationalError: cannot commit - no transaction is active
        if self.host is not None:
            # Lazy.
            raise NotImplementedError
        self.conn = sqlite3.connect(self.db_path, isolation_level=None)
        # self.conn = sqlite3.connect(self.db_path)
        assert self.conn.isolation_level is None

        self.conn.row_factory = sqlite3.Row
        self._cursor = self.conn.cursor()

    def run_sql_file(self, db_path, sql_path):
        """
        Use subprocess.run(...) to run a .sql file;
        This way, so we get actual line numbers when something fails

        e.g.

        sqlite3 blaz.db -bail -echo -cmd '.read sqlite/tables.sql' -cmd '.quit'
        """
        proc = subprocess.run(["sqlite3", db_path,
                               '-bail', '-echo',
                               '-cmd', '.read {path}'.format(path=sql_path),
                               '-cmd', '.quit',
                               ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            if proc.stdout is not None:
                print(proc.stdout.decode('utf-8'))
            if proc.stderr is not None:
                print(proc.stderr.decode('utf-8'))
            print("ERROR: failed to run sql file @ {path}; ret={ret}".format(
                ret=proc.returncode, path=db_path))
            sys.exit(1)

def conn_kwargs_as_set(conn_kwargs_dict):
    return frozenset((k, v) for k, v in conn_kwargs_dict.items())

def conn_kwargs_as_dict(conn_kwargs_set):
    return dict((k, v) for k, v in conn_kwargs_set)

class _ConnectionPoolManager:
    def __init__(self, debug=False):
        self.debug = debug
        # Mapping from connection arguments to a pool of connections opened with those arguments.
        #   { conn_kwargs } -> ConnectionPool
        self.connection_pools = dict()
        # Close a pool of connections once no one is using it anymore.
        #   { conn_kwargs } -> refcnt
        self.refcnt = dict()

    def _create_connection_pool(self, maxconn, conn_kwargs, debug=False):
        key = conn_kwargs_as_set(conn_kwargs)
        assert key not in self.connection_pools
        self.connection_pools[key] = ConnectionPool(
            conn_kwargs=conn_kwargs,
            maxconn=maxconn,
            debug=self.debug or debug,
        )
        # psycopg2.pool.SimpleConnectionPool(
        # minconn=1, maxconn=maxconn,
        # **conn_kwargs)
        return self.connection_pools[key]

    def _get_connection_pool(self, conn_kwargs):
        key = conn_kwargs_as_set(conn_kwargs)
        return self.connection_pools[key]

    def _close_connection_pool(self, conn_kwargs):
        key = conn_kwargs_as_set(conn_kwargs)
        pool = self.connection_pools[key]
        pool.closeall()
        del self.connection_pools[key]

    def get_connection_pool(self, maxconn=5, new_process=False, existing=False, debug=False, **kwargs):
        conn_kwargs = kwargs
        pool = self._maybe_create_connection_pool(
            maxconn, conn_kwargs,
            new_process=new_process,
            existing=existing,
            debug=debug,
        )
        key = conn_kwargs_as_set(conn_kwargs)
        if key not in self.refcnt or new_process:
            self.refcnt[key] = 0
        self.refcnt[key] += 1
        return pool

    def current_connection_pool(self, debug=False, **kwargs):
        pool = self.get_connection_pool(existing=True, debug=debug, **kwargs)
        return pool

    def put_connection_pool(self, pool, conn_kwargs):
        key = conn_kwargs_as_set(conn_kwargs)
        assert self.refcnt[key] > 0
        self.refcnt[key] -= 1
        pool_refcnt = self.refcnt[key]
        if self.refcnt[key] == 0:
            assert pool == self.connection_pools[key]
            self._close_connection_pool(conn_kwargs)
            del self.refcnt[key]
        return pool_refcnt

    def _maybe_create_connection_pool(self, maxconn, conn_kwargs, new_process=False, existing=False, debug=False):
        key = conn_kwargs_as_set(conn_kwargs)
        if key not in self.connection_pools and existing:
            raise RuntimeError(
                textwrap.dedent("""\
                RL-Scope ERROR: couldn't find an existing "with GetConnectionPool(...)" in the current stack-trace; did you forget to make one?
                    Connection details:
                {msg}
                {pools}
                """).format(
                    msg=textwrap.indent(pprint.pformat(conn_kwargs), prefix='    '*2),
                    pools=textwrap.indent(pprint.pformat(self.connection_pools), prefix='    '*2),
                ).rstrip())
        if key not in self.connection_pools:
            pool = self._create_connection_pool(maxconn, conn_kwargs, debug=debug)
            return pool
        if new_process:
            # Close "stale" parent connections
            self._close_connection_pool(conn_kwargs)
            # Create new connection pool for child
            self._create_connection_pool(maxconn, conn_kwargs)
        return self._get_connection_pool(conn_kwargs)

ConnectionPoolManager = _ConnectionPoolManager()

class GetConnectionPool:
    def __init__(self, conn_kwargs, maxconn=5, new_process=False, debug=False):
        self.conn_kwargs = conn_kwargs
        self.maxconn = maxconn
        self.new_process = new_process
        self.debug = debug

    def __enter__(self):
        self.pool = ConnectionPoolManager.get_connection_pool(
            maxconn=self.maxconn,
            new_process=self.new_process,
            debug=self.debug,
            **self.conn_kwargs,
        )
        return self.pool

    def __exit__(self, exc_type, exc_val, exc_tb):
        # If pool refcnt is 0, close all the connections allocated in the pool.
        # If connections "leak out" from the pool scope, hopefully
        # we'll get an error when people attempt to use them(?)
        return ConnectionPoolManager.put_connection_pool(self.pool, self.conn_kwargs)

class ConnectionPool:
    """
    Each single-threaded python "Process", upon being launched, has its own TracesConnectionPool.

    We would like to use a global variable to store a singleton TracesConnectionPool
    that is used across the "Process".
    PROBLEM:
    - On Linux (maybe not Windows) the new process is made via a fork(), so we may end up
      attempting to reuse "stale" connections from the parent...
      IDEALLY: Process creation should look like this:
          Process = fork()
          if child:
              # Close "stale" parent connections
              TracesConnectionPool.closeall()
          else:
              # Parent continues to use its pool.
              pass

    This class implements a programming pattern to handle this:

    with GetConnectionPool(# Arguments for creating SQL connection object.
                           host=..., user=..., password=...,
                           maxconn=5,
                           new_process=True) as conn_pool:

        # NOTE: nested functions/classes can use the same "with" pattern;
        # the returned pool will be identical.
        # The pool will be closed when the root "with GetConnectionPool" exits.

        conn = conn_pool.getconn()
        # ... use conn as usual ...
        # If you forget to call this, the pool will STILL be terminated
        # after exiting the "with GetConnectionPool".
        conn_pool.putconn(conn)


    # Code that ASSUMES a pool has been created externally for it:
    conn = GetConnection(host=..., user=..., password=...)
    PutConnection(conn,
        host=..., user=..., password=...)

    :param maxconn
        Anything above 5 sounds like a connection leak to me based on my usage.

    """
    def __init__(self, conn_kwargs, maxconn=5, close_eagerly=True, debug=False):
        self.maxconn = maxconn
        # (5 connections per pool)*(32 cores) == 160 connections >> 100 (default max postgres connections)...
        # so lets close eagerly to prevent that.
        self.close_eagerly = close_eagerly
        self.debug = debug
        self.conn_kwargs = conn_kwargs
        self.free_connections = []
        self.used_connections = []
        self.closed = False

    @property
    def n_connections(self):
        return len(self.used_connections)

    def putconn(self, conn, from_conn=False):
        assert not self.closed
        assert conn in self.used_connections
        self.used_connections.remove(conn)
        if self.close_eagerly:
            if not from_conn:
                conn.close(from_pool=True)
        else:
            self.free_connections.append(conn)

    def getconn(self):
        assert not self.closed
        assert self.n_connections <= self.maxconn
        if len(self.free_connections) > 0:
            conn = self.free_connections.pop()
        else:
            assert 'debug' not in self.conn_kwargs
            conn = sql_create_connection(pool=self, **self.conn_kwargs, debug=self.debug)
        self.used_connections.append(conn)
        return conn

    def closeall(self):
        """
        Close all the connections allocated in the pool.
        If connections "leak out" from the pool scope, hopefully
        we'll get an error when people attempt to use them(?)
        """
        for conn in self.used_connections:
            conn.close(from_pool=True)
        for conn in self.free_connections:
            conn.close(from_pool=True)
        self.used_connections = []
        self.free_connections = []
        self.closed = True
        assert self.n_connections == 0

    def __str__(self):
        bldr = ToStringBuilder(obj=self)
        for key, value in self.conn_kwargs.items():
            bldr.add_param(key, value)

        return bldr.to_string()

    def __repr__(self):
        return str(self)

class SQLCategoryTimesReader:
    """
    Read category times format from SQLite database containing
    all the trace files collected from running an ML-script.

    Category times format:

        A dict, where the keys are a single category name (not a combination),
        and the values are a list of raw start/end times for that category:
        {
            'Python': [(start[0][0], end[0][0]), (start[0][1], end[0][1]), ...],
            'GPU': [(start[1][0], end[1][0]), (start[1][1], end[1][1]), ...],
            ...
        }

        NOTE: start/end tuples are in sorted order (by their start time)
    """
    def __init__(self, db_path, host=None, user=None, password=None, debug_ops=False, debug=False):
        self.db_path = db_path
        self.host = host
        self.user = user
        self.password = password
        self._conn = None
        self.parse_debug = False
        self.debug_ops = debug_ops
        self.debug = debug

        self._steps = dict()

    def steps(self, process_name, machine_name, bench_name, debug=False):
        return list(range(self.num_steps(process_name, machine_name, bench_name, debug=debug)))

    @property
    def conn(self):
        self._maybe_create_conn()
        return self._conn

    def _maybe_create_conn(self):
        if self._conn is None:
            self._conn = get_sql_connection(db_path=self.db_path, host=self.host, user=self.user, password=self.password, debug=self.debug or self.debug_ops)

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def util_devices(self, machine_name=None):
        """
        Select devices that have CPU/GPU utilization info recorded.
        :return:
        """
        c = self.conn.cursor
        query = textwrap.dedent("""
        SELECT *
        FROM Device AS d1
            NATURAL JOIN Machine AS m
        WHERE
            {machine_clause} AND
            EXISTS (
                SELECT *
                FROM DeviceUtilization AS du
                WHERE du.device_id = d1.device_id
            )
        """.format(
            machine_clause=sql_machine_clause(machine_name, 'm', indents=1, allow_none=True),
        ))
        params = None
        sql_exec_query(c, query, params, self.__class__, self.debug_ops)
        rows = c.fetchall()
        devices = [Device.from_row(row) for row in rows]
        return devices

    def util_samples(self, device):
        """
        Select devices that have CPU/GPU utilization info recorded.
        :return:
        """
        assert type(device) == Device
        c = self.conn.cursor
        query = textwrap.dedent("""
        SELECT 
            du.start_time_us/1e6 AS start_time_sec, du.util
        FROM 
            Device AS d1
            NATURAL JOIN Machine AS m
            NATURAL JOIN DeviceUtilization AS du
        WHERE
            d1.device_id = {p}
        ORDER BY start_time_us ASC 
        """.format(p=sql_placeholder()))
        params = (device.device_id,)
        sql_exec_query(c, query, params, self.__class__, self.debug_ops)
        rows = c.fetchall()
        samples = {
            'util':[],
            'start_time_sec':[],
        }
        for row in rows:
            samples['util'].append(row['util'])
            samples['start_time_sec'].append(row['start_time_sec'])
        return samples

    @property
    def trace_start_time_sec(self):
        """
        Return the earliest epoch_sec of any traced Event / utilization sample.
        This determines where the heat-scale starts showing values.
        """
        c = self.conn.cursor
        query = textwrap.dedent("""
        SELECT MIN(time_us)/1e6 AS start_time_sec
        FROM 
        (
        SELECT MIN(start_time_us) AS time_us FROM Event
        UNION
        SELECT MIN(start_time_us) AS time_us FROM DeviceUtilization
        ) AS tbl
        """)
        params = None
        sql_exec_query(c, query, params, self.__class__, self.debug_ops)
        row = c.fetchone()
        return row['start_time_sec']

    def training_progress(self, *args, allow_none=False, **kwargs):
        training_progresses = self.training_progresses(*args, **kwargs)
        # Return the last call to rlscope.prof.report_progress(...)
        def by_end_training_time_us(training_progress):
            return training_progress.end_training_time_us
        training_progresses.sort(key=by_end_training_time_us)
        if allow_none and len(training_progresses) == 0:
            logger.warning("Didn't find any TrainingProgress rows for args={args}, kwargs={kwargs}".format(
                args=args,
                kwargs=kwargs))
            return None
        return training_progresses[-1]
        # assert len(training_progresses) == 1
        # return training_progresses[0]

    def training_progresses(
            self,
            machine_name=None,
            process_name=None,
            phase_name=None,
            debug=False):
        c = self.conn.cursor
        query = textwrap.dedent("""
        SELECT 
            tp.total_timesteps,
            tp.start_trace_time_us,

            tp.start_percent_complete,
            tp.start_num_timesteps,
            tp.start_training_time_us,

            tp.end_percent_complete,
            tp.end_training_time_us,
            tp.end_num_timesteps,
            
            m.machine_name,
            m.machine_id,
            p.process_name, 
            p.process_id, 
            ph.phase_name, 
            ph.phase_id
        FROM
            TrainingProgress as tp
            NATURAL JOIN Phase AS ph
            NATURAL JOIN Process AS p
            NATURAL JOIN Machine AS m
        WHERE
            {machine_clause} AND
            {process_clause} AND
            {phase_clause}
        ORDER BY
            tp.end_training_time_us
        """).format(
            machine_clause=sql_machine_clause(machine_name, 'm', indents=1, allow_none=True),
            process_clause=sql_process_clause(process_name, 'p', indents=1, allow_none=True),
            phase_clause=sql_phase_clause(phase_name, 'ph', indents=1, allow_none=True),
        )
        sql_exec_query(c, query, debug=debug)
        rows = [TrainingProgress.from_row(row) for row in c.fetchall()]
        return rows

    def phase(self, *args, **kwargs):
        phases = self.phases(*args, **kwargs)
        assert len(phases) == 1
        return phases[0]

    def phases(
            self,
            machine_name=None,
            process_name=None,
            phase_name=None,
            debug=False):
        c = self.conn.cursor
        query = textwrap.dedent("""
        SELECT 
            m.machine_name,
            m.machine_id,
            p.process_name, 
            p.process_id, 
            ph.phase_name, 
            ph.phase_id, 
            MIN(e.start_time_us) AS phase_start_time_us, 
            MAX(e.end_time_us) AS phase_end_time_us

        FROM
            Event AS e
            NATURAL JOIN Phase AS ph
            NATURAL JOIN Process AS p
            NATURAL JOIN Machine AS m
        WHERE
            {machine_clause} AND
            {process_clause} AND
            {phase_clause}
        GROUP BY
            m.machine_name, m.machine_id, p.process_name, p.process_id, ph.phase_name, ph.phase_id
        """).format(
            machine_clause=sql_machine_clause(machine_name, 'm', indents=1, allow_none=True),
            process_clause=sql_process_clause(process_name, 'p', indents=1, allow_none=True),
            phase_clause=sql_phase_clause(phase_name, 'ph', indents=1, allow_none=True),
        )
        sql_exec_query(c, query, debug=debug)
        rows = [Phase.from_row(row) for row in c.fetchall()]
        return rows

    def operation(self, *args, **kwargs):
        phases = self.phases(*args, **kwargs)
        assert len(phases) == 1
        return phases[0]

    def operations(
        self,
        machine_name=None,
        process_name=None,
        phase_name=None,
        debug=False):
        c = self.conn.cursor
        query = textwrap.dedent("""
        SELECT 
            e.event_name as operation_name,
            m.machine_name,
            m.machine_id,
            p.process_name, 
            p.process_id, 
            ph.phase_name, 
            ph.phase_id
        FROM
            Event AS e
            NATURAL JOIN Phase AS ph
            NATURAL JOIN Process AS p
            NATURAL JOIN Machine AS m
            NATURAL JOIN Category AS c
        WHERE
            {machine_clause} AND
            {process_clause} AND
            {phase_clause} AND
            c.category_name = '{CATEGORY_OPERATION}'
        GROUP BY
            m.machine_name, m.machine_id, 
            p.process_name, p.process_id,
            ph.phase_name, ph.phase_id,
            operation_name
        """).format(
            machine_clause=sql_machine_clause(machine_name, 'm', indents=1, allow_none=True),
            process_clause=sql_process_clause(process_name, 'p', indents=1, allow_none=True),
            phase_clause=sql_phase_clause(phase_name, 'ph', indents=1, allow_none=True),
            CATEGORY_OPERATION=constants.CATEGORY_OPERATION,
        )
        sql_exec_query(c, query, debug=debug)
        rows = [Operation.from_row(row) for row in c.fetchall()]
        return rows

    def keep_steps(self, process_name, machine_name, bench_name, skip_first_step=True, debug=False):
        steps = self.steps(process_name, machine_name, bench_name, debug=debug)

        # Skip the first step, since it includes profiler initialization stuff.
        # In particular, the libcupti NVIDIA library gets loaded on-demand during
        # the first traced step, and this can take 2 seconds to load!
        # (We had this bug before...)
        if skip_first_step and len(steps) > 1:
            keep_steps = steps[1:]
        else:
            keep_steps = steps

        return keep_steps

    @property
    def directory(self):
        return _d(self.db_path)

    def op_names(self, debug_ops=False):
        return self.bench_names(debug_ops)

    def bench_names(self, debug_ops=False):
        c = self.conn.cursor
        c.execute("""
        SELECT DISTINCT e.event_name FROM Event AS e NATURAL JOIN Category AS c
        WHERE 
            c.category_name = '{CATEGORY_OPERATION}' AND
            {debug_ops_clause}
        ORDER BY e.event_name
        """.format(
            CATEGORY_OPERATION=constants.CATEGORY_OPERATION,
            debug_ops_clause=sql_debug_ops_clause(debug_ops, 'e'),
         ))
        bench_names = [row['event_name'] for row in c.fetchall()]
        return bench_names

    @property
    def categories(self):
        c = self.conn.cursor
        c.execute("""
        SELECT category_name FROM Category
        ORDER BY category_name ASC
        """)
        category_names = [row['category_name'] for row in c.fetchall()]
        return category_names

    def process_names(self, machine_name, debug=False):
        c = self.conn.cursor
        query = textwrap.dedent("""
        SELECT process_name 
            FROM Process
            NATURAL JOIN Machine
        WHERE machine_name = '{machine_name}'
        ORDER BY process_name ASC
        """.format(
            machine_name=machine_name))
        sql_exec_query(c, query, debug=debug)
        process_names = [row['process_name'] for row in c.fetchall()]
        return process_names

    def process(self, *args, **kwargs):
        processes = self.processes(*args, **kwargs)
        assert len(processes) == 1
        return processes[0]

    def processes(self, machine_name=None, process_name=None):
        c = self.conn.cursor
        query = textwrap.dedent("""
        SELECT *
        FROM Process AS p
            NATURAL JOIN Machine AS m
        WHERE
            {machine_clause} AND
            {process_clause}
        """.format(
            machine_clause=sql_machine_clause(machine_name, 'm', indents=1, allow_none=True),
            process_clause=sql_process_clause(process_name, 'p', indents=1, allow_none=True),
        ))
        params = None
        sql_exec_query(c, query, params, self.__class__, self.debug_ops)
        rows = c.fetchall()
        processes = [Process.from_row(row) for row in rows]
        return processes

    def machine(self, *args, **kwargs):
        machines = self.machines(*args, **kwargs)
        assert len(machines) == 1
        return machines[0]

    def machines(self, machine_name=None):
        c = self.conn.cursor
        query = textwrap.dedent("""
        SELECT *
        FROM Machine AS m
        WHERE
            {machine_clause}
        """.format(
            machine_clause=sql_machine_clause(machine_name, 'm', indents=1, allow_none=True),
        ))
        params = None
        sql_exec_query(c, query, params, self.__class__, self.debug_ops)
        rows = c.fetchall()
        machines = [Machine.from_row(row) for row in rows]
        return machines

    @property
    def machine_names(self):
        c = self.conn.cursor
        c.execute("""
        SELECT machine_name 
        FROM Machine
        ORDER BY machine_name ASC
        """)
        machine_names = [row['machine_name'] for row in c.fetchall()]
        return machine_names


    DEBUG_FETCH_STEPS = False
    def _fetch_steps(self, process_name, machine_name, bench_name, debug=False):
        if process_name in self._steps and bench_name in self._steps[process_name]:
            return

        start_fetch_t = time.time()
        c = self.conn.cursor
        query = """
        SELECT e1.event_name, e1.start_time_us, e1.duration_us
        FROM Event AS e1
            NATURAL JOIN Category AS c
            NATURAL JOIN Process AS p
            NATURAL JOIN Machine AS m
        WHERE 
            c.category_name = '{CATEGORY_OPERATION}' AND
            e1.event_name = {p} AND
            p.process_name = {p} AND 
            {machine_clause}
        ORDER BY e1.start_time_us ASC 
        """.format(
            CATEGORY_OPERATION=constants.CATEGORY_OPERATION,
            # PROFILING_DUMP_TRACE=PROFILING_DUMP_TRACE,
            # NOTE: We do NOT want to select any steps of an operation that overlap at all with a DUMP event.
            # indents=3 since {overlap_clause} above has 3 indent-levels in front of it.
            # overlap_clause=sql_overlap_clause('e1', 'e2', indents=3),
            machine_clause=sql_machine_clause(machine_name, 'm', indents=1, allow_none=True),
            p=sql_placeholder(),
        )
        params = (bench_name, process_name)
        sql_exec_query(c, query, params,
                       debug=True,
                       # debug=debug,
        )

        rows = rows_as_ktime(c.fetchall())
        if machine_name not in self._steps:
            self._steps[machine_name] = dict()
        if process_name not in self._steps[machine_name]:
            self._steps[machine_name][process_name] = dict()
        self._steps[machine_name][process_name][bench_name] = rows
        if debug:
            logger.info("fetch_steps(proc={proc}, op={op}) fetched {n} rows.".format(
                proc=process_name,
                op=bench_name,
                n=len(self._steps[machine_name][process_name][bench_name])))
        end_fetch_t = time.time()
        sec_fetch = end_fetch_t - start_fetch_t
        if debug or SQLCategoryTimesReader.DEBUG_FETCH_STEPS:
            logger.info("> fetch_steps process={proc}, op={op} took {sec} seconds".format(
                proc=process_name,
                op=bench_name,
                sec=sec_fetch,
            ))

    def num_steps(self, process_name, machine_name, bench_name, debug=False):
        """
        We don't record step numbers in the database.
        Instead, steps are an index into the i-th time this operation occurs in the entire ML-script.

        We tend to want to skip the 1-st time the operation occurs / is profiled, since
        it will include load-time overheads (libcupti).
        """
        self._fetch_steps(process_name, machine_name, bench_name, debug=debug)
        return len(self._steps[machine_name][process_name][bench_name])

    def step_event(self, step, process_name, machine_name, bench_name):
        self._fetch_steps(process_name, machine_name, bench_name)
        return self._steps[machine_name][process_name][bench_name][step]

        # Swallow any excess arguments
    DEBUG_EACH_OP_INSTANCE = False
    def each_op_instance(self, bench_name, machine_name,
                         filter_op=True,
                         group_by_device=DEFAULT_group_by_device,
                         ignore_categories=DEFAULT_ignore_categories,
                         debug=DEFAULT_debug,
                         skip_first_step=True,
                         show_progress=True):
        start_t = time.time()
        process_names = self.process_names(machine_name)
        for process_name in process_names:

            keep_steps = self.keep_steps(process_name, machine_name, bench_name, skip_first_step, debug=debug)
            if debug:
                logger.info("keep_steps = {steps}".format(steps=keep_steps))
            if bench_name == NO_BENCH_NAME:
                pprint.pprint({
                    'name':'SQLCategoryTimesReader.each_op_instance',
                    'keep_steps':keep_steps})

            for step in progress(keep_steps,
                                 desc=as_progress_label("each_op_instance", process_name),
                                 show_progress=show_progress):
                category_times = self.parse(step, process_name, machine_name, bench_name,
                                            filter_op=filter_op,
                                            group_by_device=group_by_device, ignore_categories=ignore_categories, debug=debug,
                                            show_progress=show_progress)
                end_t = time.time()
                sec = end_t - start_t
                if SQLCategoryTimesReader.DEBUG_EACH_OP_INSTANCE:
                    # - As bad as 56 seconds in split_op_stacks
                    #   yield(process=loop_train_eval, step=0)
                    #   Why is step 0 so slow?
                    # - often 0.02 seconds, 0.10 seconds
                    logger.info("> each_op_instance yield(process={proc}, step={step}) took {sec} seconds".format(
                        proc=process_name,
                        step=step,
                        sec=sec,
                    ))
                yield process_name, step, category_times
                start_t = time.time()

    def _parse_timeline_memo_path(self):
        return _j(self.directory, '{klass}.parse_timeline.pickle'.format(
            klass=self.__class__.__name__,
        ))

    def parse_timeline(self,
                       process_name=None,
                       phase_name=None,
                       machine_name=None,
                       start_time_us=None,
                       end_time_us=None,
                       visible_overhead=False,
                       # pre_reduce=None,
                       timer=None,
                       debug=DEFAULT_debug,
                       debug_memoize=False):
        """
        Return all the Event's belonging to specific process(es), and a specific phase(s).
        The format of returned events is "process_cpu_gpu_category_times" format; i.e.
        {
          <process_cat>: <Event's sorted by Event.start_time>,
          ...
        }

        where process_cat =
            set(
              process_name,
              set(                                             # <-- NOTE: nested set for CPU/GPU/operation
                  CPU if Event.category is a CPU-category,
                  GPU if Event.category is a GPU-category,
                  Event.name if Event.category is a 'operation',
              ),
            )

        PSEUDOCODE:
        for proc in self.process_names:
            events[proc] = Select all events for <process_name> (i.e. over its min/max time range)
            events[proc] = process_op_nest(events[proc]) to get the right operations for constants.CATEGORY_OPERATION.

        # Replace Python/C++/CUDA-C with CPU category.
        # Keep GPU category.
        category_times = dict()
        for proc in events.keys():
            for event in events[proc]:
                if row['category_name'] in Python/C++/CUDA-C/etc:
                    category = 'CPU'
                else:
                    assert category == 'GPU'

                # process information is no longer required.
                category_times_add_time(
                    category_times, row['device_name'], ktime, group_by_device, category=category)

        # In caller:
        # - They will compute a single repetition of times for the different categories of overlap.
        overlap = ComputeOverlap(category_times)

        :param process_name:
        :param group_by_device:
        :param ignore_categories:
        :param debug:
        :return:
        """

        # if should_load_memo(debug_memoize, self._parse_timeline_memo_path()):
        #     ret = load_memo(debug_memoize, self._parse_timeline_memo_path())
        #     return ret

        category_times = dict()
        # operation_types = set()
        # Categories NOT including the operation-type categories (that replaced constants.CATEGORY_OPERATION)
        # categories = set()
        # proc_types = set()

        # if process_name is None and ( start_time_us is not None or end_time_us is not None ):
        #     # [None]
        #     process_names = [process_name]
        # elif process_name is not None:
        #     process_names = [process_name]
        # else:
        #     process_names = self.process_names
        #
        # for process_name in process_names:
        assert process_name is not None
        assert machine_name is not None
        assert phase_name is not None

        process_label = "process={proc}".format(
            proc=process_name)
        """
        {
            <category_name>: <events belonging to process_name, sorted by start_sec>,
            ...
        }
        """
        op_category_times = self.process_events_split(
            process_name=process_name,
            phase_name=phase_name,
            machine_name=machine_name,
            category_name=constants.CATEGORY_OPERATION,
            start_time_us=start_time_us,
            end_time_us=end_time_us,
            timer=timer,
            debug=debug,
            debug_label=process_label,
            fetchall=True,
        )
        # if timer is not None:
        #     timer.end_operation('sql_reader = SQLCategoryTimesReader(...)')

        process_eo_times_dict = self.process_events_split_eo_times(
            process_name=process_name,
            phase_name=phase_name,
            machine_name=machine_name,
            start_time_us=start_time_us,
            end_time_us=end_time_us,
            timer=timer,
            debug=debug,
            debug_label=process_label,
        )

        # if timer is not None:
        #     timer.end_operation('process_events_split(...)')
        # assert len(proc_events) > 0
        # assert len(proc_category_times) > 0
        # assert len(proc_category_times[constants.CATEGORY_OPERATION]) > 0

        for machine_name in op_category_times.keys():
            for proc in op_category_times[machine_name].keys():

                # assert constants.CATEGORY_OPERATION in proc_category_times
                if constants.CATEGORY_OPERATION in op_category_times[machine_name][proc]:
                    """
                    Replace proc_category_times[constants.CATEGORY_OPERATION], handle operation nesting.
                    We are assuming a single process is single-threaded here, so any operation 
                    nesting is form a single-threaded "call-stack".
                    """
                    op_category_times[machine_name][proc][constants.CATEGORY_OPERATION] = process_op_nest_single_thread(
                        op_category_times[machine_name][proc][constants.CATEGORY_OPERATION],
                        show_progress=debug,
                        debug_label=process_label)
                    # Doesn't speed anything up on "test_load"
                    # proc_category_times[constants.CATEGORY_OPERATION] = process_op_nest_parallel(
                    #     proc_category_times[constants.CATEGORY_OPERATION],
                    #     show_progress=debug,
                    #     debug_label=process_label)
                    assert len(op_category_times[machine_name][proc][constants.CATEGORY_OPERATION]) > 0

        if timer is not None:
            timer.end_operation('parse_timeline(...): flatten operations')

        binned_op_category_times = dict()
        for machine_name in op_category_times.keys():
            for proc in op_category_times[machine_name].keys():
                bin_category_times_single_thread(
                    # process_name, proc_category_times,
                    op_category_times[machine_name][proc],
                    visible_overhead=visible_overhead,
                    pre_reduce=pre_reduce_op_event,
                    # categories=categories, operation_types=operation_types,
                    category_times=binned_op_category_times,
                    timer=timer,
                    debug=debug)
                # Doesn't speed anything up on "test_load"
                # bin_category_times_parallel(
                #     process_name, proc_category_times,
                #     categories, category_times, operation_types, debug)

        if timer is not None:
            timer.end_operation('parse_timeline(...): bin operations into separate categories')

        for category_key in list(binned_op_category_times.keys()):
            binned_op_category_times[category_key] = EventsAsEOTimes(binned_op_category_times[category_key])

        if timer is not None:
            timer.end_operation('parse_timeline(...): convert operation times to eo_times')

        common_keys = set.intersection(set(binned_op_category_times.keys()), set(process_eo_times_dict.keys()))
        assert len(common_keys) == 0
        # Add operation eo_times.
        process_eo_times_dict.update(binned_op_category_times)

        if timer is not None:
            total_num_events = sum(len(eo_times) for category_key, eo_times in process_eo_times_dict.items())
            timer.record_throughput('events', total_num_events)

        # proc_category = proc_as_category(process_name)
        # proc_types.add(proc_category)

        # Merge all the process-specific events into a single category_times dict.

        # Q: How do we parallelize this part?
        # A: All we are doing is iterating over every event, and replacing its category.
        #    Then, we are sticking it back inside a single category_times dict.
        #    Regardless of how we partition the work, merging results from multiple workers
        #    involves writing a merge function for category_times.
        #    i.e. merged_category_times = merge(ctimes_1[c], ctimes_2[c],
        #                                       by=event.start_time_usec)
        #    I'm not sure if we'll overcome the single-threaded merge time...
        # for machine_name in op_category_times.keys():
        #     for proc in op_category_times[machine_name].keys():
        #         bin_category_times_single_thread(
        #             # process_name, proc_category_times,
        #             op_category_times[machine_name][proc],
        #             visible_overhead=visible_overhead,
        #             # pre_reduce=pre_reduce,
        #             # categories=categories, operation_types=operation_types,
        #             category_times=category_times,
        #             timer=timer,
        #             debug=debug)
        #         # Doesn't speed anything up on "test_load"
        #         # bin_category_times_parallel(
        #         #     process_name, proc_category_times,
        #         #     categories, category_times, operation_types, debug)

        # if timer is not None:
        #     timer.end_operation('parse_timeline(...): pre_reduce category_times (e.g. into "CPU/GPU" categories)')

        # Sanity check: Events are all sorted.
        # for category, events in category_times.items():
        #     for e1, e2 in zip(events, events[1:]):
        #         assert e1.start_time_usec <= e2.start_time_usec
        # if timer is not None:
        #     timer.end_operation('parse_timeline(...): sanity check')

        # assert len(operation_types) > 0

        # if debug:
        #     logger.info("> DEBUG: parse_timeline: ")
        #     pprint.pprint({
        #         'proc_types':proc_types,
        #         'operation_types':operation_types,
        #         'categories':categories,
        #     }, indent=2)

        # ret = category_times, categories, operation_types, proc_types
        ret = process_eo_times_dict
        # maybe_memoize(debug_memoize, ret, self._parse_timeline_memo_path())
        return ret

    def total_trace_time_sec(self, debug=False):
        """
        Select min(start_time_us) as, max(end_time_us) from Event
        (i.e. across all processes)
        """
        c = self.conn.cursor
        query = textwrap.dedent("""
        SELECT MIN(start_time_us) AS min_us, MAX(end_time_us) AS max_us
        FROM Event
        """)
        sql_exec_query(c, query, klass=self.__class__, debug=debug)
        row = c.fetchone()
        total_time_us = row['max_us'] - row['min_us']
        NumberType = type(total_time_us)
        total_time_sec = total_time_us/NumberType(constants.MICROSECONDS_IN_SECOND)
        return total_time_sec

    def query_trace_period(self,
                          process_name=None,
                          phase_name=None,
                          machine_name=None,
                          debug=False):
        """
        Get the start/end time-bound for a given (machine, process, phase)
        """
        c = self.conn.cursor
        query = textwrap.dedent("""
        SELECT 
            CAST(MIN(e1.start_time_us) - 1 AS BIGINT) as start_time_us,
            CAST(MAX(e1.end_time_us) + 1 AS BIGINT) as end_time_us,
            COUNT(*) as total_events
        FROM 
            Category AS c1
            NATURAL JOIN Event as e1
            NATURAL JOIN Process as p1
            NATURAL JOIN Phase as ph1
            NATURAL JOIN Machine as m 
        WHERE 
            {process_clause} AND
            {phase_clause} AND
            {machine_clause} AND
            -- Skip "process operation" events.
            NOT {process_op_clause}
        """).format(
            process_clause=sql_process_clause(process_name, 'p1', indents=1, allow_none=True),
            phase_clause=sql_phase_clause(phase_name, 'ph1', indents=1, allow_none=True),
            machine_clause=sql_machine_clause(machine_name, 'm', indents=1, allow_none=True),
            process_op_clause=sql_process_op_clause('e1'),
            p=sql_placeholder(),
        )
        params = None
        sql_exec_query(c, query, params, self.__class__, debug)
        rows = c.fetchall()
        assert len(rows) == 1
        row = rows[0]
        row = dict(row)
        row.update({
            'process_name': process_name,
            'machine_name': machine_name,
            'phase_name': phase_name,
        })
        trace_period = TracePeriod.from_row(row)
        return trace_period

    def _sql_process_events_split(
        self,
        process_name=None,
        phase_name=None,
        machine_name=None,
        category_name=None,
        start_time_us=None,
        end_time_us=None,
        # timer=None,
        # debug=False,
        # debug_label=None,
        # fetchall=True
        ):
        # """"
        # SELECT d1.device_name, c1.category_name, e1.event_name, e1.start_time_us, e1.duration_us
        #     , e1.event_id
        #     , p1.process_name, ph1.phase_name, m.machine_name
        #     , e1.pyprof_filename
        #     , e1.pyprof_line_no
        #     , e1.pyprof_function
        #     , e1.pyprof_line_description
        # FROM
        #     Category AS c1
        #     NATURAL JOIN Event as e1
        #     NATURAL JOIN Process as p1
        #     NATURAL JOIN Phase as ph1
        #     NATURAL JOIN Machine as m
        #     NATURAL LEFT JOIN Device as d1
        # """
        query = textwrap.dedent("""
            SELECT d1.device_name, c1.category_name, e1.event_name, e1.start_time_us, e1.duration_us
                , e1.event_id
                , p1.process_name, ph1.phase_name, m.machine_name
            FROM 
                Category AS c1
                NATURAL JOIN Event as e1
                NATURAL JOIN Process as p1
                NATURAL JOIN Phase as ph1
                NATURAL JOIN Machine as m 
                NATURAL LEFT JOIN Device as d1
            WHERE 
                {process_clause} AND
                {phase_clause} AND
                {machine_clause} AND
                {category_clause} AND
                -- Keep events within a EventSplit(start, end) time range.
                {event_split_range_clause} AND
                -- Skip "process operation" events.
                NOT {process_op_clause}
            ORDER BY 
                m.machine_name, p1.process_name, c1.category_name, e1.start_time_us ASC 
            """).format(
            process_clause=sql_process_clause(process_name, 'p1', indents=1, allow_none=True),
            phase_clause=sql_phase_clause(phase_name, 'ph1', indents=1, allow_none=True),
            machine_clause=sql_machine_clause(machine_name, 'm', indents=1, allow_none=True),
            category_clause=sql_category_clause(category_name, 'c1', indents=1, allow_none=True),
            process_op_clause=sql_process_op_clause('e1'),
            event_split_range_clause=sql_event_split_range_clause('e1', start_time_us, end_time_us, indents=1),
        )
        return query

    def process_events_split_eo_times(self,
                             process_name,
                             phase_name=None,
                             machine_name=None,
                             start_time_us=None,
                             end_time_us=None,
                             timer=None,
                             debug=False,
                             debug_label=None):

        assert ( start_time_us is None and end_time_us is None ) or \
               ( start_time_us is not None and end_time_us is not None )

        assert process_name is not None

        c = self.conn.cursor

        process_eo_times = dict()

        # ##
        # ## Do separate fetch queries for each category.
        # ## TODO: ammenable to being parallelized
        # ##
        # category_names = self.categories
        # for category_name in category_names:
        #     if category_name in {constants.CATEGORY_OPERATION}:
        #         continue
        #     # NOTE: we can do each of these in parallel.
        #     # Q: Can we share np.array's efficiently between workers...? I don't think so.
        #     category_cursor = c
        #     # category_cursor = self.conn.get_cursor()
        #     query = self._sql_process_events_split(
        #         process_name=process_name,
        #         phase_name=phase_name,
        #         machine_name=machine_name,
        #         category_name=category_name,
        #         start_time_us=start_time_us,
        #         end_time_us=end_time_us,
        #     )
        #     rows = RowIterator(query, category_cursor, fetchall=True)
        #     # rows = RowIterator(query, category_cursor)
        #     def count_timer(sql_op):
        #         if timer is not None:
        #             timer.end_operation("process_events_split_eo_times(...): convert SQL rows to numpy format; category={cat}: {sql_op}".format(
        #                 sql_op=sql_op,
        #                 cat=category_name,
        #             ))
        #     count = rows.count(timer=count_timer)
        #     def select_timer(sql_op):
        #         if timer is not None:
        #             timer.end_operation("process_events_split_eo_times(...): convert SQL rows to numpy format; category={cat}, rows={n}: {sql_op}".format(
        #                 sql_op=sql_op,
        #                 cat=category_name,
        #                 n=count,
        #             ))
        #     eo_times = RowsAsEOTimes(rows, timer=select_timer)
        #     if timer is not None:
        #         timer.end_operation("process_events_split_eo_times(...): convert SQL rows to numpy format; category={cat}, rows={n}: EO_TIMES READY".format(
        #             cat=category_name,
        #             n=count,
        #         ))
        #     category_key = CategoryKey(
        #         ops=frozenset(),
        #         non_ops=frozenset([category_name]),
        #         procs=frozenset([process_name]))
        #     assert category_key not in process_eo_times
        #     process_eo_times[category_key] = eo_times
        #
        #     # self.conn.put_cursor(category_cursor)

        ##
        ## Do single SQL query to fetch all events
        ## NOT ammenable to being parallelized.
        ##
        category_cursor = c
        # category_cursor = self.conn.get_cursor()
        query = self._sql_process_events_split(
            process_name=process_name,
            phase_name=phase_name,
            machine_name=machine_name,
            start_time_us=start_time_us,
            end_time_us=end_time_us,
        )
        rows = RowIterator(query, category_cursor, fetchall=True,
                           # debug=True,
                           )
        # rows = RowIterator(query, category_cursor)
        def count_timer(sql_op):
            if timer is not None:
                timer.end_operation("process_events_split_eo_times(...): convert SQL rows to numpy format (ALL categories); {sql_op}".format(
                    sql_op=sql_op,
                ))
        total_count = rows.count(timer=count_timer)
        def select_timer(sql_op):
            if timer is not None:
                timer.end_operation("process_events_split_eo_times(...): convert SQL rows to numpy format (ALL categories); rows={n}: {sql_op}".format(
                    sql_op=sql_op,
                    n=total_count,
                ))
        def groupby_key(row):
            return (row['category_name'],)
        row_iter = rows.each_row(timer=select_timer)
        for (category_name,), category_rows_iter in itertools.groupby(row_iter, key=groupby_key):
            if category_name == constants.CATEGORY_OPERATION:
                continue
            category_key = CategoryKey(
                ops=frozenset(),
                non_ops=frozenset([category_name]),
                procs=frozenset([process_name]))
            category_rows = list(category_rows_iter)
            eo_times = RowsAsEOTimes(category_rows, timer=select_timer)
            cat_count = len(category_rows)
            if timer is not None:
                timer.end_operation("process_events_split_eo_times(...): convert SQL rows to numpy format; category={cat}, rows={n}: EO_TIMES READY".format(
                    cat=category_name,
                    n=cat_count,
                ))
            assert category_key not in process_eo_times
            process_eo_times[category_key] = eo_times

            # self.conn.put_cursor(category_cursor)

        return process_eo_times

    def process_events_split(self,
                       process_name=None,
                       phase_name=None,
                       machine_name=None,
                       category_name=None,
                       start_time_us=None,
                       end_time_us=None,
                       timer=None,
                       debug=False,
                       debug_label=None,
                       fetchall=True):

        assert ( start_time_us is None and end_time_us is None ) or \
               ( start_time_us is not None and end_time_us is not None )

        c = self.conn.cursor

        query = self._sql_process_events_split(
            process_name=process_name,
            phase_name=phase_name,
            machine_name=machine_name,
            category_name=category_name,
            start_time_us=start_time_us,
            end_time_us=end_time_us,
        )

        params = None
        sql_exec_query(c, query, params, self.__class__, debug)

        rows = sql_fetch_rows(c, fetchall, debug=debug)
        if timer is not None:
            timer.end_operation('process_events_split(...): SQL query operation event rows')

        event_split = None
        if start_time_us is not None and end_time_us is not None:
            event_split = EventSplit(start_time_us=start_time_us, end_time_us=end_time_us)

        if fetchall and event_split is not None and debug:
            logger.info("{event_split} has {n} event-rows".format(
                event_split=event_split,
                n=len(rows),
            ))

        process_category_times = self._rows_as_category_times(
            rows,
            event_split=event_split,
            timer=timer,
            debug=debug,
            debug_label=debug_label)
        if timer is not None:
            timer.end_operation('process_events_split(...): convert operation event rows to category_times dict')
        return process_category_times

    def _rows_as_category_times(self, rows, event_split=None, timer=None, debug=False, debug_label=None):
        process_category_times = dict()
        def groupby_key(row):
            return (row['machine_name'], row['process_name'])
        for (machine_name, process_name), process_rows in itertools.groupby(rows, key=groupby_key):
            if machine_name not in process_category_times:
                process_category_times[machine_name] = dict()
            assert process_name not in process_category_times[machine_name]
            process_category_times[machine_name][process_name] = dict()
            self._add_event_rows_to_category_times(
                process_category_times[machine_name][process_name], process_rows,
                event_split=event_split,
                timer=timer,
                debug=debug,
                # show_progress=debug,
                show_progress=False,
                debug_label=debug_label)

        return process_category_times

    def process_events(self,
                       process_name=None,
                       phase_name=None,
                       machine_name=None,
                       ignore_categories=None,
                       start_time_us=None,
                       end_time_us=None,
                       op_name=None,
                       keep_categories=None,
                       debug=False,
                       debug_label=None,
                       debug_ops=False,
                       fetchall=True):
        """
        Query ALL events for a particular process_name.
        If process_name is None, return events across ALL processes.

        Return format is "category_times":
        {
            <category_name>: <events belonging to process_name, sorted by start_sec>,
            ...
        }

        :param process_name:
            If process_name is None, return events across ALL processes.
        :param phase_name:
            If phase_name is None, return events across ALL phases.
        :param machine_name:
            If machine_name is None, return events across ALL machines.
        :param ignore_categories:
            Ignore events belonging to this category.
        :param keep_categories:
            Only keep events belonging to this category.
        :param debug:
        :param debug_ops:
        :param fetchall:
        :return:
        """

        c = self.conn.cursor

        @pythunk
        def get_op_clause():
            return """
            EXISTS (
                SELECT * 
                FROM Event as op_event
                    NATURAL JOIN Category as c2
                WHERE 
                    c2.category_name = '{CATEGORY_OPERATION}' AND
                    op_event.event_name = '{op}' AND 
                    {op_event_subsumes_e1_clause}
            )
            """.format(
                op_event_subsumes_e1_clause=sql_overlap_clause('op_event', 'e1', overlap_type='subsumes', indents=3),
                CATEGORY_OPERATION=constants.CATEGORY_OPERATION,
                op=op_name,
            )
        keep_op_clause = op_name is not None
        op_clause = maybe_clause(get_op_clause().maybe_eval(keep_op_clause), keep=keep_op_clause, indents=3)

        query = textwrap.dedent("""
        SELECT d1.device_name, c1.category_name, e1.event_name, e1.start_time_us, e1.duration_us
            , e1.event_id
            , p1.process_name, ph1.phase_name, m.machine_name
            , e1.pyprof_filename
            , e1.pyprof_line_no
            , e1.pyprof_function
            , e1.pyprof_line_description
        FROM 
            Category AS c1
            NATURAL JOIN Event as e1
            NATURAL JOIN Process as p1
            NATURAL JOIN Phase as ph1
            NATURAL JOIN Machine as m 
            NATURAL LEFT JOIN Device as d1
        WHERE 
            {process_clause} AND
            {phase_clause} AND
            {machine_clause} AND
            {debug_ops_clause} AND
            {ignore_clause} AND
            -- Keep events within a start/end time range (e.g. overlap with a specific process phase).
            {event_range_clause} AND
            -- Only keep events that are subsumed by an <op> event.
            -- e.g. Only keep pyprof events that belong to the set_operation('tree_search') annotation
            {op_clause}
        ORDER BY 
            p1.process_name, e1.start_time_us ASC 
        """).format(
            ignore_clause=sql_ignore_clause('c1', ignore_categories, keep_categories, indents=1),
            process_clause=sql_process_clause(process_name, 'p1', indents=1, allow_none=True),
            phase_clause=sql_phase_clause(phase_name, 'ph1', indents=1, allow_none=True),
            machine_clause=sql_machine_clause(machine_name, 'm', indents=1, allow_none=True),
            debug_ops_clause=sql_debug_ops_clause(debug_ops, 'e1', indents=1),
            event_range_clause=sql_event_range_clause('e1', start_time_us, end_time_us, indents=1),
            op_clause=op_clause,
            p=sql_placeholder(),
        )

        params = None
        sql_exec_query(c, query, params, self.__class__, debug)

        rows = sql_fetch_rows(c, fetchall, debug=debug)

        process_category_times = self._rows_as_category_times(
            rows,
            debug=debug,
            debug_label=debug_label)
        return process_category_times

    def _add_event_rows_to_category_times(self, category_times, rows,
                                          event_split=None,
                                          group_by_device=DEFAULT_group_by_device,
                                          timer=None,
                                          debug=False,
                                          show_progress=False,
                                          debug_label=None):
        # rows = c.fetchall()
        # for row in rows:
        if debug_label is None:
            progress_label = '_add_event_rows_to_category_times'
        else:
            progress_label = '_add_event_rows_to_category_times: {debug_label}'.format(
                debug_label=debug_label)
        for row in progress(rows, desc=progress_label, show_progress=show_progress):
            ktime = row_as_ktime(row, event_split=event_split)
            category_times_add_time(category_times, row['device_name'], ktime, group_by_device, category=row['category_name'])

    DEBUG_PARSE = False
    def parse(self, step, process_name, machine_name, bench_name,
              filter_op=True,
              group_by_device=DEFAULT_group_by_device,
              ignore_categories=DEFAULT_ignore_categories,
              debug=DEFAULT_debug,
              show_progress=False):
        """
        This is for reading a operation-instance ("step") at-a-time, for a particular process.
        e.g.
           Give me ALL events across ALL categories
           that overlap with the bench_name='tree_search' operation,
           for process_name=loop_train_eval

        In order to read a "chunk" of the timeline trace at a time, we probably
        want to expose another method.

        Also, we do NOT consider process_id's here at all.

        # PSEUDOCODE:
        rows = SELECT category_name, start_time_us, duration_us FROM Category NATURAL JOIN Event
        for category_name, start_time_us, duration_us in rows:
            category_times[category_name].append((start_time_us, duration_us))

        :param bench_name:
        :return:
        """
        if SQLCategoryTimesReader.DEBUG_PARSE:
            start_parse_t = time.time()
            start_get_events = time.time()
        assert bench_name != NO_BENCH_NAME
        n_steps = self.num_steps(process_name, machine_name, bench_name)
        assert 0 <= step < n_steps

        op_event = self.step_event(step, process_name, machine_name, bench_name)

        parse_debug = debug or self.parse_debug
        if SQLCategoryTimesReader.DEBUG_PARSE:
            end_get_events = time.time()
            sec_get_events = end_get_events - start_get_events
            logger.info("> parse.get_events process={proc}, op={op}, step={step} took {sec} seconds".format(
                proc=process_name,
                op=bench_name,
                step=step,
                sec=sec_get_events,
            ))

        if parse_debug:
            logger.info("> step={step}, process={proc}, op={bench}, time={time}".format(
                step=step, proc=process_name, bench=bench_name, time=op_event))

        return self.events_that_overlap_with(
            op_event, process_name,
            bench_name=bench_name,
            filter_op=filter_op,
            step=step,
            group_by_device=group_by_device,
            ignore_categories=ignore_categories,
            debug=debug,
            show_progress=show_progress)

    def event_by_id(self, event_id, debug=False):
        c = self.conn.cursor
        query = textwrap.dedent("""
        SELECT event_id, device_name, category_name, process_name, event_name, start_time_us, duration_us
        FROM 
            Category
            NATURAL JOIN Event
            NATURAL JOIN Process
            NATURAL LEFT JOIN Device
        WHERE 
            event_id = {p}
        """.format(
            p=sql_placeholder(),
        ))
        params = (
            event_id,
        )
        sql_exec_query(c, query, params, self.__class__, debug)
        row = c.fetchone()
        ktime = row_as_event(row)
        return ktime

    def events_that_overlap_with(self, op_event, process_name,
                                 # For debugging self.parse()
                                 bench_name=None,
                                 filter_op=True,
                                 step=None,
                                 group_by_device=DEFAULT_group_by_device,
                                 ignore_categories=DEFAULT_ignore_categories,
                                 debug=DEFAULT_debug,
                                 show_progress=False):
        return self.events_by_time_range(
            op_event.start_time_usec, op_event.end_time_usec, process_name,
            bench_name=bench_name,
            filter_op=filter_op,
            step=step,
            group_by_device=group_by_device,
            ignore_categories=ignore_categories,
            debug=debug,
            show_progress=show_progress)

    def events_by_time_range(self, start_time_usec, end_time_usec, process_name,
                                 # For debugging self.parse()
                                 bench_name=None,
                                 filter_op=True,
                                 step=None,
                                 group_by_device=DEFAULT_group_by_device,
                                 ignore_categories=DEFAULT_ignore_categories,
                                 debug=DEFAULT_debug,
                                 show_progress=False):

        c = self.conn.cursor

        """
        We want to select all the events from all categories, where the event occurs during the operation <bench_name>. 
        
        WHERE EXISTS (
            SELECT * FROM Category as c_in NATURAL JOIN Event e_in 
            WHERE c_in.category_name = 'Operation' AND e_in.event_name = {p}
            e_in.start_time_us <= e_out.start_time_us AND e_out.start_time_us <= e_in.start_time_us + e_in.duration_us
        )
        Arguments: (? = bench_name)
        - This checks whether the given event occurs during the operation <bench_name>.
          An event E occurs during an operation, if there exists an 'Operation' event 
          that surrounds its start time E.start_us OR its end time E.end_us.
        - However, currently we only ever want to select a SINGLE "step" at a time, 
          so we aren't using this.
        """

        #
        # For each time this op-type was called / for each step:
        #   Split up op-type events based on nesting pattern. <-- handle nesting
        #   Compute the overlap of this op-type instance with all other categories
        #
        # NOTE: We DON'T want to keep any overlap for OTHER op-types,
        # since those will be calculated separately.
        # - One way of handling this; after splitting up by op-types, sweep through
        #   the op-types and remove anything that's not equal to the current op-type.
        #

        # e_out.event_name != 'train_loop' AND
        query = textwrap.dedent("""
        SELECT device_name, category_name, event_name, start_time_us, duration_us, 
               process_name, phase_name, event_id
        FROM 
            Category AS c
            NATURAL JOIN Event
            NATURAL JOIN Process
            NATURAL JOIN Phase
            NATURAL LEFT JOIN Device
        WHERE 
            process_name = {p} AND ( 
                ( {p} <= start_time_us AND start_time_us <= {p} ) OR
                ( {p} <= end_time_us AND end_time_us <= {p} )
            ) AND
            {ignore_clause}
        ORDER BY start_time_us ASC 
        """.format(
            ignore_clause=sql_ignore_clause('c', ignore_categories, indents=1),
            p=sql_placeholder(),
        ))
        params = (
            process_name,
            start_time_usec, end_time_usec,
            start_time_usec, end_time_usec,
        )

        if SQLCategoryTimesReader.DEBUG_PARSE:
            start_parse_query_t = time.time()
        sql_exec_query(c, query, params, self.__class__, debug)
        category_times = dict()
        debug_label = "process={proc}, step={step}".format(
            proc=process_name,
            step=step,
        )
        rows = c.fetchall()
        if SQLCategoryTimesReader.DEBUG_PARSE:
            end_parse_query_t = time.time()
            sec_parse_query = end_parse_query_t - start_parse_query_t
            logger.info("> parse.query process={proc}, op={op}, step={step} took {sec} seconds".format(
                proc=process_name,
                op=bench_name,
                step=step,
                sec=sec_parse_query,
            ))
            start_add_event_rows_t = time.time()
        self._add_event_rows_to_category_times(
            category_times, rows, group_by_device,
            # category_times, c, group_by_device,
            debug=debug or show_progress,
            # Gets in the way of each_op_instance progress bar.
            # show_progress=debug or show_progress,
            show_progress=False,
            debug_label=debug_label)
        if SQLCategoryTimesReader.DEBUG_PARSE:
            end_add_event_rows_t = time.time()
            sec_add_event_rows_t = end_add_event_rows_t - start_add_event_rows_t
            logger.info("> parse.add_event_rows process={proc}, op={op}, step={step} took {sec} seconds".format(
                proc=process_name,
                op=bench_name,
                step=step,
                sec=sec_add_event_rows_t,
            ))

        # if i == 0 and self.debug:
        if debug:
            # Q: What do does train_loop look like, overlapped with all its fellow operation-types?
            json_path = _j(self.directory, "SQLCategoryTimesReader{proc}{step}{bench}.debug.json".format(
                proc=process_suffix(process_name),
                step=step_suffix(step, allow_none=True),
                bench=bench_suffix(bench_name)))
            logger.info("> DEBUG: dump trace events BEFORE process_op_nest @ {path}".format(path=json_path))
            dump_category_times(category_times, json_path, print_log=False)

        if SQLCategoryTimesReader.DEBUG_PARSE:
            start_process_op_nest_t = time.time()
        filter_by_op = None
        if filter_op:
            filter_by_op = bench_name
        category_times[constants.CATEGORY_OPERATION] = process_op_nest_single_thread(category_times[constants.CATEGORY_OPERATION],
                                                             filter_by_op=filter_by_op,
                                                             # Gets in the way of each_op_instance progress bar.
                                                             # show_progress=debug or show_progress,
                                                             show_progress=SQLCategoryTimesReader.DEBUG_PARSE,
                                                             debug_label=debug_label)
        if SQLCategoryTimesReader.DEBUG_PARSE:
            end_process_op_nest_t = time.time()
            sec_process_op_nest_t = end_process_op_nest_t - start_process_op_nest_t
            logger.info("> parse.process_op_nest process={proc}, op={op}, step={step} took {sec} seconds".format(
                proc=process_name,
                op=bench_name,
                step=step,
                sec=sec_process_op_nest_t,
            ))

            end_parse_t = time.time()
            sec_parse = end_parse_t - start_parse_t
            logger.info("> parse process={proc}, op={op}, step={step} took {sec} seconds".format(
                proc=process_name,
                op=bench_name,
                step=step,
                sec=sec_parse,
            ))
            sec_total = sec_get_events + sec_add_event_rows_t + sec_process_op_nest_t + sec_parse_query
            logger.info("> total parse subops = {sec} seconds".format(
                sec=sec_total,
            ))

        return category_times

def bin_category_times_old(
    process_name,
    category,
    events,
    categories=None,
    operation_types=None,
    category_times=None,
    debug=False):

    if categories is None:
        categories = set()
    if operation_types is None:
        operation_types = set()
    if category_times is None:
        category_times = dict()

    proc_category = proc_as_category(process_name)

    progress_label = "parse_timeline: process={proc}, category={cat}".format(
        proc=process_name, cat=category)
    for event in progress(events, desc=progress_label, show_progress=debug):
        if category in constants.CATEGORIES_CPU:
            cat = constants.CATEGORY_CPU
            categories.add(cat)
        elif category == constants.CATEGORY_GPU:
            cat = constants.CATEGORY_GPU
            categories.add(cat)
        elif category == constants.CATEGORY_OPERATION:
            cat = event.name
            operation_types.add(cat)
        else:
            # Q: What about category operation...?
            # We want to KEEP the operation category so we can determine
            # overlap between q_backward/q_forward across processes...
            #
            # I think all we need to do is replace "constants.CATEGORY_OPERATION" for an event
            # with event.name (it's operation-type).
            # Then, when we go to plot the category_times data, we "remove" any operation
            # names from the category (forming an operation_key), and group the data
            # into an operation-specific dict for plotting.
            #
            # We can still test a single process trace without handling this.
            # (graph should be the same with fewer categories: CPU, GPU, CPU + GPU)
            raise RuntimeError("Not sure how to categorize {cat} into CPU or GPU.".format(
                cat=category))

        new_category = frozenset([cat, proc_category])
        if new_category not in category_times:
            category_times[new_category] = []
        # NOTE: if we bin the CPU/GPU/operation events into separate lists,
        # we can use merge_sorted, which may be faster.
        #
        # However, merge_sorted approach will probably cause more allocations...
        insort(category_times[new_category], event, key=lambda event: event.start_time_usec)

    return category_times, categories, operation_types

class EventCategoryMapper:
    """
    Map an Event and an old_category to a new_category, to be used in the category_times format:
    {
        <new_category>: <events sorted by event.start_time>,
        ...
    }

    Events belonging to an old_category are processed in serial order.
    EventCategoryMapper's can track useful state related to a stream of events belonging to a category.
    """
    pass

# class EventCategoryMapper(EventCategoryMapper):
#     def __init__(self):
#         self.categories = set()
#         self.operation_types = set()
#
#     @staticmethod
#     def as_proc_resource_category(event, old_category, state):
#         if old_category in constants.CATEGORIES_CPU:
#             cat = constants.CATEGORY_CPU
#             self.categories.add(cat)
#         elif old_category == constants.CATEGORY_GPU:
#             cat = constants.CATEGORY_GPU
#             self.categories.add(cat)
#         elif old_category == constants.CATEGORY_OPERATION:
#             cat = event.name
#             # logger.info("> operation_types.add {cat}".format(cat=cat))
#             self.operation_types.add(cat)
#         else:
#             # Q: What about category operation...?
#             # We want to KEEP the operation category so we can determine
#             # overlap between q_backward/q_forward across processes...
#             #
#             # I think all we need to do is replace "constants.CATEGORY_OPERATION" for an event
#             # with event.name (it's operation-type).
#             # Then, when we go to plot the category_times data, we "remove" any operation
#             # names from the category (forming an operation_key), and group the data
#             # into an operation-specific dict for plotting.
#             #
#             # We can still test a single process trace without handling this.
#             # (graph should be the same with fewer categories: CPU, GPU, CPU + GPU)
#             raise RuntimeError("Not sure how to categorize {cat} into CPU or GPU.".format(
#                 cat=old_category))
#
#         new_category = frozenset([cat, proc_category])

# def default_pre_reduce(category, event, process_name):
#     if category == constants.CATEGORY_OPERATION:
#         new_category = CategoryKey(
#             ops=frozenset([event.name]),
#             non_ops=frozenset(),
#             procs=frozenset([process_name]))
#     else:
#         new_category = CategoryKey(
#             ops=frozenset(),
#             non_ops=frozenset([category]),
#             procs=frozenset([process_name]))
#     return new_category

def pre_reduce_op_event(category, event, process_name):
    assert category == constants.CATEGORY_OPERATION
    new_category = CategoryKey(
        ops=frozenset([event.name]),
        non_ops=frozenset(),
        procs=frozenset([process_name]))
    return new_category

def bin_category_times(
    # process_name,
    category,
    events,
    visible_overhead=False,
    pre_reduce=None,
    # categories=None,
    # operation_types=None,
    # category_times=None,
    debug=False):
    """
    Given a list of Event's, redistribute them into category_times[<cat>]
    where cat =
        set(
          process_name,
          set(                                             # <-- NOTE: nested set for CPU/GPU/operation
              CPU if Event.category is a CPU-category,
              GPU if Event.category is a GPU-category,
              Event.name if Event.category is a 'operation',
          ),
        )

    We are preparing events for CPU/GPU/operation overlap-processing.
    """

    # if pre_reduce is None:
    #     pre_reduce = default_pre_reduce

    # if categories is None:
    #     categories = set()
    # if operation_types is None:
    #     operation_types = set()

    # if category_times is None:
    #     category_times = dict()
    #     use_insort = True
    # else:
    #     use_insort = False
    category_times = dict()

    # proc_category = proc_as_category(process_name)

    progress_label = "parse_timeline: category={cat}".format(
        cat=category)
    # progress_label = "parse_timeline: process={proc}, category={cat}".format(
    #     proc=process_name, cat=category)
    for event in progress(events, desc=progress_label, show_progress=debug):

        # if category in constants.CATEGORIES_CPU:
        #     cat = constants.CATEGORY_CPU
        #     categories.add(cat)
        # elif category == constants.CATEGORY_GPU:
        #     cat = constants.CATEGORY_GPU
        #     categories.add(cat)
        # elif category == constants.CATEGORY_OPERATION:
        #     cat = event.name
        #     # logger.info("> operation_types.add {cat}".format(cat=cat))
        #     operation_types.add(cat)
        # else:
        #     # Q: What about category operation...?
        #     # We want to KEEP the operation category so we can determine
        #     # overlap between q_backward/q_forward across processes...
        #     #
        #     # I think all we need to do is replace "constants.CATEGORY_OPERATION" for an event
        #     # with event.name (it's operation-type).
        #     # Then, when we go to plot the category_times data, we "remove" any operation
        #     # names from the category (forming an operation_key), and group the data
        #     # into an operation-specific dict for plotting.
        #     #
        #     # We can still test a single process trace without handling this.
        #     # (graph should be the same with fewer categories: CPU, GPU, CPU + GPU)
        #     raise RuntimeError("Not sure how to categorize {cat} into CPU or GPU.".format(
        #         cat=category))
        #
        # new_category = frozenset([cat, proc_category])

        if pre_reduce is not None:
            new_category = pre_reduce(category, event, visible_overhead)

        if new_category is None:
            # SKIP this event entirely.
            continue

        if new_category not in category_times:
            category_times[new_category] = []
        # NOTE: if we bin the CPU/GPU/operation events into separate lists,
        # we can use merge_sorted, which may be faster.
        #
        # However, merge_sorted approach will probably cause more allocations...
        # if use_insort:
        #     insort(category_times[new_category], event, key=lambda event: event.start_time_usec)
        # else:
        # last_event = category_times[new_category][-1]
        event_list = category_times[new_category]
        if len(event_list) == 0 or event.start_time_usec > event_list[-1].end_time_usec:
            # event does NOT overlap with last_event; append it:
            #   [ last_event ]
            #                   [   event   ]
            # Modify last_event to include event:
            #   [ last_event ]  [   event   ]
            category_times[new_category].append(event)
        else:
            # event overlaps with last_event:
            #   [ last_event ]
            #          [   event   ]
            # Modify last_event to include event:
            #   [ last_event       ]
            last_event = event_list[-1]
            new_end = max(event.end_time_usec, last_event.end_time_usec)
            last_event.set_end(new_end)

    # return category_times, categories, operation_types
    return category_times

def bin_category_times_single_thread(
    # process_name,
    proc_category_times,
    visible_overhead=False,
    pre_reduce=None,
    # categories=None,
    # operation_types=None,
    category_times=None,
    timer=None,
    debug=False):
    """
    Given proc_category_times (category_times format for a specific process), redistribute the
    Event's into category_times to prepare for CPU/GPU/operation overlap-processing,
    potentially across processes.

    See bin_category_times for details.

    :param process_name:
    :param proc_category_times:
    :param categories:
    :param operation_types:
    :param category_times:
    :param debug:
    :return:
    """
    # if categories is None:
    #     categories = set()
    # if operation_types is None:
    #     operation_types = set()
    if category_times is None:
        category_times = dict()

    for i, (category, events) in enumerate(proc_category_times.items()):
        category_times_i = bin_category_times(
            # process_name,
            category, events,
            visible_overhead=visible_overhead,
            pre_reduce=pre_reduce,
            # categories=categories, operation_types=operation_types, category_times=None,
            # category_times=None,
            debug=debug)
        # NOTE: pair-wise merge is inefficient; it will create more intermediate lists than required.
        # NOTE: this merging computation is only done to support re-running event overlap with CPU/GPU labels...
        # - Alternative: run the event trace ONCE using fine-grained CPU labels,
        #   and translate overlaps into coarse-grained "CPU" label
        # - RESULT:
        #   - all the different overlap computations can simply RE-USE the event overlap results
        #     ( this is actually a HUGE deal and will speed things up a lot)
        #   - no need to perform "merging" of times (pre_reduce eliminated);
        #     if we still need a pre-reduce step, we should only need to do it once.
        merge_category_times(category_times, category_times_i, inplace=True)

    # return category_times, categories, operation_types
    return category_times

# def bin_category_times_single_thread_old(
#     process_name,
#     proc_category_times,
#     categories=None,
#     operation_types=None,
#     category_times=None,
#     debug=False):
#     if categories is None:
#         categories = set()
#     if operation_types is None:
#         operation_types = set()
#     if category_times is None:
#         category_times = dict()
#
#     for category, events in proc_category_times.items():
#         bin_category_times(process_name, category, events,
#                            categories, operation_types, category_times,
#                            debug)
#
#     return category_times, categories, operation_types

# def BinCategoryTimesWorker(process_name, category, events, debug):
# def BinCategoryTimesWorker(args):
#     process_name, category, events, debug = args
#     return bin_category_times(process_name, category, events, debug=debug)

def split_category_times_by_category(process_name, proc_category_times, debug):
    for category, events in proc_category_times.items():
        yield process_name, category, events, debug

# TODO: if this doesn't help much, we could perform more equal splits by doing:
# 1. compute total number of events
# 2. split proc_category_times into n dicts with equal events;
#    Need to make a split_category_times(category_times, n) to do that
# 3. make the Worker take proc_category_times instead of <category, events>

# def bin_category_times_parallel(
#     process_name,
#     proc_category_times,
#     categories=None,
#     category_times=None,
#     operation_types=None,
#     debug=False,
#     nprocs=None):
#     if categories is None:
#         categories = set()
#     if operation_types is None:
#         operation_types = set()
#     if category_times is None:
#         category_times = dict()
#
#     with multiprocessing.Pool(nprocs) as pool:
#         splits = list(split_category_times_by_category(process_name, proc_category_times, debug=False))
#         it = pool.imap_unordered(BinCategoryTimesWorker, splits)
#         progress_label = "bin_category_times_parallel: process={proc}".format(proc=process_name)
#         for i, (category_times_i, categories_i, operation_types_i) in enumerate(progress(it, desc=progress_label, total=len(splits), show_progress=debug)):
#             merge_category_times(category_times, category_times_i, inplace=True)
#             categories.update(categories_i)
#             operation_types.update(operation_types_i)
#
#     return category_times, categories, operation_types

def merge_events(events1, events2):
    return merge_sorted(events1, events2, key=lambda event: event.start_time_usec)

def merge_all_category_times(category_times_list):
    merged_category_times = dict()
    for category_times in category_times_list:
        merge_category_times(merged_category_times, category_times, inplace=True)
    # categories = set()
    # for category_times in category_times_list:
    #     categories.extend(category_times.keys())
    # for category in categories:
    #     merged_category_times[category] = []
    #     for category_times in category_times_list:
    #         if category not in category_times:
    #             continue
    #         merged_category_times[category] = merge_events(merged_category_times[category],
    #                                                        category_times[category])
    return merged_category_times

def merge_category_times(ctimes1, ctimes2, inplace=False):
    if inplace:
        merged_category_times = ctimes1
        category_times_list = [ctimes2]
    else:
        merged_category_times = dict()
        category_times_list = [ctimes1, ctimes2]

    categories = set()
    for category_times in category_times_list:
        categories.update(category_times.keys())

    for category in categories:
        if category not in merged_category_times:
            merged_category_times[category] = []
        for category_times in category_times_list:
            if category not in category_times:
                continue
            merged_category_times[category] = merge_events(merged_category_times[category],
                                                           category_times[category])
    return merged_category_times

def _sqlite_traces_db_path(directory):
    return _j(directory, "traces.db")

def _psql_db_config_path(directory):
    return _j(directory, "psql_config.json")

def psql_insert_csv(csv_path, table, db_name, host=None, user=None, password=None):
    def build_copy_from_sql():
        with open(csv_path) as f:
            header = f.readline()
            header = re.split(r',', header)
        col_str = ", ".join(header)
        # NOTE: When you tell postgres to execute "COPY FROM $path" it expects $path to be available on the same machine as postgres is running.
        # So, this won't work if postgres is containerized ($path isn't accessible).
        # Hence, we use psql + "COPY FROM STDIN" instead.
        copy_from_sql = textwrap.dedent("""\
        COPY {table} ({col_str})
        FROM STDIN
        DELIMITER ',' CSV HEADER;
        """.format(
            col_str=col_str,
            table=table,
        ))
        return copy_from_sql

    copy_from_sql = build_copy_from_sql()
    cmd_kwargs = psql_cmd_args(
        db_name,
        command=copy_from_sql,
        host=host, user=user, password=password)
    # Q: Do we need to disable foreign key checks?
    with open(csv_path, 'r') as f:
        subprocess.check_call(stdin=f, **cmd_kwargs)

def psql_cmd_args(db_name, command=None, host=None, user=None, password=None):
    """
    Construct args for running
    $ psql ...

    :param db_name:
      Database name
    :param command:
      An SQL string to execute.

      NOTE: this is the same as shell option:
      $ psql --command
    :return:
    """
    assert db_name is not None

    # subprocess.run(**kwargs)
    kwargs = dict()

    cmd = ['psql']

    # Construct:
    # $ psql [OPTION]... [DBNAME [USERNAME]]

    if host is not None:
        cmd.extend(['-h', host])

    if user is not None:
        cmd.extend(['-U', user])

    if command is not None:
        cmd.extend(['-c', command])

    cmd.append(db_name)

    # Pass psql password via environment variable.
    env = dict(os.environ)
    if password is not None:
        env['PGPASSWORD'] = password

    kwargs['args'] = cmd
    kwargs['env'] = env

    return kwargs

def sql_input_path(directory):
    if py_config.SQL_IMPL == 'psql':
        return _psql_db_config_path(directory)
    elif py_config.SQL_IMPL == 'sqlite':
        return _sqlite_traces_db_path(directory)
    else:
        raise NotImplementedError("Not sure what input file to use for SQL_IMPL={impl}".format(
            impl=py_config.SQL_IMPL))

def csv_inserter_path(directory, table, suffix):
    basename = "{table}{suffix_str}.csv".format(
        table=table,
        suffix_str="_{suffix}".format(suffix=suffix if suffix is not None else ''),
    )
    return _j(directory, basename)

def sql_exec_query(cursor, query, params=None, klass=None, debug=False):
    c = cursor
    if debug:
        name_str = ""
        if klass is not None:
            name_str = "{name} ".format(
                name=klass.__name__)
        logger.info("> {name_str}query:".format(
            name_str=name_str))
        logger.info(query)
        if params is not None:
            logger.info("> params:")
            pprint.pprint(params, indent=2)
    start_t = time.time()
    if params is not None:
        c.execute(query, params)
    else:
        c.execute(query)
    end_t = time.time()
    if debug:
        logger.info("> query took {sec} seconds".format(sec=end_t - start_t))

def sql_count_from(cursor, sql_query, debug=False):
    count_query = textwrap.dedent("""
            SELECT COUNT(*) FROM (
                {sql_query}
            ) AS count_table
            """).format(
        sql_query=maybe_indent(sql_query, indents=1))
    sql_exec_query(cursor, count_query, debug=debug)
    row = cursor.fetchone()
    assert len(row) == 1
    count = row[0]
    return count

def sql_operator_in(expr, values, indents=None):
    """
    SELECT column_name(s)
    FROM table_name
    WHERE column_name IN (value1, value2, ...);
          ------------------------------------
          Generate this part like this:
          ( {expr} IN (value1, value2, ...) )

    Convenience function:
      If values are strings, quotifies them.
      Adds brackets around entire expression.

    :param expr:
    :param values:
    :param indents:
    :return:
    """
    assert len(values) >= 1
    txt = textwrap.dedent("""\
        ( ( {expr} ) IN ({values}) )
        """.format(
        expr=expr,
        values=', '.join([sql_value_string(value) for value in values])
    ).rstrip())
    txt = maybe_indent(txt, indents)
    return txt

def sql_fetch_rows(c, fetchall, debug=False):
    if fetchall:
        query_start_t = time.time()
        rows = c.fetchall()
        query_end_t = time.time()
        time_sec = query_end_t - query_start_t
        if debug:
            logger.info("> query.fetchall took {sec} seconds".format(
                sec=time_sec,
            ))
    else:
        if debug:
            logger.info("> fetchall = {fetchall}".format(
                fetchall=fetchall,
            ))
        rows = c
    return rows

def sql_compose_inequality(ineq_symbol, exprs, tautology_pairs=[], indents=None):
    """
    Express composed inequalities in SQL, for example:

    Condition (not supported by SQL syntax):
        event_split.start <= e.start <= e.end <= event_split.end

    Expanded (to support SQL binary-inequality syntax):
        event_split.start <= e.start AND
                             e.start <= e.end AND
                                        e.end <= event_split.end
    Simplified (remove tautology_pairs):
        event_split.start <= e.start AND

                                        e.end <= event_split.end

    :param ineq_symbol:
        An inequality symbol; i.e. one of {'<=', '>=', '<', '>'}
    :param exprs:
        list of SQL expressions (could be constants, column-names, etc.)
    :param tautology_pairs:
        e.g. for ineq_symbol '<='
        List of (expr1, expr2), such that:
            expr1 <= expr2
        Is ALWAYS true, and can be safely removed from the "expanded" SQL expression.
    :return:
    """
    assert ineq_symbol in {'<=', '>=', '<', '>'}
    assert len(exprs) >= 2
    expr_pairs = []
    for expr1, expr2 in zip(exprs, exprs[1:]):
        if (expr1, expr2) in tautology_pairs:
            continue
        expr_pair = "( {expr1} ) {ineq} ( {expr2} )".format(
            expr1=expr1,
            ineq=ineq_symbol,
            expr2=expr2,
        )
        expr_pairs.append(expr_pair)
    sql_expr_joined = " AND ".join(expr_pairs)
    sql_expr_bracketed = "( {sql} )".format(sql=sql_expr_joined)
    sql_expr_indented = maybe_indent(sql_expr_bracketed, indents)
    return sql_expr_indented

def sql_get_conn_kwargs(db_path, host=None, user=None, password=None, debug=False):
    if host is None:
        if 'RLSCOPE_POSTGRES_HOST' in os.environ:
            host = os.environ['RLSCOPE_POSTGRES_HOST']
        elif 'PGHOST' in os.environ:
            host = os.environ['PGHOST']
        else:
            host = 'localhost'

    if user is None:
        if 'PGUSER' in os.environ:
            user = os.environ['PGUSER']
        else:
            user = get_username()

    if password is None:
        if 'PGPASSWORD' in os.environ:
            password = os.environ['PGPASSWORD']
        # No default password.

    if debug:
        logger.info("Using DB_HOST = {host}".format(host=host))

    if py_config.SQL_IMPL == 'psql':
        # TracesPostgresConnection(db_path, host=host, user=user, password=password)
        return {
            'db_config_path': db_path,
            'host': host,
            'user': user,
            'password': password,
        }
    elif py_config.SQL_IMPL == 'sqlite':
        # TracesSQLiteConnection(db_path, host=host)
        return {
            'db_path': db_path,
            'host': host,
        }
    raise NotImplementedError("Not sure how to create connection for SQL_IMPL={impl}".format(
        impl=py_config.SQL_IMPL))

def get_sql_connection(db_path, host=None, user=None, password=None, debug=False):
    """
    Make it easy to enable/disable connection pooling.
    """
    if USE_CONNECTION_POOLING:
        conn = ConnectionPoolManager.current_connection_pool(db_path=db_path, host=host, user=user, password=password, debug=debug).getconn()
        return conn
    else:
        conn = sql_create_connection(db_path=db_path, host=host, user=user, password=password, debug=debug)
        return conn

def sql_create_connection(db_path, host=None, user=None, password=None, debug=False, pool=None):
    conn_kwargs = sql_get_conn_kwargs(
        db_path,
        host=host,
        user=user,
        password=password,
        debug=debug)
    if py_config.SQL_IMPL == 'psql':
        return TracesPostgresConnection(**conn_kwargs, pool=pool)
    elif py_config.SQL_IMPL == 'sqlite':
        return TracesSQLiteConnection(**conn_kwargs, pool=pool)
    raise NotImplementedError("Not sure how to create connection for SQL_IMPL={impl}".format(
        impl=py_config.SQL_IMPL))

def sql_placeholder():
    if py_config.SQL_IMPL == 'psql':
        return "%s"
    elif py_config.SQL_IMPL == 'sqlite':
        return "?"
    raise NotImplementedError("Not sure how to create connection for SQL_IMPL={impl}".format(
        impl=py_config.SQL_IMPL))

def rows_as_ktime(rows, event_split=None):
    return [row_as_ktime(row, event_split=event_split) for row in rows]

def row_as_ktime(row, event_split=None):
    """
    Row should be a result of "Event NATURAL JOIN Category", and hence contain at least:
    - start_time_us
    - duration_us
    - event_name
    """
    TimeType = type(row['start_time_us'])
    if event_split is None:
        start_usec = row['start_time_us']
        end_usec = row['start_time_us'] + row['duration_us']
    else:
        """
        Trimming event to fit within event-split:

        ALL event types, regardless of how they were categorized during the SQL query, 
        can be trimmed using the same simple code:

        new_event = Event(
            start=max(event_split.start, e.start),
            end=min(event_split.end, e.end))
        """
        start_usec = max(
            TimeType(event_split.start_time_us),
            row['start_time_us'])
        end_usec = min(
            TimeType(event_split.end_time_us),
            row['start_time_us'] + row['duration_us'])

    ktime = KernelTime(
        start_usec=start_usec,
        end_usec=end_usec,
        name=row['event_name'],
    )
    event_maybe_add_pyprof_fields(ktime, row)
    event_maybe_add_process_phase_fields(ktime, row)
    return ktime

def event_maybe_add_pyprof_fields(ktime, row):
    if 'pyprof_function' in row and row['pyprof_function'] is not None:
        ktime.pyprof_function = row['pyprof_function']
        ktime.pyprof_line_no = row['pyprof_line_no']
        ktime.pyprof_filename = row['pyprof_filename']
        ktime.pyprof_line_description = row['pyprof_line_description']

def event_maybe_add_process_phase_fields(ktime, row):
    if 'process_name' in row:
        ktime.process_name = row['process_name']

    if 'phase_name' in row:
        ktime.phase_name = row['phase_name']

    if 'event_id' in row:
        ktime.event_id = row['event_id']

def row_as_event(row, event_split=None):
    ktime = row_as_ktime(row, event_split=event_split)
    # Add additional attributes to KernelTime.
    ktime.event_id = row['event_id']
    ktime.device_name = row['device_name']
    ktime.category_name = row['category_name']
    ktime.process_name = row['process_name']
    return ktime

class Phase:
    def __init__(
            self,
            phase_name,
            phase_start_time_us,
            phase_end_time_us,
            # Swallow any excess arguments
            **kwargs):
        self.phase_name = phase_name
        self.phase_start_time_us = phase_start_time_us
        self.phase_end_time_us = phase_end_time_us

    @staticmethod
    def from_row(row):
        return obj_from_row(Phase, row)

    @property
    def time_usec(self):
        return self.phase_end_time_us - self.phase_start_time_us

    def __str__(self):
        if hasattr(self, 'phase_start_time_us'):
            return "Phase(name={name}, start={start} us, dur={dur} us)".format(
                name=self.phase_name, start=self.phase_start_time_us, dur=self.time_usec)
        return "Phase(name={name})".format(
            name=self.phase_name)

    def __repr__(self):
        return str(self)

class Operation:
    def __init__(
        self,
        operation_name,
        # Swallow any excess arguments
        **kwargs):
        self.operation_name = operation_name

    @staticmethod
    def from_row(row):
        return obj_from_row(Operation, row)

    def __str__(self):
        return "Operation(name={name})".format(
            name=self.operation_name)

    def __repr__(self):
        return str(self)

class TracePeriod:
    def __init__(
        self,
        start_time_us,
        end_time_us,
        total_events,
        process_name=None,
        phase_name=None,
        machine_name=None,
        # Swallow any excess arguments
        **kwargs):
        self.start_time_us = start_time_us
        self.end_time_us = end_time_us
        self.total_events = total_events
        self.process_name = process_name
        self.phase_name = phase_name
        self.machine_name = machine_name

    @staticmethod
    def from_row(row):
        return obj_from_row(TracePeriod, row)

    @property
    def duration_us(self):
        return self.end_time_us - self.start_time_us

    def __str__(self):
        bldr = ToStringBuilder(obj=self)
        if self.process_name is not None:
            bldr.add_param('process', self.process_name)
        if self.machine_name is not None:
            bldr.add_param('machine', self.machine_name)
        if self.phase_name is not None:
            bldr.add_param('phase', self.phase_name)

        bldr.add_param('start', usec_string(self.start_time_us))
        bldr.add_param('end', usec_string(self.end_time_us))
        bldr.add_param('total_events', (self.total_events))

        return bldr.to_string()

    def __repr__(self):
        return str(self)

class TrainingProgress:
    def __init__(
            self,

            total_timesteps,
            start_trace_time_us,

            start_percent_complete,
            start_num_timesteps,
            start_training_time_us,

            end_percent_complete,
            end_training_time_us,
            end_num_timesteps,

            machine_name, machine_id,
            process_name, process_id,
            phase_name, phase_id,

            # Swallow any excess arguments
            **kwargs):

        self.total_timesteps = total_timesteps
        self.start_trace_time_us = start_trace_time_us

        self.start_percent_complete = start_percent_complete
        self.start_num_timesteps = start_num_timesteps
        self.start_training_time_us = start_training_time_us

        self.end_percent_complete = end_percent_complete
        self.end_training_time_us = end_training_time_us
        self.end_num_timesteps = end_num_timesteps

        self.machine_name = machine_name
        self.machine_id = machine_id

        self.process_name = process_name
        self.process_id = process_id

        self.phase_name = phase_name
        self.phase_id = phase_id


    @staticmethod
    def from_row(row):
        return obj_from_row(TrainingProgress, row)

    def __str__(self):
        return "TrainingProgress(machine={mach}, process={proc}, phase={phase}, percent_complete={perc})".format(
            mach=self.machine_name,
            proc=self.process_name,
            phase=self.phase,
            perc=( self.end_percent_complete - self.start_percent_complete ),
        )

    def __repr__(self):
        return str(self)


def process_op_nest_single_thread(op_events, filter_by_op=None,
                    debug=False,
                    show_progress=False,
                    debug_label=None):
    """
    Given nested operation-type events, have the inner-nested
    events "absorb" the outer-nested events.

    Then, filter out only the event-type of interest.


    op-type                    [op3]
                          [     op2      ]
                     [          op1          ]

    absorb =>        [op1][op2][op3][op2][op1]

    filter(op1) =>   [op1]               [op1]

    :param bench_name:
        Current op-type.
        Only keep ops of this type.
    :param op_events: List[KernelTime]
    :return:
    """

    # Pre-condition:
    # op_events are events from a SINGLE process, sorted by start_time_sec.
    procs = set()
    for event in op_events:
        procs.add(event.process_name)
    assert len(procs) == 1

    # NOTE: We could (but haven't) parallelize this by first scanning for
    # "contiguous chunks" of operations.
    # Then the Worker would take a contiguous chunk of operations,
    # convert it to an OpStack,
    # then yield events from it.

    all_events = []
    for i, op_stack in enumerate(split_op_stacks(op_events,
                                                 show_progress=show_progress,
                                                 debug_label=debug_label,
                                                 each_push=False)):
        if py_config.DEBUG_SPLIT_STACK_OPS:
            start_i = len(all_events)

        for event in op_stack.get_absored_ops():
            if filter_by_op is not None and event.name != filter_by_op:
                continue
            if event.process_name is None:
                logger.warning("Saw process_name = {proc} for event = {ev}".format(
                    proc=event.process_name,
                    ev=event,
                ))
                assert event.process_name is not None
            all_events.append(event)

        if py_config.DEBUG_SPLIT_STACK_OPS:
            end_i = len(all_events)
            logger.info("OpStack[{i}]\n{events}".format(
                i=i,
                events=textwrap.indent(pprint.pformat(all_events[start_i:end_i]), prefix="  "),
            ))

        # logger.info(pprint_msg(all_events))
        # raise RuntimeError("Exit early")

    # raise RuntimeError("Exit early")

    return all_events

def each_stack_trace(events,
                     show_progress=False,
                     debug=False,
                     debug_label=None):
    # all_events = []
    return split_op_stacks(events,
                           show_progress=show_progress,
                           debug_label=debug_label,
                           each_push=True)

    # for op_stack in split_op_stacks(events,
    #                                 show_progress=show_progress,
    #                                 debug_label=debug_label,
    #                                 each_push=True):
    #     yield op_stack
    #     # for event in op_stack.get_absored_ops():
    #     #     if filter_by_op is not None and event.name != filter_by_op:
    #     #         continue
    #     #     all_events.append(event)

# def ProcessOpNestWorker(op_stack : OpStack, debug):
def ProcessOpNestWorker(args):
    op_stack, filter_by_op, debug = args
    events = []
    for event in op_stack.get_absored_ops():
        if filter_by_op is not None and event.name != filter_by_op:
            continue
        events.append(event)
    return events

def split_op_stacks(op_events,
                    debug_label=None,
                    show_progress=False,
                    each_push=False):

    # NOTE: It IS possible to parallelize split_op_stacks.
    # The worker processes just need to agree who covers the "chunk" on the boundary of the event split
    # e.g. if we start in the middle of a "chunk", walk forwards past the end of the chunk,
    # and assume the previous worker will handle it.
    # def as_result(op_stack):
    #     if just_op_stack:
    #         return op_stack
    #     return op_stack, filter_by_op, debug

    op_stack = None
    for i, op_event in enumerate(progress(
        op_events,
        desc=as_progress_label("split_op_stacks", debug_label),
        show_progress=show_progress)):
        if op_stack is None:
            op_stack = OpStack(op_event)
            if each_push:
                yield op_stack
        elif op_stack.ktime.subsumes(op_event):
            op_ins = op_stack.insert(op_event)
            if each_push:
                yield op_ins
        else:
            yield op_stack
            # yield as_result(op_stack)
            op_stack = OpStack(op_event)
            if each_push:
                yield op_stack

    if op_stack is not None and not each_push:
        # yield as_result(op_stack)
        yield op_stack

def process_op_nest_parallel(op_events, filter_by_op=None,
                             debug=False,
                             show_progress=False,
                             debug_label=None,
                             nprocs=None):
    """
    Given nested operation-type events, have the inner-nested
    events "absorb" the outer-nested events.

    Then, filter out only the event-type of interest.


    op-type                    [op3]
                          [     op2      ]
                     [          op1          ]

    absorb =>        [op1][op2][op3][op2][op1]

    filter(op1) =>   [op1]               [op1]

    :param bench_name:
        Current op-type.
        Only keep ops of this type.
    :param op_events: List[KernelTime]
    :return:
    """

    # def check_no_complete_overlap(op_events):
    #     for op1, op2 in progress(zip(op_events, op_events[1:]),
    #                              desc=as_progress_label("process_op_nest.check", debug_label),
    #                              show_progress=show_progress,
    #                              total=len(op_events)):
    #         assert not op1.equals(op2)
    # check_no_complete_overlap(op_events)

    # total_op_stacks = len(list(split_op_stacks(op_events, filter_by_op, debug, show_progress=True)))
    # logger.info("> total_op_stacks = {total_op_stacks}".format(
    #     total_op_stacks=total_op_stacks,
    # ))
    total_op_stacks = None

    def _split_op_stacks(op_events):
        for op_stack in split_op_stacks(op_events, filter_by_op, debug):
            # Pass extra arguments to ProcessOpNestWorker.
            yield op_stack, filter_by_op, debug

    # with multiprocessing.Pool(nprocs) as pool:
    with ProcessPoolExecutor(nprocs) as pool:
        splits = _split_op_stacks(op_events)
        all_events = []
        for i, events in enumerate(progress(
            pool.map(ProcessOpNestWorker, splits),
            desc=as_progress_label("process_op_nest_parallel", debug_label),
            total=total_op_stacks,
            show_progress=show_progress)
        ):
            all_events.extend(events)

    return all_events

class OpStack:
    """
    Helper class for process_op_nest;
    shouldn't be used by anything else.

    An OpStack is a stack of operations.
    e.g. op1, op2, op3, op4, op5 are all OpStack objects.

                   [op2] [op3] [op4] [op5]
    OpStack -> [             op1               ]

    op1 = OpStack()
    op1.sub_ops = [op2, op3, op4, op5]

    NOTE:
        OpStack assumes operations will be inserted sorted according to op.start_time.
    """
    def __init__(self, kernel_time):
        self.sub_ops = []
        self.ktime = kernel_time
        # self.last_insert_start_time = None
        self.parent = None

    @property
    def time_sec(self):
        """
                          [op3: 1 second]
                       [   op2: 5 seconds   ]
        OpStack -> [       op1: 10 seconds        ]

        >>> op3.time_sec()
        1 second
        >>> op2.stacktrace()
        5 seconds
        >>> op1.stacktrace()
        10 seconds

        :param op_stack:
        :return:
            List of OpStack entries.
        """
        return self.ktime.total_time_sec

    def stacktrace(self):
        """
                          [op3]
                       [   op2   ] [op4] [op5]
        OpStack -> [             op1               ]

        >>> op3.stacktrace()
        [op1, op2, op3]
        >>> op2.stacktrace()
        [op1, op2]
        >>> op4.stacktrace()
        [op1, op4]
        >>> op1.stacktrace()
        [op1]

        :param op_stack:
        :return:
            List of OpStack entries.
        """
        trace = []
        curframe = self
        while curframe != None:
            trace.append(curframe)
            curframe = curframe.parent
        # Put root-call is at the start of the list.
        trace.reverse()
        return trace

    def subsumes(self, op):
        return self.ktime.subsumes(op.ktime)

    def insert(self, ktime):
        # if self.last_insert_start_time is None:
        #     self.last_insert_start_time = ktime.start_time_usec
        # else:
        #     assert self.last_insert_start_time <= ktime.start_time_usec

        op1 = OpStack(ktime)
        # assert self.subsumes(op1)
        self._insert(op1)
        return op1

    def _sub_ops_key(self, op):
        return op.start_time_usec

    def _insert(self, op1):
        # assert self.subsumes(op1)
        idx = insort_right(self.sub_ops, op1, key=self._sub_ops_key, skip_insert=True)
        if idx == 0 or not self.sub_ops[idx - 1].subsumes(op1):
            op1.parent = self
            # if len(self.sub_ops) > 0:
            #     assert op1.ktime.is_after(self.sub_ops[-1].ktime)
            self.sub_ops.insert(idx, op1)
        else:
            self.sub_ops[idx - 1]._insert(op1)

    # def _insert_old(self, op1):
    #     # PROBLEM: We are doing O(n^2) insertion here...
    #     # Basically, for half of the events, everything is subsumed by a single event.
    #     # I think it's the "training_loop" annotation.
    #     #
    #     # PSEUDOCODE for fix:
    #     # # NOTE: we are inserting events in sorted order (ordered by event.start_time_usec)
    #     # # Hence, the subsume-er event will exist before the subsume-ee
    #     # idx = Do binary search on event.start_time_usec to find where we would insert the event.
    #     # if idx == 0 or not event[idx-1].subsumes(op1):
    #     #   sub_ops.insert(idx, op1)
    #     # else:
    #     #   sub_ops[idx-1]._insert(op1)
    #     assert self.subsumes(op1)
    #     for op2 in self.sub_ops:
    #         if op2.subsumes(op1):
    #             op2._insert(op1)
    #             return
    #     if len(self.sub_ops) > 0:
    #         assert op1.ktime.is_after(self.sub_ops[-1].ktime)
    #     op1.parent = self
    #     self.sub_ops.append(op1)

    @property
    def name(self):
        return self.ktime.name

    @property
    def start_time_usec(self):
        return self.ktime.start_time_usec

    @property
    def end_time_usec(self):
        return self.ktime.end_time_usec

    def get_absored_ops(self, recursive=True):
        def maybe_ktime(*args, **kwargs):
            ktime = KernelTime(*args, **kwargs,
                               name=self.name,
                               create_from=self.ktime)
            if ktime.start_time_usec < ktime.end_time_usec:
                return ktime
            return None

        if len(self.sub_ops) == 0:
            yield self.ktime
            return

        """
        
                                     Special case (2): End
        Special case (1): Start      |
        ---                       ------
           [op2]-[op3]-[op4]-[op5]
        [       |     |op1  |           ]
                -------------
                Iterative case: Between sub-ops
        """

        # Special case (1): Start
        # Space from op1 until the first sub-op.
        start_op = maybe_ktime(start_usec=self.start_time_usec,
                               end_usec=self.sub_ops[0].start_time_usec)
        if start_op is not None:
            yield start_op

        # zip([1], []):
        #   Nothing.
        # zip([1, 2], [2]):
        #   (1, 2)

        # Iterative case: Between sub-ops
        # Space between adjacent operations, which is covered only by op1.
        for op1, op2 in zip(self.sub_ops, self.sub_ops[1:]):

            if recursive:
                # Recursive case (1):
                #   Recurse on [op2], [op3], [op4] in the diagram above.
                for op in op1.get_absored_ops(True):
                    yield op

            op = maybe_ktime(start_usec=op1.end_time_usec,
                             end_usec=op2.start_time_usec)
            if op:
                yield op

        if recursive:
            # Recursive case (2):
            #   Recurse on [op5] in the diagram above.
            for op in self.sub_ops[-1].get_absored_ops(True):
                yield op

        # Special case (2): End
        # Space from op1 until the first sub-op.
        end_op = maybe_ktime(start_usec=self.sub_ops[-1].end_time_usec,
                             end_usec=self.end_time_usec)
        if end_op is not None:
            yield end_op

#
# From bisect.insort.
# Modify to add a key-function.
#
def insort_right(a, x, lo=0, hi=None,
                 key=lambda x: x,
                 skip_insert=False):
    """Insert item x in list a, and keep it sorted assuming a is sorted.

    If x is already in a, insert it to the right of the rightmost x.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if key(x) < key(a[mid]):
            hi = mid
        else:
            lo = mid+1
    if not skip_insert:
        a.insert(lo, x)
    return lo
insort = insort_right   # backward compatibility

def insort_list(xs, ys, key=lambda x: x):
    """
    :param xs:
        The list to be inserted into.
        Must be sorted.
    :param ys:
        A list (doesn't need to be sorted).
        These are the elements to be inserted into xs.
    :return:
    """
    for y in ys:
        insort(xs, y, key=key)

def merge_sorted(xs, ys, key=lambda x: x):
    """
    Merge two sorted lists.
    :param inplace
        Insert ys into xs.
    """

    i = 0
    j = 0

    # xs:         5 6 7 8
    # ys: 1 2 3 4 5       9 10 11

    # zs = []

    zs = [None]*(len(xs) + len(ys))
    m = 0

    while i < len(xs) and j < len(ys):
        if key(xs[i]) < key(ys[j]):
            # zs.append(xs[i])
            zs[m] = xs[i]
            m += 1
            i += 1
        else:
            # zs.append(ys[j])
            zs[m] = ys[j]
            m += 1
            j += 1

    if i < len(xs):
        # zs.extend(xs[i:])
        for k in range(i, len(xs)):
            zs[m] = xs[k]
            m += 1
        i += len(xs[i:])
    elif j < len(ys):
        # zs.extend(ys[j:])
        for k in range(j, len(ys)):
            zs[m] = ys[k]
            m += 1
        j += len(ys[j:])

    return zs

def sql_debug_ops_clause(debug_ops, event_alias, indents=None):
    if debug_ops:
        txt = "TRUE"
    else:
        # --debug-ops is not set.
        # DON'T show debug events.
        txt = "( NOT {e}.is_debug_event )".format(
            e=event_alias)

    txt = maybe_indent(txt, indents)
    return txt

def sql_event_range_clause(event_alias, start_time_us, end_time_us, indents=None):
    if start_time_us is None and end_time_us is None:
        txt = "TRUE"
    elif start_time_us is not None and end_time_us is not None:
        # txt = "( {e}.start_time_us <= {start} AND {end} <= {e}.end_time_us )".format(
        txt = "( {start} <= {e}.start_time_us AND {e}.end_time_us <= {end} )".format(
            e=event_alias,
            start=start_time_us,
            end=end_time_us)
    elif start_time_us is not None:
        # txt = "( {e}.start_time_us <= {start} )".format(
        txt = "( {start} <= {e}.start_time_us )".format(
            e=event_alias,
            start=start_time_us)
    else:
        assert end_time_us is not None
        # txt = "( {end} <= {e}.end_time_us )".format(
        txt = "( {e}.end_time_us <= {end} )".format(
            e=event_alias,
            end=end_time_us)

    txt = maybe_indent(txt, indents)
    return txt

def sql_event_split_range_clause(event_alias, start_time_us, end_time_us, indents=None):
    """
    4 Different cases when gather event split:

        NOTE: "tautologies" that we can remove from "Condition" to form "Simplified":
        (a) e.start <= e.end
        (b) event_split.start <= event_split.end

        1. Within event split:
            select all events e where

            Condition:
                event_split.start <= e.start <= e.end <= event_split.end

                Captures: 2, 3
                Doesn't capture: 1, 4, 5

        2. Straddle start of event split:
            select all events e where

            Condition:
                e.start <= event_split.start <= e.end <= event_split.end

                Captures: 1
                Doesn't capture: 2,3,4,5

        3. Straddle end of event split:
            select all events e where

            Condition:
                event_split.start <= e.start <= event_split.end <= e.end

                Captures: 4
                Doesn't capture: 1,2,3,5

        4. Subsumes event split:
            select all events e where

            Condition:
                e.start <= event_split.start <= event_split.end <= e.end

                Captures: 5
                Doesn't capture: 1,2,3,4

    """
    assert ( start_time_us is None and end_time_us is None ) or \
           ( start_time_us is not None and end_time_us is not None )

    if start_time_us is None and end_time_us is None:
        txt = "TRUE"
    else:
        event_split_start = start_time_us
        event_split_end = end_time_us
        assert event_split_start <= event_split_end
        e_start = "{e}.start_time_us".format(e=event_alias)
        e_end = "{e}.end_time_us".format(e=event_alias)

        tautology_pairs = [(e_start, e_end), (event_split_start, event_split_end)]
        def leq_clause(exprs):
            clause = sql_compose_inequality("<=", exprs,
                tautology_pairs=tautology_pairs,
                indents=1)
            return clause
        txt = textwrap.dedent("""\
        (
            -- When querying events spanning an "EventSplit(start={start}, end={end})", 
            -- there are 4 different ways an event could "belong" to a split.
            
            -- 1. Within event split:
            {clause_01} OR
            -- 2. Straddle start of event split:
            {clause_02} OR
            -- 3. Straddle end of event split:
            {clause_03} OR
            -- 4. Subsumes event split:
            {clause_04}
        )
        """.format(
            start=start_time_us, end=end_time_us,
            clause_01=leq_clause([event_split_start, e_start, e_end, event_split_end]),
            clause_02=leq_clause([e_start, event_split_start, e_end, event_split_end]),
            clause_03=leq_clause([event_split_start, e_start, event_split_end, e_end]),
            clause_04=leq_clause([e_start, event_split_start, event_split_end, e_end]),
        )).rstrip()

    txt = maybe_indent(txt, indents)
    return txt

def sql_process_clause(process_name, process_alias, indents=None, allow_none=False):
    return _sql_eq_clause(process_name, process_alias, 'process_name', indents, allow_none)

def sql_phase_clause(phase_name, phase_alias, indents=None, allow_none=False):
    return _sql_eq_clause(phase_name, phase_alias, 'phase_name', indents, allow_none)

def sql_machine_clause(machine_name, machine_alias, indents=None, allow_none=False):
    return _sql_eq_clause(machine_name, machine_alias, 'machine_name', indents, allow_none)

def sql_category_clause(category_name, category_alias, indents=None, allow_none=False):
    return _sql_eq_clause(category_name, category_alias, 'category_name', indents, allow_none)

def _sql_eq_clause(value, alias, field, indents=None, allow_none=False):
    def _as_value(value):
        if type(value) == str:
            return "'{value}'".format(
                value=value)
        return str(value)

    if value is None:
        if not allow_none:
            raise RuntimeError("Expected {alias}.{field} not to be None".format(
                alias=alias,
                field=field))
        txt = "TRUE"
    else:
        txt = "( {p}.{field} = {value} )".format(
            p=alias,
            field=field,
            value=_as_value(value))

    txt = maybe_indent(txt, indents)
    return txt

def sql_process_op_clause(event_alias, indents=None):
    """
    For each process, we insert a "process operation" event:
    Event(
        event_name="[ppo2_PongNoFrameskip-v4]",
        category_name="Operation",
    )
    We skip these during analysis.
    """
    txt = r"( {e}.event_name LIKE '\[%' )".format(
        e=event_alias,
    )

    txt = maybe_indent(txt, indents)
    return txt

def sql_ignore_clause(category_alias, ignore_categories=None, keep_categories=None, indents=None):
    if keep_categories is not None and ignore_categories is not None:
        raise RuntimeError("Can only provide keep_categories or ignore_categories, not both")

    if ( ignore_categories is None or len(ignore_categories) == 0 ) and \
        ( keep_categories is None or len(keep_categories) == 0 ):
        return "TRUE"

    assert ( ignore_categories is not None and len(ignore_categories) > 0 ) or \
           ( keep_categories is not None and len(keep_categories) > 0 )

    def _clause(category, ignore):
        if ignore:
            clause = "{c}.category_name != '{category}'".format(
                c=category_alias,
                category=category)
        else:
            clause = "{c}.category_name = '{category}'".format(
                c=category_alias,
                category=category)
        return clause

    def _clauses(categories, ignore):
        if ignore:
            conjunct = 'AND'
        else:
            conjunct = 'OR'
        clause = \
            "(\n" + \
            " {conjunct} \n".format(conjunct=conjunct).join([
                _clause(category, ignore) for category in categories
            ]) + \
            "\n)"
        clause = maybe_indent(clause, indents)
        return clause

    if ignore_categories is not None:
        clauses = _clauses(ignore_categories, ignore=True)
    else:
        clauses = _clauses(keep_categories, ignore=False)

    return clauses

def sql_overlap_clause(event_alias_1, event_alias_2,
                       indents=None,
                       overlap_type='any'):
    """
    Given two Event table aliases, provide a clause for determine
    if the events from those two tables overlap.
    :param overlap_type
        'any'
            Any overlap between op1 and op2, be it partial, or subsumes (in either direction)
        'subsumes'
            op1 subsumes op2
            Any overlap between op1 and op2, be it partial, or subsumes (in either direction)
        'partial'
            op1 partially overlaps op2.
            NOTE: partial overlap is symmetric
            i.e.
            "op1 partially overlaps op2" is the same as
            "op2 partially overlaps op1"
    :return:
    """

    # Q: How many ways can your overlap 2 events?
    # Partial overlap:
    #
    #   1. [ op1 ]
    #         [ op2 ]
    #      op1.start <= op2.start <= op1.end <= op2.end
    #
    #   2. [ op2 ]
    #         [ op1 ]
    #    op2.start <= op1.start <= op2.end <= op1.end
    #
    # Subsume:
    #
    #   op1 subsumes op2
    #   3. [     op1     ]
    #          [ op2 ]
    #      op2.start <= op1.start <= op1.end <= op2.end
    #
    #   op2 subsumes op1
    #   4. [     op2     ]
    #          [ op1 ]
    #    op1.start <= op2.start <= op2.end <= op1.end

    def _clause(op1, op2, otype):
        if otype == 'subsumes':
            # op1 subsumes op2
            #   3. [     op1     ]
            #          [ op2 ]
            clause = textwrap.dedent("""
                ( {op1}.start_time_us <= {op2}.start_time_us AND 
                                         {op2}.start_time_us <= {op2}.end_time_us AND 
                                                                {op2}.end_time_us <= {op1}.end_time_us )
                """.format(
                op1=op1,
                op2=op2))
        elif otype == 'partial':
            # op1 partially overlaps op2
            #   1. [ op1 ]
            #         [ op2 ]
            clause = textwrap.dedent("""
                ( {op1}.start_time_us <= {op2}.start_time_us AND 
                                         {op2}.start_time_us <= {op1}.end_time_us AND 
                                                                {op1}.end_time_us <= {op2}.end_time_us ) 
                """.format(
                op1=op1,
                op2=op2))
        else:
            raise NotImplementedError
        return clause

    e1 = event_alias_1
    e2 = event_alias_2
    if overlap_type == 'any':
        clauses = [
            _clause(e1, e2, 'partial'),
            _clause(e2, e1, 'partial'),
            _clause(e1, e2, 'subsumes'),
            _clause(e2, e1, 'subsumes'),
        ]
    elif overlap_type == 'partial':
        clauses = [
            _clause(e1, e2, 'partial'),
            _clause(e2, e1, 'partial'),
        ]
    elif overlap_type == 'subsumes':
        clauses = [
            _clause(e1, e2, 'subsumes'),
        ]
    else:
        raise NotImplementedError

    # Join clauses via OR, but make it legible (not all on 1 line, proper indent)
    clauses = " OR \n".join(clauses)
    clause = textwrap.dedent("""
    (
        {clauses}
    )
    """).format(
        clauses=maybe_indent(clauses, indents=1),
    )

    # clause = textwrap.dedent("""
    #     (
    #         ( {e1}.start_time_us <= {e2}.start_time_us AND
    #                                 {e2}.start_time_us <= {e1}.end_time_us AND
    #                                                       {e1}.end_time_us <= {e2}.end_time_us ) OR
    #         ( {e2}.start_time_us <= {e1}.start_time_us AND
    #                                 {e1}.start_time_us <= {e2}.end_time_us AND
    #                                                       {e2}.end_time_us <= {e1}.end_time_us ) OR
    #         ( {e2}.start_time_us <= {e1}.start_time_us AND
    #                                 {e1}.start_time_us <= {e1}.end_time_us AND
    #                                                       {e1}.end_time_us <= {e2}.end_time_us ) OR
    #         ( {e1}.start_time_us <= {e2}.start_time_us AND
    #                                 {e2}.start_time_us <= {e2}.end_time_us AND
    #                                                       {e2}.end_time_us <= {e1}.end_time_us )
    #     )
    #     """.format(
    #     e1=event_alias_1,
    #     e2=event_alias_2))

    # Add indent desired by caller
    clause = maybe_indent(clause, indents)

    return clause

def sql_get_source_files(klass, directory):
    """
    SQLite: We want traces.db
    Postgres: We want psql.json
    """
    src_files = []
    sql_input = sql_input_path(directory)
    if not _e(sql_input):
        raise MissingInputFiles(textwrap.dedent("""
            {klass}: Couldn't find SQL input file at {path}.
            """.format(
            klass=klass.__name__,
            path=sql_input,
        )))
    return src_files

def get_process_trace_metadata(path):
    """
    Return stuff we must insert BEFORE inserting individual events in bulk.
    That includes:
    - process_name's
    - phase_name's

    :param path:
    :return:
    """
    assert is_process_trace_file(path)

    # REALLY slow for this file:
    # /mnt/data/james/clone/dnn_tensorflow_cpp/checkpoints/minigo/vector_multiple_workers_k4000/process/loop_train_eval/phase/sgd_updates/profile.trace_2.session_1.proto
    # NOTE: we need to fix large file handling (more frequent dumps!).

    if is_pyprof_file(path) or is_dump_event_file(path):
        proto = read_pyprof_file(path)
        meta = {
            'process_name':proto.process_name,
            'phase_name':proto.phase,
            'machine_name':proto.machine_name,
        }
        return meta
    elif is_pyprof_call_times_file(path):
        call_times_data = read_pyprof_call_times_file(path)
        meta = {
            'process_name':call_times_data['process_name'],
            'phase_name':call_times_data['phase'],
            'machine_name':call_times_data['machine_name'],
        }
        return meta
    else:
        raise NotImplementedError("Not sure how to get process_name from trace-file {path}".format(
            path=path))

def get_util_metadata(path):
    """
    Return stuff we must insert BEFORE inserting individual events in bulk.
    That includes:
    - machine_name
    - device_name's
    """
    assert is_machine_util_file(path)

    proto = read_machine_util_file(path)
    device_names = list(proto.device_util.keys())
    meta = {
        'machine_name':proto.machine_name,
        'device_names':device_names,
    }
    return meta

class Device:
    def __init__(self, device_name, device_id,
                 machine_name, machine_id,
                 # Swallow any excess arguments
                 **kwargs):
        self.device_name = device_name
        self.device_id = device_id
        self.machine_name = machine_name
        self.machine_id = machine_id

    @staticmethod
    def from_row(row):
        return obj_from_row(Device, row)

    def __str__(self):
        return 'Device(name="{name}", id={id}, machine_name="{machine_name}")'.format(
            name=self.device_name,
            id=self.device_id,
            machine_name=self.machine_name,
            # machine_id=self.machine_id,
        )

    def __repr__(self):
        return str(self)

class Process:
    def __init__(self, process_name, process_id,
                 machine_name, machine_id,
                 percent_complete=None,
                 num_timesteps=None,
                 total_timesteps=None,
                 # Swallow any excess arguments
                 **kwargs):
        self.process_name = process_name
        self.process_id = process_id
        self.machine_name = machine_name
        self.machine_id = machine_id
        self.percent_complete = percent_complete
        self.num_timesteps = num_timesteps
        self.total_timesteps = total_timesteps

    @staticmethod
    def from_row(row):
        process = Process(**row)
        for attr, value in row.items():
            if not hasattr(process, attr):
                setattr(process, attr, value)
        return process

    def __str__(self):
        return 'Process(name="{name}", id={id}, machine_name="{machine_name}")'.format(
            name=self.process_name,
            id=self.process_id,
            machine_name=self.machine_name,
            # machine_id=self.machine_id,
        )

    def __repr__(self):
        return str(self)

class Machine:
    def __init__(self, machine_name, machine_id,
                 # Swallow any excess arguments
                 **kwargs):
        self.machine_name = machine_name
        self.machine_id = machine_id

    @staticmethod
    def from_row(row):
        machine = Machine(**row)
        for attr, value in row.items():
            if not hasattr(machine, attr):
                setattr(machine, attr, value)
        return machine

    def __str__(self):
        return 'Machine(name="{name}", id={id})'.format(
            name=self.machine_name,
            id=self.machine_id,
        )

    def __repr__(self):
        return str(self)

class EventSplit:
    def __init__(self, start_time_us, end_time_us):
        self.start_time_us = start_time_us
        self.end_time_us = end_time_us

    @property
    def duration_us(self):
        return self.end_time_us - self.start_time_us

    def __str__(self):
        return "EventSplit(start_us={start_us} us, duration_us={duration_us})".format(
            start_us=self.start_time_us,
            duration_us=self.duration_us,
        )

    def __repr__(self):
        return str(self)

def is_cpu_device(device_name):
    return re.search(r'Intel|AMD', device_name)

def is_gpu_device(device_name):
    return not is_cpu_device(device_name)

def usec_string(usec):
    return "{usec} us".format(usec=usec)

def sql_value_string(value):
    if type(value) == str:
        return "'{value}'".format(value=value)
    return str(value)


def EventsAsEOTimes(events):
    """
    :param events:
        KernelTime's sorted by ktime.start_time_usec
    :return:
    """
    TimeType = None
    psec_in_usec = None
    category_eo_times = np.empty(2*len(events), dtype=py_config.NUMPY_TIME_USEC_TYPE)
    for i, ktime in enumerate(events):
        if psec_in_usec is None:
            TimeType = type(ktime.start_time_usec)
            psec_in_usec = TimeType(constants.PSEC_IN_USEC)
        # Convert Decimal(usec) to int64(picosecond);
        # Should keep enough precision for accurate results, while still allow int64.
        # Picosecond decimals come from:
        # - Overhead events, whose duration is computed using an average.
        category_eo_times[i*2] = int(ktime.start_time_usec * psec_in_usec)
        category_eo_times[i*2 + 1] = int(ktime.end_time_usec * psec_in_usec)
    return category_eo_times

def RowsAsEOTimes(rows, timer):
    """
    :param events:
        KernelTime's sorted by ktime.start_time_usec
    :return:
    """
    TimeType = None
    psec_in_usec = None
    if isinstance(rows, RowIterator):
        count = len(rows)
        row_iter = rows.each_row(timer=timer)
    else:
        count = len(rows)
        row_iter = rows
    category_eo_times = np.empty(2*count, dtype=py_config.NUMPY_TIME_USEC_TYPE)
    for i, row in enumerate(row_iter):
        if psec_in_usec is None:
            TimeType = type(row['start_time_us'])
            psec_in_usec = TimeType(constants.PSEC_IN_USEC)
        # Convert Decimal(usec) to int64(picosecond);
        # Should keep enough precision for accurate results, while still allow int64.
        # Picosecond decimals come from:
        # - Overhead events, whose duration is computed using an average.
        category_eo_times[i*2] = int(row['start_time_us'] * psec_in_usec)
        end_time_usec = row['start_time_us'] + row['duration_us']
        category_eo_times[i*2 + 1] = int(end_time_usec * psec_in_usec)
    return category_eo_times

def AsNumbaEOTimes(
    eo_times_dict,
    # category_times,
    category_to_idx, idx_to_category):
    """

    category_times = {
        'A': [(t1, t2), (t3, t4), ...],
        'B': [(t5, t6), (t7, t8), ...],
        ...
    }

    category_to_idx = {
        'A': 0,
        'B': 1,
    }

    =>

    eo_times = [
        # 'A'
        [t1, t2, t3, t4, ...],
        # 'B'
        [t5, t6, t7, t8, ...],
        ...
    ]

    :param category_times:
    :param category_to_idx:
    :param idx_to_category:
    :return:
    """

    # if debug:
    #     logger.info("converting to NumbaEvent's: {msg}".format(
    #         msg=pprint_msg({
    #             'category_times': category_times,
    #         }),
    #     ))

    eo_times = []
    for idx in sorted(idx_to_category.keys()):
        category_key = idx_to_category[idx]
        category_eo_times = eo_times_dict[category_key]
        # times_by_start = category_times[category_key]
        # category_eo_times = EventsAsEOTimes(times_by_start)
        eo_times.append(category_eo_times)
    return eo_times

def category_to_idx_maps(categories):
    categories_order = sorted(categories)
    category_to_idx = dict()
    idx_to_category = dict()
    for i, category in enumerate(categories_order):
        category_to_idx[category] = i
        idx_to_category[i] = category
    return category_to_idx, idx_to_category

class RowIterator:
    def __init__(self, select_query, cursor, RowKlass=None, debug=False, fetchall=False):
        """
        :param select_query:
        :param cursor:
        :param RowKlass:
        :param debug:
        :param fetchall:
            If true, fetch rows into memory all at once.
            If false, stream rows using database cursor.
        """
        self.select_query = select_query
        self.cursor = cursor
        self.debug = debug
        self.RowKlass = RowKlass
        self.fetchall = fetchall
        self._rows = None
        self._count = None
        self._iterating_rows = False

    def _run_select(self, timer=None):
        sql_exec_query(self.cursor, self.select_query, klass=self.__class__, debug=self.debug)
        rows = sql_fetch_rows(self.cursor,
                              fetchall=self.fetchall, debug=self.debug)
        if timer is not None:
            if callable(timer):
                timer("SELECT")
            else:
                timer.end_operation("SELECT")

        # Should be a cursor, not a list of rows.
        if self.fetchall:
            # Fetch rows into memory
            assert type(rows) == list
            # assert self._count is None or len(rows) == self._count
        else:
            # Iterate over cursor.
            assert type(rows) != list
            assert rows == self.cursor

        return rows

    def _each_row_fetchall(self, timer=None):
        assert self.fetchall
        assert not self._iterating_rows

        if self._rows is not None:
            return self._rows

        # Make it so user can ask for length of query whenever (even inside loop)
        # NOTE: hopefully this isn't expensive...
        self._rows = self._run_select(timer=timer)
        self._count = len(self._rows)

        if self.RowKlass is None:
            return self._rows
        else:
            return self._each_row_fetchall_generator()
            # self._iterating_rows = True
            # for row in self._rows:
            #     obj = self.RowKlass.from_row(row)
            #     yield obj
            # self._iterating_rows = False

    def _each_row_fetchall_generator(self):
        self._iterating_rows = True
        for row in self._rows:
            obj = self.RowKlass.from_row(row)
            yield obj
        self._iterating_rows = False

    def _each_row_stream(self, timer=None):
        assert not self.fetchall
        assert not self._iterating_rows

        # Make it so user can ask for length of query whenever (even inside loop)
        # NOTE: hopefully this isn't expensive...
        self._maybe_fetch_count(timer=timer)

        self._iterating_rows = True
        rows = self._run_select(timer=timer)
        for row in rows:
            if self.RowKlass is not None:
                obj = self.RowKlass.from_row(row)
            else:
                obj = row
            yield obj
        self._iterating_rows = False

    def each_row(self, timer=None):
        if self.fetchall:
            return self._each_row_fetchall(timer=timer)
        return self._each_row_stream(timer=timer)

    def count(self, timer=None):
        return self._maybe_fetch_count(timer=timer)

    def _maybe_fetch_count(self, timer=None):
        if self.fetchall:
            return self._maybe_fetch_count_fetchall(timer=timer)
        return self._maybe_fetch_count_stream(timer=timer)

    def _maybe_fetch_count_fetchall(self, timer=None):
        if self._rows is not None:
            assert self._count is not None
            return self._count

        self._rows = self._run_select(timer=timer)
        self._count = len(self._rows)
        return self._count

    def _maybe_fetch_count_stream(self, timer=None):
        if self._count is not None:
            return self._count
        # We use the same cursor to iterate over rows as we do to get a count.
        # Make sure we don't try to get a count while iterating over rows.
        assert not self._iterating_rows
        self._count = sql_count_from(self.cursor, self.select_query)
        if timer is not None:
            timer("COUNT")
        return self._count

    def __len__(self):
        return self.count()

def test_merge_sorted():
    # xs:         5 6 7 8
    # ys: 1 2 3 4 5       9 10 11
    xs = [            5, 6, 7, 8,          ]
    ys = [1, 2, 3, 4, 5,          9, 10, 11]
    expect = [1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10, 11]
    actual = merge_sorted(xs, ys)

    assert actual == expect

from rlscope.test import test_util
# import sec, T, U
class TestProcessOpNest:
    """
    Test event overlap.
    """

    def T(self, start_sec, end_sec, name=None, **kwargs):
        return test_util.T(start_sec, end_sec, name, process_name="process", **kwargs)

    def U(self, start_usec, end_usec=None, name=None, time_usec=None, **kwargs):
        return test_util.U(start_usec, end_usec, name, time_usec, process_name="process", **kwargs)

    def test_01_1_stack(self):
        T = self.T
        U = self.U
        process_op_nest = process_op_nest_single_thread
        #           [   op3   ]
        #       [       op2       ]
        # [             op1             ]
        # 0     1   2         3   4     5

        #           [   op3   ]
        #       [o2]           [o2]
        # [ op1]                   [op1 ]
        # 0     1   2         3   4     5

        op_events = [
            T(0, 5, 'op1'),
            T(1, 4, 'op2'),
            T(2, 3, 'op3'),
        ]

        # Unfiltered:
        actual = process_op_nest(op_events)
        expect = [
            T(0, 1, 'op1'),
            T(1, 2, 'op2'),
            T(2, 3, 'op3'),
            T(3, 4, 'op2'),
            T(4, 5, 'op1'),
        ]
        assert actual == expect

        # Filter by op1
        actual = process_op_nest(op_events, 'op1')
        expect = [
            T(0, 1, 'op1'),
            T(4, 5, 'op1'),
        ]
        assert actual == expect

        # Filter by op2
        actual = process_op_nest(op_events, 'op2')
        expect = [
            T(1, 2, 'op2'),
            T(3, 4, 'op2'),
        ]
        assert actual == expect

        # Filter by op3
        actual = process_op_nest(op_events, 'op3')
        expect = [
            T(2, 3, 'op3'),
        ]
        assert actual == expect

    def test_02_2_stacks(self):
        T = self.T
        U = self.U
        process_op_nest = process_op_nest_single_thread
        op_events = [
            # Stack 1
            T(0, 5, 'op1'),
            T(1, 4, 'op2'),
            T(2, 3, 'op3'),
            # Stack 2
            T(5, 8, 'op4'),
            T(6, 7, 'op5'),
        ]

        # Unfiltered:
        actual = process_op_nest(op_events)
        expect = [
            # Stack 1
            T(0, 1, 'op1'),
            T(1, 2, 'op2'),
            T(2, 3, 'op3'),
            T(3, 4, 'op2'),
            T(4, 5, 'op1'),
            # Stack 2
            T(5, 6, 'op4'),
            T(6, 7, 'op5'),
            T(7, 8, 'op4'),
        ]
        assert actual == expect

    # # Invalid input test
    # def test_03_complete_overlap(self):
    #     import pytest
    #
    #     T = self.T
    #     U = self.U
    #     process_op_nest = process_op_nest_single_thread
    #     op_events = [
    #         T(0, 1, 'op1'),
    #         T(0, 1, 'op2'),
    #     ]
    #
    #     # Unfiltered:
    #     with pytest.raises(AssertionError):
    #         actual = process_op_nest(op_events)
    #     # expect = [
    #     #     T(0, 1, 'op2'),
    #     # ]
    #     # assert actual == expect

    def test_04_multiple_sub_events(self):
        T = self.T
        U = self.U
        process_op_nest = process_op_nest_single_thread
        op_events = [
            T(0, 5, 'op1'),
            T(1, 2, 'op2'),
            T(3, 4, 'op3'),
        ]

        # Unfiltered:
        actual = process_op_nest(op_events)
        expect = [
            T(0, 1, 'op1'),
            T(1, 2, 'op2'),
            T(2, 3, 'op1'),
            T(3, 4, 'op3'),
            T(4, 5, 'op1'),
        ]
        assert actual == expect

        # Filter by op1
        actual = process_op_nest(op_events, 'op1')
        expect = [
            T(0, 1, 'op1'),
            T(2, 3, 'op1'),
            T(4, 5, 'op1'),
        ]
        assert actual == expect

        # Filter by op2
        actual = process_op_nest(op_events, 'op2')
        expect = [
            T(1, 2, 'op2'),
        ]
        assert actual == expect

        # Filter by op3
        actual = process_op_nest(op_events, 'op3')
        expect = [
            T(3, 4, 'op3'),
        ]
        assert actual == expect

    def test_05_wild_data_01(self):
        """
        Saw this data happen 'in the wild' from a collected trace.
        For some reason, process_op_nest failed.

        {
            "args": {
                "name": "train_loop"
            },
            "cat": "Op",
            "dur": "1180757",
            "name": "train_loop",
            "ph": "X",
            "pid": 1,
            "tid": 0,
            "ts": "1551209039576428"
        },
        {
            "args": {
                "name": "q_forward"
            },
            "cat": "Op",
            "dur": "1178289",
            "name": "q_forward",
            "ph": "X",
            "pid": 1,
            "tid": 0,
            "ts": "1551209039576444"
        },

        :return:
        """
        T = self.T
        U = self.U
        process_op_nest = process_op_nest_single_thread
        from decimal import Decimal as dec

        # [       train_loop      ]
        # |     [ q_forward ]     |
        # |     |           |     |
        # |     qf_start    qf_end|
        # tl_start                tl_end

        tl_start = dec(1551209039576428)
        tl_dur = dec(1180757)
        tl_end = dec(1551209039576428) + tl_dur
        qf_start = dec(1551209039576444)
        qf_dur = dec(1178289)
        qf_end = dec(1551209039576444) + qf_dur

        train_loop_ev = U(tl_start, tl_end, name='train_loop')
        q_forward_ev = U(qf_start, qf_end, name='q_forward')
        op_events = [
            train_loop_ev,
            q_forward_ev
        ]

        # Unfiltered:
        actual = process_op_nest(op_events)
        expect = [
            U(tl_start, tl_start + ( qf_start - tl_start ), 'train_loop'),
            U(qf_start, qf_start + qf_dur, 'q_forward'),
            U(qf_start + qf_dur, tl_start + tl_dur, 'train_loop'),
        ]
        assert actual == expect

        # # Filter by op1
        # actual = process_op_nest(op_events, 'op1')
        # expect = [
        #     T(0, 1, 'op1'),
        #     T(2, 3, 'op1'),
        #     T(4, 5, 'op1'),
        # ]
        # assert actual == expect
        #
        # # Filter by op2
        # actual = process_op_nest(op_events, 'op2')
        # expect = [
        #     T(1, 2, 'op2'),
        # ]
        # assert actual == expect
        #
        # # Filter by op3
        # actual = process_op_nest(op_events, 'op3')
        # expect = [
        #     T(3, 4, 'op3'),
        # ]
        # assert actual == expect
