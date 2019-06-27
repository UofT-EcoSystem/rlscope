import logging
import psutil
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import subprocess
import progressbar
import sys
import tempfile
import getpass
import psycopg2
import psycopg2.extras
import random
import string
import itertools

import sqlite3

from os.path import join as _j, dirname as _d, exists as _e

from iml_profiler.protobuf.pyprof_pb2 import Pyprof, MachineUtilization

from iml_profiler import py_config

from iml_profiler.parser.trace_events import dump_category_times

from iml_profiler.parser.readers import TFProfCategoryTimesReader, \
    DEFAULT_group_by_device, \
    DEFAULT_ignore_categories, \
    DEFAULT_debug \

import contextlib

from iml_profiler.parser.common import *

from iml_profiler.parser.stats import category_times_add_time

from iml_profiler.parser.stats import KernelTime

SQLITE_TABLE_SQL = _j(py_config.ROOT, "sqlite", "tables.sql")
SQLITE_INDICES_SQL = _j(py_config.ROOT, "sqlite", "indices.sql")

PSQL_TABLE_SQL = _j(py_config.ROOT, "postgres", "tables.sql")
PSQL_INDICES_SQL = _j(py_config.ROOT, "postgres", "indices.sql")
PSQL_CONSTRAINTS_SQL = _j(py_config.ROOT, "postgres", "constraints.sql")

def Worker_get_device_names(kwargs):
    if kwargs['debug']:
        logging.info("> Start: Worker_get_device_names tfprof_file={path}".format(path=kwargs['tfprof_file']))
    reader = TFProfCategoryTimesReader(kwargs['tfprof_file'])
    device_names = reader.get_device_names()
    if kwargs['debug']:
        pprint.pprint({
            'tfprof.device_names':device_names,
            'tfprof_file':kwargs['tfprof_file']})
        logging.info("> Stop: Worker_get_device_names tfprof_file={path}".format(path=kwargs['tfprof_file']))

    return device_names

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
        Pyprof.trace_1.prof
        tfprof.trace_1.prof
    """

    def __init__(self, directory,
                 # Swallow any excess arguments
                 debug=False,
                 debug_single_thread=False,
                 **kwargs):
        self.directory = directory
        self.conn = sql_create_connection(self.db_path)
        self.debug = debug
        self.debug_single_thread = debug_single_thread
        self.block_size = 50000

    def get_source_files(self):
        src_files = []
        for dirpath, dirnames, filenames in os.walk(self.directory):
            for base in filenames:
                path = _j(dirpath, base)
                if is_insertable_file(path):
                    src_files.append(path)
        if len(src_files) == 0:
            raise MissingInputFiles(textwrap.dedent("""
            {klass}: Couldn't find any tfprof/pyprof files root at {dir}.
            """.format(
                klass=self.__class__.__name__,
                dir=self.directory,
            )))
        return src_files

    def _remove_buggy_data(self):
        """
        TODO: BUG: sometimes, this results in a negative duration_us for Event.duration_us for Python event's.
        HACK: just filter out these rare events when building SQL database.

        For details; see CFuncWrapper.__call__
        """
        logging.info("> BUG: remove Python Events with negative Event.duration_us")
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
        """.format(CATEGORY_OPERATION=CATEGORY_OPERATION)
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
        """.format(CATEGORY_OPERATION=CATEGORY_OPERATION)
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

        src_files = self.get_source_files()
        pprint.pprint({
            'rule':self.__class__.__name__,
            'src_files':src_files,
        })

        if self.debug:
            logging.info("> Read metadata.")

        process_trace_metas = []
        for path in src_files:
            if not is_process_trace_file(path):
                continue
            if self.debug:
                logging.info("> get_process_trace_metadata path={path}".format(path=path))
            md = get_process_trace_metadata(path)
            process_trace_metas.append(md)

        if self.debug:
            logging.info("> Insert processes.")

        process_names = sorted(meta['process_name'] for meta in process_trace_metas)
        for process_name in process_names:
            self.insert_process_name(process_name)

        if self.debug:
            logging.info("> Insert phases.")

        phase_names = sorted(meta['phase_name'] for meta in process_trace_metas)
        for phase_name in phase_names:
            self.insert_phase_name(phase_name)

        if self.debug:
            logging.info("> Read util metadata.")

        # Read metadata from CPU/GPU utilization files
        # e.g. machine_util.trace_0.proto
        util_metas = []
        for path in src_files:
            if not is_machine_util_file(path):
                continue
            if self.debug:
                logging.info("> get_util_metadata path={path}".format(path=path))
            md = get_util_metadata(path)
            util_metas.append(md)
        util_machine_names = set()
        util_device_names = set()
        for util_meta in util_metas:
            util_machine_names.add(util_meta['machine_name'])
        for util_meta in util_metas:
            util_device_names.update(util_meta['device_names'])

        if self.debug:
            logging.info("> Insert util machine names.")
        for machine_name in util_machine_names:
            self.insert_machine_name(machine_name)

        if self.debug:
            logging.info("> Insert util devices.")
        for device_name in util_device_names:
            self.insert_device_name(device_name)

        if self.debug:
            logging.info("> Insert categories.")
        categories = sorted(set(CATEGORIES_ALL))
        for category in categories:
            self.insert_category_name(category)

        if self.debug:
            logging.info("> Insert tfprof device names.")

        device_names = set()

        def get_Worker_get_device_names_kwargs(tfprof_file):
            return {'tfprof_file':tfprof_file, 'debug':self.debug}

        device_names_kwargs = [get_Worker_get_device_names_kwargs(tfprof_file)
                               for tfprof_file in src_files if is_tfprof_file(tfprof_file)]

        if not self.debug_single_thread:
            device_name_pool = multiprocessing.Pool()
            imap_iter = device_name_pool.imap_unordered(Worker_get_device_names, device_names_kwargs)
        else:
            imap_iter = self.single_thread_iter(Worker_get_device_names, device_names_kwargs)

        for names in tqdm_progress(imap_iter, desc='Device names', total=len(device_names_kwargs)):
            device_names.update(names)
            # if is_tfprof_file(path):
            #     reader = TFProfCategoryTimesReader(path)
            #     device_names.update(reader.get_device_names())
        pprint.pprint({'tfprof.device_names':device_names})

        if not self.debug_single_thread:
            device_name_pool.close()
            device_name_pool.join()

        for device_name in device_names:
            self.insert_device_name(device_name)

        if self.debug:
            logging.info("> Commit.")
        self.conn.commit()


        # name_to_id_obj = NameToID(self.db_path, debug=self.debug)
        # name_to_id_obj.run()
        #
        # self.process_to_id = name_to_id_obj.process_to_id
        # self.category_to_id = name_to_id_obj.category_to_id
        # self.device_to_id = name_to_id_obj.device_to_id
        # self.machine_to_id = name_to_id_obj.machine_to_id

        pprint.pprint({
            'process_to_id':self.process_to_id,
            'category_to_id':self.category_to_id,
            'device_to_id':self.device_to_id,
            'machine_to_id':self.machine_to_id,
        })
        # sys.exit(0)

        if not self.debug_single_thread:
            pool = multiprocessing.Pool()
        # table = 'Event'
        # id_field = 'event_id'
        id_field = None
        # worker = CSVInserterWorker(
        #     self.db_path, table, self.block_size, id_field, self.directory,
        #     debug=self.debug,
        # )

        if self.debug:
            logging.info("> Insert table files.")

        def get_worker_kwargs(path):
            if is_machine_util_file(path):
                table = 'DeviceUtilization'
            else:
                table = 'Event'
            return {
                'path':path,
                'db_path':self.db_path,
                'table':table,
                'block_size':self.block_size,
                'id_field':id_field,
                'directory':self.directory,
                'debug':self.debug,
            }
        worker_kwargs = [get_worker_kwargs(path) for path in src_files]

        if not self.debug_single_thread:
            if self.debug:
                logging.info("> Insert table files using thread pool.")
            imap_iter = pool.imap_unordered(CSVInserterWorker, worker_kwargs)
        else:
            if self.debug:
                logging.info("> Insert table files using single thread.")
            imap_iter = self.single_thread_iter(CSVInserterWorker, worker_kwargs)

        with progressbar.ProgressBar(max_value=len(src_files), prefix="SQL insert") as bar:
            for i, result in enumerate(imap_iter):
                # logging.info("> i={i}, result={result}".format(
                #     i=i, result=result))
                bar.update(i)

        if not self.debug_single_thread:
            pool.close()
            pool.join()

        # for path in src_files:
        #     if is_tfprof_file(path):
        #         self.insert_tfprof_file(path)
        #     elif is_pyprof_file(path) or is_dump_event_file(path):
        #         self.insert_pyprof_file(path)
        #     else:
        #         raise NotImplementedError

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
        logging.info("> Check event data...")
        start_t = time.time()
        self._check_no_partial_or_complete_op_overlap()
        end_t = time.time()
        time_sec = end_t - start_t
        logging.info("  Took {sec} seconds".format(sec=time_sec))

    def _HACK_clean_data(self):
        self._remove_buggy_data()

    def maybe_commit(self, i):
        if (i + 1) % self.block_size == 0:
            self.conn.commit()

    # def insert_tfprof_file(self, path):
    #
    #     reader = TFProfCategoryTimesReader(path)
    #
    #     logging.info("> Insert tfprof file: {p}".format(p=path))
    #     if self.debug:
    #         reader.logging.info(sys.stdout)
    #
    #     process_id = self.insert_process_name(reader.process_name)
    #     phase_id = self.insert_phase_name(reader.phase_name)
    #
    #     inserts = []
    #     self._total_inserts = 0
    #
    #     fields = ['start_time_us',
    #               'end_time_us',
    #               'duration_us',
    #               'event_name',
    #               'category_id',
    #               'process_id',
    #               'phase_id',
    #               'device_id',
    #               'is_debug_event']
    #
    #
    #     with progressbar.ProgressBar(max_value=reader.num_all_events()) as bar, \
    #         bulk_inserter(self.conn, 'Event', self.block_size, bar, directory=self.directory,
    #                       fields=fields) as bulk:
    #
    #         for i, (device, event) in enumerate(reader.all_events(debug=True)):
    #             category, start_time_us, duration_us, name = event
    #             category_id = self.insert_category_name(category)
    #             if category == 'GPU' and self.debug:
    #                 logging.info("> category = {c}, duration_us = {duration_us}".format(
    #                     c=category,
    #                     duration_us=duration_us))
    #             device_id = self.insert_device_name(device)
    #             end_time_us = start_time_us + duration_us
    #             is_debug_event = bool(match_debug_event_name(name))
    #             # insert = {
    #             #     # 'thread_id':event.thread_id,
    #             #     'start_time_us':start_time_us,
    #             #     'end_time_us':end_time_us,
    #             #     'duration_us':duration_us,
    #             #     'event_name':name,
    #             #     'category_id':category_id,
    #             #     'process_id':process_id,
    #             #     'device_id':device_id,
    #             #     'is_debug_event':is_debug_event,
    #             # }
    #             # bulk.add_insert(insert)
    #
    #             insert = [
    #                 # 'thread_id':event.thread_id,
    #
    #                 # 'start_time_us'
    #                 start_time_us,
    #                 # 'end_time_us'
    #                 end_time_us,
    #                 # 'duration_us'
    #                 duration_us,
    #                 # 'event_name'
    #                 name,
    #                 # 'category_id'
    #                 category_id,
    #                 # 'process_id'
    #                 process_id,
    #                 # 'phase_id'
    #                 phase_id,
    #                 # 'device_id'
    #                 device_id,
    #                 # 'is_debug_event'
    #                 is_debug_event,
    #             ]
    #             bulk.add_insert(insert)

    def insert_process_name(self, process_name):
        return self._insert_name(
            'Process',
            'process_id', 'process_name',
            self.process_to_id,
            process_name)

    def insert_phase_name(self, phase_name):
        return self._insert_name(
            'Phase',
            'phase_id', 'phase_name',
            self.phase_to_id,
            phase_name)

    def insert_device_name(self, device_name):
        return self._insert_name(
            'Device',
            'device_id', 'device_name',
            self.device_to_id,
            device_name)

    def insert_machine_name(self, machine_name):
        return self._insert_name(
            'Machine',
            'machine_id', 'machine_name',
            self.machine_to_id,
            machine_name)

    def insert_category_name(self, category_name):
        return self._insert_name(
            'Category',
            'category_id', 'category_name',
            self.category_to_id,
            category_name)

    @property
    def cursor(self):
        return self.conn.cursor

    def _insert_name(self, table, id_field, name_field, name_to_id, name):
        if name in name_to_id:
            return name_to_id[name]
        c = self.cursor
        c.execute("""
        SELECT {id_field} from {table} WHERE {name_field} = {p}
        """.format(
            id_field=id_field,
            table=table,
            name_field=name_field,
            p=sql_placeholder(),
        ), (name,))
        rows = c.fetchall()
        if len(rows) == 0:
            self.conn.insert_dict(table, {
                name_field: name,
            })
            ident = c.lastrowid
        else:
            ident = rows[0][id_field]

        name_to_id[name] = ident

        return ident

    # def insert_pyprof_file(self, path):
    #     with open(path, 'rb') as f:
    #         proto = Pyprof()
    #         proto.ParseFromString(f.read())
    #
    #     logging.info("> Insert pyprof file: {p}".format(p=path))
    #     if self.debug:
    #         logging.info(proto)
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
    #             yield CATEGORY_PYTHON, python_events.events
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
        logging.info("> Create indices...")
        start_t = time.time()
        self.conn.create_indices()
        end_t = time.time()
        time_sec = end_t - start_t
        logging.info("  Took {sec} seconds".format(sec=time_sec))

    def create_constraints(self):
        logging.info("> Create constraints...")
        start_t = time.time()
        self.conn.create_constraints()
        end_t = time.time()
        time_sec = end_t - start_t
        logging.info("  Took {sec} seconds".format(sec=time_sec))

    @property
    def db_path(self):
        return sql_input_path(self.directory)

@contextlib.contextmanager
def transaction(conn):
    # print_stacktrace('> IML: BEGIN TRANSACTION')
    conn.cursor.execute('BEGIN TRANSACTION')
    try:
        yield
    except Exception as e:
        raise
    conn.cursor.execute('COMMIT')

# @contextlib.contextmanager
# # def bulk_inserter(conn, table, block_size=50000, progress_bar=None, id_field=None):
# def bulk_inserter(*args, **kwargs):
#     try:
#         # bulk = BulkInserter(*args, **kwargs)
#         bulk = CSVInserter(*args, **kwargs)
#         # bulk = BulkInserter(conn, table, block_size, progress_bar, id_field)
#         yield bulk
#     except Exception as e:
#         raise
#     bulk.finish()
#     # conn.commit()

# @contextlib.contextmanager
# def fast_inserts(conn):
#     try:
#         conn.disable_fast_inserts()
#         yield
#     except Exception as e:
#         raise
#     conn.disable_fast_inserts()

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
            # logging.info("> IDENT is None, use 0")
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
            # logging.info("> IDENT is None, use 0")
            ident = 0
        return ident

def CSVInserterWorker(kwargs):
    worker = _CSVInserterWorker(**kwargs)
    worker.run()

class NameToID:

    def __init__(self, db_path,
                 debug=False,
                 ):
        self.db_path = db_path
        self.debug = debug

    def run(self):
        self.conn = sql_create_connection(self.db_path)

        self.process_to_id = self.build_name_to_id('Process', 'process_id', 'process_name')
        self.phase_to_id = self.build_name_to_id('Phase', 'phase_id', 'phase_name')
        self.category_to_id = self.build_name_to_id('Category', 'category_id', 'category_name')
        self.device_to_id = self.build_name_to_id('Device', 'device_id', 'device_name')
        self.machine_to_id = self.build_name_to_id('Machine', 'machine_id', 'machine_name')

        # self.insert_file(self.path)
        # self.finish()

    @property
    def cursor(self):
        return self.conn.cursor

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

class _CSVInserterWorker:

    # def __init__(self, path, db_path, table, block_size=50000, id_field=None, directory=None,
    #              # fields=None,
    #              debug=False,
    #              ):
    def __init__(self, path, db_path, table, block_size=50000, id_field=None, directory=None,
                 # fields=None,
                 debug=False,
                 csv_path=None,
                 ):
        self.path = path
        self.db_path = db_path
        self.table = table
        self.block_size = block_size
        # self.progress_bar = progress_bar
        self.id_field = id_field
        self.directory = directory
        # self.fields = fields
        self.debug = debug

        if directory is None:
            self.directory = os.getcwd()
        else:
            self.directory = directory

        self.progress_bar = None

        self.csv_path = csv_path

    def _init(self):
        """
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
            # self.tmp_f = tempfile.TemporaryFile(mode='w', dir=self.directory, prefix=table, suffix=".csv")
            self.tmp_f, self.tmp_path = tempfile.mkstemp(
                dir=self.directory,
                prefix="{table}_".format(
                    table=self.table,
                ),
                # prefix="{table}{suffix_str}_".format(
                #     table=table,
                #     suffix_str="_{suffix}".format(suffix=self.suffix) if self.suffix is not None else ''),
                suffix=".csv",
                text=True)
            os.chmod(self.tmp_path, 0o664)
            self.tmp_f = os.fdopen(self.tmp_f, mode='w')

    # def __call__(self, path):
    #     self.run(path)


    def run(self):
        self._init()

        self.conn = sql_create_connection(self.db_path)

        self.process_to_id = self.build_name_to_id('Process', 'process_id', 'process_name')
        self.phase_to_id = self.build_name_to_id('Phase', 'phase_id', 'phase_name')
        self.category_to_id = self.build_name_to_id('Category', 'category_id', 'category_name')
        self.device_to_id = self.build_name_to_id('Device', 'device_id', 'device_name')
        self.machine_to_id = self.build_name_to_id('Machine', 'machine_id', 'machine_name')

        self.insert_file(self.path)
        self.finish()

    @property
    def cursor(self):
        return self.conn.cursor

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

    def insert_file(self, path):
        if is_tfprof_file(path):
            self.insert_tfprof_file(path)
        elif is_pyprof_file(path) or is_dump_event_file(path):
            self.insert_pyprof_file(path)
        elif is_pyprof_call_times_file(path):
            self.insert_pyprof_call_times_file(path)
        elif is_machine_util_file(path):
            self.insert_machine_util_file(path)
        else:
            raise NotImplementedError("Not sure how to insert into path={path} into database".format(path=path))

    def insert_tfprof_file(self, path):

        reader = TFProfCategoryTimesReader(path)

        logging.info("> Insert tfprof file: {p}".format(p=path))
        # if self.debug:
        #     reader.logging.info(sys.stdout)

        # process_id = self.insert_process_name(reader.process_name)
        process_id = self.process_to_id[reader.process_name]
        phase_id = self.phase_to_id[reader.phase_name]

        # inserts = []
        self._total_inserts = 0

        # with progressbar.ProgressBar(max_value=reader.num_all_events()) as bar, \
        #     bulk_inserter(conn, 'Event', block_size, bar, directory=directory,
        #                   fields=fields) as bulk:

        for i, (device, event) in enumerate(reader.all_events(debug=True)):

            category, start_time_us, duration_us, name = event
            # category_id = self.insert_category_name(category)
            category_id = self.category_to_id[category]
            # if category == 'GPU' and self.debug:
            #     logging.info("> category = {c}, duration_us = {duration_us}".format(
            #         c=category,
            #         duration_us=duration_us))
            # device_id = self.insert_device_name(device)
            if device not in self.device_to_id:
                logging.info("> ERROR: Couldn't find device={dev} in path={path}".format(
                    dev=device,
                    path=path))
                pprint.pprint({
                    'device_to_id':self.device_to_id})
            device_id = self.device_to_id[device]
            end_time_us = start_time_us + duration_us
            is_debug_event = bool(match_debug_event_name(name))
            insert = {
                # 'thread_id':event.thread_id,
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
            self.add_insert(insert)

    def _pyprof_each_category_events(self, pyprof_proto):
        for step, python_events in pyprof_proto.python_events.items():
            yield CATEGORY_PYTHON, python_events.events

        for step, clibs in pyprof_proto.clibs.items():
            for category, clib_events in clibs.clibs.items():
                yield category, clib_events.events

    def insert_pyprof_call_times_file(self, path):
        call_times_data = read_pyprof_call_times_file(path)

        logging.info("> Insert pyprof call_times file: {p}".format(p=path))
        # if self.debug:
        #     pprint.pprint({'call_times_data':call_times_data})

        process_name = call_times_data['process_name']
        phase_name = call_times_data['phase']

        c = self.cursor
        # Insert Process
        # process_id = self.insert_process_name(proto.process_name)
        process_id = self.process_to_id[process_name]
        phase_id = self.phase_to_id[phase_name]

        num_all_events = sum(len(epoch_duration_sec_pairs) \
                             for func_tupl, epoch_duration_sec_pairs in call_times_data['call_times'].items())

        with progressbar.ProgressBar(max_value=num_all_events) as bar:
            category = CATEGORY_PYTHON_PROFILER
            category_id = self.category_to_id[category]
            i = 0
            for func_tupl, epoch_duration_sec_pairs in call_times_data['call_times'].items():
                pyprof_filename, pyprof_line_no, pyprof_function = func_tupl
                for start_epoch_sec, duration_sec in epoch_duration_sec_pairs:
                    # Q: Should we define a separate table for python profiling data to avoid cramming file/line/func-name data into a single field?
                    # Options:
                    # - Use EventAttribute's
                    #   - Annoying/slow: extra joins (NOTE: joins may not be a big deal)
                    # - Add (usually) NULL columns for pyprof-specific fields <-- this is probably the best approach.
                    #   - Do this; assume # of Event's is LARGE ( which is definitely is ), so lets avoid joins.
                    # - Add Pyprof table with foreign-key reference from Event to Pyprof metadata row (file/line/func-name)
                    #   - Makes event inserts slower...I don't bother handling foreign key management when doing bulk inserts.
                    #   - extra joins (NOTE: joins may not be a big deal)
                    pyprof_line_description = pyprof_func_std_string(func_tupl)
                    # Insert Event
                    is_debug_event = False
                    start_time_us = sec_to_us(start_epoch_sec)
                    duration_us = sec_to_us(duration_sec)
                    end_time_us = start_time_us + duration_us
                    ins = {
                        # 'thread_id':event.thread_id,
                        'start_time_us':start_time_us,
                        'end_time_us':end_time_us,
                        'duration_us':duration_us,
                        # Q: function name alone can be vague without knowing the filename.
                        'event_name':pyprof_function,
                        'category_id':category_id,
                        'process_id':process_id,
                        'phase_id':phase_id,
                        'is_debug_event':is_debug_event,
                        'pyprof_filename':pyprof_filename,
                        'pyprof_line_no':pyprof_line_no,
                        'pyprof_function':pyprof_function,
                        'pyprof_line_description':pyprof_line_description,
                    }
                    # import ipdb; ipdb.set_trace()

                    # ipdb> pp ins
                    # {'category_id': 8,
                    #  'duration_us': 10.735000000000001,
                    #  'end_time_us': 1550795540176821.2,
                    #  'event_name': 'set_operation',
                    #  'is_debug_event': False,
                    #  'process_id': 1,
                    #  'pyprof_filename': '/mnt/data/james/clone/dnn_tensorflow_cpp/python/profiler/profilers.py',
                    #  'pyprof_function': 'set_operation',
                    #  'pyprof_line_description': '/mnt/data/james/clone/dnn_tensorflow_cpp/python/profiler/profilers.py:788(set_operation)',
                    #  'pyprof_line_no': 788,
                    #  'start_time_us': 1550795540176810.5}

                    event_id = self.insert_dict('Event', ins)
                    i += 1
                    bar.update(i)

        self.conn.commit()

    def insert_pyprof_file(self, path):
        with open(path, 'rb') as f:
            proto = Pyprof()
            proto.ParseFromString(f.read())

        logging.info("> Insert pyprof file: {p}".format(p=path))
        # if self.debug:
        #     logging.info(proto)

        c = self.cursor
        # Insert Process
        # process_id = self.insert_process_name(proto.process_name)
        process_id = self.process_to_id[proto.process_name]
        phase_id = self.phase_to_id[proto.phase]

        # categories = set()
        def insert_category_events(event_conn, eventattr_conn, category, events):
            # Insert Category
            # categories.add(category)
            # category_id = self.insert_category_name(category)
            category_id = self.category_to_id[category]
            for event in events:
                # Insert Event
                is_debug_event = bool(match_debug_event_name(event.name))
                event_id = event_conn.insert_dict('Event', {
                    'thread_id':event.thread_id,
                    'start_time_us':event.start_time_us,
                    'end_time_us':event.start_time_us + event.duration_us,
                    'duration_us':event.duration_us,
                    'event_name':event.name,
                    'category_id':category_id,
                    'process_id':process_id,
                    'phase_id':phase_id,
                    'is_debug_event':is_debug_event,
                })
                # Insert EventAttr
                # for attr_name, attr_value in event.attrs.items():
                #     attr_id = eventattr_conn.insert_dict('EventAttr', {
                #         'event_id':event_id,
                #         'attr_name':attr_name,
                #         'attr_value':attr_value,
                #     })

        num_all_events = sum(len(events) for category, events in self._pyprof_each_category_events(proto))

        with progressbar.ProgressBar(max_value=num_all_events) as bar:
            for category, events in self._pyprof_each_category_events(proto):
                insert_category_events(self, self, category, events)

        self.conn.commit()

    def insert_machine_util_file(self, path):
        proto = read_machine_util_file(path)

        logging.info("> Insert machine CPU/GPU utilization file: {p}".format(p=path))
        # if self.debug:
        #     logging.info(proto)

        machine_id = self.machine_to_id[proto.machine_name]

        def each_sample(machine_util):
            for device_name, device_util in machine_util.device_util.items():
                for sample in device_util.samples:
                    device_id = self.device_to_id[device_name]
                    yield device_id, sample.start_time_us, sample.util

        def count_samples(machine_util):
            n = 0
            for device_name, device_util in machine_util.device_util.items():
                n += len(device_util.samples)
            return n

        num_all_samples = count_samples(proto)

        with progressbar.ProgressBar(max_value=num_all_samples) as bar:
            for i, (device_id, start_time_us, util) in enumerate(each_sample(proto)):
                ins = {
                    'device_id':device_id,
                    'machine_id':machine_id,
                    'start_time_us':start_time_us,
                    'util':util,
                }
                device_util_id = self.insert_dict('DeviceUtilization', ins)
                bar.update(i)

        self.conn.commit()

    def insert_dict(self, table, insert):
        assert self.table == table
        return self.add_insert(insert)

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
            self.conn.insert_csv(self.tmp_path, self.table)
            end_t = time.time()
            # logging.info("> Loading CSV into {table} took {sec} seconds".format(
            #     table=self.table,
            #     sec=end_t - start_t))

        self.tmp_f.close()
        os.remove(self.tmp_path)
        self.conn.close()

class CSVInserter:
    def __init__(self, conn, table, block_size=50000, progress_bar=None, id_field=None, directory=None, fields=None,
                 # suffix=None,
                 csv_path=None,
                 autoincrement=None,
                 ):
        """
        Use Postgres' "COPY FROM" command to insert a csv file into the database.
        This leads to the best-possible insertion time for data.

        https://www.postgresql.org/docs/9.2/sql-copy.html

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
        # self.inserts = []
        self.fields = fields
        self.block_size = block_size
        self.total_inserts = 0
        self.progress_bar = progress_bar
        self.id_field = id_field
        self.next_id = None
        self.header_written = False

        if autoincrement is None and id_field is not None:
            self.autoincrement = AutoincrementID(conn, table, id_field)
        else:
            self.autoincrement = autoincrement
        # self.suffix = suffix

        if directory is None:
            self.directory = os.getcwd()
        else:
            self.directory = directory

        if csv_path is not None:
            self.tmp_path = csv_path
            self.tmp_f = open(self.tmp_path, 'w')
        else:
            # self.tmp_f = tempfile.TemporaryFile(mode='w', dir=self.directory, prefix=table, suffix=".csv")
            self.tmp_f, self.tmp_path = tempfile.mkstemp(
                dir=self.directory,
                prefix="{table}_".format(
                    table=table,
                ),
                # prefix="{table}{suffix_str}_".format(
                #     table=table,
                #     suffix_str="_{suffix}".format(suffix=self.suffix) if self.suffix is not None else ''),
                suffix=".csv",
                text=True)
            os.chmod(self.tmp_path, 0o664)
            self.tmp_f = os.fdopen(self.tmp_f, mode='w')

        # if self.id_field is not None:
        #     self.next_id = self._last_insert_id() + 1

    # Match TracesSQLiteConnection interface
    def insert_dict(self, table, insert):
        assert self.table == table
        return self.add_insert(insert)

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

    def _write_csv_header(self):
        line = ','.join(self.fields)
        self.tmp_f.write(line)
        self.tmp_f.write("\n")

    def add_insert(self, insert):
        ret = None

        if self.id_field is not None:
            next_id = self.autoincrement.next_id()
            if type(insert) == dict:
                insert[self.id_field] = next_id
            else:
                insert.append(next_id)
            ret = next_id
            # self.next_id += 1

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

        start_t = time.time()
        self.conn.insert_csv(self.tmp_path, self.table)
        end_t = time.time()
        logging.info("> Loading CSV into {table} took {sec} seconds".format(
            table=self.table,
            sec=end_t - start_t))

        self.tmp_f.close()
        os.remove(self.tmp_path)

class TracesPostgresConnection:
    def __init__(self, db_config_path, db_basename='traces', host='localhost'):
        self.host = host
        self.rand_suffix_len = 4
        self.db_config_path = db_config_path
        self.db_config = None
        self.db_basename = db_basename
        self.db_name = None
        self._cursor = None
        self.conn = None
        self.pg_conn = None
        self.pg_cursor = None

    def insert_csv(self, csv_path, table):
        c = self.cursor
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
        cmd = ['psql']
        if self.host is not None:
            cmd.extend(['-h', self.host])
        # if self.user is not None:
        #     cmd.extend(['-U', self.user])
        # if self.port is not None:
        #     cmd.extend(['-p', self.port])
        assert self.db_name is not None
        cmd.extend([self.db_name, '-c', copy_from_sql])
        # Q: Do we need to disable foreign key checks?
        with open(csv_path, 'r') as f:
            subprocess.check_call(cmd, stdin=f)

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

        c.executemany("INSERT INTO {table} ({colnames}) VALUES ({placeholders})".format(
            placeholders=placeholders,
            table=table,
            colnames=colnames,
        ), all_values)

    @property
    def user(self):
        return getpass.getuser()

    def create_connection(self):
        if self.conn is not None:
            return

        self._maybe_read_db_config()

        db_exists, self.conn, self._cursor = self._create_connection(self.db_name)
        return db_exists

    def _create_connection(self, db_name):
        """ create a database connection to a SQLite database """
        # conn = psycopg2.connect("dbname={db} user={user}".format(
        #     db=db_name,
        #     user=self.user,
        # ), isolation_level=None)
        try:
            conn = psycopg2.connect(
                dbname=db_name,
                user=self.user,
                host=self.host,
                isolation_level=None)
            psycopg2.connect(dbname=db_name, user=self.user, host=self.host, isolation_level=None)
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
        logging.info("> Drop existing connections to {db}".format(db=self.db_name))
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
        if self.conn is not None:
            return

        db_exists, self.conn, self._cursor = self._create_connection(self.db_name)
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

    def _maybe_close_db_connection(self):
        if self._cursor is not None:
            self._cursor.close()
            self._cursor = None

        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def _drop_database(self):
        self._maybe_create_postgres_connection()
        self._maybe_close_db_connection()
        c = self.pg_cursor
        c.execute("""
        DROP DATABASE IF EXISTS {db};
        """.format(db=self.db_name))

    def _drop_tables(self):
        tables = self.tables

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
        self.run_sql_file(self.db_name, PSQL_TABLE_SQL)

    def run_sql_file(self, db_name, sql_path):
        """
        Use subprocess.run(...) to run a .sql file;
        This way, so we get actual line numbers when something fails

        e.g.

        sqlite3 blaz.db -bail -echo -cmd '.read sqlite/tables.sql' -cmd '.quit'
        """
        with open(sql_path, 'r') as sql_f:
            psql_cmd = ['psql']
            if self.host is not None:
                psql_cmd.extend(['-h', self.host])
            psql_cmd.extend([db_name])
            proc = subprocess.run(psql_cmd,
                                  stdin=sql_f,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
        if proc.returncode != 0:
            if proc.stdout is not None:
                print(proc.stdout.decode('utf-8'))
            if proc.stderr is not None:
                print(proc.stderr.decode('utf-8'))
            print("ERROR: failed to run sql file @ {path}; ret={ret}".format(
                ret=proc.returncode, path=db_name))
            sys.exit(1)


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
    def __init__(self, db_path, debug_ops=False):
        self.db_path = db_path
        self.conn = sql_create_connection(db_path)
        self.parse_debug = False
        self.debug_ops = debug_ops

        self._steps = dict()

    def steps(self, process_name, bench_name):
        return list(range(self.num_steps(process_name, bench_name)))

    @property
    def util_devices(self):
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
            EXISTS (
                SELECT *
                FROM DeviceUtilization AS du
                WHERE du.device_id = d1.device_id
            )
        """)
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

    def process_phases(
            self,
            process_name,
            debug=False):
        c = self.conn.cursor
        query = textwrap.dedent("""
        SELECT 
            p.process_name, 
            ph.phase_name, 
            MIN(e.start_time_us) AS phase_start_time_us, 
            MAX(e.end_time_us) AS phase_end_time_us

        FROM
            Event AS e
            NATURAL JOIN Phase AS ph
            NATURAL JOIN Process AS p
        WHERE
            p.process_name = {p}
        GROUP BY
            p.process_name, ph.phase_name
        """).format(
            p=sql_placeholder(),
        )
        params = (process_name,)
        sql_exec_query(c, query, params, debug=debug)
        rows = [row_as_phase(row) for row in c.fetchall()]
        return rows

    def keep_steps(self, process_name, bench_name, skip_first_step=True):
        steps = self.steps(process_name, bench_name)

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
            CATEGORY_OPERATION=CATEGORY_OPERATION,
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

    @property
    def process_names(self):
        c = self.conn.cursor
        c.execute("""
        SELECT process_name FROM Process
        ORDER BY process_name ASC
        """)
        process_names = [row['process_name'] for row in c.fetchall()]
        return process_names


    DEBUG_FETCH_STEPS = False
    def _fetch_steps(self, process_name, bench_name):
        if process_name in self._steps and bench_name in self._steps[process_name]:
            return

        start_fetch_t = time.time()
        c = self.conn.cursor
        c.execute("""
        SELECT e1.event_name, e1.start_time_us, e1.duration_us
        FROM Event AS e1
            NATURAL JOIN Category AS c
            NATURAL JOIN Process AS p
        WHERE 
            c.category_name = '{CATEGORY_OPERATION}' AND
            e1.event_name = {p} AND
            p.process_name = {p} AND 
            {no_dump_overlap_clause}
        ORDER BY e1.start_time_us ASC 
        """.format(
            CATEGORY_OPERATION=CATEGORY_OPERATION,
            # CATEGORY_PROFILING=CATEGORY_PROFILING,
            # PROFILING_DUMP_TRACE=PROFILING_DUMP_TRACE,
            # NOTE: We do NOT want to select any steps of an operation that overlap at all with a DUMP event.
            # indents=3 since {overlap_clause} above has 3 indent-levels in front of it.
            # overlap_clause=sql_overlap_clause('e1', 'e2', indents=3),
            no_dump_overlap_clause=sql_no_dump_overlap_clause('e1', tmp_event_alias_2='e2', tmp_category_alias_2='c2', indents=1),
            p=sql_placeholder(),
        ),
            (bench_name, process_name))

        # NOT EXISTS (
        #     SELECT *
        #     FROM Event AS e2
        # NATURAL JOIN Category as c2
        # WHERE
        # c2.category_name = '{CATEGORY_PROFILING}' AND
        # e2.event_name = '{PROFILING_DUMP_TRACE}' AND
        # {overlap_clause}
        # )

        rows = rows_as_ktime(c.fetchall())
        if process_name not in self._steps:
            self._steps[process_name] = dict()
        self._steps[process_name][bench_name] = rows
        end_fetch_t = time.time()
        sec_fetch = end_fetch_t - start_fetch_t
        if SQLCategoryTimesReader.DEBUG_FETCH_STEPS:
            logging.info("> fetch_steps process={proc}, op={op} took {sec} seconds".format(
                proc=process_name,
                op=bench_name,
                sec=sec_fetch,
            ))

    def num_steps(self, process_name, bench_name):
        """
        We don't record step numbers in the database.
        Instead, steps are an index into the i-th time this operation occurs in the entire ML-script.

        We tend to want to skip the 1-st time the operation occurs / is profiled, since
        it will include load-time overheads (libcupti).
        """
        self._fetch_steps(process_name, bench_name)
        return len(self._steps[process_name][bench_name])

    def step_event(self, step, process_name, bench_name):
        self._fetch_steps(process_name, bench_name)
        return self._steps[process_name][bench_name][step]

        # Swallow any excess arguments
    DEBUG_EACH_OP_INSTANCE = False
    def each_op_instance(self, bench_name,
                         group_by_device=DEFAULT_group_by_device,
                         ignore_categories=DEFAULT_ignore_categories,
                         debug=DEFAULT_debug,
                         skip_first_step=True,
                         show_progress=True):
        start_t = time.time()
        process_names = self.process_names
        for process_name in process_names:

            keep_steps = self.keep_steps(process_name, bench_name, skip_first_step)
            if bench_name == NO_BENCH_NAME:
                pprint.pprint({
                    'name':'SQLCategoryTimesReader.each_op_instance',
                    'keep_steps':keep_steps})

            for step in progress(keep_steps,
                                 desc=as_progress_label("each_op_instance", process_name),
                                 show_progress=show_progress):
                category_times = self.parse(step, process_name, bench_name,
                                            group_by_device, ignore_categories, debug,
                                            show_progress=show_progress)
                end_t = time.time()
                sec = end_t - start_t
                if SQLCategoryTimesReader.DEBUG_EACH_OP_INSTANCE:
                    # - As bad as 56 seconds in split_op_stacks
                    #   yield(process=loop_train_eval, step=0)
                    #   Why is step 0 so slow?
                    # - often 0.02 seconds, 0.10 seconds
                    logging.info("> each_op_instance yield(process={proc}, step={step}) took {sec} seconds".format(
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
                       start_time_us=None,
                       end_time_us=None,
                       pre_reduce=None,
                       # group_by_device=DEFAULT_group_by_device,
                       ignore_categories=DEFAULT_ignore_categories,
                       debug=DEFAULT_debug,
                       debug_ops=False,
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
            events[proc] = process_op_nest(events[proc]) to get the right operations for CATEGORY_OPERATION.

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
        # Categories NOT including the operation-type categories (that replaced CATEGORY_OPERATION)
        # categories = set()
        # proc_types = set()

        if process_name is None and ( start_time_us is not None or end_time_us is not None ):
            # [None]
            process_names = [process_name]
        elif process_name is not None:
            process_names = [process_name]
        else:
            process_names = self.process_names

        for process_name in process_names:

            process_label = "process={proc}".format(
                proc=process_name)
            """
            {
                <category_name>: <events belonging to process_name, sorted by start_sec>,
                ...
            }
            """
            process_category_times = self.process_events(
                process_name=process_name,
                phase_name=phase_name,
                ignore_categories=ignore_categories,
                start_time_us=start_time_us,
                end_time_us=end_time_us,
                debug=debug,
                debug_ops=debug_ops,
                debug_label=process_label,
                # fetchall=False,
                fetchall=True,
            )
            # assert len(proc_events) > 0
            # assert len(proc_category_times) > 0
            # assert len(proc_category_times[CATEGORY_OPERATION]) > 0

            for proc in process_category_times.keys():

                # assert CATEGORY_OPERATION in proc_category_times
                if CATEGORY_OPERATION in process_category_times[proc]:
                    """
                    Replace proc_category_times[CATEGORY_OPERATION], handle operation nesting.
                    We are assuming a single process is single-threaded here, so any operation 
                    nesting is form a single-threaded "call-stack".
                    """
                    process_category_times[proc][CATEGORY_OPERATION] = process_op_nest_single_thread(
                        process_category_times[proc][CATEGORY_OPERATION],
                        show_progress=debug,
                        debug_label=process_label)
                    # Doesn't speed anything up on "test_load"
                    # proc_category_times[CATEGORY_OPERATION] = process_op_nest_parallel(
                    #     proc_category_times[CATEGORY_OPERATION],
                    #     show_progress=debug,
                    #     debug_label=process_label)
                    assert len(process_category_times[proc][CATEGORY_OPERATION]) > 0

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
            for proc in process_category_times.keys():
                bin_category_times_single_thread(
                    # process_name, proc_category_times,
                    process_category_times[proc],
                    pre_reduce=pre_reduce,
                    # categories=categories, operation_types=operation_types,
                    category_times=category_times, debug=debug)
                # Doesn't speed anything up on "test_load"
                # bin_category_times_parallel(
                #     process_name, proc_category_times,
                #     categories, category_times, operation_types, debug)

        # Sanity check: Events are all sorted.
        for category, events in category_times.items():
            for e1, e2 in zip(events, events[1:]):
                assert e1.start_time_usec <= e2.start_time_usec

        # assert len(operation_types) > 0

        # if debug:
        #     logging.info("> DEBUG: parse_timeline: ")
        #     pprint.pprint({
        #         'proc_types':proc_types,
        #         'operation_types':operation_types,
        #         'categories':categories,
        #     }, indent=2)

        # ret = category_times, categories, operation_types, proc_types
        ret = category_times
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
        total_time_sec = total_time_us/NumberType(MICROSECONDS_IN_SECOND)
        return total_time_sec

    def process_events(self, process_name=None, phase_name=None,
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
                CATEGORY_OPERATION=CATEGORY_OPERATION,
                op=op_name,
            )
        keep_op_clause = op_name is not None
        op_clause = maybe_clause(get_op_clause().maybe_eval(keep_op_clause), keep=keep_op_clause, indents=3)

        query = textwrap.dedent("""
        SELECT d1.device_name, c1.category_name, e1.event_name, e1.start_time_us, e1.duration_us
            , e1.event_id
            , p1.process_name, ph1.phase_name
            , e1.pyprof_filename
            , e1.pyprof_line_no
            , e1.pyprof_function
            , e1.pyprof_line_description
        FROM 
            Category AS c1
            NATURAL JOIN Event as e1
            NATURAL JOIN Process as p1
            NATURAL JOIN Phase as ph1
            NATURAL LEFT JOIN Device as d1
        WHERE 
            {process_clause} AND
            {phase_clause} AND
            {debug_ops_clause} AND
            {ignore_clause} AND
            -- Ignore any events that overlap with when we dump profiling information.
            {no_dump_overlap_clause} AND
            -- Keep events within a start/end time range (e.g. overlap with a specific process phase).
            {event_range_clause} AND
            -- Only keep events that are subsumed by an <op> event.
            -- e.g. Only keep pyprof events that belong to the set_operation('tree_search') annotation
            {op_clause}
        ORDER BY 
            p1.process_name, e1.start_time_us ASC 
        """).format(
            ignore_clause=sql_ignore_clause('c1', ignore_categories, keep_categories, indents=1),
            # CATEGORY_PROFILING=CATEGORY_PROFILING,
            # PROFILING_DUMP_TRACE=PROFILING_DUMP_TRACE,
            # overlap_clause=sql_overlap_clause('e1', 'dump_event', indents=3),
            no_dump_overlap_clause=sql_no_dump_overlap_clause('e1', 'e2', 'c2', indents=1),
            process_clause=sql_process_clause(process_name, 'p1', indents=1, allow_none=True),
            phase_clause=sql_phase_clause(phase_name, 'ph1', indents=1, allow_none=True),
            debug_ops_clause=sql_debug_ops_clause(debug_ops, 'e1', indents=1),
            event_range_clause=sql_event_range_clause('e1', start_time_us, end_time_us, indents=1),
            op_clause=op_clause,
            p=sql_placeholder(),
        )

        # NOT EXISTS (
        #     SELECT *
        #     FROM Event as dump_event
        # NATURAL JOIN Category as c2
        # WHERE
        # c2.category_name = '{CATEGORY_PROFILING}' AND
        # dump_event.event_name = '{PROFILING_DUMP_TRACE}' AND
        # {overlap_clause}
        # ) AND

        # query = textwrap.dedent("""
        # SELECT device_name, category_name, event_name, start_time_us, duration_us
        # FROM
        #   Category AS c
        #   NATURAL JOIN Event as e
        #   NATURAL JOIN Process as p
        #   NATURAL LEFT JOIN Device as d
        # ORDER BY start_time_us ASC
        # """.format(
        #     ignore_clause=ignore_clause,
        # ))

        params = None
        sql_exec_query(c, query, params, self.__class__, debug)

        if fetchall:
            query_start_t = time.time()
            rows = c.fetchall()
            query_end_t = time.time()
            time_sec = query_end_t - query_start_t
            if debug:
                logging.info("> query took {sec} seconds".format(
                    sec=time_sec,
                ))
        else:
            if debug:
                logging.info("> fetchall = {fetchall}".format(
                    fetchall=fetchall,
                ))
            rows = c

        # start_t = time.time()

        # process_category_times:
        # process_name -> CATEGORY_* -> times
        process_category_times = dict()
        for process_name, process_rows in itertools.groupby(rows, key=lambda row: row['process_name']):
            assert process_name not in process_category_times
            process_category_times[process_name] = dict()
            self._add_event_rows_to_category_times(
                process_category_times[process_name], process_rows,
                debug=debug,
                show_progress=debug,
                debug_label=debug_label)

        # end_t = time.time()
        # if debug:
        #     time_sec = end_t - start_t
        #     logging.info("> process_name={proc}: _add_event_rows_to_category_times took {sec} seconds".format(
        #         proc=process_name,
        #         sec=time_sec,
        #     ))

        return process_category_times


    def _add_event_rows_to_category_times(self, category_times, rows,
                                          group_by_device=DEFAULT_group_by_device,
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
            ktime = row_as_ktime(row)
            category_times_add_time(category_times, row['device_name'], ktime, group_by_device, category=row['category_name'])

    DEBUG_PARSE = False
    def parse(self, step, process_name, bench_name,
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
        n_steps = self.num_steps(process_name, bench_name)
        assert 0 <= step < n_steps

        op_event = self.step_event(step, process_name, bench_name)

        parse_debug = debug or self.parse_debug
        if SQLCategoryTimesReader.DEBUG_PARSE:
            end_get_events = time.time()
            sec_get_events = end_get_events - start_get_events
            logging.info("> parse.get_events process={proc}, op={op}, step={step} took {sec} seconds".format(
                proc=process_name,
                op=bench_name,
                step=step,
                sec=sec_get_events,
            ))

        if parse_debug:
            logging.info("> step={step}, process={proc}, op={bench}, time={time}".format(
                step=step, proc=process_name, bench=bench_name, time=op_event))

        return self.events_that_overlap_with(
            op_event, process_name,
            bench_name=bench_name,
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
                                 step=None,
                                 group_by_device=DEFAULT_group_by_device,
                                 ignore_categories=DEFAULT_ignore_categories,
                                 debug=DEFAULT_debug,
                                 show_progress=False):
        return self.events_by_time_range(
            op_event.start_time_usec, op_event.end_time_usec, process_name,
            bench_name=bench_name,
            step=step,
            group_by_device=group_by_device,
            ignore_categories=ignore_categories,
            debug=debug,
            show_progress=show_progress)

    def events_by_time_range(self, start_time_usec, end_time_usec, process_name,
                                 # For debugging self.parse()
                                 bench_name=None,
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
        # import ipdb; ipdb.set_trace()
        category_times = dict()
        debug_label = "process={proc}, step={step}".format(
            proc=process_name,
            step=step,
        )
        rows = c.fetchall()
        if SQLCategoryTimesReader.DEBUG_PARSE:
            end_parse_query_t = time.time()
            sec_parse_query = end_parse_query_t - start_parse_query_t
            logging.info("> parse.query process={proc}, op={op}, step={step} took {sec} seconds".format(
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
            logging.info("> parse.add_event_rows process={proc}, op={op}, step={step} took {sec} seconds".format(
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
            logging.info("> DEBUG: dump trace events BEFORE process_op_nest @ {path}".format(path=json_path))
            dump_category_times(category_times, json_path, print_log=False)

        if SQLCategoryTimesReader.DEBUG_PARSE:
            start_process_op_nest_t = time.time()
        category_times[CATEGORY_OPERATION] = process_op_nest_single_thread(category_times[CATEGORY_OPERATION],
                                                             filter_by_op=bench_name,
                                                             # Gets in the way of each_op_instance progress bar.
                                                             # show_progress=debug or show_progress,
                                                             show_progress=SQLCategoryTimesReader.DEBUG_PARSE,
                                                             debug_label=debug_label)
        if SQLCategoryTimesReader.DEBUG_PARSE:
            end_process_op_nest_t = time.time()
            sec_process_op_nest_t = end_process_op_nest_t - start_process_op_nest_t
            logging.info("> parse.process_op_nest process={proc}, op={op}, step={step} took {sec} seconds".format(
                proc=process_name,
                op=bench_name,
                step=step,
                sec=sec_process_op_nest_t,
            ))

            end_parse_t = time.time()
            sec_parse = end_parse_t - start_parse_t
            logging.info("> parse process={proc}, op={op}, step={step} took {sec} seconds".format(
                proc=process_name,
                op=bench_name,
                step=step,
                sec=sec_parse,
            ))
            sec_total = sec_get_events + sec_add_event_rows_t + sec_process_op_nest_t + sec_parse_query
            logging.info("> total parse subops = {sec} seconds".format(
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
        if category in CATEGORIES_CPU:
            cat = CATEGORY_CPU
            categories.add(cat)
        elif category == CATEGORY_GPU:
            cat = CATEGORY_GPU
            categories.add(cat)
        elif category == CATEGORY_OPERATION:
            cat = event.name
            operation_types.add(cat)
        else:
            # Q: What about category operation...?
            # We want to KEEP the operation category so we can determine
            # overlap between q_backward/q_forward across processes...
            #
            # I think all we need to do is replace "CATEGORY_OPERATION" for an event
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
#         if old_category in CATEGORIES_CPU:
#             cat = CATEGORY_CPU
#             self.categories.add(cat)
#         elif old_category == CATEGORY_GPU:
#             cat = CATEGORY_GPU
#             self.categories.add(cat)
#         elif old_category == CATEGORY_OPERATION:
#             cat = event.name
#             # logging.info("> operation_types.add {cat}".format(cat=cat))
#             self.operation_types.add(cat)
#         else:
#             # Q: What about category operation...?
#             # We want to KEEP the operation category so we can determine
#             # overlap between q_backward/q_forward across processes...
#             #
#             # I think all we need to do is replace "CATEGORY_OPERATION" for an event
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

def default_pre_reduce(category, event, process_name):
    if category == CATEGORY_OPERATION:
        new_category = CategoryKey(
            ops=frozenset([event.name]),
            non_ops=frozenset(),
            procs=frozenset([process_name]))
    else:
        new_category = CategoryKey(
            ops=frozenset(),
            non_ops=frozenset([category]),
            procs=frozenset([process_name]))
    return new_category

def bin_category_times(
    # process_name,
    category,
    events,
    pre_reduce=None,
    # categories=None,
    # operation_types=None,
    category_times=None,
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

    if pre_reduce is None:
        pre_reduce = default_pre_reduce

    # if categories is None:
    #     categories = set()
    # if operation_types is None:
    #     operation_types = set()

    if category_times is None:
        category_times = dict()
        use_insort = True
    else:
        use_insort = False

    # proc_category = proc_as_category(process_name)

    progress_label = "parse_timeline: category={cat}".format(
        cat=category)
    # progress_label = "parse_timeline: process={proc}, category={cat}".format(
    #     proc=process_name, cat=category)
    for event in progress(events, desc=progress_label, show_progress=debug):

        # if category in CATEGORIES_CPU:
        #     cat = CATEGORY_CPU
        #     categories.add(cat)
        # elif category == CATEGORY_GPU:
        #     cat = CATEGORY_GPU
        #     categories.add(cat)
        # elif category == CATEGORY_OPERATION:
        #     cat = event.name
        #     # logging.info("> operation_types.add {cat}".format(cat=cat))
        #     operation_types.add(cat)
        # else:
        #     # Q: What about category operation...?
        #     # We want to KEEP the operation category so we can determine
        #     # overlap between q_backward/q_forward across processes...
        #     #
        #     # I think all we need to do is replace "CATEGORY_OPERATION" for an event
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

        # new_category = pre_reduce(category, event, process_name)
        new_category = pre_reduce(category, event)
        if new_category is None:
            # SKIP this event entirely.
            continue

        if new_category not in category_times:
            category_times[new_category] = []
        # NOTE: if we bin the CPU/GPU/operation events into separate lists,
        # we can use merge_sorted, which may be faster.
        #
        # However, merge_sorted approach will probably cause more allocations...
        if use_insort:
            insort(category_times[new_category], event, key=lambda event: event.start_time_usec)
        else:
            category_times[new_category].append(event)

    # return category_times, categories, operation_types
    return category_times

def bin_category_times_single_thread(
    # process_name,
    proc_category_times,
    pre_reduce=None,
    # categories=None,
    # operation_types=None,
    category_times=None,
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
            pre_reduce=pre_reduce,
            # categories=categories, operation_types=operation_types, category_times=None,
            category_times=None,
            debug=debug)
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
def BinCategoryTimesWorker(args):
    process_name, category, events, debug = args
    return bin_category_times(process_name, category, events, debug=debug)

def split_category_times_by_category(process_name, proc_category_times, debug):
    for category, events in proc_category_times.items():
        yield process_name, category, events, debug

# TODO: if this doesn't help much, we could perform more equal splits by doing:
# 1. compute total number of events
# 2. split proc_category_times into n dicts with equal events;
#    Need to make a split_category_times(category_times, n) to do that
# 3. make the Worker take proc_category_times instead of <category, events>

def bin_category_times_parallel(
    process_name,
    proc_category_times,
    categories=None,
    category_times=None,
    operation_types=None,
    debug=False,
    nprocs=None):
    if categories is None:
        categories = set()
    if operation_types is None:
        operation_types = set()
    if category_times is None:
        category_times = dict()

    with multiprocessing.Pool(nprocs) as pool:
        splits = list(split_category_times_by_category(process_name, proc_category_times, debug=False))
        it = pool.imap_unordered(BinCategoryTimesWorker, splits)
        progress_label = "bin_category_times_parallel: process={proc}".format(proc=process_name)
        for i, (category_times_i, categories_i, operation_types_i) in enumerate(progress(it, desc=progress_label, total=len(splits), show_progress=debug)):
            merge_category_times(category_times, category_times_i, inplace=True)
            categories.update(categories_i)
            operation_types.update(operation_types_i)

    return category_times, categories, operation_types

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
        logging.info("> {name_str}query:".format(
            name_str=name_str))
        logging.info(query)
        if params is not None:
            logging.info("> params:")
            pprint.pprint(params, indent=2)
    start_t = time.time()
    if params is not None:
        c.execute(query, params)
    else:
        c.execute(query)
    end_t = time.time()
    if debug:
        logging.info("> query took {sec} seconds".format(sec=end_t - start_t))

def sql_create_connection(db_path, host=None):
    if 'IML_POSTGRES_HOST' in os.environ and host is None:
        host = os.environ['IML_POSTGRES_HOST']
    else:
        host = 'localhost'
    logging.info("Using DB_HOST = {host}".format(host=host))
    if py_config.SQL_IMPL == 'psql':
        return TracesPostgresConnection(db_path, host=host)
    elif py_config.SQL_IMPL == 'sqlite':
        return TracesSQLiteConnection(db_path, host=host)
    raise NotImplementedError("Not sure how to create connection for SQL_IMPL={impl}".format(
        impl=py_config.SQL_IMPL))

def sql_placeholder():
    if py_config.SQL_IMPL == 'psql':
        return "%s"
    elif py_config.SQL_IMPL == 'sqlite':
        return "?"
    raise NotImplementedError("Not sure how to create connection for SQL_IMPL={impl}".format(
        impl=py_config.SQL_IMPL))

def rows_as_ktime(rows):
    return [row_as_ktime(row) for row in rows]

def row_as_ktime(row):
    """
    Row should be a result of "Event NATURAL JOIN Category", and hence contain at least:
    - start_time_us
    - duration_us
    - event_name
    """
    ktime = KernelTime(
        start_usec=row['start_time_us'],
        end_usec=row['start_time_us'] + row['duration_us'],
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

def row_as_event(row):
    ktime = row_as_ktime(row)
    # Add additional attributes to KernelTime.
    ktime.event_id = row['event_id']
    ktime.device_name = row['device_name']
    ktime.category_name = row['category_name']
    ktime.process_name = row['process_name']
    return ktime

class Phase:
    def __init__(self, phase_name):
        self.phase_name = phase_name

    def _add_row_fields(self, row):
        self.process_name = row['process_name']
        self.phase_start_time_us = row['phase_start_time_us']
        self.phase_end_time_us = row['phase_end_time_us']

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

def row_as_phase(row):
    phase = Phase(row['phase_name'])
    phase._add_row_fields(row)
    return phase

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
        for event in op_stack.get_absored_ops():
            if filter_by_op is not None and event.name != filter_by_op:
                continue
            all_events.append(event)

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
    # logging.info("> total_op_stacks = {total_op_stacks}".format(
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
        self.last_insert_start_time = None
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
        if self.last_insert_start_time is None:
            self.last_insert_start_time = ktime.start_time_usec
        else:
            assert self.last_insert_start_time <= ktime.start_time_usec

        op1 = OpStack(ktime)
        assert self.subsumes(op1)
        self._insert(op1)
        return op1

    def _insert(self, op1):
        assert self.subsumes(op1)
        for op2 in self.sub_ops:
            if op2.subsumes(op1):
                op2._insert(op1)
                return
        if len(self.sub_ops) > 0:
            assert op1.ktime.is_after(self.sub_ops[-1].ktime)
        op1.parent = self
        self.sub_ops.append(op1)
        # into = self
        # if into.ktime.subsumes(op.ktime):

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
                 key=lambda x: x):
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
    a.insert(lo, x)
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

def sql_process_clause(process_name, process_alias, indents=None, allow_none=False):
    return _sql_eq_clause(process_name, process_alias, 'process_name', indents, allow_none)

def sql_phase_clause(phase_name, phase_alias, indents=None, allow_none=False):
    return _sql_eq_clause(phase_name, phase_alias, 'phase_name', indents, allow_none)

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

def sql_no_dump_overlap_clause(event_alias_1, tmp_event_alias_2, tmp_category_alias_2, indents=None):
    """
    Make sure Event's we are selecting from event_alias_1
    do not overlap with DUMP Event's we are selecting from tmp_event_alias_2.

    :param event_alias_1:
        The event alias used for Events we are overlapping against.
    :param tmp_event_alias_2:
    :param tmp_category_alias_2:
        Aliases to use for DUMP events.
    :return:
    """
    clause = textwrap.dedent("""
        NOT EXISTS (
            SELECT * 
            FROM Event AS {e2}
                NATURAL JOIN Category as {c2}
            WHERE 
                {c2}.category_name = '{CATEGORY_PROFILING}' AND
                {e2}.event_name = '{PROFILING_DUMP_TRACE}' AND 
                {overlap_clause}
        )
        """.format(
        CATEGORY_OPERATION=CATEGORY_OPERATION,
        CATEGORY_PROFILING=CATEGORY_PROFILING,
        PROFILING_DUMP_TRACE=PROFILING_DUMP_TRACE,
        # NOTE: We do NOT want to select any steps of an operation that overlap at all with a DUMP event.
        # indents=3 since {overlap_clause} above has 3 indent-levels in front of it.
        e2=tmp_event_alias_2,
        c2=tmp_category_alias_2,
        overlap_clause=sql_overlap_clause(event_alias_1, tmp_event_alias_2, indents=indents+2),
    ))
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

    if is_tfprof_file(path):
        proto = read_tfprof_file(path)
        meta = {
            'process_name':proto.process_name,
            'phase_name':proto.phase,
        }
        return meta
    elif is_pyprof_file(path) or is_dump_event_file(path):
        proto = read_pyprof_file(path)
        meta = {
            'process_name':proto.process_name,
            'phase_name':proto.phase,
        }
        return meta
    elif is_pyprof_call_times_file(path):
        call_times_data = read_pyprof_call_times_file(path)
        meta = {
            'process_name':call_times_data['process_name'],
            'phase_name':call_times_data['phase'],
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
                 # Swallow any excess arguments
                 **kwargs):
        self.device_name = device_name
        self.device_id = device_id

    @staticmethod
    def from_row(row):
        device = Device(**row)
        for attr, value in row.items():
            if not hasattr(device, attr):
                setattr(device, attr, value)
        return device

    def __str__(self):
        return 'Device(name="{name}", id={id})'.format(
            name=self.device_name,
            id=self.device_id)

def test_merge_sorted():

    def test_01():

        # xs:         5 6 7 8
        # ys: 1 2 3 4 5       9 10 11
        xs = [            5, 6, 7, 8,          ]
        ys = [1, 2, 3, 4, 5,          9, 10, 11]
        expect = [1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10, 11]
        actual = merge_sorted(xs, ys)

        assert actual == expect
    test_01()

def test_process_op_nest():
    from test.test_util import sec, T, U

    process_op_nest = process_op_nest_single_thread

    def test_01_1_stack():
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
    test_01_1_stack()

    def test_02_2_stacks():
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
    test_02_2_stacks()

    # # Invalid input test
    # def test_03_complete_overlap():
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
    # test_03_complete_overlap()

    def test_04_multiple_sub_events():
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
    test_04_multiple_sub_events()

    def test_05_wild_data_01():
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
    test_05_wild_data_01()
