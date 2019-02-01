import os
import re
import subprocess
import progressbar
import pytest
import bisect

import sqlite3
from sqlite3 import Error

from parser.common import ProfilerParserCommonMixin
from os.path import join as _j, abspath as _a, dirname as _d, exists as _e, basename as _b

# from proto.tensorflow.core.profiler.tfprof_log_pb2 import ProfileProto
from tensorflow.core.profiler.tfprof_log_pb2 import ProfileProto
from proto.protobuf.pyprof_pb2 import Pyprof

import py_config

from parser.trace_events import dump_category_times

from parser.readers import TFProfCategoryTimesReader, \
    DEFAULT_group_by_device, \
    DEFAULT_ignore_categories, \
    DEFAULT_debug \

import contextlib

from parser.common import *

from parser.stats import category_times_add_time

from parser.stats import KernelTime

TABLE_SQL = _j(py_config.ROOT, "sqlite", "tables.sql")
INDICES_SQL = _j(py_config.ROOT, "sqlite", "indices.sql")

class SQLiteParser:
    # (ProfilerParserCommonMixin):
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
                 **kwargs):
        self.directory = directory
        self.conn = TracesSQLiteConnection(self.db_path)
        self.debug = debug
        self.block_size = 50000

    def get_source_files(self):
        src_files = []
        for dirpath, dirnames, filenames in os.walk(self.directory):
            for base in filenames:
                path = _j(dirpath, base)
                if is_tfprof_file(path) or is_pyprof_file(path):
                    src_files.append(path)
        if len(src_files) == 0:
            raise MissingInputFiles(textwrap.dedent("""
            {klass}: Couldn't find any tfprof/pyprof files root at {dir}.
            """.format(
                klass=self.__class__.__name__,
                dir=self.directory,
            )))
        return src_files

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
        c.execute("""
        SELECT * FROM Event AS op1 NATURAL JOIN Category c1
        WHERE 
        c1.category_name = '{CATEGORY_OPERATION}' AND
        EXISTS (
            SELECT * FROM Event AS op2 NATURAL JOIN Category c2
            WHERE 
            op1.process_id == op2.process_id AND
            c2.category_name = '{CATEGORY_OPERATION}' AND
            op1.start_time_us < op2.start_time_us AND
                                op2.start_time_us < op1.end_time_us AND
                                                    op1.end_time_us < op2.end_time_us
        )
        """.format(CATEGORY_OPERATION=CATEGORY_OPERATION))
        rows = c.fetchall()
        if len(rows) > 0:
            pprint.pprint({'Events with partial overlap':[dict(row) for row in rows]})
            raise RuntimeError("ERROR: Detected partial-overlap between operation-type events")

        # op1.start == op2.start and
        # op1.end   == op2.end
        c.execute("""
        SELECT * FROM Event AS op1 NATURAL JOIN Category c1
        WHERE 
        c1.category_name = '{CATEGORY_OPERATION}' AND
        EXISTS (
            SELECT * FROM Event AS op2 NATURAL JOIN Category c2
            WHERE 
            c2.category_name = '{CATEGORY_OPERATION}' AND
            op1.event_id != op2.event_id AND
            op1.start_time_us == op2.start_time_us AND
            op1.end_time_us == op2.end_time_us
        )
        """.format(CATEGORY_OPERATION=CATEGORY_OPERATION))
        rows = c.fetchall()
        if len(rows) > 0:
            pprint.pprint({'Events with complete overlap':[dict(row) for row in rows]})
            raise RuntimeError("ERROR: Detected complete-overlap between operation-type events")

    def run(self):
        # 1. Create SQLite db file
        # for each input file:
        #   if is pyprof file:
        #     ...
        #     insert into db
        #   elif if tfprof file:
        #     ...
        #     insert into db

        db_exists = _e(self.db_path)
        if db_exists:
            os.remove(self.db_path)

        self.create_db()

        self.process_to_id = dict()
        self.category_to_id = dict()
        self.device_to_id = dict()

        src_files = self.get_source_files()
        pprint.pprint({
            'rule':self.__class__.__name__,
            'src_files':src_files,
        })
        for path in src_files:
            if is_tfprof_file(path):
                self.insert_tfprof_file(path)
            elif is_pyprof_file(path):
                self.insert_pyprof_file(path)
            else:
                raise NotImplementedError

        # Create indices at the end to reduce per-insert overhead.
        self.create_indices()
        
        self._check()

        self.conn.commit()
        self.conn.close()
        
    def _check(self):
        """
        Run any post-insert checks not captured by database constraints.
        """
        self._check_no_partial_or_complete_op_overlap()

    def maybe_commit(self, i):
        if (i + 1) % self.block_size == 0:
            self.conn.commit()

    def insert_tfprof_file(self, path):

        reader = TFProfCategoryTimesReader(path)

        print("> Insert tfprof file: {p}".format(p=path))
        if self.debug:
            reader.print(sys.stdout)

        process_id = self.insert_process_name(reader.process_name)

        inserts = []
        self._total_inserts = 0

        with progressbar.ProgressBar(max_value=reader.num_all_events()) as bar, \
             bulk_inserter(self.conn, 'Event', self.block_size, bar) as bulk:
            for i, (device, event) in enumerate(reader.all_events()):
                category, start_time_us, duration_us, name = event
                category_id = self.insert_category_name(category)
                if category == 'GPU' and self.debug:
                    print("> category = {c}, duration_us = {duration_us}".format(
                        c=category,
                        duration_us=duration_us))
                device_id = self.insert_device_name(device)
                end_time_us = start_time_us + duration_us
                insert = {
                    # 'thread_id':event.thread_id,
                    'start_time_us':start_time_us,
                    'end_time_us':end_time_us,
                    'duration_us':duration_us,
                    'event_name':name,
                    'category_id':category_id,
                    'process_id':process_id,
                    'device_id':device_id,
                }
                bulk.add_insert(insert)

    def insert_process_name(self, process_name):
        return self._insert_name(
            'Process',
            'process_id', 'process_name',
            self.process_to_id,
            process_name)

    def insert_device_name(self, device_name):
        return self._insert_name(
            'Device',
            'device_id', 'device_name',
            self.device_to_id,
            device_name)

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
        SELECT {id_field} from {table} WHERE {name_field} = ?
        """.format(
            id_field=id_field,
            table=table,
            name_field=name_field,
        ), (name,))
        rows = c.fetchall()
        if len(rows) == 0:
            self.conn.insert_dict(table, {
                name_field: name,
            })
            ident = c.lastrowid
        else:
            ident = rows[0][id_field]

        name_to_id[name_field] = ident

        return ident

    def insert_pyprof_file(self, path):
        with open(path, 'rb') as f:
            proto = Pyprof()
            proto.ParseFromString(f.read())

        print("> Insert pyprof file: {p}".format(p=path))
        if self.debug:
            print(proto)

        c = self.cursor
        # Insert Process
        process_id = self.insert_process_name(proto.process_name)

        # categories = set()
        def insert_category_events(event_conn, eventattr_conn, category, events):
            # Insert Category
            # categories.add(category)
            category_id = self.insert_category_name(category)
            # category_id = self.category_to_id[category]
            for event in events:
                # Insert Event
                event_id = event_conn.insert_dict('Event', {
                    'thread_id':event.thread_id,
                    'start_time_us':event.start_time_us,
                    'end_time_us':event.start_time_us + event.duration_us,
                    'duration_us':event.duration_us,
                    'event_name':event.name,
                    'category_id':category_id,
                    'process_id':process_id,
                })
                # Insert EventAttr
                for attr_name, attr_value in event.attrs.items():
                    attr_id = eventattr_conn.insert_dict('EventAttr', {
                        'event_id':event_id,
                        'attr_name':attr_name,
                        'attr_value':attr_value,
                    })

        def each_category_events():
            for step, python_events in proto.python_events.items():
                yield CATEGORY_PYTHON, python_events.events

            for step, clibs in proto.clibs.items():
                for category, clib_events in clibs.clibs.items():
                    yield category, clib_events.events

        num_all_events = sum(len(events) for category, events in each_category_events())

        with progressbar.ProgressBar(max_value=num_all_events) as bar, \
            bulk_inserter(self.conn, 'EventAttr', self.block_size, bar) as event_attr_bulk, \
            bulk_inserter(self.conn, 'Event', self.block_size, progress_bar=None, id_field='event_id') as event_bulk:
            for category, events in each_category_events():
                insert_category_events(event_bulk, event_attr_bulk, category, events)

        self.conn.commit()

    def create_db(self):
        self.conn.run_sql_file(self.db_path, TABLE_SQL)
        assert _e(self.db_path)

        # NOTE: This doesn't provide line numbers when things fail.
        #
        # with open(TABLE_SQL) as f:
        #     script = f.read()
        # self.cursor.executescript(script)

    def create_indices(self):
        self.conn.run_sql_file(self.db_path, INDICES_SQL)

    @property
    def db_path(self):
        return traces_db_path(self.directory)

@contextlib.contextmanager
def transaction(conn):
    # print_stacktrace('> IML: BEGIN TRANSACTION')
    conn.cursor.execute('BEGIN TRANSACTION')
    try:
        yield
    except Exception as e:
        raise
    conn.cursor.execute('COMMIT')

@contextlib.contextmanager
# def bulk_inserter(conn, table, block_size=50000, progress_bar=None, id_field=None):
def bulk_inserter(*args, **kwargs):
    try:
        bulk = BulkInserter(*args, **kwargs)
        # bulk = BulkInserter(conn, table, block_size, progress_bar, id_field)
        yield bulk
    except Exception as e:
        raise
    bulk.finish()
    # conn.commit()

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
    def __init__(self, conn, table, block_size=50000, progress_bar=None, id_field=None):
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
            # print("> IDENT is None, use 0")
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

class TracesSQLiteConnection:
    def __init__(self, db_path):
        self.db_path = db_path
        self._cursor = None
        self.conn = None

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
        placeholders = ','.join('?' * len(cols))
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

class SQLiteCategoryTimesReader:
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
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = TracesSQLiteConnection(db_path)
        self.parse_debug = False

        self._steps = dict()

    def steps(self, process_name, bench_name):
        return list(range(self.num_steps(process_name, bench_name)))

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

    @property
    def bench_names(self):
        c = self.conn.cursor
        c.execute("""
        SELECT DISTINCT event_name FROM Event AS e NATURAL JOIN Category AS c
        WHERE c.category_name = '{op}' 
        ORDER BY e.event_name
        """.format(op=CATEGORY_OPERATION))
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

    def _fetch_steps(self, process_name, bench_name):
        if bench_name in self._steps:
            return

        c = self.conn.cursor
        c.execute("""
        SELECT e.event_name, e.start_time_us, e.duration_us
        FROM Event AS e 
            NATURAL JOIN Category AS c
            NATURAL JOIN Process AS p
        WHERE c.category_name = '{op}' AND
              e.event_name = ? AND
              p.process_name = ?
        ORDER BY e.start_time_us ASC 
        """.format(op=CATEGORY_OPERATION),
                  (bench_name, process_name))
        rows = rows_as_ktime(c.fetchall())
        if process_name not in self._steps:
            self._steps[process_name] = dict()
        self._steps[process_name][bench_name] = rows

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

    def each_op_instance(self, bench_name,
                         group_by_device=DEFAULT_group_by_device,
                         ignore_categories=DEFAULT_ignore_categories,
                         debug=DEFAULT_debug,
                         skip_first_step=True):
        process_names = self.process_names
        for process_name in process_names:

            keep_steps = self.keep_steps(process_name, bench_name, skip_first_step)

            for step in keep_steps:
                category_times = self.parse(step, process_name, bench_name,
                                            group_by_device, ignore_categories, debug)
                yield process_name, step, category_times

    def parse_timeline(self,
                       # group_by_device=DEFAULT_group_by_device,
                       ignore_categories=DEFAULT_ignore_categories,
                       debug=DEFAULT_debug):
        """
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
        category_times = dict()
        operation_types = set()
        # Categories NOT including the operation-type categories (that replaced CATEGORY_OPERATION)
        categories = set()

        for process_name in self.process_names:

            proc_events = self.process_events(process_name, ignore_categories,
                                              debug=debug,
                                              # fetchall=False,
                                              fetchall=True,
                                              )
            # assert len(proc_events) > 0
            proc_category_times = dict()
            self._add_event_rows_to_category_times(proc_category_times, proc_events)
            assert len(proc_category_times) > 0
            assert len(proc_category_times[CATEGORY_OPERATION]) > 0

            assert CATEGORY_OPERATION in proc_category_times
            proc_category_times[CATEGORY_OPERATION] = process_op_nest(proc_category_times[CATEGORY_OPERATION])
            assert len(proc_category_times[CATEGORY_OPERATION]) > 0

            # Merge all the process-specific events into a single category_times dict.
            for category, events in proc_category_times.items():
                for event in events:
                    if category in CATEGORIES_CPU:
                        new_category = CATEGORY_CPU
                        categories.add(new_category)
                    elif category == CATEGORY_GPU:
                        new_category = CATEGORY_GPU
                        categories.add(new_category)
                    elif category == CATEGORY_OPERATION:
                        new_category = event.name
                        operation_types.add(new_category)
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
                    if new_category not in category_times:
                        category_times[new_category] = []
                    # NOTE: if we bin the CPU/GPU/operation events into separate lists,
                    # we can use merge_sorted, which may be faster.
                    #
                    # However, merge_sorted approach will probably cause more allocations...
                    insort(category_times[new_category], event, key=lambda event: event.start_time_usec)
                # insort_list(category_times[new_category], events, key=lambda event: event.start_time_usec)

        # Sanity check: Events are all sorted.
        for category, events in category_times.items():
            for e1, e2 in zip(events, events[1:]):
                assert e1.start_time_usec <= e2.start_time_usec

        assert len(operation_types) > 0

        if debug:
            print("> DEBUG: parse_timeline: ")
            pprint.pprint({
                'operation_types':operation_types,
                'categories':categories,
            }, indent=2)

        return category_times, categories, operation_types

    def _query(self, query, params=None, debug=False):
        c = self.conn.cursor
        if debug:
            print("> {name} query:".format(name=self.__class__.__name__))
            print(query)
            if params is not None:
                print("> params:")
                pprint.pprint(params, indent=2)
        if params is not None:
            c.execute(query, params)
        else:
            c.execute(query)

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
        self._query(query, debug=debug)
        row = c.fetchone()
        total_time_us = row['max_us'] - row['min_us']
        total_time_sec = total_time_us/MICROSECONDS_IN_SECOND
        return total_time_sec

    def process_events(self, process_name, ignore_categories,
                       debug=False,
                       fetchall=True):
        ignore_clause = self._ignore_clause(ignore_categories)
        c = self.conn.cursor
        query = textwrap.dedent("""
        SELECT device_name, category_name, event_name, start_time_us, duration_us
        FROM 
          Category AS c 
          NATURAL JOIN Event as e
          NATURAL JOIN Process as p
          NATURAL LEFT JOIN Device as d
        WHERE p.process_name = ?
        {ignore_clause}
        ORDER BY start_time_us ASC 
        """.format(
            ignore_clause=ignore_clause,
        ))

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


        params = (
            process_name,
        )
        if debug:
            print("> {name} query:".format(name=self.__class__.__name__))
            print(query)
            print("> params:")
            pprint.pprint(params, indent=2)
        c.execute(query, params)
        if fetchall:
            ret = c.fetchall()
        else:
            ret = c
        return ret

    def _ignore_clause(self, ignore_categories):
        def ignore_and_clause(category):
            and_clause = textwrap.dedent("""\
            AND ( category_name != '{category}' )\
            """.format(category=category))
            return and_clause

        # ignore_cats = list(ignore_categories) + [CATEGORY_OPERATION]
        ignore_clause = "\n".join([ignore_and_clause(category) for category in ignore_categories])

        return ignore_clause

    def _add_event_rows_to_category_times(self, category_times, rows,
                                          group_by_device=DEFAULT_group_by_device):
        # rows = c.fetchall()
        # for row in rows:
        for row in rows:
            ktime = row_as_ktime(row)
            category_times_add_time(category_times, row['device_name'], ktime, group_by_device, category=row['category_name'])


    def parse(self, step, process_name, bench_name,
              group_by_device=DEFAULT_group_by_device,
              ignore_categories=DEFAULT_ignore_categories,
              debug=DEFAULT_debug):
        """
        JAMES NOTE: This is for reading a operation-instance (step) at-a-time.
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
        assert bench_name != NO_BENCH_NAME
        n_steps = self.num_steps(process_name, bench_name)
        assert 0 <= step < n_steps

        op_event = self.step_event(step, process_name, bench_name)

        parse_debug = debug or self.parse_debug

        if parse_debug:
            print("> step={step}, process={proc}, op={bench}, time={time}".format(
                step=step, proc=process_name, bench=bench_name, time=op_event))

        c = self.conn.cursor

        """
        We want to select all the events from all categories, where the event occurs during the operation <bench_name>. 
        
        WHERE EXISTS (
            SELECT * FROM Category as c_in NATURAL JOIN Event e_in 
            WHERE c_in.category_name = 'Operation' AND e_in.event_name = ?
            e_in.start_time_us <= e_out.start_time_us AND e_out.start_time_us <= e_in.start_time_us + e_in.duration_us
        )
        Arguments: (? = bench_name)
        - This checks whether the given event occurs during the operation <bench_name>.
          An event E occurs during an operation, if there exists an 'Operation' event 
          that surrounds its start time E.start_us OR its end time E.end_us.
        - However, currently we only ever want to select a SINGLE "step" at a time, 
          so we aren't using this.
        """

        ignore_clause = self._ignore_clause(ignore_categories)

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
        SELECT device_name, category_name, event_name, start_time_us, duration_us
        FROM 
          Category
          NATURAL JOIN Event
          NATURAL JOIN Process
          NATURAL LEFT JOIN Device
        WHERE process_name = ? AND ( 
            ( ? <= start_time_us AND start_time_us <= ? ) OR
            ( ? <= end_time_us AND end_time_us <= ? )
        )
        {ignore_clause}
        ORDER BY start_time_us ASC 
        """.format(
            ignore_clause=ignore_clause,
        ))
        if parse_debug:
            print("> {name} query:".format(name=self.__class__.__name__))
            print(query)
        c.execute(query, (
            process_name,
            op_event.start_time_usec, op_event.end_time_usec,
            op_event.start_time_usec, op_event.end_time_usec,
        ))
        category_times = dict()
        self._add_event_rows_to_category_times(category_times, c, group_by_device)

        # if i == 0 and self.debug:
        if parse_debug:
            # Q: What do does train_loop look like, overlapped with all its fellow operation-types?
            json_path = _j(self.directory, "SQLiteCategoryTimesReader{proc}{step}{bench}.debug.json".format(
                proc=process_suffix(process_name),
                step=step_suffix(step),
                bench=bench_suffix(bench_name)))
            print("> DEBUG: dump trace events BEFORE process_op_nest @ {path}".format(path=json_path))
            dump_category_times(category_times, json_path, print_log=False)

        category_times[CATEGORY_OPERATION] = process_op_nest(category_times[CATEGORY_OPERATION],
                                                             filter_by_op=bench_name)

        return category_times

def traces_db_path(directory):
    return _j(directory, "traces.db")

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
    return ktime

def process_op_nest(op_events, filter_by_op=None):
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

    def check_no_complete_overlap(op_events):
        for op1, op2 in zip(op_events, op_events[1:]):
            assert not op1.equals(op2)

    check_no_complete_overlap(op_events)

    op_stack = None
    events = []
    for i, op_event in enumerate(op_events):
        if op_stack is None:
            op_stack = OpStack(op_event)
        elif op_stack.ktime.subsumes(op_event):
            op_stack.insert(op_event)
        else:
            for event in op_stack.get_absored_ops():
                events.append(event)
            op_stack = OpStack(op_event)

    if op_stack is not None:
        for event in op_stack.get_absored_ops():
            events.append(event)

    if filter_by_op is not None:
        events = [event for event in events if event.name == filter_by_op]

    return events

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

    def _insert(self, op1):
        assert self.subsumes(op1)
        for op2 in self.sub_ops:
            if op2.subsumes(op1):
                op2._insert(op1)
                return
        if len(self.sub_ops) > 0:
            assert op1.ktime.is_after(self.sub_ops[-1].ktime)
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
            ktime = KernelTime(*args, **kwargs, name=self.name)
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
    from test.test_util import sec, T

    def test_01_1_stack():
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

    # Invalid input test
    def test_03_complete_overlap():
        op_events = [
            T(0, 1, 'op1'),
            T(0, 1, 'op2'),
        ]

        # Unfiltered:
        with pytest.raises(AssertionError):
            actual = process_op_nest(op_events)
        # expect = [
        #     T(0, 1, 'op2'),
        # ]
        # assert actual == expect
    test_03_complete_overlap()

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
