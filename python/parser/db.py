import os
import re
import subprocess

import sqlite3
from sqlite3 import Error

from parser.common import ProfilerParserCommonMixin
from os.path import join as _j, abspath as _a, dirname as _d, exists as _e, basename as _b

# from proto.tensorflow.core.profiler.tfprof_log_pb2 import ProfileProto
from tensorflow.core.profiler.tfprof_log_pb2 import ProfileProto
from proto.protobuf.pyprof_pb2 import Pyprof

import py_config

from parser.readers import TFProfCategoryTimesReader

import contextlib

from parser.common import *

from parser.stats import category_times_add_time

from parser.stats import KernelTime

TABLE_SQL = _j(py_config.ROOT, "sqlite", "tables.sql")

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
        # self.parser = parser
        # self.args = args
        self.directory = directory
        self.conn = TracesSQLiteConnection(self.db_path)
        self.debug = debug

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

        with transaction(self.conn):
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

        self.conn.close()

    def insert_tfprof_file(self, path):

        reader = TFProfCategoryTimesReader(path)

        if self.debug:
            print("> Insert tfprof file: {p}".format(p=path))
            reader.print(sys.stdout)

        process_id = self.insert_process_name(reader.process_name)
        for device, event in reader.all_events():
            category, start_time_us, duration_us, name = event
            category_id = self.insert_category_name(category)
            if category == 'GPU' and self.debug:
                print("> category = {c}, duration_us = {duration_us}".format(
                    c=category,
                    duration_us=duration_us))
            device_id = self.insert_device_name(device)
            self.conn.insert_dict('Event', {
                # 'thread_id':event.thread_id,
                'start_time_us':start_time_us,
                'duration_us':duration_us,
                'event_name':name,
                'category_id':category_id,
                'process_id':process_id,
                'device_id':device_id,
            })

        self.conn.commit()

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

        if self.debug:
            print("> Insert pyprof file: {p}".format(p=path))
            print(proto)

        c = self.cursor

        # Insert Process
        process_id = self.insert_process_name(proto.process_name)
        # Insert Category
        # categories = set()
        # for step, clibs in proto.clibs.items():
        #     for category, clib_events in clibs.clibs.items():
        # categories = list(proto.clibs.keys())
        # for category in categories:
        #     self.insert_category_name(category)

        def insert_category_events(category, events):
            categories.add(category)
            category_id = self.insert_category_name(category)
            # category_id = self.category_to_id[category]
            for event in events:
                # Insert Event
                self.conn.insert_dict('Event', {
                    'thread_id':event.thread_id,
                    'start_time_us':event.start_time_us,
                    'duration_us':event.duration_us,
                    'event_name':event.name,
                    'category_id':category_id,
                    'process_id':process_id,
                })
                event_id = c.lastrowid
                # Insert EventAttr
                for attr_name, attr_value in event.attrs.items():
                    self.conn.insert_dict('EventAttr', {
                        'event_id':event_id,
                        'attr_name':attr_name,
                        'attr_value':attr_value,
                    })
                    attr_id = c.lastrowid

        categories = set()

        for step, python_events in proto.python_events.items():
            insert_category_events(CATEGORY_PYTHON, python_events.events)

        for step, clibs in proto.clibs.items():
            for category, clib_events in clibs.clibs.items():
                insert_category_events(category, clib_events.events)

                # categories.add(category)
                # category_id = self.insert_category_name(category)
                # # category_id = self.category_to_id[category]
                # for event in clib_events.events:
                #     # Insert Event
                #     self.conn.insert_dict('Event', {
                #         'thread_id':event.thread_id,
                #         'start_time_us':event.start_time_us,
                #         'duration_us':event.duration_us,
                #         'event_name':event.name,
                #         'category_id':category_id,
                #         'process_id':process_id,
                #     })
                #     event_id = c.lastrowid
                #     # Insert EventAttr
                #     for attr_name, attr_value in event.attrs.items():
                #         self.conn.insert_dict('EventAttr', {
                #             'event_id':event_id,
                #             'attr_name':attr_name,
                #             'attr_value':attr_value,
                #         })
                #         attr_id = c.lastrowid

        self.conn.commit()

    def create_db(self):
        self.conn.run_sql_file(self.db_path, TABLE_SQL)
        assert _e(self.db_path)

        # NOTE: This doesn't provide line numbers when things fail.
        #
        # with open(TABLE_SQL) as f:
        #     script = f.read()
        # self.cursor.executescript(script)

    @property
    def db_path(self):
        return traces_db_path(self.directory)

@contextlib.contextmanager
def transaction(conn):
    try:
        yield
    except Exception as e:
        raise
    conn.commit()

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

class TracesSQLiteConnection:
    def __init__(self, db_path):
        self.db_path = db_path
        self._cursor = None
        self.conn = None

    def commit(self):
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
        sorted_cols = sorted(dic.keys())
        placeholders = ','.join('?' * len(sorted_cols))
        colnames = ','.join(sorted_cols)
        values = [dic[col] for col in sorted_cols]
        c.execute("INSERT INTO {table} ({colnames}) VALUES ({placeholders})".format(
            placeholders=placeholders,
            table=table,
            colnames=colnames,
        ), values)

    def create_connection(self):
        """ create a database connection to a SQLite database """
        if self.conn is not None:
            return

        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._cursor = self.conn.cursor()
        # print(sqlite3.version)

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

        self._steps = dict()

    def steps(self, bench_name):
        return list(range(self.num_steps(bench_name)))

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

    def _fetch_steps(self, bench_name):
        if bench_name in self._steps:
            return

        c = self.conn.cursor
        c.execute("""
        SELECT e.event_name, e.start_time_us, e.duration_us FROM Event AS e NATURAL JOIN Category AS c
        WHERE c.category_name = '{op}' AND
              e.event_name = ?
        ORDER BY e.start_time_us ASC 
        """.format(op=CATEGORY_OPERATION),
                  (bench_name,))
        rows = rows_as_ktime(c.fetchall())
        self._steps[bench_name] = rows

    def num_steps(self, bench_name):
        """
        We don't record step numbers in the database.
        Instead, steps are an index into the i-th time this operation occurs in the entire ML-script.

        We tend to want to skip the 1-st time the operation occurs / is profiled, since
        it will include load-time overheads (libcupti).
        """
        self._fetch_steps(bench_name)
        return len(self._steps[bench_name])

    def step_event(self, step, bench_name):
        self._fetch_steps(bench_name)
        return self._steps[bench_name][step]

    def parse(self, step, bench_name,
              group_by_device=False, include_dummy=False, debug=False):
        """
        # PSEUDOCODE:
        rows = SELECT category_name, start_time_us, duration_us FROM Category NATURAL JOIN Event
        for category_name, start_time_us, duration_us in rows:
            category_times[category_name].append((start_time_us, duration_us))

        :param bench_name:
        :return:
        """
        assert bench_name != NO_BENCH_NAME
        n_steps = self.num_steps(bench_name)
        assert 0 <= step < n_steps

        op_event = self.step_event(step, bench_name)

        if debug:
            print("> step={step}, op={bench}, time={time}".format(
                step=step, bench=bench_name, time=op_event))

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
          that surrounds its start time E.start.us.
        - However, currently we only ever want to select a SINGLE "step" at a time, 
          so we aren't using this.
        """
        dummy_where = ""
        if not include_dummy:
            dummy_where = textwrap.dedent("""\
            AND ( c_out.category_name != '{CATEGORY_DUMMY_EVENT}' )\
            """.format(CATEGORY_DUMMY_EVENT=CATEGORY_DUMMY_EVENT))

        query = textwrap.dedent("""
        SELECT device_name, category_name, event_name, start_time_us, duration_us
        FROM 
          Category AS c_out 
          NATURAL JOIN Event as e_out
          NATURAL LEFT JOIN Device as d_out
        WHERE ( ? <= e_out.start_time_us AND e_out.start_time_us <= ? )
        {dummy_where}
        AND ( c_out.category_name != '{CATEGORY_OPERATION}' )
        ORDER BY start_time_us ASC 
        """.format(
            dummy_where=dummy_where,
            CATEGORY_OPERATION=CATEGORY_OPERATION,
        ))
        if debug:
            print("> {name} query:".format(name=self.__class__.__name__))
            print(query)
        c.execute(query, (op_event.start_time_usec, op_event.end_time_usec))
        rows = c.fetchall()
        category_times = dict()
        for row in rows:
            ktime = row_as_ktime(row)
            category_times_add_time(category_times, row['device_name'], ktime, group_by_device, category=row['category_name'])

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