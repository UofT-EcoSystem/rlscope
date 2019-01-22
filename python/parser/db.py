import os
import re
import subprocess

import sqlite3
from sqlite3 import Error

from parser.common import ProfilerParserCommonMixin
from os.path import join as _j, abspath as _a, dirname as _d, exists as _e, basename as _b

from proto.tensorflow.core.profiler.tfprof_log_pb2 import ProfileProto
from proto.protobuf.pyprof_pb2 import Pyprof

import py_config

from parser.tfprof import TFProfCategoryTimesReader

from parser.common import *

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
        self.conn = None
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

        self.create_db()
        self.create_connection()

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

        self.cursor.close()
        self.conn.commit()
        self.conn.close()

    def insert_tfprof_file(self, path):
        reader = TFProfCategoryTimesReader(path)
        process_id = self.insert_process_name(reader.process_name)
        for device, event in reader.all_events():
            category, start_us, duration_us, name = event
            category_id = self.insert_category_name(category)
            device_id = self.insert_device_name(reader.process_name)
            self.insert_dict('Event', {
                # 'thread_id':event.thread_id,
                'start_time_us':start_us,
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
            self.insert_dict(table, {
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

        categories = set()
        for step, clibs in proto.clibs.items():
            for category, clib_events in clibs.clibs.items():
                categories.add(category)
                category_id = self.insert_category_name(category)
                # category_id = self.category_to_id[category]
                for event in clib_events.events:
                    # Insert Event
                    self.insert_dict('Event', {
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
                        self.insert_dict('EventAttr', {
                            'event_id':event_id,
                            'attr_name':attr_name,
                            'attr_value':attr_value,
                        })
                        attr_id = c.lastrowid

        self.conn.commit()

    def insert_dict(self, table, dic):
        c = self.cursor
        sorted_cols = sorted(dic.keys())
        placeholders = ','.join('?' * len(sorted_cols))
        colnames = ','.join(sorted_cols)
        values = [dic[col] for col in sorted_cols]
        c.execute("INSERT INTO {table} ({colnames}) VALUES ({placeholders})".format(
            placeholders=placeholders,
            table=table,
            colnames=colnames,
        ), values)

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
                print(proc.stdout)
            if proc.stderr is not None:
                print(proc.stderr)
            print("ERROR: failed to run sql file @ {path}; ret={ret}".format(
                ret=proc.returncode, path=db_path))
            sys.exit(1)

    def create_db(self):
        self.run_sql_file(self.db_path, TABLE_SQL)
        assert _e(self.db_path)

        # NOTE: This doesn't provide line numbers when things fail.
        #
        # with open(TABLE_SQL) as f:
        #     script = f.read()
        # self.cursor.executescript(script)

    def create_connection(self):
        """ create a database connection to a SQLite database """
        if self.conn is not None:
            return

        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        print(sqlite3.version)

    @property
    def db_path(self):
        return _j(self.directory, "traces.db")

def is_tfprof_file(path):
    base = _b(path)
    m = re.search(r'profile{bench}{trace}.proto'.format(
        bench=BENCH_SUFFIX_RE,
        trace=TRACE_SUFFIX_RE,
    ), base)
    return m

def is_pyprof_file(path):
    base = _b(path)
    m = re.search(r'pyprof{bench}{trace}.proto'.format(
        bench=BENCH_SUFFIX_RE,
        trace=TRACE_SUFFIX_RE,
    ), base)
    return m

def is_config_file(path):
    base = _b(path)
    m = re.search(r'config{bench}{trace}.json'.format(
        bench=BENCH_SUFFIX_RE,
        trace=TRACE_SUFFIX_RE,
    ), base)
    return m
