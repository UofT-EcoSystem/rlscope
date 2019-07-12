-- noinspection SqlNoDataSourceInspectionForFile

-- These tables map directly to the protobuf data format we use for data collection.
--
-- see: protobuf/pyprof.proto

CREATE TABLE Event (
    event_id SERIAL PRIMARY KEY,
    thread_id BIGINT,

--     start_time_us BIGINT NOT NULL,
--     end_time_us BIGINT NOT NULL,
--     duration_us BIGINT NOT NULL,

    -- Use float/double instead of integer since for python call-times,
    -- individual function-calls can take sub-microsecond time
    -- (e.g. list operations).
    -- TODO: read https://en.wikipedia.org/wiki/Denormal_number
    -- These numbers shouldn't be too small.
    -- Not sure about them being to large though...
--     start_time_us FLOAT NOT NULL,
--     end_time_us FLOAT NOT NULL,
--     duration_us FLOAT NOT NULL,

    -- We need nanosecond scale precision for short python profile function calls.
    --
    -- https://www.postgresql.org/docs/10/datatype-numeric.html#DATATYPE-FLOAT
    -- Suggestion: use numeric type to preserve precision.
    --
    -- Q: Can we specify needed decimal-digits precision / # of non-decimal-digits?
    -- Would that be faster?
    --
    -- Double precision is not good enough for nanosecond scale:
    --   "1550795540.17681"
    --   select 1550795540.1768074::double precision;
    --   "1550795540.17681"
    --   select 1550795540.176808::double precision;
    --   (1550795540.176808 − 1550795540.1768074)×1e9 = 600 ns...
    --   But postgres rounds these to the same number!
    --   That's why the start time of now_us() and time.time() are identical in the trace.
    --
    -- Numeric precision works!
    --   "1550795540.1768074"
    --   select 1550795540.1768074::numeric;
    --   "1550795540.176808"
    --   select 1550795540.176808::numeric;
    start_time_us NUMERIC NOT NULL,
    end_time_us NUMERIC NOT NULL,
    duration_us NUMERIC NOT NULL,

    -- A short name to show in the chrome://tracing view of Chromium web browser.
    event_name TEXT,

    category_id INTEGER NOT NULL,
    -- Detailed python profiler file/line-number/function information.
    -- e.g.
    --   ('Lib/test/my_test_profile.py', 225, 'helper'),
    --   ('~', 0, "<method 'append' of 'list' objects>"),
    pyprof_filename TEXT,
    pyprof_line_no INTEGER,
    pyprof_function TEXT,
    -- Concatenated version of; make it easier to do "LIKE" queries.
    pyprof_line_description TEXT,

    -- Not present in protobuf (currently);
    -- however to support tracing multiple processes, we will need this.
    process_id INTEGER NOT NULL,

    -- The "phase" this training event belongs to.
    -- The phase covered by a script may change during training.
    -- E.g. a single script could handle "simulator" and "gradient_update" phases.
    phase_id INTEGER NOT NULL,

    -- This is specific to tfprof.
    -- pyprof doesn't use it.
    --
    -- To mimic TensorFlow, TraceEvents should shows the same
    -- category with different device_id in a different "section".
    device_id INTEGER,

    -- NOTE: This is a boolean, but SQLite just uses 0/1 integers for those.
    is_debug_event BOOLEAN NOT NULL

--     FOREIGN KEY(process_id) REFERENCES Process(process_id),
--     FOREIGN KEY(category_id) REFERENCES Category(category_id),
--     FOREIGN KEY(device_id) REFERENCES Device(device_id)
);

CREATE TABLE EventAttr (
  event_id INTEGER NOT NULL,
  attr_name TEXT NOT NULL,
  attr_value TEXT NOT NULL,
--   FOREIGN KEY(event_id) REFERENCES Event(event_id),
  PRIMARY KEY(event_id, attr_name)
);

CREATE TABLE Category (
  category_id SERIAL NOT NULL PRIMARY KEY,
  category_name TEXT
--   UNIQUE (category_name)
);

CREATE TABLE Process (
  process_id SERIAL NOT NULL PRIMARY KEY,
  process_name TEXT,
  parent_process_id INTEGER,
  machine_id INTEGER NOT NULL,
  -- ProcessMetadata.TrainingProgress
  percent_complete FLOAT,
  num_timesteps INTEGER,
  total_timesteps INTEGER
--   UNIQUE (process_name)
--   FOREIGN KEY(parent_process_id) REFERENCES Process(process_id),
--   FOREIGN KEY(machine_id) REFERENCES Machine(machine_id);
);

-- NOTE: this allows many-to-many relationships between processes.
-- I might have done this with arbitrary RPC's in mind.
-- However, those won't be supported, so screw it.
-- CREATE TABLE ProcessDependency (
--   process_id_parent SERIAL NOT NULL PRIMARY KEY,
--   process_name_parent TEXT,
--   process_id_child SERIAL NOT NULL PRIMARY KEY,
--   process_name_child TEXT,
--   PRIMARY KEY(process_id_parent, process_id_child)
-- --   UNIQUE (process_name_parent, process_name_child)
-- );

CREATE TABLE Phase (
  phase_id SERIAL NOT NULL PRIMARY KEY,
  phase_name TEXT
--   UNIQUE (phase_name)
);

CREATE TABLE Device (
  device_id SERIAL NOT NULL PRIMARY KEY,
  device_name TEXT,
  machine_id INTEGER NOT NULL
--   UNIQUE (device_name)
--   FOREIGN KEY(machine_id) REFERENCES Machine(machine_id),
);

CREATE TABLE DeviceUtilization (
  device_id INTEGER NOT NULL,
  machine_id INTEGER NOT NULL,
  start_time_us NUMERIC NOT NULL,
  util FLOAT NOT NULL,
  total_resident_memory_bytes INTEGER NOT NULL;
);

CREATE TABLE Machine (
  machine_id SERIAL NOT NULL PRIMARY KEY,
  machine_name TEXT
--   UNIQUE (machine_name)
);
