-- These tables map directly to the protobuf data format we use for data collection.
--
-- see: protobuf/pyprof.proto

CREATE TABLE Event (
    event_id SERIAL PRIMARY KEY,
    thread_id BIGINT,
    start_time_us BIGINT NOT NULL,
    end_time_us BIGINT NOT NULL,
    duration_us BIGINT NOT NULL,
    event_name TEXT,

    category_id INTEGER NOT NULL,

    -- Not present in protobuf (currently);
    -- however to support tracing multiple processes, we will need this.
    process_id INTEGER NOT NULL,

    -- This is specific to tfprof.
    -- pyprof doesn't use it.
    --
    -- To mimic TensorFlow, TraceEvents should shows the same
    -- category with different device_id in a different "section".
    device_id INTEGER,

    -- NOTE: This is a boolean, but SQLite just uses 0/1 integers for those.
    is_debug_event INTEGER NOT NULL

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
  process_name TEXT
--   UNIQUE (process_name)
);

CREATE TABLE Device (
  device_id SERIAL NOT NULL PRIMARY KEY,
  device_name TEXT
--   UNIQUE (device_name)
);

