-- These tables map directly to the protobuf data format we use for data collection.
--
-- see: protobuf/pyprof.proto

CREATE TABLE Event (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id INTEGER,
    start_time_us INTEGER NOT NULL,
    duration_us INTEGER NOT NULL,
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

    FOREIGN KEY(process_id) REFERENCES Process(process_id),
    FOREIGN KEY(category_id) REFERENCES Category(category_id),
    FOREIGN KEY(device_id) REFERENCES Device(device_id)
);

CREATE TABLE EventAttr (
  event_id INTEGER NOT NULL,
  attr_name TEXT NOT NULL,
  attr_value TEXT NOT NULL,
  FOREIGN KEY(event_id) REFERENCES Event(event_id),
  PRIMARY KEY(event_id, attr_name)
);

CREATE TABLE Category (
  category_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  category_name TEXT
);

CREATE TABLE Process (
  process_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  process_name TEXT,
  UNIQUE (process_name)
);

CREATE TABLE Device (
  device_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  device_name TEXT,
  UNIQUE (device_name)
);
