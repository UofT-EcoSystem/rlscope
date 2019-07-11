-- Indexes for "class SQLCategoryTimesReader"
CREATE INDEX index_event_name ON Event (event_name);
CREATE INDEX index_is_debug_event ON Event (is_debug_event);
CREATE INDEX index_start_time_us ON Event (start_time_us);
CREATE INDEX index_end_time_us ON Event (end_time_us);
CREATE INDEX index_phase_id ON Event (phase_id);
CREATE INDEX index_category_name ON Category (category_name);
CREATE INDEX index_process_name ON Process (process_name);
CREATE INDEX index_machine_name ON Machine (machine_name);
CREATE INDEX index_start_time_us_end_time_us ON Event (start_time_us, end_time_us);
