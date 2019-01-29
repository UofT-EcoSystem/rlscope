-- Indexes for "class SQLiteCategoryTimesReader"
CREATE INDEX index_start_time_us ON Event (start_time_us);
CREATE INDEX index_end_time_us ON Event (end_time_us);
CREATE INDEX index_category_name ON Category (category_name);
