ALTER TABLE Event ADD CONSTRAINT Event_fk_01
  FOREIGN KEY (process_id) REFERENCES Process(process_id);
ALTER TABLE Event ADD CONSTRAINT Event_fk_02
  FOREIGN KEY(category_id) REFERENCES Category(category_id);
ALTER TABLE Event ADD CONSTRAINT Event_fk_03
  FOREIGN KEY(device_id) REFERENCES Device(device_id);

ALTER TABLE EventAttr ADD CONSTRAINT EventAttr_fk_01
  FOREIGN KEY(event_id) REFERENCES Event(event_id);

ALTER TABLE Category ADD CONSTRAINT Category_fk_01
  UNIQUE(category_name);

ALTER TABLE Process ADD CONSTRAINT Process_fk_01
  UNIQUE(process_name);

ALTER TABLE Device ADD CONSTRAINT Device_fk_01
  UNIQUE(device_name);
