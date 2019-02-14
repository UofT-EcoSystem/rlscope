# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: protobuf/pyprof.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='protobuf/pyprof.proto',
  package='iml.pyprof',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x15protobuf/pyprof.proto\x12\niml.pyprof\"\xb7\x02\n\x06Pyprof\x12\r\n\x05steps\x18\x01 \x03(\x03\x12;\n\rpython_events\x18\x02 \x03(\x0b\x32$.iml.pyprof.Pyprof.PythonEventsEntry\x12,\n\x05\x63libs\x18\x03 \x03(\x0b\x32\x1d.iml.pyprof.Pyprof.ClibsEntry\x12\x14\n\x0cprocess_name\x18\x04 \x01(\t\x12\r\n\x05phase\x18\x05 \x01(\x03\x1aM\n\x11PythonEventsEntry\x12\x0b\n\x03key\x18\x01 \x01(\x03\x12\'\n\x05value\x18\x02 \x01(\x0b\x32\x18.iml.pyprof.PythonEvents:\x02\x38\x01\x1a?\n\nClibsEntry\x12\x0b\n\x03key\x18\x01 \x01(\x03\x12 \n\x05value\x18\x02 \x01(\x0b\x32\x11.iml.pyprof.CLibs:\x02\x38\x01\"\xaf\x01\n\x05\x45vent\x12\x11\n\tthread_id\x18\x01 \x01(\x03\x12\x15\n\rstart_time_us\x18\x02 \x01(\x03\x12\x13\n\x0b\x64uration_us\x18\x03 \x01(\x03\x12\x0c\n\x04name\x18\x04 \x01(\t\x12+\n\x05\x61ttrs\x18\x05 \x03(\x0b\x32\x1c.iml.pyprof.Event.AttrsEntry\x1a,\n\nAttrsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"z\n\x05\x43Libs\x12+\n\x05\x63libs\x18\x01 \x03(\x0b\x32\x1c.iml.pyprof.CLibs.ClibsEntry\x1a\x44\n\nClibsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12%\n\x05value\x18\x02 \x01(\x0b\x32\x16.iml.pyprof.CLibEvents:\x02\x38\x01\"/\n\nCLibEvents\x12!\n\x06\x65vents\x18\x01 \x03(\x0b\x32\x11.iml.pyprof.Event\"1\n\x0cPythonEvents\x12!\n\x06\x65vents\x18\x01 \x03(\x0b\x32\x11.iml.pyprof.Event\"\xbe\x01\n\tAttrValue\x12\x0b\n\x01s\x18\x02 \x01(\x0cH\x00\x12\x0b\n\x01i\x18\x03 \x01(\x03H\x00\x12\x0b\n\x01\x66\x18\x04 \x01(\x02H\x00\x12\x0b\n\x01\x62\x18\x05 \x01(\x08H\x00\x12/\n\x04list\x18\x01 \x01(\x0b\x32\x1f.iml.pyprof.AttrValue.ListValueH\x00\x1a\x43\n\tListValue\x12\t\n\x01s\x18\x02 \x03(\x0c\x12\r\n\x01i\x18\x03 \x03(\x03\x42\x02\x10\x01\x12\r\n\x01\x66\x18\x04 \x03(\x02\x42\x02\x10\x01\x12\r\n\x01\x62\x18\x05 \x03(\x08\x42\x02\x10\x01\x42\x07\n\x05valueb\x06proto3')
)




_PYPROF_PYTHONEVENTSENTRY = _descriptor.Descriptor(
  name='PythonEventsEntry',
  full_name='iml.pyprof.Pyprof.PythonEventsEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='iml.pyprof.Pyprof.PythonEventsEntry.key', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='iml.pyprof.Pyprof.PythonEventsEntry.value', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=207,
  serialized_end=284,
)

_PYPROF_CLIBSENTRY = _descriptor.Descriptor(
  name='ClibsEntry',
  full_name='iml.pyprof.Pyprof.ClibsEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='iml.pyprof.Pyprof.ClibsEntry.key', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='iml.pyprof.Pyprof.ClibsEntry.value', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=286,
  serialized_end=349,
)

_PYPROF = _descriptor.Descriptor(
  name='Pyprof',
  full_name='iml.pyprof.Pyprof',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='steps', full_name='iml.pyprof.Pyprof.steps', index=0,
      number=1, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='python_events', full_name='iml.pyprof.Pyprof.python_events', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='clibs', full_name='iml.pyprof.Pyprof.clibs', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='process_name', full_name='iml.pyprof.Pyprof.process_name', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='phase', full_name='iml.pyprof.Pyprof.phase', index=4,
      number=5, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_PYPROF_PYTHONEVENTSENTRY, _PYPROF_CLIBSENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=38,
  serialized_end=349,
)


_EVENT_ATTRSENTRY = _descriptor.Descriptor(
  name='AttrsEntry',
  full_name='iml.pyprof.Event.AttrsEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='iml.pyprof.Event.AttrsEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='iml.pyprof.Event.AttrsEntry.value', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=483,
  serialized_end=527,
)

_EVENT = _descriptor.Descriptor(
  name='Event',
  full_name='iml.pyprof.Event',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='thread_id', full_name='iml.pyprof.Event.thread_id', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='start_time_us', full_name='iml.pyprof.Event.start_time_us', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='duration_us', full_name='iml.pyprof.Event.duration_us', index=2,
      number=3, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='name', full_name='iml.pyprof.Event.name', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='attrs', full_name='iml.pyprof.Event.attrs', index=4,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_EVENT_ATTRSENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=352,
  serialized_end=527,
)


_CLIBS_CLIBSENTRY = _descriptor.Descriptor(
  name='ClibsEntry',
  full_name='iml.pyprof.CLibs.ClibsEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='iml.pyprof.CLibs.ClibsEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='iml.pyprof.CLibs.ClibsEntry.value', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=583,
  serialized_end=651,
)

_CLIBS = _descriptor.Descriptor(
  name='CLibs',
  full_name='iml.pyprof.CLibs',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='clibs', full_name='iml.pyprof.CLibs.clibs', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_CLIBS_CLIBSENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=529,
  serialized_end=651,
)


_CLIBEVENTS = _descriptor.Descriptor(
  name='CLibEvents',
  full_name='iml.pyprof.CLibEvents',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='events', full_name='iml.pyprof.CLibEvents.events', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=653,
  serialized_end=700,
)


_PYTHONEVENTS = _descriptor.Descriptor(
  name='PythonEvents',
  full_name='iml.pyprof.PythonEvents',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='events', full_name='iml.pyprof.PythonEvents.events', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=702,
  serialized_end=751,
)


_ATTRVALUE_LISTVALUE = _descriptor.Descriptor(
  name='ListValue',
  full_name='iml.pyprof.AttrValue.ListValue',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='s', full_name='iml.pyprof.AttrValue.ListValue.s', index=0,
      number=2, type=12, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='i', full_name='iml.pyprof.AttrValue.ListValue.i', index=1,
      number=3, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\020\001'), file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='f', full_name='iml.pyprof.AttrValue.ListValue.f', index=2,
      number=4, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\020\001'), file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='b', full_name='iml.pyprof.AttrValue.ListValue.b', index=3,
      number=5, type=8, cpp_type=7, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\020\001'), file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=868,
  serialized_end=935,
)

_ATTRVALUE = _descriptor.Descriptor(
  name='AttrValue',
  full_name='iml.pyprof.AttrValue',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='s', full_name='iml.pyprof.AttrValue.s', index=0,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='i', full_name='iml.pyprof.AttrValue.i', index=1,
      number=3, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='f', full_name='iml.pyprof.AttrValue.f', index=2,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='b', full_name='iml.pyprof.AttrValue.b', index=3,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='list', full_name='iml.pyprof.AttrValue.list', index=4,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_ATTRVALUE_LISTVALUE, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='value', full_name='iml.pyprof.AttrValue.value',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=754,
  serialized_end=944,
)

_PYPROF_PYTHONEVENTSENTRY.fields_by_name['value'].message_type = _PYTHONEVENTS
_PYPROF_PYTHONEVENTSENTRY.containing_type = _PYPROF
_PYPROF_CLIBSENTRY.fields_by_name['value'].message_type = _CLIBS
_PYPROF_CLIBSENTRY.containing_type = _PYPROF
_PYPROF.fields_by_name['python_events'].message_type = _PYPROF_PYTHONEVENTSENTRY
_PYPROF.fields_by_name['clibs'].message_type = _PYPROF_CLIBSENTRY
_EVENT_ATTRSENTRY.containing_type = _EVENT
_EVENT.fields_by_name['attrs'].message_type = _EVENT_ATTRSENTRY
_CLIBS_CLIBSENTRY.fields_by_name['value'].message_type = _CLIBEVENTS
_CLIBS_CLIBSENTRY.containing_type = _CLIBS
_CLIBS.fields_by_name['clibs'].message_type = _CLIBS_CLIBSENTRY
_CLIBEVENTS.fields_by_name['events'].message_type = _EVENT
_PYTHONEVENTS.fields_by_name['events'].message_type = _EVENT
_ATTRVALUE_LISTVALUE.containing_type = _ATTRVALUE
_ATTRVALUE.fields_by_name['list'].message_type = _ATTRVALUE_LISTVALUE
_ATTRVALUE.oneofs_by_name['value'].fields.append(
  _ATTRVALUE.fields_by_name['s'])
_ATTRVALUE.fields_by_name['s'].containing_oneof = _ATTRVALUE.oneofs_by_name['value']
_ATTRVALUE.oneofs_by_name['value'].fields.append(
  _ATTRVALUE.fields_by_name['i'])
_ATTRVALUE.fields_by_name['i'].containing_oneof = _ATTRVALUE.oneofs_by_name['value']
_ATTRVALUE.oneofs_by_name['value'].fields.append(
  _ATTRVALUE.fields_by_name['f'])
_ATTRVALUE.fields_by_name['f'].containing_oneof = _ATTRVALUE.oneofs_by_name['value']
_ATTRVALUE.oneofs_by_name['value'].fields.append(
  _ATTRVALUE.fields_by_name['b'])
_ATTRVALUE.fields_by_name['b'].containing_oneof = _ATTRVALUE.oneofs_by_name['value']
_ATTRVALUE.oneofs_by_name['value'].fields.append(
  _ATTRVALUE.fields_by_name['list'])
_ATTRVALUE.fields_by_name['list'].containing_oneof = _ATTRVALUE.oneofs_by_name['value']
DESCRIPTOR.message_types_by_name['Pyprof'] = _PYPROF
DESCRIPTOR.message_types_by_name['Event'] = _EVENT
DESCRIPTOR.message_types_by_name['CLibs'] = _CLIBS
DESCRIPTOR.message_types_by_name['CLibEvents'] = _CLIBEVENTS
DESCRIPTOR.message_types_by_name['PythonEvents'] = _PYTHONEVENTS
DESCRIPTOR.message_types_by_name['AttrValue'] = _ATTRVALUE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Pyprof = _reflection.GeneratedProtocolMessageType('Pyprof', (_message.Message,), dict(

  PythonEventsEntry = _reflection.GeneratedProtocolMessageType('PythonEventsEntry', (_message.Message,), dict(
    DESCRIPTOR = _PYPROF_PYTHONEVENTSENTRY,
    __module__ = 'protobuf.pyprof_pb2'
    # @@protoc_insertion_point(class_scope:iml.pyprof.Pyprof.PythonEventsEntry)
    ))
  ,

  ClibsEntry = _reflection.GeneratedProtocolMessageType('ClibsEntry', (_message.Message,), dict(
    DESCRIPTOR = _PYPROF_CLIBSENTRY,
    __module__ = 'protobuf.pyprof_pb2'
    # @@protoc_insertion_point(class_scope:iml.pyprof.Pyprof.ClibsEntry)
    ))
  ,
  DESCRIPTOR = _PYPROF,
  __module__ = 'protobuf.pyprof_pb2'
  # @@protoc_insertion_point(class_scope:iml.pyprof.Pyprof)
  ))
_sym_db.RegisterMessage(Pyprof)
_sym_db.RegisterMessage(Pyprof.PythonEventsEntry)
_sym_db.RegisterMessage(Pyprof.ClibsEntry)

Event = _reflection.GeneratedProtocolMessageType('Event', (_message.Message,), dict(

  AttrsEntry = _reflection.GeneratedProtocolMessageType('AttrsEntry', (_message.Message,), dict(
    DESCRIPTOR = _EVENT_ATTRSENTRY,
    __module__ = 'protobuf.pyprof_pb2'
    # @@protoc_insertion_point(class_scope:iml.pyprof.Event.AttrsEntry)
    ))
  ,
  DESCRIPTOR = _EVENT,
  __module__ = 'protobuf.pyprof_pb2'
  # @@protoc_insertion_point(class_scope:iml.pyprof.Event)
  ))
_sym_db.RegisterMessage(Event)
_sym_db.RegisterMessage(Event.AttrsEntry)

CLibs = _reflection.GeneratedProtocolMessageType('CLibs', (_message.Message,), dict(

  ClibsEntry = _reflection.GeneratedProtocolMessageType('ClibsEntry', (_message.Message,), dict(
    DESCRIPTOR = _CLIBS_CLIBSENTRY,
    __module__ = 'protobuf.pyprof_pb2'
    # @@protoc_insertion_point(class_scope:iml.pyprof.CLibs.ClibsEntry)
    ))
  ,
  DESCRIPTOR = _CLIBS,
  __module__ = 'protobuf.pyprof_pb2'
  # @@protoc_insertion_point(class_scope:iml.pyprof.CLibs)
  ))
_sym_db.RegisterMessage(CLibs)
_sym_db.RegisterMessage(CLibs.ClibsEntry)

CLibEvents = _reflection.GeneratedProtocolMessageType('CLibEvents', (_message.Message,), dict(
  DESCRIPTOR = _CLIBEVENTS,
  __module__ = 'protobuf.pyprof_pb2'
  # @@protoc_insertion_point(class_scope:iml.pyprof.CLibEvents)
  ))
_sym_db.RegisterMessage(CLibEvents)

PythonEvents = _reflection.GeneratedProtocolMessageType('PythonEvents', (_message.Message,), dict(
  DESCRIPTOR = _PYTHONEVENTS,
  __module__ = 'protobuf.pyprof_pb2'
  # @@protoc_insertion_point(class_scope:iml.pyprof.PythonEvents)
  ))
_sym_db.RegisterMessage(PythonEvents)

AttrValue = _reflection.GeneratedProtocolMessageType('AttrValue', (_message.Message,), dict(

  ListValue = _reflection.GeneratedProtocolMessageType('ListValue', (_message.Message,), dict(
    DESCRIPTOR = _ATTRVALUE_LISTVALUE,
    __module__ = 'protobuf.pyprof_pb2'
    # @@protoc_insertion_point(class_scope:iml.pyprof.AttrValue.ListValue)
    ))
  ,
  DESCRIPTOR = _ATTRVALUE,
  __module__ = 'protobuf.pyprof_pb2'
  # @@protoc_insertion_point(class_scope:iml.pyprof.AttrValue)
  ))
_sym_db.RegisterMessage(AttrValue)
_sym_db.RegisterMessage(AttrValue.ListValue)


_PYPROF_PYTHONEVENTSENTRY._options = None
_PYPROF_CLIBSENTRY._options = None
_EVENT_ATTRSENTRY._options = None
_CLIBS_CLIBSENTRY._options = None
_ATTRVALUE_LISTVALUE.fields_by_name['i']._options = None
_ATTRVALUE_LISTVALUE.fields_by_name['f']._options = None
_ATTRVALUE_LISTVALUE.fields_by_name['b']._options = None
# @@protoc_insertion_point(module_scope)
