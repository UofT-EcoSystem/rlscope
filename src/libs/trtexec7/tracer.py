#!/usr/bin/env python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

'''
Print a trtexec timing trace from a JSON file

Given a JSON file containing a trtexec timing trace,
this program prints the trace in CSV table format.
Each row represents an entry point in the trace.

The columns, as indicated by the header, respresent
one of the metric recorded. The output format can
be optionally converted to a format suitable for
GNUPlot.
'''

import sys
import json
import argparse

# import csv
import pandas as pd
import re
import logging
logger = logging.getLogger(__name__)

timestamps = ['startInMs', 'endInMs', 'startComputeMs', 'endComputeMs', 'startOutMs', 'endOutMs']

intervals = ['inMs', 'computeMs', 'outMs', 'latencyMs', 'endToEndMs']

other_metrics = ['stream']

allMetrics = timestamps + intervals + other_metrics

defaultMetrics = ",".join(allMetrics)

descriptions = ['start input', 'end input', 'start compute', 'end compute', 'start output',
                'end output', 'input', 'compute', 'output', 'latency', 'end to end latency']

# metricsDescription = pu.combineDescriptions('Possible metrics (all in ms) are:',
#                                              allMetrics, descriptions)



# def skipTrace(trace, start):
#     ''' Skip trace entries until start time '''
#
#     for t in range(len(trace)):
#         if trace[t]['startComputeMs'] >= start:
#             return trace[t:]
#
#     return []
#
#
#
# def hasTimestamp(metrics):
#     ''' Check if features have at least one timestamp '''
#
#     for timestamp in timestamps:
#         if timestamp in metrics:
#             return True
#     return False;
#
#
#
# def avgData(data, avg, times):
#     ''' Average trace entries (every avg entries) '''
#
#     averaged = []
#     accumulator = []
#     r = 0
#
#     for row in data:
#         if r == 0:
#             for m in row:
#                 accumulator.append(m)
#         else:
#             for m in row[times:]:
#                 accumulator[t] += m
#
#         r += 1
#         if r == avg:
#             for t in range(times, len(row)):
#                 accumulator[t] /= avg
#             averaged.append(accumulator)
#             accumulator = []
#             r = 0
#
#     return averaged

# class ChromeTracerWriter:
#     def __init__(self):

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name', metavar='filename', help='Trace file.')
    args = parser.parse_args()

    with open(args.name) as f:
        trace = json.load(f)

    if len(trace) != 0:
        header = set(trace[0].keys())
        for i, record in enumerate(trace):
            missing_keys = set(header).difference(record.keys())
            if len(missing_keys) > 0:
                parser.error("{path} was missing some fields in record {i}: {fields}".format(
                    i=i,
                    path=args.name,
                    fields=missing_keys,
                ))
        header = sorted(header)
        # writer = csv.writer(sys.stdout)
        # writer.writerow(header)
        data = dict()
        for i, record in enumerate(trace):
            for k in header:
                if k not in data:
                    data[k] = []
                data[k].append(record[k])
            # row = [record[k] for k in header]
            # writer.writerow(row)
        df = pd.DataFrame(data=data)
        csv_path = re.sub(r'\.json$', '.csv', args.name)
        assert csv_path != args.name

        ordered_cols = timestamps
        idx_map = dict()
        for i, colname in enumerate(ordered_cols):
            idx_map[colname] = i
        def column_key(colname):
            if colname in idx_map:
                # Ordered columns appear in-order.
                return (idx_map[colname], colname)
            # Remaining unordered columns appear in lexicographic order.
            return (len(idx_map), colname)
        cols = list(df.keys())
        cols.sort(key=column_key)
        logger.info(f"cols = {cols}")
        df = df[cols]
        if "startInMs" in df:
            df.sort_values(by=["startInMs"], inplace=True)
        df.to_csv(csv_path, index=False)
        logger.info("Output csv to {path}".format(path=csv_path))
    else:
        logger.info("JSON trace file @ {path} was empty; didn't output anything".format(path=args.name))
        sys.exit(1)

def log_msg(tag, msg):
    return f"[{tag}] {msg}"

def trt_log_msg(msg):
    return log_msg('TRT', msg)

def setup_logging():
    format = 'PID={process}/{processName} @ {funcName}, {filename}:{lineno} :: {asctime} {levelname}: {message}'
    logging.basicConfig(format=format, style='{')
    logger.setLevel(logging.INFO)

if __name__ == '__main__':
    sys.exit(main())
