"""
Indexing output files from ``rls-analyze`` to make it easier to select *.venn_js.json files.
"""
from rlscope.profiler.rlscope_logging import logger
import re
import copy
import numpy as np
import pprint
import json
import textwrap

from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b

TITLE_ORDER = ['overlap_type', 'machine', 'process', 'phase', 'resource_overlap', 'operation']
SEL_ORDER = {
    'CategoryOverlap': ['machine', 'process', 'phase', 'resource_overlap', 'operation'],
    'ResourceOverlap': ['machine', 'process', 'phase'],
    'ResourceSubplot': ['machine', 'process', 'phase'],
    'OperationOverlap': ['machine', 'process', 'phase', 'resource_overlap'],
    'HeatScale': ['machine_name', 'device_name'],
    'ProfilingOverhead': ['machine', 'process', 'phase'],
}
KEY_TYPE = {
    'CategoryOverlap': ['category'],
    'ResourceOverlap': ['resource_overlap'],
    'ResourceSubplot': ['resource_overlap'],
    'OperationOverlap': ['operation'],
}
# Sadly, we don't currently handle "correcting" ResourceSubplot.
# Total time is really [CPU, GPU]; to correct this we should handle it like we handle ResourceOverlap:
# First subtract from the largest time (CPU), then subtract remaining stuff with CPU in it i.e. "Total"/[CPU, GPU].
OVERLAP_PLOT_TYPES = set(SEL_ORDER.keys()).difference(['HeatScale', 'ResourceSubplot'])

class _DataIndex:
    """
    Access pre-generated venn_js JSON files needed for plotting stuff.

    Useful functionality provided by this class:
    - iterate through available venn_js files belonging to a "subtree"
      of the files in a consistent order;
      e.g.
      - all CatergoryOverlap files
      - all CatergoryOverlap files belonging to a given process
      - all CatergoryOverlap files belonging to a given (process, phase)
      - etc.
    """
    def __init__(self, index, directory, debug=False):
        self.index = index
        self.directory = directory
        self.debug = debug

    def set_debug(self, debug):
        self.debug = debug

    def each_file(self, selector, skip_missing_fields=False, debug=False):
        """
        :param selector:
            Sub-select the "sub-tree" of files you wish to consider.
            {
            'overlap_type': 'CategoryOverlap',
            'process': 'loop_train_eval',
            'phase': 'evaluate',
            'resource_overlap': ['CPU'],
            }
        :return:
        """
        if 'overlap_type' in selector:
            field = 'overlap_type'
        elif 'plot_type' in selector:
            field = 'plot_type'
        else:
            assert False

        for md, entry in _sel_idx(self.index, selector, field, skip_missing_fields=skip_missing_fields, debug=debug):
            md = dict(md)
            md[field] = selector[field]
            sel_order = SEL_ORDER[selector[field]]
            if self.debug:
                pprint.pprint({'entry':entry})
            ident = self.md_id(md, sel_order, skip_missing_fields=skip_missing_fields)
            new_entry = self._adjust_paths(entry)
            if self.debug:
                pprint.pprint({'md':md, 'entry':new_entry, 'ident':ident})
            yield md, new_entry, ident

    def _adjust_paths(self, old_entry):
        entry = dict(old_entry)
        for key in list(entry.keys()):
            if re.search(r'_path$', key):
                entry[key] = _j(self.directory, entry[key])
        return entry

    def get_files(self, selector, skip_missing_fields=False, debug=False):
        return list(self.each_file(selector, skip_missing_fields=skip_missing_fields, debug=debug))

    def available_values(self, selector, field, can_ignore=False, skip_missing_fields=False, debug=False):
        files = self.get_files(selector, skip_missing_fields=skip_missing_fields, debug=debug)
        if len(files) > 0:
            md, entry, indent = files[0]
            if field not in md:
                if can_ignore:
                    return []
                choices = sorted(md.keys())
                raise ValueError((
                    "_DataIndex.available_values(selector={sel}, field={field}): "
                    "couldn't find \"{field}\"; choices for field are {choices}"
                ).format(
                    field=field,
                    choices=choices,
                    sel=selector,
                ))
        values = sorted(set([md[field] for md, entry, ident in files]))
        return values

    def get_file(self, selector, skip_missing_fields=False, debug=False):
        files = list(self.each_file(selector, skip_missing_fields=skip_missing_fields, debug=debug))
        assert len(files) == 1
        return files[0]

    def read_metadata(self, selector, path_field='venn_js_path'):
        md, entry, ident = self.get_file(selector)
        path = entry[path_field]
        with open(path) as f:
            js = json.load(f)
        metadata = js['metadata']
        return metadata

    def get_title(self, md, total_sec=None):
        """
        Plot:{}, Process:{proc}, Phase:{phase}, ResourceOverlap:{resource_overlap}, Operation:{op}, [TotalSec:{total_sec}]

        :param md:
        :return:
        """

        def pretty_field(field):
            if field == 'overlap_type':
                return 'Plot'
            pretty = field

            # pretty = re.sub(r'(^[a-z])', r'\u', pretty)
            # pretty = re.sub(r'(^[a-z])', r'\u$1', pretty)
            # pretty = re.sub(r'(:?_([a-z]))', r'\u$1', pretty)

            def to_upper(match):
                return match.group(1).upper()
            pretty = re.sub(r'(^[a-z])', to_upper, pretty)
            pretty = re.sub(r'(^[a-z])', to_upper, pretty)
            pretty = re.sub(r'(:?_([a-z]))', to_upper, pretty)

            return pretty

        value_sep = ', '
        def pretty_value(value):
            if type(value) in {list, tuple, set, frozenset}:
                return "[" + value_sep.join(sorted(value)) + "]"

            return str(value)

        sub_title_sep = ', '

        fields = [field for field in TITLE_ORDER if field in md]
        values = [md[field] for field in fields]
        if total_sec is not None:
            fields.append('TotalSec')
            values.append(np.round(total_sec, 2))
        title = sub_title_sep.join([
            "{field}:{value}".format(
                field=pretty_field(field),
                value=pretty_value(value),
            ) for field, value in zip(fields, values)
        ])

        return title

    def md_id(self, md, sel_order, skip_missing_fields=False):
        """
        Valid <div id=*> format for id's in HTML:

        "
        The value must not contain any space characters.
        HTML 4: ID and NAME tokens must begin with a letter ([A-Za-z])
        and may be followed by any number of
        letters, digits ([0-9]), hyphens ("-"), underscores ("_"), colons (":"), and periods (".").
        "

               field_value_sep                       sub_id_sep
               -                                     -

        process_selfplay_worker_0-phase_default_phase-resources_CPU_GPU-operation_q_forward

        ------- -----------------                                  -
        field   value                                              value_sep

        -------------------------
        sub_id

        :param md:
        :param sel_order:
        :return:
        """
        for field in sel_order:
            if self.debug:
                pprint.pprint({'field': field, 'md': md, 'sel_order':sel_order})
            if not skip_missing_fields:
                assert field in md

        field_value_sep = '_'
        sub_id_sep = '-'
        value_sep = '_'

        def value_str(value):
            if type(value) in {list, tuple, frozenset}:
                return value_sep.join([str(x) for x in sorted(value)])
            return str(value)

        def sub_id(field):
            return "{field}{field_value_sep}{val}".format(
                field=field,
                field_value_sep=field_value_sep,
                val=value_str(md[field]),
            )

        def id_str():
            return sub_id_sep.join([sub_id(field) for field in sel_order if field in md])

        ident = id_str()

        return ident

def _sel_idx(idx, selector, field, skip_missing_fields=False, debug=False):
    """
    Generalize this code across all the different OverlapType's:

    # Sub-selecting venn_js_path's from CategoryOverlap data.
    md = dict()
    for overlap_type, process_data in sel(self.index, 'overlap_type'):
        md['overlap_type'] = overlap_type
        for process, phase_data in sel(process_data, 'process'):
            md['process'] = process
            for phase, resource_overlap_data in sel(phase_data, 'phase'):
                md['phase'] = phase
                for resource_overlap, operation_data in sel(resource_overlap_data, 'resource_overlap'):
                    md['resource_overlap'] = resource_overlap
                    for operation, operation_data in sel(operation_data, 'operation'):
                        md['operation'] = operation
                        # md = {
                        #     'overlap_type': overlap_type,
                        #     'process': process,
                        #     'phase': phase,
                        #     'resource_overlap': resource_overlap,
                        #     'operation': operation,
                        # }
                        yield dict(md), operation_data

    :param self:
    :param overlap_type:
    :return:
    """
    md = dict()
    level = 0
    for overlap, subtree in _sel(selector, idx, field, skip_missing_fields=skip_missing_fields, debug=debug):
        sel_order = SEL_ORDER[overlap]
        for md, entry in _sel_all(selector, sel_order, level, md, subtree, skip_missing_fields=skip_missing_fields, debug=debug):
            yield md, entry

def _sel(selector, idx, sel_field, skip_missing_fields=False, debug=False):
    """
    Given a subtree (idx) key-ed by sel_field values, iterate over sub-subtree whose values match selector.

    :param selector:
    :param idx:
        A subtree of the INDEX, where:
        keys = values of type 'sel_field'

        Initially when _sel is first called, the idx is INDEX, and the subtree is key-ed by plot-type (e.g. ResourceSubplot).

    :param sel_field:
        Initially when _sel is first called, sel_field = 'overlap_type'.

    :return:
    """

    if debug:
        logger.info("> _sel:")
        logger.info(textwrap.indent(pprint.pformat(locals()), prefix='  '))

    if sel_field in selector and callable(selector[sel_field]):
        for value, subtree in idx.items():
            if selector[sel_field](value):
                yield value, subtree
        return

    if sel_field in selector:
        value = selector[sel_field]
        if value not in idx:
            return
        subtree = idx[value]
        yield value, subtree
        return

    for value, subtree in idx.items():
        yield value, subtree


def _sel_all(selector, sel_order, level, md, subtree, skip_missing_fields=False, debug=False):
    """
    Given a subtree key-ed like:
    subtree = {
      sel_order[level]: {
          <sel_order[level] value>: ...
      }
    }
    Recursively iterate according to sel_order[i], using selector
    to decide which subtrees to visit at each level.

    :param selector:
    :param idx:
        A subtree of the INDEX, where:
        keys = values of type 'sel_field'

        Initially when _sel is first called, the idx is INDEX, and the subtree is key-ed by plot-type (e.g. ResourceSubplot).

    :param sel_field:
        Initially when _sel is first called, sel_field = 'overlap_type'.

    :return:
    """
    if debug:
        logger.debug(f"level = {level}")
    while True:
        if level == len(sel_order):
            yield dict(md), subtree
            return
        field = sel_order[level]

        if field not in subtree and skip_missing_fields:
            # Subtree is missing field, but there's only one choice of field-value to use.
            logger.warning("Skipping field={field} since it is missing".format(field=field))
            level += 1
        elif field in subtree:
            break
        else:
            raise RuntimeError("Didn't find field={field} in selector; options are {fields}".format(
                field=field,
                fields=sorted(subtree.keys()),
            ))

    for value, next_subtree in _sel(selector, subtree[field], field, skip_missing_fields=skip_missing_fields, debug=debug):
        md[field] = value
        for md, entry in _sel_all(selector, sel_order, level + 1, md, next_subtree, skip_missing_fields=skip_missing_fields, debug=debug):
            yield md, entry


class TestSel:

    def test_sel_01(self):
        INDEX = {
            'ResourceSubplot':
                {'process': {'loop_init': {'phase': {'bootstrap': {'venn_js_path': 'ResourceSubplot.process_loop_init.phase_bootstrap.venn_js.json'},
                                                                    'default_phase': {'venn_js_path': 'ResourceSubplot.process_loop_init.phase_default_phase.venn_js.json'}}},
                                            'loop_selfplay': {'phase': {'default_phase': {'venn_js_path': 'ResourceSubplot.process_loop_selfplay.phase_default_phase.venn_js.json'},
                                                                        'selfplay_workers': {'venn_js_path': 'ResourceSubplot.process_loop_selfplay.phase_selfplay_workers.venn_js.json'}}},
                                            'loop_train_eval': {'phase': {'default_phase': {'venn_js_path': 'ResourceSubplot.process_loop_train_eval.phase_default_phase.venn_js.json'},
                                                                          'evaluate': {'venn_js_path': 'ResourceSubplot.process_loop_train_eval.phase_evaluate.venn_js.json'},
                                                                          'sgd_updates': {'venn_js_path': 'ResourceSubplot.process_loop_train_eval.phase_sgd_updates.venn_js.json'}}},
                                            'selfplay_worker_0': {'phase': {'default_phase': {'venn_js_path': 'ResourceSubplot.process_selfplay_worker_0.phase_default_phase.venn_js.json'},
                                                                            'selfplay_worker_0': {'venn_js_path': 'ResourceSubplot.process_selfplay_worker_0.phase_selfplay_worker_0.venn_js.json'}}},
                                            'selfplay_worker_1': {'phase': {'default_phase': {'venn_js_path': 'ResourceSubplot.process_selfplay_worker_1.phase_default_phase.venn_js.json'},
                                                                            'selfplay_worker_1': {'venn_js_path': 'ResourceSubplot.process_selfplay_worker_1.phase_selfplay_worker_1.venn_js.json'}}}}},

        }

        for i, (value, subtree) in enumerate(_sel({'overlap_type':'ResourceSubplot'}, INDEX, 'overlap_type')):
            assert i == 0
            assert subtree == INDEX['ResourceSubplot']

        for i, (value, subtree) in enumerate(_sel(
                {'overlap_type':'ResourceSubplot', 'process':'loop_train_eval'},
                INDEX['ResourceSubplot']['process'],
                'process')):
            assert i == 0
            assert subtree == INDEX['ResourceSubplot']['process']['loop_train_eval']

        actual_entries = []
        actual_mds = []
        expected_entries = [
            {'venn_js_path': 'ResourceSubplot.process_loop_train_eval.phase_default_phase.venn_js.json'},
            {'venn_js_path': 'ResourceSubplot.process_loop_train_eval.phase_evaluate.venn_js.json'},
            {'venn_js_path': 'ResourceSubplot.process_loop_train_eval.phase_sgd_updates.venn_js.json'},
        ]
        md = dict()
        md['overlap_type'] = 'ResourceSubplot'
        # {'overlap_type':'ResourceSubplot', 'process':'loop_train_eval', 'phase':'evaluate'},
        for i, (md, entry) in enumerate(_sel_all(
                {'overlap_type':'ResourceSubplot', 'process':'loop_train_eval'},
                # SEL_ORDER['ResourceSubplot'],
                ['process', 'phase'],
                0,
                md,
                INDEX['ResourceSubplot'])):
            actual_entries.append(entry)
            actual_mds.append(md)

        # actual_entries.sort()
        # actual_mds.sort()
        for expected_entry in expected_entries:
            assert expected_entry in actual_entries

    def test_sel_02(self):
        INDEX = {
            'OperationOverlap': {
                'machine':
                    {'eco11': {
                        'process': {'dqn_PongNoFrameskip-v4': {
                            'phase': {
                                'dqn_PongNoFrameskip-v4': {
                                    'resource_overlap': {
                                        ('CPU',): {
                                            'venn_js_path': 'OperationOverlap.process_dqn_PongNoFrameskip-v4.phase_dqn_PongNoFrameskip-v4.resources_CPU.venn_js.json'},
                                        ('CPU', 'GPU'): {
                                            'venn_js_path': 'OperationOverlap.process_dqn_PongNoFrameskip-v4.phase_dqn_PongNoFrameskip-v4.resources_CPU_GPU.venn_js.json'}
                                    }}}}}}}}}
        data_index = _DataIndex(INDEX, "madeup_dir")
        selector = {
            'overlap_type':'OperationOverlap',
        }

        list_resource_overlap_types = set()
        files = list(data_index.each_file(selector))
        for md, entry, ident in files:
            list_resource_overlap_types.add(copy.copy(md['resource_overlap']))
        # pprint.pprint({'files':files})
        # print()

        foreach_resource_overlap_types = set()
        for md, entry, ident in data_index.each_file(selector):
            foreach_resource_overlap_types.add(copy.copy(md['resource_overlap']))
            # pprint.pprint({'md': md, 'entry': entry, 'ident': ident})

        expect_resource_overlap_types = {
            ('CPU',),
            ('CPU', 'GPU'),
        }
        assert foreach_resource_overlap_types == expect_resource_overlap_types

        assert list_resource_overlap_types == foreach_resource_overlap_types
