import re
import numpy as np
import pprint
import json

from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b

import plot_index_data

TITLE_ORDER = ['overlap_type', 'process', 'phase', 'resource_overlap', 'operation']
SEL_ORDER = {
    'CategoryOverlap': ['process', 'phase', 'resource_overlap', 'operation'],
    'ResourceOverlap': ['process', 'phase'],
    'ResourceSubplot': ['process', 'phase'],
    'OperationOverlap': ['process', 'phase', 'resource_overlap'],
    'HeatScale': ['device_name'],
}

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

    def each_file(self, selector):
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

        for md, entry in _sel_idx(self.index, selector, field):
            md[field] = selector[field]
            sel_order = SEL_ORDER[selector[field]]
            if self.debug:
                pprint.pprint({'entry':entry})
            ident = self.md_id(md, sel_order)
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

    def get_file(self, selector):
        files = list(self.each_file(selector))
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

    def md_id(self, md, sel_order):
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
            return sub_id_sep.join([sub_id(field) for field in sel_order])

        ident = id_str()

        return ident

def _sel_idx(idx, selector, field):
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
    for overlap, subtree in _sel(selector, idx, field):
        sel_order = SEL_ORDER[overlap]
        for md, entry in _sel_all(selector, sel_order, level, md, subtree):
            yield md, entry

def _sel(selector, idx, sel_field):
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

def _sel_all(selector, sel_order, level, md, subtree):
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
    if level == len(sel_order):
        yield dict(md), subtree
        return
    field = sel_order[level]
    for value, next_subtree in _sel(selector, subtree[field], field):
        md[field] = value
        for md, entry in _sel_all(selector, sel_order, level + 1, md, next_subtree):
            yield md, entry


DataIndex = _DataIndex(plot_index_data.INDEX, plot_index_data.DIRECTORY)

# DIRECTORY = "/mnt/data/james/clone/baselines.checkpoints/checkpoints/minigo/vector_test_multiple_workers_k4000/to_index"
# INDEX = \
#     {'CategoryOverlap': {'process': {'loop_init': {'phase': {'bootstrap': {'resource_overlap': {('CPU',): {'operation': {'bootstrap': {'venn_js_path': 'CategoryOverlap.process_loop_init.phase_bootstrap.ops_bootstrap.venn_js.json'}}}}}}},
#                                      'loop_train_eval': {'phase': {'evaluate': {'resource_overlap': {('CPU',): {'operation': {'eval_game': {'venn_js_path': 'CategoryOverlap.process_loop_train_eval.phase_evaluate.ops_eval_game.venn_js.json'},
#                                                                                                                               'load_network': {'venn_js_path': 'CategoryOverlap.process_loop_train_eval.phase_evaluate.ops_load_network.venn_js.json'},
#                                                                                                                               'tree_search': {'venn_js_path': 'CategoryOverlap.process_loop_train_eval.phase_evaluate.ops_tree_search.venn_js.json'},
#                                                                                                                               'tree_search_loop': {'venn_js_path': 'CategoryOverlap.process_loop_train_eval.phase_evaluate.ops_tree_search_loop.venn_js.json'}}}}},
#                                                                    'sgd_updates': {'resource_overlap': {('CPU',): {'operation': {'estimator_save_model': {'venn_js_path': 'CategoryOverlap.process_loop_train_eval.phase_sgd_updates.ops_estimator_save_model.venn_js.json'},
#                                                                                                                                  'estimator_train': {'venn_js_path': 'CategoryOverlap.process_loop_train_eval.phase_sgd_updates.ops_estimator_train.venn_js.json'},
#                                                                                                                                  'gather': {'venn_js_path': 'CategoryOverlap.process_loop_train_eval.phase_sgd_updates.ops_gather.venn_js.json'},
#                                                                                                                                  'train': {'venn_js_path': 'CategoryOverlap.process_loop_train_eval.phase_sgd_updates.ops_train.venn_js.json'}}}}}}},
#                                      'selfplay_worker_0': {'phase': {'selfplay_worker_0': {'resource_overlap': {('CPU',): {'operation': {'load_network': {'venn_js_path': 'CategoryOverlap.process_selfplay_worker_0.phase_selfplay_worker_0.ops_load_network.venn_js.json'},
#                                                                                                                                          'selfplay_loop': {'venn_js_path': 'CategoryOverlap.process_selfplay_worker_0.phase_selfplay_worker_0.ops_selfplay_loop.venn_js.json'},
#                                                                                                                                          'tree_search': {'venn_js_path': 'CategoryOverlap.process_selfplay_worker_0.phase_selfplay_worker_0.ops_tree_search.venn_js.json'},
#                                                                                                                                          'tree_search_loop': {'venn_js_path': 'CategoryOverlap.process_selfplay_worker_0.phase_selfplay_worker_0.ops_tree_search_loop.venn_js.json'}}}}}}},
#                                      'selfplay_worker_1': {'phase': {'selfplay_worker_1': {'resource_overlap': {('CPU',): {'operation': {'load_network': {'venn_js_path': 'CategoryOverlap.process_selfplay_worker_1.phase_selfplay_worker_1.ops_load_network.venn_js.json'},
#                                                                                                                                          'selfplay_loop': {'venn_js_path': 'CategoryOverlap.process_selfplay_worker_1.phase_selfplay_worker_1.ops_selfplay_loop.venn_js.json'},
# DataIndex = _DataIndex(INDEX, DIRECTORY)

# DIRECTORY = "/mnt/data/james/clone/baselines.checkpoints/checkpoints/minigo/vector_test_multiple_workers_k4000/to_index"
INDEX = \
    {
        # 'CategoryOverlap': {'process': {'loop_init': {'phase': {'bootstrap': {'resource_overlap': {('CPU',): {'operation': {'bootstrap': {'venn_js_path': 'CategoryOverlap.process_loop_init.phase_bootstrap.ops_bootstrap.venn_js.json'}}}}}}},
        #                                 'loop_train_eval': {'phase': {'evaluate': {'resource_overlap': {('CPU',): {'operation': {'eval_game': {'venn_js_path': 'CategoryOverlap.process_loop_train_eval.phase_evaluate.ops_eval_game.venn_js.json'},
        #                                                                                                                          'load_network': {'venn_js_path': 'CategoryOverlap.process_loop_train_eval.phase_evaluate.ops_load_network.venn_js.json'},
        #                                                                                                                          'tree_search': {'venn_js_path': 'CategoryOverlap.process_loop_train_eval.phase_evaluate.ops_tree_search.venn_js.json'},
        #                                                                                                                          'tree_search_loop': {'venn_js_path': 'CategoryOverlap.process_loop_train_eval.phase_evaluate.ops_tree_search_loop.venn_js.json'}}}}},
        #                                                               'sgd_updates': {'resource_overlap': {('CPU',): {'operation': {'estimator_save_model': {'venn_js_path': 'CategoryOverlap.process_loop_train_eval.phase_sgd_updates.ops_estimator_save_model.venn_js.json'},
        #                                                                                                                             'estimator_train': {'venn_js_path': 'CategoryOverlap.process_loop_train_eval.phase_sgd_updates.ops_estimator_train.venn_js.json'},
        #                                                                                                                             'gather': {'venn_js_path': 'CategoryOverlap.process_loop_train_eval.phase_sgd_updates.ops_gather.venn_js.json'},
        #                                                                                                                             'train': {'venn_js_path': 'CategoryOverlap.process_loop_train_eval.phase_sgd_updates.ops_train.venn_js.json'}}}}}}},
        #                                 'selfplay_worker_0': {'phase': {'selfplay_worker_0': {'resource_overlap': {('CPU',): {'operation': {'load_network': {'venn_js_path': 'CategoryOverlap.process_selfplay_worker_0.phase_selfplay_worker_0.ops_load_network.venn_js.json'},
        #                                                                                                                                     'selfplay_loop': {'venn_js_path': 'CategoryOverlap.process_selfplay_worker_0.phase_selfplay_worker_0.ops_selfplay_loop.venn_js.json'},
        #                                                                                                                                     'tree_search': {'venn_js_path': 'CategoryOverlap.process_selfplay_worker_0.phase_selfplay_worker_0.ops_tree_search.venn_js.json'},
        #                                                                                                                                     'tree_search_loop': {'venn_js_path': 'CategoryOverlap.process_selfplay_worker_0.phase_selfplay_worker_0.ops_tree_search_loop.venn_js.json'}}}}}}},
        #                                 'selfplay_worker_1': {'phase': {'selfplay_worker_1': {'resource_overlap': {('CPU',): {'operation': {'load_network': {'venn_js_path': 'CategoryOverlap.process_selfplay_worker_1.phase_selfplay_worker_1.ops_load_network.venn_js.json'},
        #                                                                                                                                     'selfplay_loop': {'venn_js_path': 'CategoryOverlap.process_selfplay_worker_1.phase_selfplay_worker_1.ops_selfplay_loop.venn_js.json'},
        #                                                                                                                                     'tree_search': {'venn_js_path': 'CategoryOverlap.process_selfplay_worker_1.phase_selfplay_worker_1.ops_tree_search.venn_js.json'},
        #                                                                                                                                     'tree_search_loop': {'venn_js_path': 'CategoryOverlap.process_selfplay_worker_1.phase_selfplay_worker_1.ops_tree_search_loop.venn_js.json'}}}}}}}}},

        # 'OperationOverlap': {'process': {'loop_init': {'phase': {'bootstrap': {'resource_overlap': {('CPU',): {'venn_js_path': 'OperationOverlap.process_loop_init.phase_bootstrap.resources_CPU.venn_js.json'},
        #                                                                                             ('CPU', 'GPU'): {'venn_js_path': 'OperationOverlap.process_loop_init.phase_bootstrap.resources_CPU_GPU.venn_js.json'},
        #                                                                                             ('GPU',): {'venn_js_path': 'OperationOverlap.process_loop_init.phase_bootstrap.resources_GPU.venn_js.json'}}}}},
        #                                  'loop_selfplay': {'phase': {'default_phase': {'resource_overlap': {('CPU',): {'venn_js_path': 'OperationOverlap.process_loop_selfplay.phase_default_phase.resources_CPU.venn_js.json'},
        #                                                                                                     ('CPU', 'GPU'): {'venn_js_path': 'OperationOverlap.process_loop_selfplay.phase_default_phase.resources_CPU_GPU.venn_js.json'},
        #                                                                                                     ('GPU',): {'venn_js_path': 'OperationOverlap.process_loop_selfplay.phase_default_phase.resources_GPU.venn_js.json'}}},
        #                                                              'selfplay_workers': {'resource_overlap': {('CPU',): {'venn_js_path': 'OperationOverlap.process_loop_selfplay.phase_selfplay_workers.resources_CPU.venn_js.json'},
        #                                                                                                        ('CPU', 'GPU'): {'venn_js_path': 'OperationOverlap.process_loop_selfplay.phase_selfplay_workers.resources_CPU_GPU.venn_js.json'},
        #                                                                                                        ('GPU',): {'venn_js_path': 'OperationOverlap.process_loop_selfplay.phase_selfplay_workers.resources_GPU.venn_js.json'}}}}},
        #                                  'loop_train_eval': {'phase': {'default_phase': {'resource_overlap': {('CPU',): {'venn_js_path': 'OperationOverlap.process_loop_train_eval.phase_default_phase.resources_CPU.venn_js.json'},
        #                                                                                                       ('CPU', 'GPU'): {'venn_js_path': 'OperationOverlap.process_loop_train_eval.phase_default_phase.resources_CPU_GPU.venn_js.json'},
        #                                                                                                       ('GPU',): {'venn_js_path': 'OperationOverlap.process_loop_train_eval.phase_default_phase.resources_GPU.venn_js.json'}}},
        #                                                                'evaluate': {'resource_overlap': {('CPU',): {'venn_js_path': 'OperationOverlap.process_loop_train_eval.phase_evaluate.resources_CPU.venn_js.json'},
        #                                                                                                  ('CPU', 'GPU'): {'venn_js_path': 'OperationOverlap.process_loop_train_eval.phase_evaluate.resources_CPU_GPU.venn_js.json'},
        #                                                                                                  ('GPU',): {'venn_js_path': 'OperationOverlap.process_loop_train_eval.phase_evaluate.resources_GPU.venn_js.json'}}},
        #                                                                'sgd_updates': {'resource_overlap': {('CPU',): {'venn_js_path': 'OperationOverlap.process_loop_train_eval.phase_sgd_updates.resources_CPU.venn_js.json'},
        #                                                                                                     ('CPU', 'GPU'): {'venn_js_path': 'OperationOverlap.process_loop_train_eval.phase_sgd_updates.resources_CPU_GPU.venn_js.json'},
        #                                                                                                     ('GPU',): {'venn_js_path': 'OperationOverlap.process_loop_train_eval.phase_sgd_updates.resources_GPU.venn_js.json'}}}}},
        #                                  'selfplay_worker_0': {'phase': {'default_phase': {'resource_overlap': {('CPU',): {'venn_js_path': 'OperationOverlap.process_selfplay_worker_0.phase_default_phase.resources_CPU.venn_js.json'},
        #                                                                                                         ('CPU', 'GPU'): {'venn_js_path': 'OperationOverlap.process_selfplay_worker_0.phase_default_phase.resources_CPU_GPU.venn_js.json'},
        #                                                                                                         ('GPU',): {'venn_js_path': 'OperationOverlap.process_selfplay_worker_0.phase_default_phase.resources_GPU.venn_js.json'}}},
        #                                                                  'selfplay_worker_0': {'resource_overlap': {('CPU',): {'venn_js_path': 'OperationOverlap.process_selfplay_worker_0.phase_selfplay_worker_0.resources_CPU.venn_js.json'},
        #                                                                                                             ('CPU', 'GPU'): {'venn_js_path': 'OperationOverlap.process_selfplay_worker_0.phase_selfplay_worker_0.resources_CPU_GPU.venn_js.json'},
        #                                                                                                             ('GPU',): {'venn_js_path': 'OperationOverlap.process_selfplay_worker_0.phase_selfplay_worker_0.resources_GPU.venn_js.json'}}}}},
        #                                  'selfplay_worker_1': {'phase': {'default_phase': {'resource_overlap': {('CPU',): {'venn_js_path': 'OperationOverlap.process_selfplay_worker_1.phase_default_phase.resources_CPU.venn_js.json'},
        #                                                                                                         ('CPU', 'GPU'): {'venn_js_path': 'OperationOverlap.process_selfplay_worker_1.phase_default_phase.resources_CPU_GPU.venn_js.json'},
        #                                                                                                         ('GPU',): {'venn_js_path': 'OperationOverlap.process_selfplay_worker_1.phase_default_phase.resources_GPU.venn_js.json'}}},
        #                                                                  'selfplay_worker_1': {'resource_overlap': {('CPU',): {'venn_js_path': 'OperationOverlap.process_selfplay_worker_1.phase_selfplay_worker_1.resources_CPU.venn_js.json'},
        #                                                                                                             ('CPU', 'GPU'): {'venn_js_path': 'OperationOverlap.process_selfplay_worker_1.phase_selfplay_worker_1.resources_CPU_GPU.venn_js.json'},
        #                                                                                                             ('GPU',): {'venn_js_path': 'OperationOverlap.process_selfplay_worker_1.phase_selfplay_worker_1.resources_GPU.venn_js.json'}}}}}}},

        # 'ResourceOverlap': {'process': {'loop_init': {'phase': {'bootstrap': {'venn_js_path': 'ResourceOverlap.process_loop_init.phase_bootstrap.venn_js.json'},
        #                                                         'default_phase': {'venn_js_path': 'ResourceOverlap.process_loop_init.phase_default_phase.venn_js.json'}}},
        #                                 'loop_selfplay': {'phase': {'default_phase': {'venn_js_path': 'ResourceOverlap.process_loop_selfplay.phase_default_phase.venn_js.json'},
        #                                                             'selfplay_workers': {'venn_js_path': 'ResourceOverlap.process_loop_selfplay.phase_selfplay_workers.venn_js.json'}}},
        #                                 'loop_train_eval': {'phase': {'default_phase': {'venn_js_path': 'ResourceOverlap.process_loop_train_eval.phase_default_phase.venn_js.json'},
        #                                                               'evaluate': {'venn_js_path': 'ResourceOverlap.process_loop_train_eval.phase_evaluate.venn_js.json'},
        #                                                               'sgd_updates': {'venn_js_path': 'ResourceOverlap.process_loop_train_eval.phase_sgd_updates.venn_js.json'}}},
        #                                 'selfplay_worker_0': {'phase': {'default_phase': {'venn_js_path': 'ResourceOverlap.process_selfplay_worker_0.phase_default_phase.venn_js.json'},
        #                                                                 'selfplay_worker_0': {'venn_js_path': 'ResourceOverlap.process_selfplay_worker_0.phase_selfplay_worker_0.venn_js.json'}}},
        #                                 'selfplay_worker_1': {'phase': {'default_phase': {'venn_js_path': 'ResourceOverlap.process_selfplay_worker_1.phase_default_phase.venn_js.json'},
        #                                                                 'selfplay_worker_1': {'venn_js_path': 'ResourceOverlap.process_selfplay_worker_1.phase_selfplay_worker_1.venn_js.json'}}}}},

        'ResourceSubplot': {'process': {'loop_init': {'phase': {'bootstrap': {'venn_js_path': 'ResourceSubplot.process_loop_init.phase_bootstrap.venn_js.json'},
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

def test_sel():

    for i, (value, subtree) in enumerate(_sel({'overlap_type':'ResourceSubplot'}, INDEX, 'overlap_type')):
        print(" HI1 ")
        assert i == 0
        assert subtree == INDEX['ResourceSubplot']

    for i, (value, subtree) in enumerate(_sel(
            {'overlap_type':'ResourceSubplot', 'process':'loop_train_eval'},
            INDEX['ResourceSubplot']['process'],
            'process')):
        print(" HI2 ")
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
            SEL_ORDER['ResourceSubplot'],
            0,
            md,
            INDEX['ResourceSubplot'])):
        actual_entries.append(entry)
        actual_mds.append(md)
    # actual_entries.sort()
    # actual_mds.sort()
    for expected_entry in expected_entries:
        assert expected_entry in actual_entries
