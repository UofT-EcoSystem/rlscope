import plot_index_data as idx

idx.INDEX

DIRECTORY = "/mnt/data/james/clone/baselines.checkpoints/checkpoints/minigo/vector_test_multiple_workers_k4000/to_index"
INDEX = \
    {'CategoryOverlap': {'process': {'loop_init': {'phase': {'bootstrap': {'resource_overlap': {('CPU',): {'operation': {'bootstrap': {'venn_js_path': 'CategoryOverlap.process_loop_init.phase_bootstrap.ops_bootstrap.venn_js.json'}}}}}}},
                                     'loop_train_eval': {'phase': {'evaluate': {'resource_overlap': {('CPU',): {'operation': {'eval_game': {'venn_js_path': 'CategoryOverlap.process_loop_train_eval.phase_evaluate.ops_eval_game.venn_js.json'},
                                                                                                                              'load_network': {'venn_js_path': 'CategoryOverlap.process_loop_train_eval.phase_evaluate.ops_load_network.venn_js.json'},
                                                                                                                              'tree_search': {'venn_js_path': 'CategoryOverlap.process_loop_train_eval.phase_evaluate.ops_tree_search.venn_js.json'},
                                                                                                                              'tree_search_loop': {'venn_js_path': 'CategoryOverlap.process_loop_train_eval.phase_evaluate.ops_tree_search_loop.venn_js.json'}}}}},
                                                                   'sgd_updates': {'resource_overlap': {('CPU',): {'operation': {'estimator_save_model': {'venn_js_path': 'CategoryOverlap.process_loop_train_eval.phase_sgd_updates.ops_estimator_save_model.venn_js.json'},
                                                                                                                                 'estimator_train': {'venn_js_path': 'CategoryOverlap.process_loop_train_eval.phase_sgd_updates.ops_estimator_train.venn_js.json'},
                                                                                                                                 'gather': {'venn_js_path': 'CategoryOverlap.process_loop_train_eval.phase_sgd_updates.ops_gather.venn_js.json'},
                                                                                                                                 'train': {'venn_js_path': 'CategoryOverlap.process_loop_train_eval.phase_sgd_updates.ops_train.venn_js.json'}}}}}}},
                                     'selfplay_worker_0': {'phase': {'selfplay_worker_0': {'resource_overlap': {('CPU',): {'operation': {'load_network': {'venn_js_path': 'CategoryOverlap.process_selfplay_worker_0.phase_selfplay_worker_0.ops_load_network.venn_js.json'},
                                                                                                                                         'selfplay_loop': {'venn_js_path': 'CategoryOverlap.process_selfplay_worker_0.phase_selfplay_worker_0.ops_selfplay_loop.venn_js.json'},
                                                                                                                                         'tree_search': {'venn_js_path': 'CategoryOverlap.process_selfplay_worker_0.phase_selfplay_worker_0.ops_tree_search.venn_js.json'},
                                                                                                                                         'tree_search_loop': {'venn_js_path': 'CategoryOverlap.process_selfplay_worker_0.phase_selfplay_worker_0.ops_tree_search_loop.venn_js.json'}}}}}}},
                                     'selfplay_worker_1': {'phase': {'selfplay_worker_1': {'resource_overlap': {('CPU',): {'operation': {'load_network': {'venn_js_path': 'CategoryOverlap.process_selfplay_worker_1.phase_selfplay_worker_1.ops_load_network.venn_js.json'},
                                                                                                                                         'selfplay_loop': {'venn_js_path': 'CategoryOverlap.process_selfplay_worker_1.phase_selfplay_worker_1.ops_selfplay_loop.venn_js.json'},
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
    def __init__(self, index, directory):
        self.index = index
        self.directory = directory

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
        assert 'overlap_type' in selector
        for md, entry in self._sel_idx(selector):
            sel_order = DataIndex.SEL_ORDER[md['overlap_type']]
            ident = self.md_id(md, sel_order)
            yield md, entry, ident

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
            assert field in md

        field_value_sep = '_'
        sub_id_sep = '-'
        value_sep = '_'

        def value_str(value):
            if type(value) in {list, tuple, frozenset}:
                return value_sep.join([str(x) for x in sorted(value)])

        def sub_id(field):
            return "{field}{field_value_sep}{val}".format(
                field_value_sep=field_value_sep,
                val=value_str(md[field]),
            )

        def id_str():
            return sub_id_sep.join([sub_id(field) for field in sel_order])

        ident = id_str()

        return ident

    SEL_ORDER = {
        'CategoryOverlap': ['process', 'phase', 'resource_overlap', 'operation'],
        'ResourceOverlap': ['process', 'phase'],
        'ResourceSubplot': ['process', 'phase'],
        'OperationOverlap': ['process', 'phase', 'resource_overlap'],
    }
    def _sel_idx(self, selector):
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

        def sel(idx, sel_field):
            if sel_field in selector:
                value = selector[sel_field]
                subtree = idx[value]
                yield value, subtree
                return
            for value, subtree in idx.items():
                yield value, subtree

        def _sel_all(sel_order, level, md, subtree):
            if level == len(sel_order):
                yield dict(md), subtree
                return
            field = sel_order[level]
            for value, next_subtree in sel(subtree, field):
                md[field] = value
                for md, entry in _sel_all(sel_order, level + 1, md, next_subtree):
                    yield md, entry

        md = dict()
        level = 0
        for overlap, data in sel(self.index, 'overlap_type'):
            sel_order = _DataIndex.SEL_ORDER[overlap]
            for md, entry in _sel_all(sel_order, level, md):
                yield md, entry

DataIndex = _DataIndex(idx.INDEX, idx.DIRECTORY)
# DataIndex = _DataIndex(INDEX, DIRECTORY)
