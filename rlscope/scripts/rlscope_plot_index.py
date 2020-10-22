"""
Import index of ``*.venn_js.json`` files needed for multi-process visualization.
rlscope_plot_index_data.py is generated by
:py:mod:`rlscope.scripts.generate_rlscope_plot_index`.`
"""
import rlscope_plot_index_data

from rlscope.parser.plot_index import _DataIndex

DataIndex = _DataIndex(rlscope_plot_index_data.INDEX, rlscope_plot_index_data.DIRECTORY)
