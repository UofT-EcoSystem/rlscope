"""
Reading overlap results from ``rls-analyze``.
"""
from rlscope.parser.common import *

class CategoryKey:
    def __init__(self):
        self.procs = frozenset()
        self.ops = frozenset()
        self.non_ops = frozenset()

    @staticmethod
    def from_js(obj):
        self = CategoryKey()
        assert obj['typename'] == 'CategoryKey'
        self.procs = frozenset(obj['procs'])
        self.ops = frozenset(obj['ops'])
        self.non_ops = frozenset(obj['non_ops'])
        return self

    def __eq__(self, rhs):
        lhs = self
        return lhs.procs == rhs.procs and \
               lhs.ops == rhs.ops and \
               lhs.non_ops == rhs.non_ops

    def __hash__(self):
        return hash((self.procs, self.ops, self.non_ops))

    def __str__(self):
        bldr = ToStringBuilder(obj=self)
        bldr.add_param('procs', self.procs)
        bldr.add_param('ops', self.ops)
        bldr.add_param('non_ops', self.non_ops)
        return bldr.to_string()

    def __repr__(self):
        return str(self)

# class OverlapResult:
#     def __init__(self):
#         self.procs = frozenset()
#         self.ops = frozenset()
#         self.non_ops = frozenset()
#
#     @staticmethod
#     def from_js(obj):
#         self = OverlapResult()
#         self.overlap_map = dict()
#         assert obj['typename'] == 'CategoryKey'
#         self.procs = frozenset(obj['procs'])
#         self.ops = frozenset(obj['ops'])
#         self.non_ops = frozenset(obj['non_ops'])
#         return self

def from_js(obj, mutable=True):
    if type(obj) == dict and 'typename' in obj:
        if obj['typename'] == 'dict':
            return dict_from_js(obj, mutable=mutable)
        elif obj['typename'] in JS_TYPENAME_TO_KLASS:
            Klass = JS_TYPENAME_TO_KLASS[obj['typename']]
            parsed = Klass.from_js(obj)
            return parsed
        else:
            raise NotImplementedError("Not sure how to parse js object with typename={typename}".format(typename=obj['typename']))
    elif type(obj) == list:
        if mutable:
            return [from_js(x, mutable=mutable) for x in obj]
        else:
            return tuple(from_js(x, mutable=mutable) for x in obj)
    else:
        return obj

def dict_from_js(obj, mutable=True):
    assert obj['typename'] == 'dict'
    d = dict()
    for key, value in obj['key_value_pairs']:
        parsed_key = from_js(key, mutable=False)
        # if type(parsed_key) == list:
        #     parsed_key = tuple(parsed_key)
        # elif type(parsed_key) == set:
        #     parsed_key = frozenset(parsed_key)
        d[parsed_key] = from_js(value, mutable=mutable)
    return d

JS_TYPENAME_TO_KLASS = {
    'CategoryKey': CategoryKey,
    # 'OverlapResult': OverlapResult,
}
