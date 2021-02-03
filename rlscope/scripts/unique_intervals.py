# -*- coding: utf-8 -*-
"""
Speedup event overlap computation using numba to JIT compile the algorithm using LLVM.

This script is a simple example for experimenting, testing, and visualizing the core algorithm.
"""
import time
import argparse
import textwrap
import numpy as np
from rlscope.parser.plot_utils import setup_matplotlib
setup_matplotlib()
import matplotlib.pyplot as plt
import pprint

from rlscope.parser.common import *

from rlscope.profiler.rlscope_logging import logger

from rlscope import py_config
if py_config.USE_NUMBA:
    import numba
    import numba as nb

UNICODE_TIMES = u"\u00D7"
BITS_PER_BYTE = 8

"""
Trying out different optimizations:

    (1) Using bitsets instead of numpy true/false arrays:
        RESULT: slowdown
        
        > k=12 categories with n=1000 events in each list (iters=5, reps=5)
        > Total time spent benchmarking = 9.634756803512573 seconds
        > with numba: 0.0035594630055129526 seconds
        > without numba: 0.32853033738210796 seconds
        > Speedup: 92.29772493021419 ×

        > k=12 categories with n=10000 events in each list (iters=5, reps=5)
        > Total time spent benchmarking = 85.38820958137512 seconds
        > with numba: 0.032505753077566625 seconds
        > without numba: 3.2078955063596366 seconds
        > Speedup: 98.6870077645892 ×
        
Adding features:
        (1) Add OverlapMetadata:

        > k=12 categories with n=1000 events in each list (iters=5, reps=5)
        > Total time spent benchmarking = 17.913370370864868 seconds
        > with numba: 0.006963567808270454 seconds
        > without numba: 0.5863135676831007 seconds
        > Speedup: 84.19729423568631 ×

"""

def GenerateIntervals(n, max_length=10, mean_wait=15):
    """
    Generate n start/end intervals.
    The start and length of the intervals are randomly generated.

    >>> GenerateIntervals(3)
    (array([19, 42, 59]), array([28, 43, 61]))

    :param n:
    :param max_length:
        length of randomly generated intervals are in [0..max_length)
        (i.e. does NOT include max_length)
    :param mean_wait:
    :return:

    """
    lengths = np.random.randint(0, max_length, size=n)
    # NOTE: this will ensure that none of the intervals we generate overlap
    # within each other.
    waits = np.random.poisson(mean_wait,size=n)
    
    starts = waits.cumsum()
    starts[1:] += lengths[:-1].cumsum()    
    ends = starts + lengths
    
    for i in range(1,len(ends)):
        assert ends[i-1] <= starts[i]
        
    return starts, ends

def PlotIntervals(starts, ends, y=0, 
                  lw=10, marker_scale=3, **kwargs):
    if 'color' not in kwargs:
        kwargs['color'] = plt.gca()._get_lines.get_next_color()        
    ys = np.repeat(y, len(starts))
    plt.hlines(ys, starts, ends, lw=lw, **kwargs)
    kwargs.pop('label', None)
    edgesize = marker_scale * lw**2
    plt.scatter(starts, ys, marker='|', s=edgesize, **kwargs)
    plt.scatter(ends, ys, marker='|', s=edgesize, **kwargs)

def PlotCategories(categories, start_times, end_times):
    plt.figure()
    k = len(categories)
    assert k == len(start_times) and  k == len(end_times)
    
    for y, (category, starts, ends) in enumerate(zip(categories, start_times, end_times)):
        PlotIntervals(starts, ends, y=y, label=category)
    plt.yticks(range(k), categories)

# See https://stackoverflow.com/a/5347492       
def Interleave(a, b):
    """Interleave numpy arrays"""
    c = np.empty((a.size + b.size,), dtype=a.dtype)
    c[0::2] = a
    c[1::2] = b
    return c

def Deinterleave(c):
    """Deinterleave numpy arrays"""
    assert (c.size % 2) == 0
    a = np.empty((c.size//2,), dtype=c.dtype)
    b = np.empty((c.size//2,), dtype=c.dtype)
    a = c[0::2]
    b = c[1::2]
    return a,b

"""
Individual categories have one bit set:
    Category[0] = 0b0001
    Category[1] = 0b0010
    Category[2] = 0b0100
    Category[3] = 0b1000
    ...

Set Union:
    { Category[0], Category[3] } = 0b1001

Removing elements from the set:
    Remove Category[0] from { Category[0], Category[3] } = 0b1000
                                                                -
                                                                flipped this bit to 0
Adding elements to the set:
    Add Category[1] from { Category[0], Category[3] } = 0b1011
                                                            -
                                                            flipped this bit to 1
"""
if py_config.USE_NUMBA:
    @numba.njit
    def bitset_add(bitset, idx):
        return bitset | (1 << idx)
    @numba.njit
    def bitset_remove(bitset, idx):
        return bitset & ~(1 << idx)
    @numba.njit
    def bitset_contains(bitset, idx):
        return bitset & (1 << idx)
    @numba.njit
    def bitset_union(bitset1, bitset2):
        return bitset1 | bitset2
    @numba.njit
    def bitset_empty_set():
        return 0
    @numba.njit
    def bitset_full_set(N):
        """
        e.g. N = 4
        0b1111
            == 0b10000 - 1
            == (1 << N) - 1


        :param N:
            Number of elements in the set.
        :return:
            A bitset containing all members {0, 1, 2, ..., N-1}
        """
        return (1 << N) - 1
    @numba.njit
    def bitset_is_empty(bitset):
        return bitset == 0
    @numba.njit
    def bitset_indices(bitset):
        bits_left = bitset
        # if py_config.NUMBA_DISABLE_JIT:
        #     indices = []
        # else:
        #     indices = numba.typed.List.empty_list(nb.uint64)
        indices = []
        idx = 0
        while not bitset_is_empty(bits_left):
            if bitset_contains(bitset, idx):
                indices.append(idx)
                bits_left = bitset_remove(bits_left, idx)
            idx += 1
        return indices

def bitset_np_bool_vector(bitset, N):
    """
    Convert bit-vector into numpy true/false bool vector.
    e.g.
    N = 6
    bitset = 0b1101
    [True, False, True, True, False, False]
     -----------------------  ------------
             0b1001              2 leading
                                   zeros

    NOTE: reversed order between bit-vector and numpy array.

    :param bitset:
    :param N:
    :return:
    """
    return np.array([bitset_contains(bitset, i) for i in range(N)], dtype=bool)

def num_bits_numpy_type(DType):
    """
    How many bits are in a numpy integer data-type (e.g. np.int64)?

    >>> num_bits_numpy_type(np.int64)
    64

    :param dtype:
    :return:
    """
    # Q: Should we use unsigned?  Does it matter?
    # Yes probably; np.packedbits uses uint8.
    assert issubclass(DType, np.signedinteger) or issubclass(DType, np.unsignedinteger)
    # assert DType in [np.int64, np.int32, np.int16]
    num_bytes = DType(0).nbytes
    return num_bytes * BITS_PER_BYTE

if py_config.USE_NUMBA:
    @numba.njit
    def implUniqueSplit(index, lengths, times,

                        outputs,
                        output_cats,

                        # cur_cat,
                        out_numba_overlap,
                        out_overlap_metadata,

                        # include_empty_time,
                        ):
        """
        Preconditions:
        - Events in each category have no self-intersection.

        N = total number of intervals across all categories ( = np.sum(lengths) )
        k = number of categories

        :param index:
            np.array of integers of size k (all zeros)
            k = number of categories
        :param lengths:
            array of k integers
            length of each list i
            For each category, the number of intervals/"times".
        :param times:
            # Example format:
            #
            # Each event-array is sorted by start time.
            [
                # Category A:
                array( [ 14 , 15 , 36 , 45 , 63 , 67 , 78 , 80 , 101 , 103 ] ) ,
                #        -------
                #        first event.
                #        start_time = 14
                #        (even position)
                #        end_time = 15
                #        (odd position)
                # Category B:
                array( [ 18 , 19 , 33 , 37 , 49 , 52 , 69 , 69 , 82  , 82  ] ) ,
                # Category C:
                array( [ 22 , 26 , 36 , 43 , 62 , 69 , 84 , 92 , 103 , 105 ] ) ,
            ]
            List of k arrays.
            Each array has integers with start time in even positions, and end time in odd positions.
            Each array must be sorted by start time (individually).
        :param outputs:
            Input/output variable
            Array of N integers initialized to zero, where N is the sum of lengths.
        :param output_cats:
            (N x k) array of booleans, initialized to false.
            TODO: change to bitarrays?
            For each event, which categories are active?
            This is useful for creating visualizations (and other stuffs).

            NOTE: if two events start at the same time, they will be represented at the SAME location in output_cats.
            Also, we preallocate N entries in output_cats, but in reality, the number of "filled" entries will
            be smaller when two events start at the same time.
            "outputs" tells us how many entries in output_cats are filled.

        :param cur_cat:
            Scratch variable.
            This is a "bit-vector" of the current active categories.
            Array of k booleans, all initialized to false.

        :return:
            Number of valid intervals in the output and output_cats array.
            1 <= ... <= N

            NOTE: we don't output this format.
            {
                [A, B] : 120 seconds
            }
        """

        # Q: What integer type are we using to represent start/end timestamps?
        # Currently, we using int64 to represent microsecond timestamps.
        # However, in the future we may try to be more clever by something smaller like int16,
        # by subtracting the very first start-time from all start/end events.
        time_type = times[0].dtype
        cur_output = 0
        min_time_value = np.iinfo(time_type).min
        max_time_value = np.iinfo(time_type).max
        last_time = min_time_value

        # cur_cat = py_config.NUMBA_CATEGORY_KEY_TYPE(0)
        cur_cat = 0

        while (index < lengths).any():
            min_cat = 0
            # min_cat = py_config.NUMBA_CATEGORY_KEY_TYPE(0)
            min_time = max_time_value
            # Find the non-empty category with the next minimum start/end time.
            for i in range(len(index)):
                # Check we haven't exhausted the intervals in the category.
                if index[i] < lengths[i]:
                    # Non-empty category.
                    if times[i][index[i]] <= min_time:
                        min_cat = i
                        min_time = times[i][index[i]]
                        # logger.info(": {msg}".format(msg=pprint_msg({
                        #     'cond': "({left}) times[i][index[i]] <= min_time ({right})".format(
                        #         left=times[i][index[i]],
                        #         right=min_time,
                        #     ),
                        #     'min_cat': min_cat,
                        #     'min_time': min_time,
                        # })))

            # logger.info(": {msg}".format(msg=pprint_msg({
            #     'min_cat': min_cat,
            #     'min_time': min_time,
            #     # 'index[min_cat]': index[min_cat],
            #     # 'times[min_cat]': times[min_cat],
            #     'cur_cat': "{cur_cat:b}".format(cur_cat=cur_cat),
            #     'time_type': 'start' if (index[min_cat] % 2 == 0) else 'end',
            # })))

            # min_cat = the Category with the next smallest time (could be a start or end time)
            # min_time = the next smallest time (NOT the index, it's the time itself)

            # (index[min_cat] % 2) == 0
            # This checks if it is a start_time (even index).

            # Skip empty intervals.
            #
            # An empty interval has the start time equal to the end-time.
            #
            #   (index[min_cat] % 2) == 0
            #   - This checks that we're currently looking at a start-time of an interval
            #
            #   min_time == times[min_cat][index[min_cat]+1]:
            #   - times[min_cat][index[min_cat]+1] is the end-time of the interval we're look at,
            #     and min_time is the start-time.
            #   - this is just checking "if start_time == end_time"
            if (index[min_cat] % 2) == 0 and min_time == times[min_cat][index[min_cat]+1]:
                index[min_cat] += 2
                continue

            # if not include_empty_time and bitset_is_empty(cur_cat):

            time_chunk = min_time - last_time
            if last_time != min_time_value and time_chunk > 0:
                # Q: Does Dict have default values...?
                if cur_cat not in out_numba_overlap:
                    out_numba_overlap[cur_cat] = 0
                out_numba_overlap[cur_cat] += time_chunk

            # Update current list of active categories.
            #
            is_start = (index[min_cat] % 2 == 0)
            # assert is_start != cur_cat[min_cat]
            # cur_cat[min_cat] = is_start
            if is_start:
                cur_cat = bitset_add(cur_cat, min_cat)
            else:
                start_time_usec = times[min_cat][index[min_cat]-1]
                end_time_usec = min_time

                # out_overlap_metadata.add_event(cur_cat, start_time_usec, end_time_usec)
                if cur_cat not in out_overlap_metadata:
                    out_overlap_metadata[cur_cat] = NumbaRegionMetadata(cur_cat)
                out_overlap_metadata[cur_cat].add_event(start_time_usec, end_time_usec)

                overlap_region = out_overlap_metadata[cur_cat]

                # if out_overlap_metadata[cur_cat].start_time_usec == 0 or start_time_usec < out_overlap_metadata[cur_cat].start_time_usec:
                #     out_overlap_metadata[cur_cat].start_time_usec = start_time_usec
                # if out_overlap_metadata[cur_cat].end_time_usec == 0 or end_time_usec > out_overlap_metadata[cur_cat].end_time_usec:
                #     out_overlap_metadata[cur_cat].end_time_usec = end_time_usec

                if overlap_region.start_time_usec == 0 or start_time_usec < overlap_region.start_time_usec:
                    overlap_region.start_time_usec = start_time_usec
                if overlap_region.end_time_usec == 0 or end_time_usec > overlap_region.end_time_usec:
                    overlap_region.end_time_usec = end_time_usec

                cur_cat = bitset_remove(cur_cat, min_cat)

            # Can have multiple categories entering and leaving, so just make sure we keep things correct
            if last_time == min_time:
                # Start of new interval which is the same as the previous interval.
                # output_cats[cur_output-1, min_cat] = is_start
                output_cats[cur_output-1] = bitset_add(output_cats[cur_output-1], min_cat)
                pass
            else:
                # Normal case:
                # Insert event if there is a change from last time
                outputs[cur_output] = min_time
                # output_cats[cur_output, :] = cur_cat
                output_cats[cur_output] = cur_cat
                cur_output += 1
                # last_time = min_time

            last_time = min_time
            index[min_cat] += 1

        return cur_output


    Category_Numpy_DType = py_config.NUMPY_CATEGORY_KEY_TYPE
    Category_Numba_Type = py_config.NUMBA_CATEGORY_KEY_TYPE
    Time_Numba_Type = py_config.NUMBA_TIME_USEC_TYPE

def UniqueSplits(
    times, use_numba=True):
    k = len(times)

    time_type = times[0].dtype
    index = np.zeros(k, dtype=int)
    lengths = np.array([ len(t) for t in times ], dtype=int)
    if use_numba:
        times = tuple(times)
    
    outputs = np.zeros(lengths.sum(), dtype=time_type)
    # cur_cat = np.zeros(k, dtype=bool)
    # output_cats = np.zeros((len(outputs), k), dtype=bool)
    # We can represent a set of at most 64 elements using a 64-bit integer...
    # Ideally we would instead use numpy's bit-vector representation.
    # https://stackoverflow.com/questions/5602155/numpy-boolean-array-with-1-bit-entries
    # (see np.packbits
    assert num_bits_numpy_type(Category_Numpy_DType) >= k
    output_cats = np.zeros(len(outputs), dtype=Category_Numpy_DType)

    # numba_overlap_metadata = numba.typed.Dict.empty(
    #     key_type=np.float64,
    #     value_type=NumbaRegionMetadata_type,
    # )

    out_numba_overlap = numba.typed.Dict.empty(
        key_type=Category_Numba_Type,
        value_type=Time_Numba_Type,
    )

    # NOTE: This causes an assertion to go off inside numba!
    # python: /root/miniconda2/conda-bld/llvmdev_1559156562364/work/lib/IR/DataLayout.cpp:680:
    # unsigned int llvm::DataLayout::getAlignment(llvm::Type*, bool) const: Assertion `Ty->isSized() &&
    # "Cannot getTypeInfo() on a type that is unsized!"' failed.

    # NOTE: This is significantly slower ( Speedup: 84.19729423568631 × with --timing-num-intervals $((10**3)) )
    # So, we "manually inline"  NumbaOverlapMetadata
    # out_overlap_metadata = NumbaOverlapMetadata()
    out_overlap_metadata = numba.typed.Dict.empty(
        key_type=py_config.NUMBA_CATEGORY_KEY_TYPE,
        value_type=NumbaRegionMetadata_type,
    )

    implementation = implUniqueSplit if use_numba else implUniqueSplit.py_func
    # implementation(
    cur_output = implementation(
        index, lengths, times,

        outputs,
        output_cats,

        # cur_cat,
        out_numba_overlap,
        out_overlap_metadata,
    )

    return out_numba_overlap, out_overlap_metadata, outputs[:cur_output], output_cats[:cur_output]
    # return outputs[:cur_output]

# NumbaOverlapMetadata_type = numba.deferred_type()
# NumbaRegionMetadata_type = numba.deferred_type()

if py_config.USE_NUMBA:
    # https://numba.pydata.org/numba-doc/dev/user/jitclass.html
    NumbaRegionMetadata_Fields = [
        ('category_key', py_config.NUMBA_CATEGORY_KEY_TYPE),
        ('start_time_usec', py_config.NUMBA_TIME_USEC_TYPE),
        ('end_time_usec', py_config.NUMBA_TIME_USEC_TYPE),
        ('num_events', nb.uint64),
    ]
    @numba.jitclass(NumbaRegionMetadata_Fields)
    class NumbaRegionMetadata:
        """
        Numbified version of RegionMetadata class.
        RegionMetadata is used to track the following statistics about each overlap region:

            self.start_time_usec = None
            self.end_time_usec = None
            self.num_events = 0
        - min(event.start_time)
        - max(event.start_time)
        - number of events in overlap region

        NOTE: I'm not sure if these statistics are entirely necessary, but they're just nice-to-have for debugging.
        """
        def __init__(self, category_key):
            self.category_key = category_key
            self.start_time_usec = 0
            self.end_time_usec = 0
            self.num_events = 0

        def add_event(self, start_time_usec, end_time_usec):
            if self.start_time_usec == 0 or start_time_usec < self.start_time_usec:
                self.start_time_usec = start_time_usec

            if self.end_time_usec == 0 or end_time_usec > self.end_time_usec:
                self.end_time_usec = end_time_usec

            self.num_events += 1
    # if not py_config.NUMBA_DISABLE_JIT:
    #     NumbaRegionMetadata_type.define(NumbaRegionMetadata.class_type.instance_type)
    if not py_config.NUMBA_DISABLE_JIT:
        NumbaRegionMetadata_type = NumbaRegionMetadata.class_type.instance_type
    else:
        NumbaRegionMetadata_type = None


# NumbaOverlapMetadata_Fields = [
#     ('regions', numba.types.DictType(py_config.NUMBA_CATEGORY_KEY_TYPE, NumbaRegionMetadata_type)),
# ]
# @numba.jitclass(NumbaOverlapMetadata_Fields)
# class NumbaOverlapMetadata:
#     """
#     Numbafied version of OverlapMetadata.
#     """
#     def __init__(self):
#         # CategoryKey -> RegionMetadata
#         # if py_config.NUMBA_DISABLE_JIT:
#         #     self.regions = dict()
#         # else:
#         self.regions = numba.typed.Dict.empty(
#             key_type=py_config.NUMBA_CATEGORY_KEY_TYPE,
#             value_type=NumbaRegionMetadata_type,
#         )
#
#     def add_event(self, category_key, start_time_usec, end_time_usec):
#         if category_key not in self.regions:
#             self.regions[category_key] = NumbaRegionMetadata(category_key)
#
#         self.regions[category_key].add_event(start_time_usec, end_time_usec)
# if not py_config.NUMBA_DISABLE_JIT:
#     NumbaOverlapMetadata_type = NumbaOverlapMetadata.class_type.instance_type
# else:
#     NumbaOverlapMetadata_type = None

def PlotOutput(outputs, output_categories, categories):
    plt.ylim(ymin=-1)
    for i, (x, cats) in enumerate(zip(outputs, output_categories)):
        plt.axvline(x, color='k', lw=1, ls='dashed')
        # cats = a numpy "bit-vector"
        # s = ''.join( letter for c, letter in zip(cats, categories) if c )

        # cats_bool_vector = cats
        cats_bool_vector = bitset_np_bool_vector(cats, len(categories))

        active_categories = list(np.array(categories)[cats_bool_vector])

        # s = ''.join( letter for c, letter in zip(cats, categories) if c )

        s = ''.join(active_categories)
        y = -0.9 + (i % 3) * 0.25
        plt.text(x+0.1, y, s)

def Experiment(figure_basename, categories, gen_func,
               interactive=False,
               use_numba=True):
    starts, ends = zip(*[ gen_func() for _ in categories ] )
    PlotCategories(categories, starts, ends)
    times = [ Interleave(s, e) for s,e in zip(starts, ends) ]
    # UniqueSplits(times, use_numba=use_numba)
    # outputs = UniqueSplits(times, use_numba=use_numba)
    out_numba_overlap, out_overlap_metadata, outputs, output_categories = UniqueSplits(times, use_numba=use_numba)
    PlotOutput(outputs, output_categories, categories)
    ShowOrSave(
        figure_basename,
        interactive=interactive,
    )

def ShowOrSave(base, interactive=False, ext='png'):
    if interactive:
        plt.show()
    else:
        path = '{base}.{ext}'.format(base=base, ext=ext)
        print("> Save figure to {path}".format(path=path))
        plt.savefig(path)

def Timing(n=10**5, k=12, iterations=5, repeats=5):
    import timeit
    categories = [ 'C%d' for i in range(k) ]
    times = [ Interleave(*GenerateIntervals(n)) for _ in range(k) ]

    variables = dict(globals(), **locals())

    print("> Timing configuration:")
    print(textwrap.dedent("""
    num_categories: {num_categories}
        Number of categories to generate intervals for.
    timing_num_intervals: {timing_num_intervals}
        Number of intervals to generate for EACH category.
    iterations: {iterations}
        Iterations for python timeit module.
    repeats: {repeats}
        Repetitions for python timeit module.
    """).format(
        timing_num_intervals=n,
        num_categories=k,
        iterations=iterations,
        repeats=repeats,
    ))
    """
    Some runs with varying numbers of intervals. 
    The effect of increasing the number of intervals is that more time will be spent in the numbafied inner loop, 
    rather than starting a new iterations from pure-python inside UniqueSplits.
    NOTE: UniqueSplits includes several things:
    - initializing zeroed np.array's for all the "output" variables (e.g. k*N array for output_cats)
      Q: Does this actually take long...?
    - Hence, you obtain LARGER speedups when you are analyzing a larger number of events. 
      This is good to know, since it tells us in the real implementation how many events to 
      assign each worker thread.  Currently, the split is coarse-grained since its time interval based, not # of events, 
      so there could be imbalance.
    
        > k=12 categories with n=10 events in each list (iters=5, reps=5)
        > with numba: 0.0003420809283852577 seconds
        > without numba: 0.0037674466148018837 seconds
        > Speedup: 11.013319662646955 ×
        
        > k=12 categories with n=100 events in each list (iters=5, reps=5)
        > with numba: 0.000641842745244503 seconds
        > without numba: 0.037069130968302486 seconds
        > Speedup: 57.75422600466007 ×
        
        > k=12 categories with n=1000 events in each list (iters=5, reps=5)
        > Total time spent benchmarking = 10.256434202194214 seconds
        > with numba: 0.003667216468602419 seconds
        > without numba: 0.36136230640113354 seconds
        > Speedup: 98.53858082690417 ×
        
        > k=12 categories with n=10000 events in each list (iters=5, reps=5)
        > Total time spent benchmarking = 98.15659523010254 seconds
        > with numba: 0.03406465379521251 seconds
        > without numba: 3.6920439190231265 seconds
        > Speedup: 108.38342703315574 ×
        
        > k=12 categories with n=100000 events in each list (iters=5, reps=5)
        > Total time spent bechmarking = 952.9324278831482 seconds
        > with numba: 0.34007849236950277 seconds
        > without numba: 36.205823552422224 seconds
        > Speedup: 106.46313826010437 ×
    """

    bench_start_t = time.time()

    # For repeats=5, timeit will return a list of 5 floats.
    # Each is the total time (seconds) it took to run iterations=5 in a loop.

    # Precompile with numba and warm up the CPU and cache
    UniqueSplits(times)
    with_numba = timeit.repeat('UniqueSplits(times)',
                               repeat=repeats, number=iterations,
                               globals=variables)


    # Warm up the CPU and cache
    UniqueSplits(times, use_numba=False)
    without = timeit.repeat('UniqueSplits(times, use_numba=False)',
                            repeat=repeats, number=iterations,
                            globals=variables)

    bench_end_t = time.time()

    print("> k={k} categories with n={n} events in each list (iters={iters}, reps={reps})".format(
        k=k,
        n=n,
        iters=iterations,
        reps=repeats,
    ))
    with_numba = min(with_numba)/iterations
    without_numba = min(without)/iterations
    speedup = without_numba / with_numba

    print("> Total time spent benchmarking = {sec} seconds".format(
        sec=bench_end_t - bench_start_t,
    ))

    print("> with numba: {sec} seconds".format(
        sec=with_numba))

    print("> without numba: {sec} seconds".format(
        sec=without_numba))

    print("> Speedup: {speedup} {UNICODE_TIMES}".format(
        speedup=speedup,
        UNICODE_TIMES=UNICODE_TIMES))

if py_config.USE_NUMBA:
    class TestBitset:
        def test_bitset_01_empty_set(self):
            empty_set = bitset_empty_set()
            indices = bitset_indices(empty_set)
            assert indices == []

        def test_bitset_02_one_elem_set(self):
            empty_set = bitset_empty_set()
            one_elem_set = bitset_add(empty_set, 0)
            indices = bitset_indices(one_elem_set)
            assert indices == [0]

        def test_bitset_03_two_elem_set(self):
            empty_set = bitset_empty_set()
            one_elem_set = bitset_add(empty_set, 0)
            two_elem_set = bitset_add(one_elem_set, 2)
            indices = bitset_indices(two_elem_set)
            assert indices == [0, 2]

if __name__ == '__main__':
    np.random.seed(475)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=[
            'unit-test',
            'perf',
        ],
        required=True,
        help=textwrap.dedent("""
    What should we run?
    
    unit-test:
        Randomly generate an event trace, and compute the overlap on it.
    
    perf:
        Compare JIT-compiled runtime to pure python runtime.
        NOTE: this can take ~ 10 minutes to run.
    """))
    parser.add_argument(
        "--interactive",
        action='store_true',
        help=textwrap.dedent("""
    Use plot.show() to show matplotlib interactive plot viewer
    (default: saves to png files)
    """))
    parser.add_argument(
        "--timing-num-intervals",
        type=int,
        # NOTE: originally Mickey used 10**5 intervals for each category.
        # However, 10**4 returns similar results and takes around 100 seconds
        # to run the whole benchmark, whereas 10**5 takes a really long time.
        default=10**4,
        help=textwrap.dedent("""
    For "--mode perf", how many intervals in total should we randomly generate for EACH category?
    See also:
        - For number of categories, see --timing-num-categories.
    """))
    parser.add_argument(
        "--timing-num-categories",
        type=int,
        default=12,
        help=textwrap.dedent("""
    For "--mode perf", how many categories are there?
    
    See also:
        - For number of intervals in each category, see 
    """))
    parser.add_argument(
        "--timing-num-iterations",
        type=int,
        default=5,
        help=textwrap.dedent("""
    For "--mode perf", how many iterations to perform?
    For details, refer to timeit python module:
    https://docs.python.org/2/library/timeit.html
    """))
    parser.add_argument(
        "--timing-num-repeats",
        type=int,
        default=5,
        help=textwrap.dedent("""
    For "--mode perf", how many repeats to perform?
    For details, refer to timeit python module:
    https://docs.python.org/2/library/timeit.html
    """))
    parser.add_argument(
        "--disable-numba",
        action='store_true',
        help=textwrap.dedent("""
    # For unit-test, skip JIT compilation, and run in pure python mode
    Use plot.show() to show matplotlib interactive plot viewer
    (default: saves to png files)
    """))
    args = parser.parse_args()

    use_numba = not args.disable_numba
    if not use_numba:
        print("> Disabling numba; running in pure python mode.")

    if args.mode == 'unit-test':
        Experiment(
            'A_B_C', [ 'A', 'B', 'C' ], lambda: GenerateIntervals(5),
            interactive=args.interactive,
            use_numba=use_numba)
        # Experiment(
        #     'X_Y_Z_W', [ 'X', 'Y', 'Z', 'W' ], lambda: GenerateIntervals(np.random.randint(8,12), mean_wait=5),
        #     interactive=args.interactive,
        #     use_numba=use_numba)
    elif args.mode == 'perf':
        """
        Example output:
            12 categories with 100000 events in each list
            with numba: 0.3409351079724729 seconds
            without it:   37.32773629501462 seconds
        """
        Timing(
            n=args.timing_num_intervals,
            k=args.timing_num_categories,
            iterations=args.timing_num_iterations,
            repeats=args.timing_num_repeats,
        )
    else:
        raise NotImplementedError("Not sure how to run --mode={mode}".format(
            mode=args.mode))
