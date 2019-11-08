# -*- coding: utf-8 -*-
"""
Speedup event overlap computation using numba to JIT compile the algorithm using LLVM.

This script is a simple example for experimenting, testing, and visualizing the core algorithm.
"""
import time
import argparse
import textwrap
import numpy as np
import matplotlib.pyplot as plt
import numba
import pprint

UNICODE_TIMES = u"\u00D7"

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
    

@numba.njit
def implUniqueSplit(index, lengths, times, outputs, output_cats, cur_cat):
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
    last_time = np.iinfo(time_type).min
    max_time_value = np.iinfo(time_type).max

    while (index < lengths).any():
        min_cat = 0
        min_time = max_time_value
        # Find the non-empty category with the next minimum start/end time.
        for i in range(len(index)):
            if index[i] < lengths[i]:
                # Non-empty category.
                if times[i][index[i]] <= min_time:
                    min_cat = i
                    min_time = times[i][index[i]]

        # (index[min_cat] % 2) == 0
        # This checks if it is a start_time (even index).
        
        # Skip empty intervals
        if (index[min_cat] % 2) == 0 and min_time == times[min_cat][index[min_cat]+1]:
            index[min_cat] += 2
            continue          
        
        # Update current list of categories
        is_start = index[min_cat] % 2 == 0
        assert is_start != cur_cat[min_cat]
        cur_cat[min_cat] = is_start        
        
        # Can have multiple categories entering and leaving, so just make sure we keep things correct
        if last_time == min_time:
            # Start of new interval which is the same as the previous interval.
            output_cats[cur_output-1, min_cat] = is_start
        else:
            # Normal case:
            # Insert event if there is a change from last time
            outputs[cur_output] = min_time                        
            output_cats[cur_output, :] = cur_cat
            cur_output += 1
            last_time = min_time
            
        index[min_cat] += 1
        
    return cur_output


def UniqueSplits(categories, times, use_numba=True):
    k = len(categories)
    time_type = times[0].dtype
    index = np.zeros(k, dtype=int)
    lengths = np.array([ len(t) for t in times ], dtype=int)
    if use_numba:
        times = tuple(times)
    
    outputs = np.zeros(lengths.sum(), dtype=time_type)
    cur_cat = np.zeros(k, dtype=bool)
    output_cats = np.zeros((len(outputs), k), dtype=bool)
        
    implementation = implUniqueSplit if use_numba else implUniqueSplit.py_func 
    cur_output = implementation(index, lengths, times, outputs, output_cats, cur_cat)
                
    return outputs[:cur_output], output_cats[:cur_output]

def PlotOutput(outputs, output_categories, categories):
    plt.ylim(ymin=-1)
    for i, (x, cats) in enumerate(zip(outputs, output_categories)):
        plt.axvline(x, color='k', lw=1, ls='dashed')
        s = ''.join( letter for c, letter in zip(cats, categories) if c )
        y = -0.9 + (i % 3) * 0.25
        plt.text(x+0.1, y, s)

def Experiment(figure_basename, categories, gen_func,
               interactive=False,
               use_numba=True):
    starts, ends = zip(*[ gen_func() for _ in categories ] )
    PlotCategories(categories, starts, ends)
    times = [ Interleave(s, e) for s,e in zip(starts, ends) ]
    outputs, output_categories = UniqueSplits(categories, times, use_numba=use_numba)
    PlotOutput(outputs, output_categories, categories)
    ShowOrSave(
        figure_basename,
        interactive=interactive,
    )

def ShowOrSave(base, interactive=False, ext='png'):
    if args.interactive:
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
    UniqueSplits(categories, times)
    with_numba = timeit.repeat('UniqueSplits(categories, times)',
                               repeat=repeats, number=iterations,
                               globals=variables)


    # Warm up the CPU and cache
    UniqueSplits(categories, times, use_numba=False)
    without = timeit.repeat('UniqueSplits(categories, times, use_numba=False)',
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
