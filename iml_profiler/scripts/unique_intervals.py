# -*- coding: utf-8 -*-
"""
Speedup event overlap computation using numba to JIT compile the algorithm using LLVM.

This script is a simple example for experimenting, testing, and visualizing the core algorithm.
"""
import argparse
import textwrap
import numpy as np
import matplotlib.pyplot as plt
import numba

np.random.seed(475)

def GenerateIntervals(n, max_length=10, mean_wait=15):
    lengths = np.random.randint(0, max_length, size=n)
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
    
    time_type = times[0].dtype
    cur_output = 0
    last_time = np.iinfo(time_type).min
    max_time_value = np.iinfo(time_type).max

    while (index < lengths).any():
        min_cat = 0
        min_time = max_time_value
        for i in range(len(index)):
            if index[i] < lengths[i]:
                if times[i][index[i]] <= min_time:
                    min_cat = i
                    min_time = times[i][index[i]]
        
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
            output_cats[cur_output-1, min_cat] = is_start
        else:
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
               interactive=False):
    starts, ends = zip(*[ gen_func() for _ in categories ] )
    PlotCategories(categories, starts, ends)
    times = [ Interleave(s, e) for s,e in zip(starts, ends) ]
    outputs, output_categories = UniqueSplits(categories, times)
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

    print('%d categories with %d events in each list' % (k, n))
    print('with numba:', min(with_numba)/iterations, 'seconds')
    print('without it:  ', min(without)/iterations, 'seconds')

if __name__ == '__main__':
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
    args = parser.parse_args()

    if args.mode == 'unit-test':
        Experiment(
            'A_B_C', [ 'A', 'B', 'C' ], lambda: GenerateIntervals(5),
            interactive=args.interactive)
        Experiment(
            'X_Y_Z_W', [ 'X', 'Y', 'Z', 'W' ], lambda: GenerateIntervals(np.random.randint(8,12), mean_wait=5),
            interactive=args.interactive)
    elif args.mode == 'perf':
        """
        Example output:
            12 categories with 100000 events in each list
            with numba: 0.3409351079724729 seconds
            without it:   37.32773629501462 seconds
        """
        Timing()
    else:
        raise NotImplementedError("Not sure how to run --mode={mode}".format(
            mode=args.mode))
