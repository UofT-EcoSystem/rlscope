class _FreqParser:
    """
    > Running on CPU: 14
    > cycles[0] = 12021055605 cycles
    > seconds[0] = 5 sec
    > cpu_freq[0] = 2.40421 GHz
    > cycles[1] = 12021049029 cycles
    > seconds[1] = 5 sec
    > cpu_freq[1] = 2.40421 GHz
    > cycles[2] = 12021048798 cycles
    > seconds[2] = 5 sec
    > cpu_freq[2] = 2.40421 GHz
    > cycles[3] = 12021049617 cycles
    > seconds[3] = 5 sec
    > cpu_freq[3] = 2.40421 GHz
    > cycles[4] = 12021049059 cycles
    > seconds[4] = 5 sec
    > cpu_freq[4] = 2.40421 GHz
    > cycles[5] = 12021051003 cycles
    > seconds[5] = 5 sec
    > cpu_freq[5] = 2.40421 GHz
    > cycles[6] = 12021049767 cycles
    > seconds[6] = 5 sec
    > cpu_freq[6] = 2.40421 GHz
    > cycles[7] = 12021049392 cycles
    > seconds[7] = 5 sec
    > cpu_freq[7] = 2.40421 GHz
    > cycles[8] = 12021050112 cycles
    > seconds[8] = 5 sec
    > cpu_freq[8] = 2.40421 GHz
    > cycles[9] = 12021050535 cycles
    > seconds[9] = 5 sec
    > cpu_freq[9] = 2.40421 GHz
    > Mean CPU frequency: 2.40421 GHz
    > Std CPU frequency: 3.77961e-07 GHz
    > Num measurements: 10

    {
    'repetitions':10,
    'cpu_freq_ghz':[ ... ]
    'cpu_freq_ghz_mean':...,
    'cpu_freq_ghz_std':...,
    'cpu_id':...,
    }
    """
    def __init__(self, parser, args,
                 rep_array_name_regex=r"cpu_freq",
                 stat_name_regex=r"CPU frequency",
                 field_name='cpu_freq_ghz'):
        self.parser = parser
        self.args = args
        self.rep_array_name_regex = rep_array_name_regex
        self.stat_name_regex = stat_name_regex
        self.field_name = field_name
        self.mean_name = '{f}_mean'.format(f=self.field_name)
        self.std_name = '{f}_std'.format(f=self.field_name)

    def parse(self, it, all_results):

        # self.results = dict()
        def store(*args, **kwargs):
            store_group(all_results, *args, **kwargs)

        for line in line_iter(it):
            m = re.search(r'> Running on CPU: (?P<cpu_id>\d+)', line)
            if m:
                store(m)
                continue

            m = re.search(r'> Num measurements: (?P<repetitions>\d+)', line)
            if m:
                store(m)
                continue

            regex = r'> {array_name}\[\d+\]\s*[=:]\s*(?P<freq_ghz>{float}) GHz'.format(
                float=float_re,
                array_name=self.rep_array_name_regex)
            m = re.search(regex, line)
            if m:
                store_as(all_results,
                         self.field_name,
                         float(m.group('freq_ghz')),
                         store_type='list')
                continue

            m = re.search(r'> Mean {stat_name}\s*[=:]\s*(?P<freq_ghz_mean>{float}) GHz'.format(
                float=float_re,
                stat_name=self.stat_name_regex), line)
            if m:
                store_as(all_results,
                         self.mean_name,
                         float(m.group('freq_ghz_mean')))
                assert type(all_results[self.mean_name]) == float
                continue

            m = re.search(r'> Std {stat_name}\s*[=:]\s*(?P<freq_ghz_std>{float}) GHz'.format(
                float=float_re,
                stat_name=self.stat_name_regex), line)
            if m:
                store_as(all_results,
                         self.std_name,
                         float(m.group('freq_ghz_std')))
                continue

        expected_keys = set([
            'repetitions',
            self.field_name,
            self.mean_name,
            self.std_name,
            'cpu_id',
        ])
        missing_keys = expected_keys.difference(set(all_results.keys()))
        assert len(missing_keys) == 0

        assert type(all_results[self.mean_name]) == float
        return all_results

class CPUFreqParser(_FreqParser):
    def __init__(self, parser, args):
        super().__init__(parser, args,
                         rep_array_name_regex=r"cpu_freq",
                         stat_name_regex=r"CPU frequency",
                         field_name='cpu_freq_ghz')

class TSCFreqParser(_FreqParser):
    def __init__(self, parser, args):
        super().__init__(parser, args,
                         rep_array_name_regex=r"TSC frequency",
                         stat_name_regex=r"TSC frequency",
                         field_name='tsc_freq_ghz')

class CUDAMicroParser:
    def __init__(self, parser, args, microbench_name, start_header_regex, result_format):
        self.parser = parser
        self.args = args
        self.microbench_name = microbench_name
        self.start_header_regex = start_header_regex
        assert result_format in ['vector']
        self.result_format = result_format

    def parse(self, it, all_results):
        if self.result_format == 'vector':
            self._parse_vector(it, all_results)
        else:
            raise NotImplementedError

    def parse_specific(self, results, line, it):
        return False

    def _parse_vector(self, it, all_results):
        """
        CudaLaunch latency:
        CUDA kernel launch with 1 blocks of 256 threads
        > CudaLaunch latencies:
        Time for running operations on the stream, measuring GPU-side.
                      GPU
                        0  3.90 +/-   0.04
                        1  3.54 +/-   0.00
                        2  3.51 +/-   0.01
                        3  4.44 +/-   0.01

        Time for scheduling operations to the stream (not running them), measuring CPU-side.
                      CPU
                        0  3.18 +/-   0.04
                        1  3.17 +/-   0.04
                        2  3.22 +/-   0.11
                        3  3.17 +/-   0.07

        Time for scheduling + running operations on the stream, measuring CPU-side.
                      CPU
                        0  7.16 +/-   0.06
                        1  6.79 +/-   0.04
                        2  6.82 +/-   0.12
                        3  7.70 +/-   0.07

        {
            'thread_blocks': 1,
            'threads_per_block': 256,
            'gpu_time_usec': {
                'device': {
                    0: (3.90, 0.04),
                    1: (3.54, 0.00),
                    ...
                },
            },
            'cpu_sched_time_usec': {
                'device': {
                    0: (3.18, 0.04),
                    ...
                },
            },
            'cpu_time_usec': {
                'device': {
                    0: (7.16, 0.06),
                    ...
                },
            },
        },
        """
        gpu_time_header_regex = r'(?:Time for running operations on the stream, measuring GPU-side)'
        cpu_sched_time_header_regex = r'(?:Time for scheduling operations to the stream \(not running them\), measuring CPU-side)'
        cpu_time_header_regex = r'(?:Time for scheduling \+ running operations on the stream, measuring CPU-side)'

        def parse_vector(vector, it):
            for line in it:
                m = re.search(r'\s*(?P<device>\d+)\s+(?P<mean>{float})\s+\+/-\s+(?P<std>{float})'.format(float=float_re), line)
                if m:
                    device = int(m.group('device'))
                    mean = float(m.group('mean'))
                    std = float(m.group('std'))
                    put_key(vector, 'device', dict())
                    put_key(vector['device'], device, (mean, std))
                    continue

                m = re.search('^\s*$', line)
                if m:
                    break
            return vector

        def parse_time_vector(results, it, header_regex, time_name):
            m = re.search(header_regex, line)
            if m:
                put_key(results, time_name, dict())
                parse_vector(results[time_name], it)
                return True
            return False

        results = dict()
        # with open(filename) as f:
        #     it = line_iter(f)
        for line in it:
            m = re.search(self.start_header_regex, line)
            if m:
                # Start of CudaLaunch experiment
                break

        for line in it:

            if self.args.debug:
                print("> {klass}, line :: {line}".format(
                    klass=self.__class__.__name__,
                    line=line))

            if self.parse_specific(results, line, it):
                continue

            if parse_time_vector(results, it, gpu_time_header_regex, 'gpu_time_usec'):
                continue
            if parse_time_vector(results, it, cpu_sched_time_header_regex, 'cpu_sched_time_usec'):
                continue
            if parse_time_vector(results, it, cpu_time_header_regex, 'cpu_time_usec'):
                # End of CudaLaunch results
                break

        put_key(all_results, self.microbench_name, results)

        return results

class CUDALaunchParser(CUDAMicroParser):
    def __init__(self, parser, args):
        super().__init__(parser, args,
                         microbench_name='cuda_launch',
                         start_header_regex=r'^CudaLaunch latency:',
                         result_format='vector')

    def parse_specific(self, results, line, it):
        m = re.search(r'CUDA kernel launch with (?P<thread_blocks>\d+) blocks of (?P<threads_per_block>\d+) threads', line)
        if m:
            put_key(results, 'thread_blocks', int(m.group('thread_blocks')))
            put_key(results, 'threads_per_block', int(m.group('threads_per_block')))
            return True
        return False

class CUDAD2HParser(CUDAMicroParser):
    def __init__(self, parser, args):
        super().__init__(parser, args,
                         microbench_name='d2h',
                         start_header_regex=r'^> Device-to-Host latencies',
                         result_format='vector')

class CUDAH2DParser(CUDAMicroParser):
    def __init__(self, parser, args):
        super().__init__(parser, args,
                         microbench_name='h2d',
                         start_header_regex=r'^> Host-to-Device latencies',
                         result_format='vector')

class _BenchmarkFreq:
    def __init__(self, parser, args, ParserType, exec_path):
        self.args = args
        self.parser = parser
        self.ParserType = ParserType
        self.exec_path = exec_path

    @property
    def dump_path(self):
        raise NotImplementedError

    def error(self, msg):
        print("ERROR: {msg}".format(msg=msg))
        sys.exit(1)

    def run_freq(self):
        print("> Running {name} microbenchmarks; output to {path}".format(
            name=type(self).__name__,
            path=self.dump_path))
        if not _e(self.exec_path):
            self.error("Couldn't find CPUFreq microbenchmark executable {exec}; please run 'cmake' from {dir}".format(
                exec=self.exec_path,
                dir=_j(py_config.ROOT, 'build')))
        os.makedirs(_d(self.dump_path), exist_ok=True)
        with open(self.dump_path, 'w') as f:
            cmdline = self.cmdline_array()
            subprocess.check_call(cmdline, stderr=subprocess.PIPE, stdout=f)

    def cmdline_array(self):
        return [self.exec_path]

    def parse_freq(self):
        parser = self.ParserType(self.parser, self.args)
        with open(self.dump_path) as f:
            it = line_iter(f)
            all_results = dict()
            parser.parse(it, all_results)
        return all_results

    def run(self):
        args = self.args
        parser = self.parser

        # PSEUDOCODE:
        # - run p2pBandwidthLatencyTest > cuda_microbench.txt
        # - nice to have, but not a priority:
        #   for num_floating_point_operations in powers_of_two(1, 2, 4, ...):
        #     run cuda-launch microbenchmark
        #     record y=usec/num_floating_point_operations
        #            x=num_floating_point_operations
        #     # NOTE: Just record it for GPU 0

        # if _e(self.dump_path) and not self.args.replace:
        # Ignore replace.
        if _e(self.dump_path):
            print("> Skip {name}; {path} already exists".format(
                name=type(self).__name__,
                path=self.dump_path))
        else:
            self.run_freq()

        self.results = self.parse_freq()

# class BenchmarkCPUFreq(_BenchmarkFreq):
#     def __init__(self, parser, args):
#         exec_path = _j(py_config.ROOT, 'build', 'cpufreq')
#         super().__init__(parser, args, CPUFreqParser, exec_path)
#
#     @property
#     def dump_path(self):
#         return _j(self.args.directory, 'cpufreq.txt')

# class BenchmarkTSCFreq(_BenchmarkFreq):
#     def __init__(self, parser, args):
#         exec_path = _j(py_config.ROOT, 'build', 'clocks')
#         super().__init__(parser, args, TSCFreqParser, exec_path)
#
#     def cmdline_array(self):
#         return [self.exec_path, '--measure_tsc_freq']
#
#     @property
#     def dump_path(self):
#         return _j(self.args.directory, 'tsc_freq.txt')


class BenchmarkCUDA:
    def __init__(self, parser, args):
        self.args = args
        self.parser = parser

    @property
    def _cuda_microbench_path(self):
        return _j(self.args.directory, 'cuda_microbench.txt')

    def error(self, msg):
        print("ERROR: {msg}".format(msg=msg))
        sys.exit(1)

    def cuda_microbench_all(self, bench_name):
        print("> Running CUDA microbenchmarks; output to {path}".format(
            path=self._cuda_microbench_path(bench_name)))
        CUDA_MICROBENCH_EXEC = _j(py_config.ROOT, 'cpp', 'p2pBandwidthLatencyTest', 'p2pBandwidthLatencyTest')
        if not _e(CUDA_MICROBENCH_EXEC):
            self.error("Couldn't find CUDA microbenchmark executable {exec}; please run 'make' from {dir}".format(
                exec=CUDA_MICROBENCH_EXEC,
                dir=_d(CUDA_MICROBENCH_EXEC)))
        with open(self._cuda_microbench_path(bench_name), 'w') as f:
            subprocess.check_call([CUDA_MICROBENCH_EXEC], stderr=subprocess.PIPE, stdout=f)

    def parse_cuda_microbench(self, bench_name):
        parsers = [
            CUDAH2DParser(self.parser, self.args),
            CUDAD2HParser(self.parser, self.args),
            CUDALaunchParser(self.parser, self.args),
        ]
        with open(self._cuda_microbench_path(bench_name)) as f:
            it = line_iter(f)

            all_results = dict()
            for parser in parsers:
                parser.parse(it, all_results)

        return all_results

    def run(self, bench_name):
        args = self.args
        parser = self.parser

        # PSEUDOCODE:
        # - run p2pBandwidthLatencyTest > cuda_microbench.txt
        # - nice to have, but not a priority:
        #   for num_floating_point_operations in powers_of_two(1, 2, 4, ...):
        #     run cuda-launch microbenchmark
        #     record y=usec/num_floating_point_operations
        #            x=num_floating_point_operations
        #     # NOTE: Just record it for GPU 0

        if not args.plot:
            # If we don't JUST want to plot our results.
            if _e(self._cuda_microbench_path(bench_name)) and not self.args.replace:
                print("> Skip CUDA microbenchmarks; {path} already exists".format(path=self._cuda_microbench_path(bench_name)))
            else:
                self.cuda_microbench_all(bench_name)

        self.plot_benchmarks()
        return

    def plot_benchmarks(self, bench_name):
        self.results = self.parse_cuda_microbench(bench_name)

        xs = []
        ys = []
        yerr = []
        # Plot the latency seen by the CUDA API user
        # (don't care how long it ran on the GPU itself, or how long operations get scheduled for).
        time_measurement = 'cpu_time_usec'
        for microbench_name in CUDA_MICROBENCH_NAMES:
            mean, std = self.results[microbench_name][time_measurement]['device'][self.args.gpu]
            # ys.append(mean/MICROSECONDS_IN_SECOND)
            # yerr.append(std/MICROSECONDS_IN_SECOND)
            ys.append(mean)
            yerr.append(std)
            xs.append(microbench_name)

        ys_microseconds = ys
        yerr_microseconds = yerr

        def as_seconds(micros):
            return [x/MICROSECONDS_IN_SECOND for x in micros]

        ys_seconds = as_seconds(ys)
        yerr_seconds = as_seconds(yerr)

        def _plot(ys, yerr, png_basename, ylabel):
            plot_xs_vs_ys(xs, ys, yerr, CUDA_MICROBENCH_NAME_TO_PRETTY,
                          png_basename=png_basename,
                          xlabel='CUDA operation',
                          ylabel=ylabel,
                          title="Latency of GPU operations",
                          directory=self.args.directory)

        _plot(ys_microseconds, yerr_microseconds,
              'cuda.microseconds.png', 'Time (microseconds)')
        _plot(ys_seconds, yerr_seconds,
              'cuda.seconds.png', 'Time (seconds)')

def plot_xs_vs_ys(
    xs,
    ys,
    yerr,
    name_to_pretty,
    png_basename,
    directory=None,
    log_scale=True,
    std_label=True,
    xlabel=None,
    ylabel=None,
    title=None,
    show=False):
    # c_only=False, python_overhead=False):
    """
                     |
                     |
    Time (seconds)   |
                     |
                     |---------------------------------------------
                       micros[0].name     .....

    PROBLEM: not sure what scale each operation will be on.
    TODO: we want each operation to reflect the proportion of operations that run in DQN.
    """
    assert len(xs) == len(ys) == len(yerr)

    png_path = _j(*[x for x in [directory, png_basename] if x is not None])

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    # https://matplotlib.org/examples/api/barchart_demo.html
    width = 0.35
    errorbar_capsize = 10

    xs = [name_to_pretty[name] for name in xs]
    xs = sort_xs_by_ys(xs, ys)
    yerr = sort_xs_by_ys(yerr, ys)
    ys = sorted(ys)

    ind = np.arange(len(xs))/2.

    lists = {'xs':xs,
             'ys':ys,
             'yerr':yerr,
             'ind':ind}
    positive, negative = separate_plus_minus_by(ys, lists)

    def label_bars(rects, xs, ys, yerr, positive):
        """
        Attach a text label above each bar displaying its height
        """
        assert len(rects) == len(xs) == len(ys)
        for rect, x, y, err in zip(rects, xs, ys, yerr):
            # Are we talking about the same bar?
            assert rect.get_height() == y
            # assert rect.get_x() == x

            if std_label:
                bar_label = "{y:f} +/- {std:f}".format(y=y, std=err)
            else:
                bar_label = "{y:f}".format(y=y)

            if positive:
                # Place it above the top of the bar.
                y_pos = 1.05*y
            else:
                # Bar faces downward, place it above the "bottom" of the bar.
                y_pos = 0.05*max(ys)

            ax.text(rect.get_x() + rect.get_width()/2.,
                    y_pos,
                    bar_label,
                    ha='center', va='bottom')

    def add_to_plot(plot_data, color):

        if len(plot_data['ys']) == 0:
            return

        any_is_positive = any(y > 0 for y in plot_data['ys'])
        all_is_positive = all(y > 0 for y in plot_data['ys'])
        assert ( any_is_positive and all_is_positive ) or \
               ( not any_is_positive and not all_is_positive )

        rects1 = ax.bar(plot_data['ind'], plot_data['ys'], width, color=color, yerr=plot_data['yerr'],
                        # bottom=smallest_y,
                        error_kw={
                            'capsize':errorbar_capsize,
                        })

        label_bars(rects1, plot_data['xs'], plot_data['ys'], plot_data['yerr'], positive=any_is_positive)

    add_to_plot(positive, color='r')
    add_to_plot(negative, color='r')

    # import ipdb; ipdb.set_trace()
    ax.set_xticks(ind)
    ax.set_xticklabels(xs)

    if log_scale and not any(y < 0 for y in ys):
        ax.set_yscale("log")
    else:
        print("> WARNING: Saw negative value for {png}; using regular scale instead of log-scale".format(png=png_path))

    ax.set(xlabel=xlabel,
           ylabel=ylabel,
           title=title)

    ax.legend()
    ax.grid()
    print("> Save plot to {path}".format(path=png_path))
    fig.savefig(png_path)
    if show:
        plt.show()


def separate_plus_minus_by(ys, lists):

    def append_to_list(new_lists, i):
        for key in lists.keys():
            new_lists[key].append(lists[key][i])

    def mk_lists():
        return dict((key, []) for key in lists.keys())
    positive = mk_lists()
    negative = mk_lists()

    for i, y in enumerate(ys):
        if y > 0:
            append_to_list(positive, i)
        else:
            append_to_list(negative, i)

    return positive, negative


