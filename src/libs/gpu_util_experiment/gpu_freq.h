//
// Created by jgleeson on 2020-01-23.
//

#ifndef IML_GPU_FREQ_H
#define IML_GPU_FREQ_H

#include <pthread.h>

#include <cuda.h>
#include <cuda_runtime.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreorder"
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#pragma GCC diagnostic pop

#include <cmath>
#include <thread>
#include <chrono>
#include <vector>
#include <string>

#include <boost/process.hpp>

#include "range_sampling.h"

#include "gpu_util_experiment.h"

#include "common_util.h"

//#define GPU_CLOCK_MIN_SAMPLE_TIME_SEC (10)
#define GPU_CLOCK_MIN_SAMPLE_TIME_SEC (5)
#define GPU_CLOCK_REPETITIONS (3)
#define GPU_CLOCK_INIT_GPU_SLEEP_CYCLES (1000)
#define GPU_CLOCK_MIN_GUESS_TIME_SEC (1)

#define MICROSECONDS_IN_SECOND (1000000)

//std::stringstream _ss
//_ss << __FILE__ << ":" << __LINE__ << " @ " << __func__ << ": CUDA Failed with (err=" << err << "): " << err_str;
//RLS_LOG("GPU_UTIL", "{}", _ss.str());


namespace rlscope {

template<class Container>
double Average(Container &buffer) {
    double summation = 0;
    int n = 0;
    for (auto x : buffer) {
        summation += x;
        n += 1;
    }
    return summation / ((double) n);
}

template<class Container>
double Std(Container &buffer) {
    auto avg = Average(buffer);
    double summation = 0;
    int n = 0;
    for (auto x : buffer) {
        summation += pow(x - avg, 2);
        n += 1;
    }
    return sqrt(summation / ((double) n));
}

class TimeHistogram {
public:
    using nano_t = uint64_t;

    std::string _name;
    // nanoseconds -> count
    std::map<nano_t, uint64_t> _hist;

    TimeHistogram(const std::string &name) :
            _name(name) {
    }

    template<typename T>
    T _RoundUp(T x, T multiple) const {

        return ((x + multiple - 1) / multiple) * multiple;
    }

    template<typename T>
    T _RoundDown(T x, T multiple) const {
        return (x / multiple) * multiple;
    }

    // Round to to closest multiple:
    //   remainder = x % multiple
    //   if remainder > multiple/2:
    //     _RoundUp(x, multiple)
    //   else:
    //     _RoundDown(x, multiple)
    template<typename T>
    T _RoundClosest(T x, T multiple) const {
        auto remainder = x % multiple;
        if (2 * remainder > multiple) {
            return _RoundUp(x, multiple);
        }
        return _RoundDown(x, multiple);
    }

    void CountNanoseconds(nano_t nanosec) {
        // If nanosec >= 1 second:
        //   round to closest second
        // elif nanosec >= 1 ms:
        //   round to closest ms
        // else nanosec >= 1 us:
        //   round to closest us
#define NANO_IN_SEC static_cast<nano_t>(1*1000*1000*1000)
#define NANO_IN_MS static_cast<nano_t>(1*1000*1000)
#define NANO_IN_US static_cast<nano_t>(1*1000)


        nano_t record_nano;
        if (nanosec >= NANO_IN_SEC) {
            record_nano = _RoundClosest(nanosec, NANO_IN_SEC);
        } else if (nanosec >= NANO_IN_MS) {
            record_nano = _RoundClosest(nanosec, NANO_IN_MS);
//    } else if (nanosec >= NANO_IN_US) {
        } else {
            record_nano = _RoundClosest(nanosec, NANO_IN_US);
        }
        _hist[record_nano] += 1;
    }

    template<typename OStream, typename NanoType>
    void _PrintHumanTime(OStream &out, NanoType nanosec, int indent) const {
        PrintIndent(out, indent);
        if (nanosec >= NANO_IN_SEC) {
            double sec = static_cast<double>(nanosec) / static_cast<double>(NANO_IN_SEC);
            out << sec << " sec";
        } else if (nanosec >= NANO_IN_MS) {
            double ms = static_cast<double>(nanosec) / static_cast<double>(NANO_IN_MS);
            out << ms << " ms";
        } else {
            double us = static_cast<double>(nanosec) / static_cast<double>(NANO_IN_US);
            out << us << " us";
        }
    }

    template<typename OStream>
    void Print(OStream &out, int indent) const {
        PrintIndent(out, indent);
        out << "TimeHistogram: name = " << _name << ", size = " << _hist.size();

        std::vector<nano_t> times_nano;
        times_nano.reserve(_hist.size());
        for (const auto &pair : _hist) {
            auto const &nanosec = pair.first;
            auto const count = pair.second;
            for (uint64_t i = 0; i < count; i++) {
                times_nano.push_back(nanosec);
            }
        }
        auto avg_nano_dbl = Average(times_nano);
        auto std_nano_dbl = Std(times_nano);

        out << "\n";
        PrintIndent(out, indent + 1);
        out << "Average time = ";
        _PrintHumanTime(out, avg_nano_dbl, indent);

        out << "\n";
        PrintIndent(out, indent + 1);
        out << "Stdev time = ";
        _PrintHumanTime(out, std_nano_dbl, indent);

        if (_hist.size() > 0) {
            out << "\n";
            PrintIndent(out, indent + 1);
            out << "Histogram:";
        }

        uint64_t total_count = 0;
        for (const auto &pair : _hist) {
            auto const count = pair.second;
            total_count += count;
        }
        for (const auto &pair : _hist) {
            auto const &nanosec = pair.first;
            auto const count = pair.second;
            double percent = 100 * static_cast<double>(count) / static_cast<double>(total_count);

            out << "\n";
            PrintIndent(out, indent + 2);
            out << "Time = [";
            _PrintHumanTime(out, nanosec, indent);
            out << "]: " << percent << "% (" << count << "/" << total_count << ")";
        }
    }

    template<typename OStream>
    friend OStream &operator<<(OStream &os, const TimeHistogram &obj) {
        obj.Print(os, 0);
        return os;
    }

};

using clock_value_t = long long;
// https://www.softwariness.com/articles/monotonic-clocks-windows-and-posix/
//  using std::chrono::steady_clock;
using time_type = std::chrono::steady_clock::time_point;

double elapsed_sec(time_type start, time_type stop);

time_type time_now();

struct CudaStreamWrapper {
    cudaStream_t _handle;

    void synchronize();

    CudaStreamWrapper();

    ~CudaStreamWrapper();
};

class CudaStream {
public:
    // We want to inherit shared_ptr copy/move constructor functionality, but we need new/delete
    // to create a cudaStream_t using cudaStreamCreate/cudaStreamDestroy.
    uint64_t _stream_id;
    std::shared_ptr<CudaStreamWrapper> _stream;

//  cudaStream_t _stream;
    CudaStream();

    cudaStream_t get() const;

    void synchronize();

    void _set_name(const std::string &name);

    void set_stream_id(uint64_t stream_id);
};

struct CudaContextWrapper {
    CUcontext _handle;
    unsigned int _flags;
    CUdevice _dev = 0;

    CudaContextWrapper();

    CudaContextWrapper(unsigned int flags);

    void _Init();

    ~CudaContextWrapper();

    void synchronize();
};

class CudaContext {
public:
    std::shared_ptr<CudaContextWrapper> _context;

    CudaContext();

    CudaContext(unsigned int flags);

    CUcontext get() const;

    void synchronize();
};

template<typename T>
struct CudaHostBufferWrapper {
    T *_handle;
    size_t _n_elems;

    CudaHostBufferWrapper(size_t n_elems) :
            _handle(nullptr),
            _n_elems(n_elems) {
        cudaError_t ret;
        RUNTIME_API_CALL_MAYBE_EXIT(cudaMallocHost(&_handle, size_bytes()));
        if (_handle == nullptr) {
          std::stringstream ss;
          ss << "Failed to allocate a CUDA host buffer of " << size_bytes() << " bytes";
          MyStatus status = MyStatus(error::INVALID_ARGUMENT, ss.str());
          IF_BAD_STATUS_EXIT_WITH(status);
          // RLS_LOG("GPU_UTIL", "Failed to allocate a CUDA host buffer of {} bytes", size_bytes());
        }
        assert(_handle != nullptr);
    }

    size_t size_bytes() const {
        return _n_elems * sizeof(T);
    }

    ~CudaHostBufferWrapper() {
        cudaError_t ret;
        if (_handle) {
            RUNTIME_API_CALL_MAYBE_EXIT(cudaFreeHost(_handle));
            _handle = nullptr;
        }
    }
};

template<typename T>
class CudaHostBuffer {
public:
    // We want to inherit shared_ptr copy/move constructor functionality, but we need new/delete
    // to create a cudaStream_t using cudaStreamCreate/cudaStreamDestroy.
    std::shared_ptr<CudaHostBufferWrapper<T>> _buffer;
    size_t _n_elems;
//  cudaStream_t _stream;

    CudaHostBuffer(size_t n_elems) :
            _buffer(new CudaHostBufferWrapper<T>(n_elems)),
            _n_elems(n_elems) {
    }

    T *get() const {
        return _buffer->_handle;
    }

    size_t num_elems() const {
        return _n_elems;
    }

};

//struct GPUClockResult {
//  double avg_mhz;
//  double std_mhz;
//};

// We want to expose a function for sleeping for a given number of cycles
// (used by GPUClockFreq).
// Sleeping for a given number of seconds requires GPUClockFreq's calculation of GPU clock rate.
class GPUSleeper {
public:
//  CudaStream _stream;
    CudaHostBuffer<int64_t> _output;

    GPUSleeper() :
            _output(1) {
    }

    void gpu_sleep_cycles(CudaStream stream, clock_value_t sleep_cycles, bool sync);

    void gpu_sleep_cycles_sync(CudaStream stream, clock_value_t sleep_cycles);

};

class GPUKernel {
public:
    GPUUtilExperimentArgs args;

    GPUKernel() = default;

    GPUKernel(GPUUtilExperimentArgs args) :
            args(args) {
    }

    // TODO: this needs to be called AFTER CUDA context has been allocated...
    virtual MyStatus Init() = 0;

    virtual MyStatus ResetPass() {
        return MyStatus::OK();
    }

    virtual MyStatus CheckArgs() = 0;

    virtual std::unique_ptr<GPUKernel> clone() const = 0;

    virtual void RunSync(CudaStream stream) = 0;

    virtual void RunAsync(CudaStream stream) = 0;

    virtual MyStatus DumpKernelInfo(int thread_id, CudaStream stream) = 0;
};

class GPUComputeKernel : public GPUKernel {
public:
    struct RunCtx {
        CudaHostBuffer<int64_t> output;

        RunCtx(size_t n_elems) :
                output(n_elems) {
        }
    };

    std::unique_ptr<RunCtx> run_ctx;

    GPUComputeKernel(GPUUtilExperimentArgs args
    ) : GPUKernel(args) {
    }

    void _gpu_compute_kernel(CudaStream stream, bool sync);

    virtual MyStatus Init() override;

    virtual MyStatus CheckArgs() override;

    virtual std::unique_ptr<GPUKernel> clone() const override;

    virtual void RunSync(CudaStream stream) override;

    virtual void RunAsync(CudaStream stream) override;

    virtual MyStatus DumpKernelInfo(int thread_id, CudaStream stream) override;

};


struct GPUThreadSchedInfo {
    // A numeric identifier that tells us which CUDA stream this kernel was launched to.
    // Useful for determining which CPU thread this kernel is launched from (given we know which thread uses which CUDA stream).
    uint64_t stream_id;
    // kernel_id = i if this is the i-th kernel launched to this stream.
    uint64_t kernel_id;
    // Determined by the GPU-side hardware scheduler.
    // We can record these to determine how the GPU multiplexes kernels.
    uint32_t sm_id;
    uint32_t warp_id;
    uint32_t lane_id;
    // GPU-side nanosecond timestamp collected using the special %globaltimer register.
    // We can use this to establish global order of GPU-side events.
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-globaltimer
    uint64_t globaltimer_ns;
};

struct GPUThreadSchedInfoVectors {
    std::vector<decltype(GPUThreadSchedInfo::stream_id)> stream_id;
    std::vector<decltype(GPUThreadSchedInfo::kernel_id)> kernel_id;
    std::vector<decltype(GPUThreadSchedInfo::sm_id)> sm_id;
    std::vector<decltype(GPUThreadSchedInfo::warp_id)> warp_id;
    std::vector<decltype(GPUThreadSchedInfo::lane_id)> lane_id;
    std::vector<decltype(GPUThreadSchedInfo::globaltimer_ns)> globaltimer_ns;

    static GPUThreadSchedInfoVectors FromStructArray(const GPUThreadSchedInfo *sched_info, size_t n_elems) {
        GPUThreadSchedInfoVectors self;
        self.stream_id.reserve(n_elems);
        self.kernel_id.reserve(n_elems);
        self.sm_id.reserve(n_elems);
        self.warp_id.reserve(n_elems);
        self.lane_id.reserve(n_elems);
        self.globaltimer_ns.reserve(n_elems);
        for (size_t i = 0; i < n_elems; i++) {
            self.stream_id.push_back(sched_info[i].stream_id);
            self.kernel_id.push_back(sched_info[i].kernel_id);
            self.sm_id.push_back(sched_info[i].sm_id);
            self.warp_id.push_back(sched_info[i].warp_id);
            self.lane_id.push_back(sched_info[i].lane_id);
            self.globaltimer_ns.push_back(sched_info[i].globaltimer_ns);
        }
        return self;
    }
};

class GPUComputeSchedInfoKernel : public GPUKernel {
public:
    // WARNING: If you add fields, you MUST update GPUComputeSchedInfoKernel::clone
    cudaDeviceProp device_prop;
    uint64_t gpu_base_timestamp_ns;
    uint64_t device_base_timestamp_ns;
    timestamp_us cpu_base_timestamp_us;
    int64_t trace_id;

    struct RunCtx {
        GPUUtilExperimentArgs args;
        cudaDeviceProp device_prop;
        uint64_t kernel_id;
        CudaHostBuffer<int64_t> output;
        CudaHostBuffer<GPUThreadSchedInfo> sched_info;

        RunCtx(
                GPUUtilExperimentArgs args,
                cudaDeviceProp device_prop,
                size_t n_elems
        ) :
                args(args),
                device_prop(device_prop),
                kernel_id(0),
                output(n_elems),
                sched_info(ComputeNumTotalSchedSamples(args, device_prop)) {
        }

        void ResetPass() {
            kernel_id = 0;
        }

        static size_t ComputeNumTotalSchedSamples(
                const GPUUtilExperimentArgs &args,
                const cudaDeviceProp &device_prop);


    };

    std::unique_ptr<RunCtx> run_ctx;

    GPUComputeSchedInfoKernel(GPUUtilExperimentArgs args
    ) :
            GPUKernel(args),
            gpu_base_timestamp_ns(0),
            device_base_timestamp_ns(0),
            trace_id(0)
    {
    }

    void _gpu_compute_kernel(CudaStream stream, bool sync);

    virtual MyStatus Init() override;
    virtual MyStatus ResetPass() override;

    virtual MyStatus CheckArgs() override;

    virtual std::unique_ptr<GPUKernel> clone() const override;

    virtual void RunSync(CudaStream stream) override;

    virtual void RunAsync(CudaStream stream) override;

    virtual MyStatus DumpKernelInfo(int thread_id, CudaStream stream) override;

};

class GPUClockFreq : public GPUKernel {
public:
    GPUSleeper _gpu_sleeper;

    std::vector<double> _time_secs;
    std::vector<double> _freq_mhz;
    double _avg_mhz;
    double _std_mhz;
//  std::vector<clock_value_t> time_cycles;
    // Number of cycles to sleep for on the GPU.
    clock_value_t _sleep_cycles;
//  int _repetitions;
//  GPUClockResult _result;
//  std::string _directory;

    GPUClockFreq(GPUUtilExperimentArgs args
    ) :
            GPUKernel(args), _avg_mhz(0.), _std_mhz(0.), _sleep_cycles(GPU_CLOCK_INIT_GPU_SLEEP_CYCLES) {
    }

    virtual MyStatus Init() override;

    virtual std::unique_ptr<GPUKernel> clone() const override;

    virtual void RunSync(CudaStream stream) override;

    virtual void RunAsync(CudaStream stream) override;

    virtual MyStatus CheckArgs() override;

    virtual MyStatus DumpKernelInfo(int thread_id, CudaStream stream) override;

    void guess_cycles(CudaStream stream);

    void iter(CudaStream stream, time_type *start_t, time_type *end_t);

    double freq_mhz(double time_sec);

    void gpu_sleep_sec(CudaStream stream, double seconds);

    void _gpu_sleep_us(CudaStream stream, int64_t usec, bool sync);

    void gpu_sleep_us(CudaStream stream, int64_t usec);

    void gpu_sleep_us_sync(CudaStream stream, int64_t usec);

    void run();

    MyStatus dump_json() const;

    MyStatus load_json(const std::string &path);

    std::string json_path() const;

    std::string json_basename() const;
};


MyStatus GetGPUKernel(GPUUtilExperimentArgs args, std::unique_ptr<GPUKernel> *gpu_kernel);

struct SyncBlock;
struct SharedMem;

class GPUKernelRunner {
public:
    struct RunContext {
        // NOTE: We DON'T make these member variables, since we need to create it AFTER separate threads have started running
        // (the context is BOUND to the current thread once created with cuCtxCreate).
        int _thread_id;
        std::unique_ptr<CudaContext> _context;
        std::unique_ptr<CudaStream> _stream;

        RunContext(int thread_id, bool cuda_context, unsigned int cuda_context_flags) :
                _thread_id(thread_id) {
            if (cuda_context) {
                // Create a per-thread CUDA context.
                // NEED to create CUcontext before anything else (e.g., streams).
                _context.reset(new CudaContext(cuda_context_flags));
            }
            _stream.reset(new CudaStream());
            // https://stackoverflow.com/questions/44266820/how-can-i-access-the-numeric-stream-ids-seen-in-nvprof-using-a-cudastream-t
            // It's no clear how to obtain the 0-based stream ids reported by nvprof, so instead assign them manually.
            _stream->set_stream_id(_thread_id);
        }

        void synchronize() {
            _stream->synchronize();
            if (_context) {
                _context->synchronize();
            }
        }
    };

    // Run context is not created until run() starts (i.e. AFTER thread is created).
    // std::unique_ptr<RunContext> _run_ctx;
    std::unique_ptr<std::thread> _async_thread;
    std::unique_ptr<boost::process::child> _async_process;

    GPUUtilExperimentArgs args;

    std::unique_ptr<GPUKernel> _gpu_kernel;
    int _thread_id;


    GPUKernelRunner(
            int thread_id,
            std::unique_ptr<GPUKernel> gpu_kernel,
            GPUUtilExperimentArgs args
    ) :
            args(args), _gpu_kernel(std::move(gpu_kernel)), _thread_id(thread_id) {
    }

    void DelayUs(int64_t usec);

    void run_child_thread();
    void run_child_process();

    void SetCurrentThreadName(const std::string &name);

    void run_async_thread();

    void wait_thread();

    void run_async_process(int thread_id);

    void wait_process(int thread_id);

    MyStatus _run_child_pass(SyncBlock *sync_block, RunContext* _run_ctx);

    void
    _setup_child_thread(std::unique_ptr<RunContext> *_run_ctx, rlscope::SharedMem *shared_mem,
                        SyncBlock **sync_block);
};

struct SharedMem {
public:
    std::string _name;
    size_t _size_bytes;
    boost::interprocess::managed_shared_memory _segment;

    SharedMem() : _size_bytes(0) {

    }

private:
    SharedMem(const std::string &name, size_t size_bytes) :
            _name(name),
            _size_bytes(size_bytes) {
    }

    SharedMem(const std::string &name) :
            _name(name),
            _size_bytes(0) {
    }

public:
    static SharedMem Child(const std::string &name) {
        SharedMem shared_mem(name);
        shared_mem._segment = boost::interprocess::managed_shared_memory(boost::interprocess::open_only, name.c_str());
        shared_mem._size_bytes = shared_mem._segment.get_size();
        return shared_mem;
    }

    static SharedMem Parent(const std::string &name, size_t size_bytes) {
        SharedMem shared_mem(name, size_bytes);
        // FUN C++ FACT: destructors WON'T be called during SIGINT (ctrl-c).
        // So, we CANNOT depend on a named shared memory region being cleaned up.
        // HACK: remove it if it exists.
        boost::interprocess::shared_memory_object::remove(name.c_str());
        shared_mem._segment = boost::interprocess::managed_shared_memory(boost::interprocess::create_only, name.c_str(),
                                                                         size_bytes);
        return shared_mem;
    }

    template<typename OStream>
    void Print(OStream &out, int indent) const {
        PrintIndent(out, indent);
        out << "SharedMem(name=" << _name << ", size_bytes=" << _size_bytes << ")";
    }

    template<typename OStream>
    friend OStream &operator<<(OStream &os, const SharedMem &obj) {
        obj.Print(os, 0);
        return os;
    }

};

#define BARRIER_NAME_MAX_LEN 256
struct InterProcessBarrier {
    boost::interprocess::interprocess_mutex mutex;
    boost::interprocess::interprocess_condition barrier_limit_break;

    // Current generation's barrier_reason.
    char barrier_reason[BARRIER_NAME_MAX_LEN];

    size_t num_threads;
    // Number of threads that need to call arrive_and_wait before all threads are awoken,
    // for the CURRENT <generation>.
    size_t n_threads_left;

    // https://stackoverflow.com/a/27118537
    // Need this to make the barrier "multi-use".
    size_t generation;

    InterProcessBarrier(size_t num_threads) :
            num_threads(num_threads),
            n_threads_left(num_threads),
            generation(0) {
    }

    void arrive_and_wait(const std::string& reason) {
        boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> lock(mutex);
        auto gen = generation;
        assert(n_threads_left >= 1);

        if (n_threads_left == num_threads) {
            assert(reason.size() < BARRIER_NAME_MAX_LEN);
            strncpy(barrier_reason, reason.c_str(), reason.size());
            barrier_reason[reason.size()] = '\0';
        } else {
            if (reason != barrier_reason) {
                std::cerr << "ERROR: saw thread arriving at barrier with reason=\"" << reason << "\", but there are already "
                          << (num_threads - n_threads_left) << " threads waiting at barrier with reason=\"" << barrier_reason << "\""
                          << std::endl;
               assert(reason == barrier_reason);
            }
        }

        n_threads_left -= 1;
        if (n_threads_left == 0) {
            n_threads_left = num_threads;
            generation += 1;
            barrier_limit_break.notify_all();
        } else {
            barrier_limit_break.wait(lock, [this, gen] {
               return gen != this->generation;
            });
        }
    }

    template<typename OStream>
    void Print(OStream &out, int indent) const {
        PrintIndent(out, indent);
        out << "InterProcessBarrier(n_threads_left=" << n_threads_left << ", num_threads=" << num_threads << ", generation=" << generation << ")";
    }

    template<typename OStream>
    friend OStream &operator<<(OStream &os, const InterProcessBarrier &obj) {
        obj.Print(os, 0);
        return os;
    }

};

struct SyncBlock {
    size_t n_threads;

    InterProcessBarrier barrier;

    std::atomic<bool> should_stop;

    size_t num_passes;

    SyncBlock(size_t n_threads) :
            n_threads(n_threads),
            barrier(n_threads),
            should_stop(false),
            num_passes(0) {
    }

    ~SyncBlock();

    static SyncBlock *Child(SharedMem *shared_mem, const std::string &name) {
        std::pair<SyncBlock *, boost::interprocess::managed_shared_memory::size_type> res;
        res = shared_mem->_segment.find<SyncBlock>(name.c_str());
        //Length should be 1
        assert(res.second == 1);
        return res.first;
    }

    template<typename ...Args>
    static SyncBlock *Parent(SharedMem *shared_mem, const std::string &name, Args &&...args) {
        return shared_mem->_segment.construct<SyncBlock>
                (name.c_str())  //name of the object
                (std::forward<Args>(args)...);            //ctor first argument
    }


    template<typename OStream>
    void Print(OStream &out, int indent) const {
        PrintIndent(out, indent);
        out << "SyncBlock(barrier=" << barrier << ", should_stop=" << should_stop << ")";
    }

    template<typename OStream>
    friend OStream &operator<<(OStream &os, const SyncBlock &obj) {
        obj.Print(os, 0);
        return os;
    }

};

#define SYNC_BLOCK_NAME "SyncBlock"
//#define SYNC_BLOCK_INIT_COUNTER 1337
#define SHARED_MEM_NAME "SharedMem"
#define SHARED_MEM_SIZE_BYTES (1*1024*1024)

class ThreadedGPUKernelRunner {
public:
    SharedMem _shared_mem;
    SyncBlock *_sync_block;

//  // "Number of kernels to launch per-thread."
//  int64_t _n_launches;
//  // "Time between kernel launches in microseconds"
//  int64_t _kernel_delay_us;
//  // "Duration of kernel in microseconds"
//  int64_t _kernel_duration_us;
//  // "How to long to run for (in seconds)"
//  double _run_sec;
//  size_t _num_threads;
//  bool _processes;
//  // After launching a kernel, wait for it to finish.
//  // Useful for running really long kernels (e.g., 10 sec)
//  // without creating a giant queue of kernel launches (e.g., delay=1us)
//  bool _sync;
//  bool _cuda_context;
//  bool _internal_is_child;

    GPUUtilExperimentArgs args;

    std::unique_ptr<GPUKernel> _gpu_kernel;
//  std::string _directory;
//  bool _debug;

    ThreadedGPUKernelRunner(
            std::unique_ptr<GPUKernel> gpu_kernel,
            GPUUtilExperimentArgs args

//      int64_t n_launches,
//      int64_t kernel_delay_us,
//      int64_t kernel_duration_us,
//      double run_sec,
//      size_t num_threads,
//      bool processes,
//      bool sync,
//      bool cuda_context,
//      bool internal_is_child,
//      const std::string& directory,
//      bool debug
    ) :
            _sync_block(nullptr),
            args(args),
//      _n_launches(n_launches),
//      _kernel_delay_us(kernel_delay_us),
//      _kernel_duration_us(kernel_duration_us),
//      _run_sec(run_sec),
//      _num_threads(num_threads),
//      _processes(processes),
//      _sync(sync),
//      _cuda_context(cuda_context),
//      _internal_is_child(internal_is_child),
            _gpu_kernel(std::move(gpu_kernel))
//      _directory(directory),
//      _debug(debug)
    {
    }

    void run();

    void run_parent();
};


} // namespace rlscope

#endif //IML_GPU_FREQ_H
