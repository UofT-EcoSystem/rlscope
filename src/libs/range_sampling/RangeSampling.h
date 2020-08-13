//
// Created by jgleeson on 2020-05-14.
//
#pragma once

#include "common_util.h"

#include <cupti_target.h>
#include <cupti.h>
#include <cupti_profiler_target.h>
#include <nvperf_target.h>
#include <nvperf_host.h>

#include "range_sampling/range_sampling.pb.h"

#include <vector>
#include <list>
#include <string>
#include <regex>
#include <chrono>
#include <functional>

#include <boost/asio/thread_pool.hpp>
#include <boost/asio/post.hpp>

// Allocate extra 1MB on the left/right of each profiling data buffer (config data, scratch buffer, etc.).
// Check the 1MB regions regularly for buffer overruns.
// #define CONFIG_CHECK_PROF_BUFFER_OVERFLOW

namespace rlscope {

std::vector<std::string> get_DEFAULT_METRICS();
std::string get_DEFAULT_METRICS_STR();

struct RangeNode {
  RangeNode *parent;
  std::string name;
  // push(A), push(B)
  // => A is a parent-of B
  //    B is a child-of A
  std::map<std::string, std::shared_ptr<RangeNode>> children;

  RangeNode(const std::string &name) :
      parent(nullptr),
      name(name) {
  }

  RangeNode(RangeNode *parent, const std::string &name) :
      parent(parent),
      name(name) {
  }

};

struct RangeTree;
struct RangeTreeStats { 
  size_t max_nesting_levels{0};
  size_t max_unique_ranges{0};
  size_t max_num_ranges{0};
  // Range name length (according to NVIDIA profiling API's definition...I think).
  // range = [training_loop, q_forward]
  // range_name = "training_loop/q_forward"
  // len(range_name) + 1 = 23 + 1
  //                     = 24
  size_t max_range_name_length{0};
  bool initialized{false};

  RangeTreeStats() = default;
  RangeTreeStats(const RangeTree& range_tree);
};
struct RangeTree {
  using Stack = std::list<const RangeNode*>;


  // Not a "real" push() annotation.  Use to handle multiple root push() annotations.
  std::shared_ptr<RangeNode> root;
  RangeNode *cur_node;
  size_t cur_depth;
  size_t cur_range_name_length;
  size_t cur_num_ranges;

  RangeTreeStats stats;
  RangeTreeStats recorded_stats;

  RangeTree() : root(new RangeNode("[ROOT]")),
                cur_node(root.get()),
                cur_depth(0),
                cur_range_name_length(0),
                stats(*this),
                recorded_stats()
  {
  }

  const RangeTreeStats& RecordedStats() const;

  MyStatus Push(const std::string &name, bool update_stats);
  void Pop();

  void StartPass(bool update_stats);
  void EndPass(bool update_stats);

  void _RecordStats();

  RangeTree::Stack CurStack() const;

  using EachStackSeenCb = std::function<void(const RangeTree::Stack&)>;
  void EachStackSeen(EachStackSeenCb func) const;
  void _EachStackSeen(RangeNode* node, RangeTree::Stack stack, EachStackSeenCb func) const;
  template <class OStream>
  void PrintStack(OStream& out, int indent, RangeTree::Stack node_stack) {
    int i = 0;
    PrintIndent(out, indent);
    out << "[";
    for (auto const& node : node_stack) {
      if (i != 0) {
        out << ", ";
      }
      out << node->name;
      i += 1;
    }
    out << "]";
  }
  template <class OStream>
  void PrintStacks(OStream& out, int indent) {
    size_t i = 0;
    this->EachStackSeen([&i, &out, indent, this] (const RangeTree::Stack& stack) {
      PrintIndent(out, indent);
      out << (i + 1) << ": ";
      this->PrintStack(out, 0, stack);
      out << "\n";
      i += 1;
    });
    PrintIndent(out, indent);
    out << "Saw " << i << " stacks in total.";
  }

  size_t CurRangeNameLength() const;

  void _UpdateStatsOnPush(bool was_insert);

  void _UpdateStatsOnPop();
};

template <typename T, T guard_value, size_t guard_size_bytes, T dflt_value = T()>
struct GuardedBuffer {
  //
  // [  guard_size bytes  ][  buffer  ][  guard_size bytes  ]
  //          0xbe                              0xbe
  //
  size_t _n_elems;
  size_t _alloc_count;
//  T _dflt_value;
//  T _guard_value;
  size_t _guard_size_elems;
  size_t _guard_size_bytes;
  std::vector<T> _buffer;

  GuardedBuffer() {
    _n_elems = 0;
    _alloc_count = 0;
    _init_bytes_and_elems();
  }

  GuardedBuffer(size_t n_elems
      // size_t guard_size_bytes, T guard_value, T dflt_value = T()
          )
  {
    _n_elems = n_elems;
    _alloc_count = 0;
//    _dflt_value = dflt_value;
    size_t _alloc_count;
//    _guard_value = guard_value;
    _init_bytes_and_elems();
    _init_buffer();
  }

  void _init_bytes_and_elems() {
    _guard_size_elems = ( (guard_size_bytes + sizeof(T) - 1) / sizeof(T) );
    _guard_size_bytes = _guard_size_elems*sizeof(T);
  }

  void _init_buffer() {
    _buffer.resize(actual_size());
    _alloc_count += 1;
    if (!(_alloc_count <= 1)) {
      std::cerr << "Saw more than one allocation of buffer (alloc_count = " << _alloc_count  << ")." << std::endl;
      rlscope::DumpStacktrace(std::cerr, 4);
      assert(_alloc_count <= 1);
    }
    // Left guard bytes = _guard_value
    std::fill_n(_buffer.begin(), _guard_size_elems, guard_value);
    // Usable bytes = _dflt_value
    std::fill_n(_buffer.begin() + _guard_size_elems, _n_elems, dflt_value);
    // Right guard bytes = _guard_value
    std::fill_n(_buffer.begin() + _guard_size_elems + _n_elems, _guard_size_elems, guard_value);
    // Sanity check.
    check();
  }

  void resize(size_t n_elems) {
    _n_elems = n_elems;
    _init_buffer();
  }

  // Return USABLE data (don't include left/right guard bytes)
  inline size_t size() const {
    return _n_elems;
  }

  // INCLUDE guard bytes.
  inline size_t actual_size() const {
    return _guard_size_elems*2 + _n_elems;
  }

  inline void check_i(size_t i) const {
    if (_buffer[i] != guard_value) {
      std::cerr << "Saw corrupted data in buffer "
                << "(usable size = " << _n_elems*sizeof(T)
                << ", left/right guard bytes = " << _guard_size_bytes << ")." << std::endl;
      const char* location = nullptr;
      const char* offset_sign = nullptr;
      size_t array_offset = 0;
      if (i < _guard_size_bytes) {
        location = "[left buffer guard]";
        array_offset = _guard_size_elems - i;
        offset_sign = "-";
      } else if (i < _guard_size_bytes + _n_elems) {
        // Shouldn't be checking usable area.
        assert(false);
      } else {
        location = "[right buffer guard]";
        array_offset = i - (_guard_size_bytes + _n_elems);
        offset_sign = "+";
      }
      std::cerr << "  Buffer offset ('+' = beyond end, '-' = before start): "
                << location << " " << offset_sign << array_offset << std::endl;
      rlscope::DumpStacktrace(std::cerr, 4);
      assert(_buffer[i] == guard_value);
    }
  }

  void check() const {
    if (_buffer.size() == 0) {
      // Default constructed buffer; hasn't been resize()ed yet.
      return;
    }
    // Left guard bytes.
    for (size_t i = 0; i < _guard_size_elems; i++) {
      check_i(i);
    }
    // Right guard bytes.
    for (size_t i = _guard_size_elems + _n_elems; i < _guard_size_elems; i++) {
      check_i(i);
    }
  }

  // Return USABLE data (don't include left/right guard bytes)
  inline const T* data() const {
    return _buffer.data() + _guard_size_elems;
  }
  inline T* data() {
    return _buffer.data() + _guard_size_elems;
  }
  T& operator[](size_t i) {
    assert(i >= 0);
    assert(i < _n_elems);
    return _buffer[i + _guard_size_elems];
  }
  const T& operator[](size_t i) const {
    assert(i >= 0);
    assert(i < _n_elems);
    return _buffer[i + _guard_size_elems];
  }

};
#define KB_BYTES (1024)
#define MB_BYTES (1024*KB_BYTES)
#define GUARD_BYTE (0xbe)
using GuardedByteBuffer = GuardedBuffer<uint8_t, GUARD_BYTE, MB_BYTES>;

#ifdef CONFIG_CHECK_PROF_BUFFER_OVERFLOW
using ProfilingByteBuffer = GuardedByteBuffer;
#else
using ProfilingByteBuffer = std::vector<uint8_t>;
#endif // CONFIG_CHECK_PROF_BUFFER_OVERFLOW

struct ConfigData {
  std::string chipName;
  std::vector<std::string> metricNames;
  uint16_t counter_data_max_num_nesting_levels;

  ProfilingByteBuffer configImage;

  ConfigData() :
      counter_data_max_num_nesting_levels(0) {
  }

  ConfigData(
      const std::string &chipName,
      const std::vector<std::string> &metricNames,
      uint16_t counter_data_max_num_nesting_levels) :
      chipName(chipName),
      metricNames(metricNames),
      counter_data_max_num_nesting_levels(counter_data_max_num_nesting_levels) {
  }

  size_t size_bytes() const {
    return configImage.size() * sizeof(configImage[0]);
  }

  std::unique_ptr<iml::ConfigDataProto> AsProto();

  MyStatus Init();

//    MyStatus _GetRawMetricRequests(NVPA_MetricsContext* pMetricsContext,
//                               const std::vector<std::string>& metricNames,
//                               std::vector<NVPA_RawMetricRequest>& rawMetricRequests,
//                               std::vector<std::string>& temp);


  MyStatus _InitConfigImage();

  size_t size() const {
    return configImage.size();
  }

#ifdef CONFIG_CHECK_PROF_BUFFER_OVERFLOW
  void _Check() const {
    configImage.check();
  }
#endif

};

struct CounterData {
  std::string chipName;
  std::vector<std::string> metricNames;
  uint32_t counter_data_max_num_ranges;
  uint32_t counter_data_maxRangeNameLength;

  ProfilingByteBuffer counterDataImage;
  ProfilingByteBuffer counterDataScratchBuffer;
  ProfilingByteBuffer counterDataImagePrefix;

  CounterData() :
      counter_data_max_num_ranges(0),
      counter_data_maxRangeNameLength(0) {
  }

  size_t size() const {
    return counterDataImage.size() + counterDataScratchBuffer.size() + counterDataImagePrefix.size();
  }

  size_t size_bytes() const {
    // We only end up storing counterDataImage in the protobuf.
    return counterDataImage.size() * sizeof(counterDataImage[0]);
  }

  CounterData(
      const std::string &chipName,
      const std::vector<std::string> &metricNames,
      uint32_t counter_data_max_num_ranges,
      uint32_t counter_data_maxRangeNameLength)
      : chipName(chipName),
        metricNames(metricNames),
        counter_data_max_num_ranges(counter_data_max_num_ranges),
        counter_data_maxRangeNameLength(counter_data_maxRangeNameLength)
  {
  }

  MyStatus Init();

  MyStatus _InitConfigImagePrefix();

  MyStatus getNumRangeCollected(size_t *numRanges) const;

#ifdef CONFIG_CHECK_PROF_BUFFER_OVERFLOW
  void _Check() const {
    counterDataImage.check();
    counterDataScratchBuffer.check();
    counterDataImagePrefix.check();
  }
#endif

};

struct CUPTIProfilerState {
  size_t counter_data_max_num_ranges;

  bool _endPassParams_allPassesSubmitted;
  bool _profiler_running;
  bool _pass_running;

  CUPTIProfilerState() :
      counter_data_max_num_ranges(0) {
    _ConstructorInit();
  }

  CUPTIProfilerState(size_t counter_data_max_num_ranges) :
      counter_data_max_num_ranges(counter_data_max_num_ranges) {
    _ConstructorInit();
  }

  void _ConstructorInit() {
    _endPassParams_allPassesSubmitted = false;
    _profiler_running = false;
    _pass_running = false;
  }

  bool HasNextPass() const;

  MyStatus StartPass(ConfigData &config_data);

  MyStatus EndPass();

  MyStatus StartProfiling(ConfigData &config_data, CounterData &counter_data);

  MyStatus StopProfiling(ConfigData &config_data, CounterData &counter_data);

  MyStatus Flush(ConfigData &config_data, CounterData &counter_data);

  MyStatus NextPass(ConfigData &config_data, CounterData &counter_data);

  MyStatus _InitConfig(ConfigData &config_data);
};


struct GPUHwCounterSamplerState {
  std::string directory;
  ConfigData config_data;
  CounterData counter_data;
  rlscope::timestamp_us start_profiling_t;
  rlscope::timestamp_us stop_profiling_t;

//    timestamp_us start_profiling_t;
//    timestamp_us stop_profiling_t;

  bool CanDump() const;

  std::unique_ptr<iml::CounterDataProto> AsProto();

  size_t size_bytes() const {
    return config_data.size_bytes() + counter_data.size_bytes();
  }

#ifdef CONFIG_CHECK_PROF_BUFFER_OVERFLOW
  void _Check() const {
    config_data._Check();
    counter_data._Check();
  }
#endif

};

// State needed to create a GPUHwCounterSampleProto.
struct GPUHwCounterSamplerProtoState {
  int _trace_id;
  std::string _directory;
  std::string _dump_suffix;
  std::string _chip_name;
  std::vector<std::string> _metrics;
  ConfigData _config_data;
  std::list<std::unique_ptr<GPUHwCounterSamplerState>> _samples;
  size_t _num_passes;

  GPUHwCounterSamplerProtoState(
      int trace_id, const std::string &directory, const std::string &dump_suffix, const std::string &chip_name,
      const std::vector<std::string> &metrics, const ConfigData &config_data,
      std::list<std::unique_ptr<GPUHwCounterSamplerState>> &&samples, size_t num_passes
  ) :
      _trace_id(trace_id), _directory(directory), _dump_suffix(dump_suffix), _chip_name(chip_name), _metrics(metrics),
      _config_data(config_data), _samples(std::move(samples)), _num_passes(num_passes) {
  }

  std::unique_ptr<iml::GPUHwCounterSampleProto> AsProto();

  std::string DumpPath() const;

  MyStatus DumpSync();
};

#define GPU_HW_COUNTER_SAMPLER_MODE_PROFILE 0
#define GPU_HW_COUNTER_SAMPLER_MODE_CONFIG 1
#define GPU_HW_COUNTER_SAMPLER_MODE_EVAL 2

enum GPUHwCounterSamplerMode {
  PROFILE = 0,
  CONFIG = 1,
  EVAL = 2,
};

class GPUHwCounterSampler {
public:
//  enum Mode {
//    PROFILE,
//    CONFIG,
//    EVAL,
//  };

  static const size_t MaxSampleFileSizeBytes;

  static const char* ModeString(GPUHwCounterSamplerMode mode);

  int _device;
  GPUHwCounterSamplerMode _mode;
  std::string _directory;
  std::string _dump_suffix;

  std::list<std::unique_ptr<GPUHwCounterSamplerState>> _samples;

  std::string _chip_name;

  std::vector<std::string> _metrics;

  RangeTree _range_tree;
  // Max depth of any one stack.
//    uint16_t max_nesting_levels;
  // Total number of unique "stacks".
//    uint16_t max_unique_ranges;

  GPUHwCounterSamplerState state;
  CUPTIProfilerState profiler_state;

  size_t _pass_idx;
//    uint64_t trace_id;
  uint64_t _next_trace_id;
  bool _initialized;
  int _initialized_device;
  bool RLS_GPU_HW_SKIP_PROF_API;

  boost::asio::thread_pool _pool{4};

  size_t _size_bytes;

  bool _enabled;
  bool _running_pass;

  CUcontext _context;

  GPUHwCounterSampler() {
    _device = -1;
    _ConstructorInit();
  }

  GPUHwCounterSampler(int device, std::string directory, std::string dump_suffix) {
    _device = device;
    _directory = directory;
    _dump_suffix = dump_suffix;
    _ConstructorInit();
  }

  void _ConstructorInit() {
    _mode = EVAL;
    _pass_idx = 0;
    _next_trace_id = 0;
    _initialized = false;
    _initialized_device = -1;
    RLS_GPU_HW_SKIP_PROF_API = is_yes("RLS_GPU_HW_SKIP_PROF_API", false);
    _size_bytes = 0;
    _enabled = true;
    _running_pass = false;
    _context = nullptr;
  }

  void SetDirectory(std::string &directory);

  void SetDevice(int device);

  size_t size_bytes() const {
    return _size_bytes;
  }

  GPUHwCounterSamplerProtoState AsProtoState();

  MyStatus CheckCUPTIProfilingAPISupported();

  MyStatus StartConfig(const std::vector<std::string> &metrics);

  MyStatus _CheckInitialized(const char* file, int lineno, const char* func) const;

  MyStatus StartProfiling();

  MyStatus StopProfiling();

  bool HasNextPass() const;

  size_t NumPasses() const;

  MyStatus StartPass();

  MyStatus EndPass();

  GPUHwCounterSamplerMode Mode() const;

  bool CanRecord() const;

  static const std::regex FilenameRegex;

  // TODO: load protobuf file.
  // We can keep config in each protobuf file...for now.
  bool IsProtoFile(const boost::filesystem::path &path);


  size_t MaxNestingLevels() const;
  size_t UseMaxNestingLevels() const;
  size_t MaxUniqueRanges() const;
  size_t MaxRangeNameLength() const;
  size_t UseMaxRangeNameLength() const;
  size_t MaxNumRanges() const;
  size_t UseMaxNumRanges() const;



//    bool IsEnabled();
  MyStatus Push(const std::string &operation);

  MyStatus Pop();
//    MyStatus Start();
//    MyStatus Stop();
//    MyStatus Print();

//    MyStatus SetMetadata(const char* directory, const char* process_name, const char* machine_name, const char* phase_name) override;
//    MyStatus AsyncDump() override;
//    MyStatus AwaitDump() override;
  MyStatus Init();

  ~GPUHwCounterSampler();

//    MyStatus _SyncDumpWithState(GPUHwCounterSamplerState&& dump_state_);

  MyStatus _InitSamplerState();

  MyStatus _MaybeDump(bool *dumped);

  MyStatus _NextSamplerState();

  MyStatus _DumpAsync();

  MyStatus _DumpSync();

  MyStatus DumpSync();

  MyStatus DumpAsync();

  MyStatus _Dump(bool sync);

  MyStatus AwaitDump();

  MyStatus _MaybeRecordSample(bool *recorded);

  MyStatus RecordSample();

  bool CanDump() const;

  MyStatus PrintCSV(std::ostream &out, bool &printed_header);

  MyStatus PrintCSV(std::ostream &out, const iml::GPUHwCounterSampleProto &proto, bool &printed_header);

  MyStatus
  PrintCSV(std::ostream &out, const std::string &chipName, const uint8_t *counterDataImage,
           size_t counterDataImageSize,
           const std::vector<std::string> &metricNames, bool &printed_header,
           const std::vector<std::string> &extra_headers = {},
           const std::vector<std::string> &extra_fields = {});

  bool ShouldDump() const;

  MyStatus Disable();

  bool Enabled() const;

#ifdef CONFIG_CHECK_PROF_BUFFER_OVERFLOW
  void _Check() const {
    state._Check();
  }
#endif

};

} // namespace rlscope

