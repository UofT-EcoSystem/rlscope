//
// Created by jgleeson on 2020-05-14.
//
#ifndef CUPTI_RANGE_SAMPLING_H
#define CUPTI_RANGE_SAMPLING_H

#include "common_util.h"

#include <cupti_target.h>
#include <cupti_profiler_target.h>
#include <nvperf_target.h>
#include <nvperf_host.h>

#include "range_sampling/range_sampling.pb.h"

#include <vector>
#include <list>
#include <string>
#include <regex>
#include <chrono>

#include <boost/asio/thread_pool.hpp>
#include <boost/asio/post.hpp>

namespace rlscope {

struct RangeNode {
    RangeNode* parent;
    std::string name;
    // push(A), push(B)
    // => A is a parent-of B
    //    B is a child-of A
    std::map<std::string, std::shared_ptr<RangeNode>> children;

    RangeNode(const std::string& name) :
            parent(nullptr),
            name(name) {
    }

    RangeNode(RangeNode* parent, const std::string& name) :
            parent(parent),
            name(name) {
    }

};

struct RangeTree {
    size_t max_nesting_levels;
    size_t max_unique_ranges;

    // Not a "real" push() annotation.  Use to handle multiple root push() annotations.
    std::shared_ptr<RangeNode> root;
    RangeNode* cur_node;
    size_t cur_depth;

    RangeTree() :
            max_nesting_levels(0)
            , max_unique_ranges(0)
            , root(new RangeNode("[ROOT]"))
            , cur_node(root.get())
            , cur_depth(0)
    {
    }

    void Push(const std::string& name);
    void Pop();

    void _UpdateStatsOnPush(bool was_insert);
    void _UpdateStatsOnPop();
};

struct ConfigData {
    std::string chipName;
    std::vector<std::string> metricNames;
    uint16_t counter_data_max_num_nesting_levels;

    std::vector<uint8_t> configImage;

    ConfigData() :
            counter_data_max_num_nesting_levels(0)
    {
    }

    ConfigData(
            const std::string& chipName,
            const std::vector<std::string>& metricNames,
            uint16_t counter_data_max_num_nesting_levels) :
            chipName(chipName),
            metricNames(metricNames),
            counter_data_max_num_nesting_levels(counter_data_max_num_nesting_levels)
    {
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

};

struct CounterData {
    std::string chipName;
    std::vector<std::string> metricNames;
    uint32_t counter_data_max_num_ranges;

    std::vector<uint8_t> counterDataImage;
    std::vector<uint8_t> counterDataScratchBuffer;
    std::vector<uint8_t> counterDataImagePrefix;

    CounterData() :
            counter_data_max_num_ranges(0)
    {
    }

    size_t size() const {
        return counterDataImage.size() + counterDataScratchBuffer.size() + counterDataImagePrefix.size();
    }

    size_t size_bytes() const {
        // We only end up storing counterDataImage in the protobuf.
        return counterDataImage.size() * sizeof(counterDataImage[0]);
    }

    CounterData(
            const std::string& chipName
            , const std::vector<std::string>& metricNames
            , uint32_t counter_data_max_num_ranges) :
            chipName(chipName)
            , metricNames(metricNames)
            , counter_data_max_num_ranges(counter_data_max_num_ranges) {
    }

    MyStatus Init();

    MyStatus _InitConfigImagePrefix();

    MyStatus getNumRangeCollected(size_t* numRanges) const;

};

struct CUPTIProfilerState {
    size_t counter_data_max_num_ranges;

    bool _endPassParams_allPassesSubmitted;

    CUPTIProfilerState() :
            counter_data_max_num_ranges(0),
            _endPassParams_allPassesSubmitted(false) {
    }

    CUPTIProfilerState(size_t counter_data_max_num_ranges) :
            counter_data_max_num_ranges(counter_data_max_num_ranges) {
    }

    bool HasNextPass() const;

    MyStatus StartPass(ConfigData& config_data);
    MyStatus EndPass();

    MyStatus StartProfiling(ConfigData& config_data, CounterData& counter_data);
    MyStatus StopProfiling(ConfigData& config_data, CounterData& counter_data);
    MyStatus Flush(ConfigData& config_data, CounterData& counter_data);
    MyStatus NextPass(ConfigData& config_data, CounterData& counter_data);

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
            int trace_id
            , const std::string& directory
            , const std::string& dump_suffix
            , const std::string& chip_name
            , const std::vector<std::string>& metrics
            , const ConfigData& config_data
            , std::list<std::unique_ptr<GPUHwCounterSamplerState>>&& samples
            , size_t num_passes
    ) :
            _trace_id(trace_id)
            , _directory(directory)
            , _dump_suffix(dump_suffix)
            , _chip_name(chip_name)
            , _metrics(metrics)
            , _config_data(config_data)
            , _samples(std::move(samples))
            , _num_passes(num_passes)
    {
    }

    std::unique_ptr<iml::GPUHwCounterSampleProto> AsProto();
    std::string DumpPath() const;

    MyStatus DumpSync();
};

class GPUHwCounterSampler {
public:
    enum Mode {
        PROFILE,
        CONFIG,
        EVAL,
    };

    static const size_t MaxSampleFileSizeBytes;

    int _device;
    GPUHwCounterSampler::Mode _mode;
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

    boost::asio::thread_pool _pool;

    size_t _size_bytes;

    bool _enabled;

    GPUHwCounterSampler(int device, std::string directory, std::string dump_suffix)
            : _device(device),
              _mode(EVAL),
              _directory(directory),
              _dump_suffix(dump_suffix),
              _pass_idx(0),
              _next_trace_id(0),
              _initialized(false),
              _pool(/*num_threads=*/4),
              _size_bytes(0),
              _enabled(true)
    {
    }

    size_t size_bytes() const {
        return _size_bytes;
    }

    GPUHwCounterSamplerProtoState AsProtoState();

    MyStatus CheckCUPTIProfilingAPISupported();
    MyStatus StartConfig(const std::vector<std::string>& metrics);
    MyStatus StartProfiling();
    MyStatus StopProfiling();
    bool HasNextPass() const;
    size_t NumPasses() const;
    MyStatus StartPass();
    MyStatus EndPass();
    bool CanRecord() const;

    static const std::regex FilenameRegex;
    // TODO: load protobuf file.
    // We can keep config in each protobuf file...for now.
    bool IsProtoFile(const boost::filesystem::path &path);

    inline size_t MaxNestingLevels() const {
        return _range_tree.max_nesting_levels;
    }
    // HACK: use twice the max nesting level seen at runtime during config; hopefully it's big enough.
    inline size_t UseMaxNestingLevels() const {
        return 2*MaxNestingLevels();
    }

    inline size_t MaxUniqueRanges() const {
        return _range_tree.max_unique_ranges;
    }
    // HACK: use twice the max unique ranges seen at runtime during config; hopefully it's big enough.
    inline size_t UseMaxUniqueRanges() const {
        return 2*MaxUniqueRanges();
    }

//    bool IsEnabled();
    MyStatus Push(const std::string& operation);
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

    MyStatus PrintCSV(std::ostream& out, bool& printed_header);

    MyStatus PrintCSV(std::ostream &out, const iml::GPUHwCounterSampleProto &proto, bool& printed_header);

    MyStatus
    PrintCSV(std::ostream &out, const std::string &chipName, const uint8_t *counterDataImage,
             size_t counterDataImageSize,
             const std::vector<std::string> &metricNames, bool &printed_header,
             const std::vector<std::string> &extra_headers = {},
             const std::vector<std::string> &extra_fields = {});

    bool ShouldDump() const;

    MyStatus Disable();

    bool Enabled() const;
};

} // namespace rlscope

#endif //CUPTI_RANGE_SAMPLING_H
