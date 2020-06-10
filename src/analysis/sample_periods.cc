//
// Created by jgleeson on 2020-01-21.
//

#include "common_util.h"
#include "analysis/sample_periods.h"
#include "analysis/trace_file_parser.h"

#include <sstream>

#include <nlohmann/json.hpp>

 using json = nlohmann::json;

namespace rlscope {

template <typename IntType>
IntType ceiling_div(IntType numerator, IntType denominator) {
  return (numerator + (denominator - 1)) / denominator;
}

nlohmann::json PollingUtil::Compute() const {
  // PSEUDOCODE:
  //
  // For process, phase, trace_file in trace_files:
  // gpu_kernels[process][phase] = sorted(gpu kernel times from trace-file)
  //
  // all_gpu_kernels = merge(gpu_kernels[process][phase] forall process, phase in gpu_kernels)
  // Sample_period_duration_us = … // cmd-line argument
  // // Q: what should we use here…?
  // // If we use last GPU kernel time, it won't be accurate and we'll need to
  // // "fill in" sample-periods later on…OK
  // last_end_us = max(end_us for start_us, end_us in all_gpu_kernels)
  // first_start_us = all_gpu_kernels[0].start_us
  // n_bins = Math.ceiling( ( first_start_us - last_end_us ) / polling_interval_us  )
  // // We can use a bit-vector here… std::vector<bool>
  // bins = std::vector<uint8>()
  // bins.resize(n_bins)
  //
  // i = 0
  // cur_period_start_us = 0
  // cur_period_end_us = cur_period_start_us + polling_interval_us
  // For start_us, end_us in all_gpu_kernels:
  //   If start_us > cur_period_end_us:
  //     i = (int) (end_us - start_us)/polling_interval_us
  //     cur_period_start_us = i * polling_interval_us
  //     cur_period_end_us = cur_period_start_us + polling_interval_us
  //   bins[i] = 1
  //
  //
  // NOTE: to parallelize, we can split all_gpu_kernels into 16 chunks, and create 16 locks
  // Thread0 will (potentially) update:
  // - bins[chunk(0)[0].start_us] …
  // - bins[chunk(0)[-1].end_us]
  //
  // Thread1 will (potentially) update:
  // - bins[chunk(1)[0].start_us] …
  // - bins[chunk(1)[-1].end_us]
  //
  // Alternative:
  // - Split, map, merge
  // - Split:
  //   - Make all_gpu_kernels read-only; the "split" just gives start/end indices into all_gpu_kernels
  // - Map:
  //   - Create a bins vector that contains all zeros.
  // - Merge
  //   - OR each entry from each worker
  //   - Q: isn't this just as slow as having one worker do everything…b/c now 1 worker needs to iterate over 16 vectors that are O(n_gpu_kernels)


  // js =
  //  {
  //    metadata: {
  //      start_time_us: ,
  //      end_time_us: ,
  //      polling_interval_us: ,
  //    }
  //    bins: [
  //      # 0/1 for each polling_interval_us between [start_time_us..end_time_us]
  //    ]
  //    training_time_us: [
  //      0*polling_interval_us + start_time_us,
  //      1*polling_interval_us + start_time_us,
  //      ...,
  //    ]
  //  }

  json js;

  CategoryKey gpu_key;
  gpu_key.non_ops.insert(CATEGORY_GPU);
  assert(category_times.eo_times.find(gpu_key) != category_times.eo_times.end());
  const auto& gpu_times = category_times.eo_times.at(gpu_key);

  TimeUsec last_end_us;
  {
    auto last_end_it = std::max_element(
        gpu_times.begin(),
        gpu_times.end(),
        [] (const EOEvent& lhs, const EOEvent& rhs) {
          return lhs.end_time_us() < rhs.end_time_us();
        });
    last_end_us = (*last_end_it).end_time_us();
  }

  TimeUsec first_start_us = (*gpu_times.begin()).start_time_us();

  assert(first_start_us <= last_end_us);
//  size_t n_bins = (last_end_us - first_start_us)/_polling_interval_us;
  size_t n_bins = ceiling_div(last_end_us - first_start_us, _polling_interval_us);

  std::vector<bool> bins;
  bins.resize(n_bins);

  auto bin_idx = [this, first_start_us, &bins] (TimeUsec time_us) -> size_t {
    assert(time_us >= first_start_us);
    size_t i = (time_us - first_start_us)/_polling_interval_us;
    return i;
  };

  size_t num_multi_bin_events = 0;
  for (const auto& event : gpu_times) {
    auto start_i = bin_idx(event.start_time_us());
    auto end_i = bin_idx(event.end_time_us());
    if (end_i != start_i) {
      num_multi_bin_events += 1;
    }
    for (size_t i = start_i; i <= end_i; i++) {
      assert(i < bins.size());
      bins[i] = true;
    }
  }

  js["metadata"]["start_time_us"] = first_start_us;
  js["metadata"]["end_time_us"] = last_end_us;
  js["metadata"]["polling_interval_us"] = _polling_interval_us;
  js["metadata"]["num_multi_bin_events"] = num_multi_bin_events;
  js["bins"] = bins;

  return js;
}

std::string PollingUtil::JSPath() const {
  boost::filesystem::path direc(_iml_directory);
  boost::filesystem::path base = JSBasename();
  return (direc / base).string();
}

std::string PollingUtil::JSBasename() const {
  std::stringstream ss;
  ss << "polling_util";
  ss << ".polling_interval_us_" << _polling_interval_us << "_us";
  ss << ".json";
  return ss.str();
}


} // namespace rlscope

