//
// Created by jgleeson on 2020-04-13.
//

#ifndef RLSCOPE_GPU_UTIL_EXPERIMENT_H
#define RLSCOPE_GPU_UTIL_EXPERIMENT_H

#include <boost/optional.hpp>
#include <boost/process.hpp>

#include "common_util.h"

namespace rlscope {

struct GPUUtilExperimentArgs {
    // Environment variables
    boost::optional<std::string> RLSCOPE_PROCESS_NAME;

    // Flags
    boost::optional<bool> FLAGS_debug;
    boost::optional<std::string> FLAGS_rlscope_directory;
    boost::optional<std::string> FLAGS_gpu_clock_freq_json;
    boost::optional<std::string> FLAGS_mode;
    boost::optional<bool> FLAGS_hw_counters;

    boost::optional<int64_t> FLAGS_n_launches;
    boost::optional<int64_t> FLAGS_kernel_delay_us;
    boost::optional<int64_t> FLAGS_kernel_duration_us;
    boost::optional<double> FLAGS_run_sec;
    boost::optional<int64_t> FLAGS_num_threads;
    boost::optional<bool> FLAGS_processes;
    boost::optional<bool> FLAGS_sync;
    boost::optional<bool> FLAGS_cuda_context;
    boost::optional<int64_t> FLAGS_repetitions;
    boost::optional<int64_t> FLAGS_samples;
    boost::optional<std::vector<std::string>> FLAGS_metrics;
    boost::optional<int32_t> FLAGS_device;
    boost::optional<bool> FLAGS_internal_is_child;
    boost::optional<int64_t> FLAGS_internal_thread_id;
    boost::optional<int64_t> FLAGS_kern_arg_iterations;
    boost::optional<int64_t> FLAGS_kern_arg_num_blocks;
    boost::optional<int64_t> FLAGS_kern_arg_threads_per_block;
    boost::optional<int64_t> FLAGS_kern_arg_iterations_per_sched_sample;
    boost::optional<std::string> FLAGS_kernel;

    boost::optional<std::string> FLAGS_gpu_sched_policy;
    // NOT a command-line arguments.
    boost::optional<unsigned int> FLAGS_cuda_context_flags;

    static GPUUtilExperimentArgs FromFlags();


    template<typename OStream, class FlagVar>
    void _PrintFlag(OStream &out, int indent, const std::string &flag_name, FlagVar FLAGS_var) const {
        out << "\n";
        PrintIndent(out, indent + 1);
        out << flag_name << " = ";
        if (!FLAGS_var.has_value()) {
            out << "None";
        } else {
            PrintValue(out, FLAGS_var.get());
        }
    }


    template<typename OStream>
    void Print(OStream &out, int indent) const {
        PrintIndent(out, indent);
        out << "GPUUtilExperimentArgs:";

//    out << "\n";
//    PrintIndent(out, indent + 1);
//    out << "FLAGS_debug = " << FLAGS_debug.get();

#define PRINT_FLAG(FLAGS_var) \
    _PrintFlag(out, indent, #FLAGS_var, FLAGS_var);

        PRINT_FLAG(FLAGS_debug);
        PRINT_FLAG(FLAGS_rlscope_directory);
        PRINT_FLAG(FLAGS_gpu_clock_freq_json);
        PRINT_FLAG(FLAGS_mode);
        PRINT_FLAG(FLAGS_hw_counters);
        PRINT_FLAG(FLAGS_n_launches);
        PRINT_FLAG(FLAGS_kernel_delay_us);
        PRINT_FLAG(FLAGS_kernel_duration_us);
        PRINT_FLAG(FLAGS_run_sec);
        PRINT_FLAG(FLAGS_num_threads);
        PRINT_FLAG(FLAGS_processes);
        PRINT_FLAG(FLAGS_sync);
        PRINT_FLAG(FLAGS_cuda_context);
        PRINT_FLAG(FLAGS_repetitions);
        PRINT_FLAG(FLAGS_samples);
        PRINT_FLAG(FLAGS_metrics);
        PRINT_FLAG(FLAGS_device);
        PRINT_FLAG(FLAGS_internal_is_child);
        PRINT_FLAG(FLAGS_internal_thread_id);
        PRINT_FLAG(FLAGS_kern_arg_iterations);
        PRINT_FLAG(FLAGS_kern_arg_num_blocks);
        PRINT_FLAG(FLAGS_kern_arg_threads_per_block);
        PRINT_FLAG(FLAGS_kern_arg_iterations_per_sched_sample);
        PRINT_FLAG(FLAGS_kernel);
        PRINT_FLAG(FLAGS_gpu_sched_policy);

#undef PRINT_FLAG

//    PrintValue();
    }

    template<typename OStream>
    friend OStream &operator<<(OStream &os, const GPUUtilExperimentArgs &obj) {
        obj.Print(os, 0);
        return os;
    }

};


boost::process::child ReinvokeProcess(const GPUUtilExperimentArgs &overwrite_args, boost::process::environment env);

}

#endif //RLSCOPE_GPU_UTIL_EXPERIMENT_H
