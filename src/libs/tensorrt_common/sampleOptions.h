/*
 * Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#ifndef TRT_SAMPLE_OPTIONS_H
#define TRT_SAMPLE_OPTIONS_H

#include <utility>
#include <stdexcept>
#include <vector>
#include <array>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <iostream>

#include "NvInfer.h"

namespace sample
{

// Build default params
constexpr int defaultMaxBatch{1};
constexpr int defaultWorkspace{16};
constexpr int defaultMinTiming{1};
constexpr int defaultAvgTiming{8};

// System default params
constexpr int defaultDevice{0};

// Inference default params
constexpr int defaultBatch{1};
constexpr int defaultStreams{1};
constexpr int defaultIterations{10};
constexpr int defaultWarmUp{200};
constexpr int defaultDuration{10};
constexpr int defaultSleep{0};

// Reporting default params
constexpr int defaultAvgRuns{10};
constexpr float defaultPercentile{99};

enum class ModelFormat {kANY, kCAFFE, kONNX, kUFF};

using Arguments = std::unordered_multimap<std::string, std::string>;

using IOFormat = std::pair<nvinfer1::DataType, nvinfer1::TensorFormats>;

using ShapeRange = std::array<nvinfer1::Dims, nvinfer1::EnumMax<nvinfer1::OptProfileSelector>()>;

struct Options
{
    virtual void parse(Arguments& arguments) = 0;
};

struct BaseModelOptions: public Options
{
    ModelFormat format{ModelFormat::kANY};
    std::string model;

    void parse(Arguments& arguments) override;

    static void help(std::ostream& out);
};

struct UffInput: public Options
{
    std::vector<std::pair<std::string, nvinfer1::Dims>> inputs;
    bool NHWC{false};

    void parse(Arguments& arguments) override;

    static void help(std::ostream& out);
};

struct ModelOptions: public Options
{
    BaseModelOptions baseModel;
    std::string prototxt;
    std::vector<std::string> outputs;
    UffInput uffInputs;

    void parse(Arguments& arguments) override;

    static void help(std::ostream& out);
};

struct BuildOptions: public Options
{
    //bool explicitBatch{false};
    int maxBatch{defaultMaxBatch}; // Parsing sets maxBatch to 0 if explicitBatch is true
    int workspace{defaultWorkspace};
    int minTiming{defaultMinTiming};
    int avgTiming{defaultAvgTiming};
    bool fp16{false};
    bool int8{false};
    bool safe{false};
    bool save{false};
    bool load{false};
    std::string engine;
    std::string calibration;
    std::unordered_map<std::string, ShapeRange> shapes;
    std::vector<IOFormat> inputFormats;
    std::vector<IOFormat> outputFormats;

    void parse(Arguments& arguments) override;

    static void help(std::ostream& out);
};

struct SystemOptions: public Options
{
    int device{defaultDevice};
    int DLACore{-1};
    bool fallback{false};
    std::vector<std::string> plugins;

    void parse(Arguments& arguments) override;

    static void help(std::ostream& out);
};

struct InferenceOptions: public Options
{
    int batch{defaultBatch}; // Parsing sets batch to 0 is shapes is not empty
    int iterations{defaultIterations};
    int warmup{defaultWarmUp};
    int duration{defaultDuration};
    int sleep{defaultSleep};
    int streams{defaultStreams};
    bool spin{false};
    bool threads{true};
    bool graph{false};
    bool skip{false};
    std::unordered_map<std::string, nvinfer1::Dims> shapes;

    void parse(Arguments& arguments) override;

    static void help(std::ostream& out);
};

struct ReportingOptions: public Options
{
    bool verbose{false};
    int avgs{defaultAvgRuns};
    float percentile{defaultPercentile};
    bool output{false};
    bool profile{false};
    std::string exportTimes{};
    std::string exportProfile{};

    void parse(Arguments& arguments) override;

    static void help(std::ostream& out);
};

struct AllOptions: public Options
{
    ModelOptions model;
    BuildOptions build;
    SystemOptions system;
    InferenceOptions inference;
    ReportingOptions reporting;
    bool helps{false};

    void parse(Arguments& arguments) override;

    static void help(std::ostream& out);
};

Arguments argsToArgumentsMap(int argc, char* argv[]);

bool parseHelp(Arguments& arguments);

void helpHelp(std::ostream& out);

// Functions to print options

std::ostream& operator<<(std::ostream& os, const BaseModelOptions& options);

std::ostream& operator<<(std::ostream& os, const UffInput& input);

std::ostream& operator<<(std::ostream& os, const IOFormat& format);

std::ostream& operator<<(std::ostream& os, const nvinfer1::Dims& dims);

std::ostream& operator<<(std::ostream& os, const ShapeRange& dims);

std::ostream& operator<<(std::ostream& os, const ModelOptions& options);

std::ostream& operator<<(std::ostream& os, const BuildOptions& options);

std::ostream& operator<<(std::ostream& os, const SystemOptions& options);

std::ostream& operator<<(std::ostream& os, const InferenceOptions& options);

std::ostream& operator<<(std::ostream& os, const ReportingOptions& options);

std::ostream& operator<<(std::ostream& os, const AllOptions& options);

// Utils to extract options

inline std::vector<std::string> splitToStringVec(const std::string& option, char separator)
{
    std::vector<std::string> options;

    for(size_t start = 0; start < option.length(); )
    {
        size_t separatorIndex = option.find(separator, start);
        if (separatorIndex == std::string::npos)
        {
            separatorIndex = option.length();
        }
        options.emplace_back(option.substr(start, separatorIndex - start));
        start = separatorIndex + 1;
    }

    return options;
}

template <typename T>
inline T stringToValue(const std::string& option)
{
    return T{option};
}

template <>
inline int stringToValue<int>(const std::string& option)
{
    return std::stoi(option);
}

template <>
inline float stringToValue<float>(const std::string& option)
{
    return std::stof(option);
}

template <>
inline bool stringToValue<bool>(const std::string& option)
{
    return true;
}

template <>
inline nvinfer1::Dims stringToValue<nvinfer1::Dims>(const std::string& option)
{
    nvinfer1::Dims dims;
    dims.nbDims = 0;
    std::vector<std::string> dimsStrings = splitToStringVec(option, 'x');
    for (const auto& d : dimsStrings)
    {
        if (d == "*")
        {
            break;
        }
        dims.d[dims.nbDims] = stringToValue<int>(d);
        ++dims.nbDims;
    }
    return dims;
}

template <>
inline nvinfer1::DataType stringToValue<nvinfer1::DataType>(const std::string& option)
{
    const std::unordered_map<std::string, nvinfer1::DataType> strToDT{{"fp32", nvinfer1::DataType::kFLOAT}, {"fp16", nvinfer1::DataType::kHALF},
                                                                      {"int8", nvinfer1::DataType::kINT8}, {"int32", nvinfer1::DataType::kINT32}};
    auto dt = strToDT.find(option);
    if (dt == strToDT.end())
    {
        throw std::invalid_argument("Invalid DataType " + option);
    }
    return dt->second;
}

template <>
inline nvinfer1::TensorFormats stringToValue<nvinfer1::TensorFormats>(const std::string& option)
{
    std::vector<std::string> optionStrings = splitToStringVec(option, '+');
    const std::unordered_map<std::string, nvinfer1::TensorFormat> strToFmt{{"chw", nvinfer1::TensorFormat::kLINEAR}, {"chw2", nvinfer1::TensorFormat::kCHW2},
                                                                           {"chw4", nvinfer1::TensorFormat::kCHW4}, {"hwc8", nvinfer1::TensorFormat::kHWC8},
                                                                           {"chw16", nvinfer1::TensorFormat::kCHW16}, {"chw32", nvinfer1::TensorFormat::kCHW32}};
    nvinfer1::TensorFormats formats{};
    for (auto f : optionStrings)
    {
        auto tf = strToFmt.find(f);
        if (tf == strToFmt.end())
        {
            throw std::invalid_argument(std::string("Invalid TensorFormat ") + f);
        }
        formats |= 1U << int(tf->second);
    }

    return formats;
}

template <>
inline IOFormat stringToValue<IOFormat>(const std::string& option)
{
    IOFormat ioFormat{};
    size_t colon = option.find(':');

    if (colon == std::string::npos)
    {
        throw std::invalid_argument(std::string("Invalid IOFormat ") + option);
    }
    ioFormat.first = stringToValue<nvinfer1::DataType>(option.substr(0, colon));
    ioFormat.second = stringToValue<nvinfer1::TensorFormats>(option.substr(colon+1));

    return ioFormat;
}

inline const char* boolToEnabled(bool enable)
{
    return enable ? "Enabled" : "Disabled";
}

template <typename T>
inline bool checkEraseOption(Arguments& arguments, const std::string& option, T& value)
{
    auto match = arguments.find(option);
    if (match != arguments.end())
    {
        value = stringToValue<T>(match->second);
        arguments.erase(match);
        return true;
    }

    return false;
}

template <typename T>
inline bool checkEraseRepeatedOption(Arguments& arguments, const std::string& option, std::vector<T>& values)
{
    auto match = arguments.equal_range(option);
    if (match.first == match.second)
    {
        return false;
    }
    auto addValue = [&values](Arguments::value_type& value) {values.emplace_back(stringToValue<T>(value.second));};
    std::for_each(match.first, match.second, addValue);
    arguments.erase(match.first, match.second);
    return true;
}

} // namespace sample

#endif // TRT_SAMPLES_OPTIONS_H
