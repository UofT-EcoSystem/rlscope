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

#include <cstring>
#include <string>
#include <vector>
#include <functional>
#include <algorithm>
#include <stdexcept>
#include <iostream>

#include "NvInfer.h"

#include "sampleOptions.h"

namespace sample
{

Arguments argsToArgumentsMap(int argc, char* argv[])
{
    Arguments arguments;
    for (int i = 1; i < argc; ++i)
    {
        auto valuePtr = strchr(argv[i], '=');
        if (valuePtr)
        {
            std::string value{valuePtr+1};
            arguments.emplace(std::string(argv[i], valuePtr-argv[i]), value);
        }
        else
        {
            arguments.emplace(argv[i], "");
        }
    }
    return arguments;
}

void BaseModelOptions::parse(Arguments& arguments)
{
    if (checkEraseOption(arguments, "--onnx", model))
    {
        format = ModelFormat::kONNX;
    }
    else if (checkEraseOption(arguments, "--uff", model))
    {
        format = ModelFormat::kUFF;
    }
    else if (checkEraseOption(arguments, "--model", model))
    {
        format = ModelFormat::kCAFFE;
    }
}

void UffInput::parse(Arguments& arguments)
{
    checkEraseOption(arguments, "--uffNHWC", NHWC);
    std::vector<std::string> args;
    if (checkEraseRepeatedOption(arguments, "--uffInput", args))
    {
        for (const auto& i: args)
        {
            std::vector<std::string> values{splitToStringVec(i, ',')};
            if (values.size() == 4)
            {
                nvinfer1::Dims3 dims{std::stoi(values[1]), std::stoi(values[2]), std::stoi(values[3])};
                inputs.emplace_back(values[0], dims);
            }
            else
            {
                throw std::invalid_argument(std::string("Invalid uffInput ") + i);
            }
        }
    }
}

void ModelOptions::parse(Arguments& arguments)
{
    baseModel.parse(arguments);

    switch (baseModel.format)
    {
    case ModelFormat::kCAFFE:
    {
        checkEraseOption(arguments, "--deploy", prototxt);
        break;
    }
    case ModelFormat::kUFF:
    {
        uffInputs.parse(arguments);
        if (uffInputs.inputs.empty())
        {
            throw std::invalid_argument("Uff models require at least one input");
        }
        break;
    }
    case ModelFormat::kONNX:
        break;
    case ModelFormat::kANY:
    {
        if (checkEraseOption(arguments, "--deploy", prototxt))
        {
            baseModel.format = ModelFormat::kCAFFE;
        }
        break;
    }
    }
    if (baseModel.format == ModelFormat::kCAFFE || baseModel.format == ModelFormat::kUFF)
    {
        std::vector<std::string> outArgs;
        if (checkEraseRepeatedOption(arguments, "--output", outArgs))
        {
            for (const auto& o: outArgs)
            {
                for (auto& v: splitToStringVec(o, ','))
                {
                    outputs.emplace_back(std::move(v));
                }
            }
        }
        if (outputs.empty())
        {
            throw std::invalid_argument("Caffe and Uff models require at least one output");
        }
    }
}

namespace
{

void insertShapes(std::unordered_map<std::string, ShapeRange>& shapes, const std::string& name, const nvinfer1::Dims& dims)
{
    std::pair<std::string, ShapeRange> profile;
    profile.first = name;
    profile.second[static_cast<size_t>(nvinfer1::OptProfileSelector::kMIN)] = dims;
    profile.second[static_cast<size_t>(nvinfer1::OptProfileSelector::kOPT)] = dims;
    profile.second[static_cast<size_t>(nvinfer1::OptProfileSelector::kMAX)] = dims;
    shapes.insert(profile);
}

}

void BuildOptions::parse(Arguments& arguments)
{
    auto getFormats = [&arguments](std::vector<IOFormat>& formatsVector, const char* argument)
    {
        std::string list;
        checkEraseOption(arguments, argument, list);
        std::vector<std::string> formats{splitToStringVec(list, ',')};
        for (const auto& f : formats)
        {
            formatsVector.push_back(stringToValue<IOFormat>(f));
        }
    };

    getFormats(inputFormats, "--inputIOFormats");
    getFormats(outputFormats, "--outputIOFormats");

    auto getShapes = [&arguments](std::unordered_map<std::string, ShapeRange>& shapes, const char* argument, nvinfer1::OptProfileSelector selector)
    {
        std::string list;
        checkEraseOption(arguments, argument, list);
        std::vector<std::string> shapeList{splitToStringVec(list, ',')};
        for (const auto& s : shapeList)
        {
            std::vector<std::string> nameRange{splitToStringVec(s, ':')};
            if (shapes.find(nameRange[0]) == shapes.end())
            {
                auto dims = stringToValue<nvinfer1::Dims>(nameRange[1]);
                insertShapes(shapes, nameRange[0], dims);
            }
            else
            {
                shapes[nameRange[0]][static_cast<size_t>(selector)] = stringToValue<nvinfer1::Dims>(nameRange[1]);
            }
        }
    };

    bool explicitBatch{false};
    checkEraseOption(arguments, "--explicitBatch", explicitBatch);
    getShapes(shapes, "--minShapes", nvinfer1::OptProfileSelector::kMIN);
    getShapes(shapes, "--optShapes", nvinfer1::OptProfileSelector::kOPT);
    getShapes(shapes, "--maxShapes", nvinfer1::OptProfileSelector::kMAX);
    explicitBatch = explicitBatch || !shapes.empty();

    int batch{0};
    checkEraseOption(arguments, "--maxBatch", batch);
    if (explicitBatch && batch)
    {
        throw std::invalid_argument("Explicit batch or dynamic shapes enabled with implicit maxBatch " + std::to_string(batch));
    }

    if (explicitBatch)
    {
        maxBatch = 0;
    }
    else
    {
        if (batch)
        {
            maxBatch = batch;
        }
    }
    
    checkEraseOption(arguments, "--workspace", workspace);
    checkEraseOption(arguments, "--minTiming", minTiming);
    checkEraseOption(arguments, "--avgTiming", avgTiming);
    checkEraseOption(arguments, "--fp16", fp16);
    checkEraseOption(arguments, "--int8", int8);
    checkEraseOption(arguments, "--safe", safe);
    checkEraseOption(arguments, "--calib", calibration);
    if (checkEraseOption(arguments, "--loadEngine", engine))
    {
        load = true;
    }
    if (checkEraseOption(arguments, "--saveEngine", engine))
    {
        save = true;
    }
    if (load && save)
    {
        throw std::invalid_argument("Incompatible load and save engine options selected");
    }
}

void SystemOptions::parse(Arguments& arguments)
{
    checkEraseOption(arguments, "--device", device);
    checkEraseOption(arguments, "--useDLACore", DLACore);
    checkEraseOption(arguments, "--allowGPUFallback", fallback);
    std::string pluginName;
    while (checkEraseOption(arguments, "--plugins", pluginName))
    {
        plugins.emplace_back(pluginName);
    }
}

void InferenceOptions::parse(Arguments& arguments)
{
    checkEraseOption(arguments, "--streams", streams);
    checkEraseOption(arguments, "--iterations", iterations);
    checkEraseOption(arguments, "--duration", duration);
    checkEraseOption(arguments, "--warmUp", warmup);
    checkEraseOption(arguments, "--useSpinWait", spin);
    checkEraseOption(arguments, "--threads", threads);
    checkEraseOption(arguments, "--useCudaGraph", graph);
    checkEraseOption(arguments, "--buildOnly", skip);

    std::string list;
    checkEraseOption(arguments, "--shapes", list);
    std::vector<std::string> shapeList{splitToStringVec(list, ',')};
    for (const auto& s : shapeList)
    {
        std::vector<std::string> shapeSpec{splitToStringVec(s, ':')};
        shapes.insert({shapeSpec[0], stringToValue<nvinfer1::Dims>(shapeSpec[1])});
    }

    int batchOpt{0};
    checkEraseOption(arguments, "--batch", batchOpt);
    if (!shapes.empty() && batchOpt)
    {
        throw std::invalid_argument("Explicit batch or dynamic shapes enabled with implicit batch " + std::to_string(batchOpt));
    }
    if (batchOpt)
    {
        batch = batchOpt;
    }
    else
    {
        if (!shapes.empty())
        {
            batch = 0;
        }
    }

}

void ReportingOptions::parse(Arguments& arguments)
{
    checkEraseOption(arguments, "--percentile", percentile);
    checkEraseOption(arguments, "--avgRuns", avgs);
    checkEraseOption(arguments, "--verbose", verbose);
    checkEraseOption(arguments, "--dumpOutput", output);
    checkEraseOption(arguments, "--dumpProfile", profile);
    checkEraseOption(arguments, "--exportTimes", exportTimes);
    checkEraseOption(arguments, "--exportProfile", exportProfile);
    if (percentile < 0 || percentile > 100)
    {
        throw std::invalid_argument(std::string("Percentile ") + std::to_string(percentile) + "is not in [0,100]");
    }
}

bool parseHelp(Arguments& arguments)
{
    bool help{false};
    checkEraseOption(arguments, "--help", help);
    return help;
}

void AllOptions::parse(Arguments& arguments)
{
    model.parse(arguments);
    build.parse(arguments);
    system.parse(arguments);
    inference.parse(arguments);

    if ((!build.maxBatch && inference.batch && inference.batch != defaultBatch) ||
        (build.maxBatch && build.maxBatch != defaultMaxBatch && !inference.batch))
    {
        // If either has selected implict batch and the other has selected explicit batch
        throw std::invalid_argument("Conflicting build and inference batch settings");
    }

    if (build.shapes.empty() && !inference.shapes.empty())
    {
        for (auto& s : inference.shapes)
        {
            insertShapes(build.shapes, s.first, s.second);
        }
        build.maxBatch = 0;
    }
    else
    {
        if (!build.shapes.empty() && inference.shapes.empty())
        {
            for (auto& s : build.shapes)
            {
                inference.shapes.insert({s.first, s.second[static_cast<size_t>(nvinfer1::OptProfileSelector::kOPT)]});
            }
        }
        if (!build.maxBatch)
        {
            inference.batch = 0;
        }
    }

    if (build.maxBatch && inference.batch)
    {
        // For implicit batch, check for compatibility and if --maxBatch is not given and inference batch is greater
        // than maxBatch, use inference batch also for maxBatch
        if (build.maxBatch != defaultMaxBatch && build.maxBatch < inference.batch)
        {
            throw std::invalid_argument("Build max batch " + std::to_string(build.maxBatch) +
                                        " is less than inference batch " + std::to_string(inference.batch));
        }
        else
        {
            if (build.maxBatch < inference.batch)
            {
                build.maxBatch = inference.batch;
            }
        }
    }

    reporting.parse(arguments);
    helps = parseHelp(arguments);

    if (!helps)
    {
        if (!build.load && model.baseModel.format == ModelFormat::kANY)
        {
            throw std::invalid_argument("Model missing or format not recognized");
        }
        if (!build.load && !build.maxBatch && model.baseModel.format != ModelFormat::kONNX)
        {
            throw std::invalid_argument("Explicit batch size not supported for Caffe and Uff models");
        }
        if (build.safe && system.DLACore >= 0)
        {
            auto checkSafeDLAFormats = [](const std::vector<IOFormat>& fmt)
            {
                return fmt.empty() ? false : std::all_of(fmt.begin(), fmt.end(), [](const IOFormat& pair)
                {
                    bool supported{false};
                    supported |= pair.first == nvinfer1::DataType::kINT8 &&
                                 pair.second == 1U << static_cast<int>(nvinfer1::TensorFormat::kCHW32);
                    supported |= pair.first == nvinfer1::DataType::kHALF &&
                                 pair.second == 1U << static_cast<int>(nvinfer1::TensorFormat::kCHW16);
                    return supported;
                });
            };
            if (!checkSafeDLAFormats(build.inputFormats) || !checkSafeDLAFormats(build.inputFormats))
            {
                throw std::invalid_argument("I/O formats for safe DLA capability are restricted to fp16:chw16 or int8:chw32");
            }
            if (system.fallback)
            {
                throw std::invalid_argument("GPU fallback (--allowGPUFallback) not allowed for safe DLA capability");
            }
        }
    }
}

std::ostream& operator<<(std::ostream& os, const BaseModelOptions& options)
{
    os << "=== Model Options ===" << std::endl;

    os << "Format: ";
    switch (options.format)
    {
    case ModelFormat::kCAFFE:
    {
        os << "Caffe";
        break;
    }
    case ModelFormat::kONNX:
    {
        os << "ONNX";
        break;
    }
    case ModelFormat::kUFF:
    {
        os << "UFF";
        break;
    }
    case ModelFormat::kANY:
        os << "*";
        break;
    }
    os << std::endl << "Model: " << options.model << std::endl;

    return os;
}

std::ostream& operator<<(std::ostream& os, const UffInput& input)
{
    os << "Uff Inputs Layout: " << (input.NHWC ? "NHWC" : "NCHW") << std::endl;
    for (const auto& i : input.inputs)
    {
        os << "Input: " << i.first << "," << i.second.d[0] << "," << i.second.d[1] << "," << i.second.d[2] << std::endl;
    }

    return os;
}

std::ostream& operator<<(std::ostream& os, const ModelOptions& options)
{
    os << options.baseModel;
    switch (options.baseModel.format)
    {
    case ModelFormat::kCAFFE:
    {
        os << "Prototxt: " << options.prototxt;
        break;
    }
    case ModelFormat::kUFF:
    {
        os << options.uffInputs;
        break;
    }
    case ModelFormat::kONNX: // Fallthrough: No options to report for ONNX or the generic case
    case ModelFormat::kANY:
        break;
    }

    os << "Output:";
    for (const auto& o : options.outputs)
    {
        os << " " << o;
    }
    os << std::endl;

    return os;
}

std::ostream& operator<<(std::ostream& os, const IOFormat& format)
{
    switch(format.first)
    {
    case nvinfer1::DataType::kFLOAT:
    {
        os << "fp32:";
        break;
    }
    case nvinfer1::DataType::kHALF:
    {
        os << "fp16:";
        break;
    }
    case nvinfer1::DataType::kINT8:
    {
        os << "int8:";
        break;
    }
    case nvinfer1::DataType::kINT32:
    {
        os << "int32:";
        break;
    }
    }

    for(int f = 0; f < nvinfer1::EnumMax<nvinfer1::TensorFormat>(); ++f)
    {
        if ((1U<<f) & format.second)
        {
            if (f)
            {
                os << "+";
            }
            switch(nvinfer1::TensorFormat(f))
            {
            case nvinfer1::TensorFormat::kLINEAR:
            {
                os << "chw";
                break;
            }
            case nvinfer1::TensorFormat::kCHW2:
            {
                os << "chw2";
                break;
            }
            case nvinfer1::TensorFormat::kHWC8:
            {
                os << "hwc8";
                break;
            }
            case nvinfer1::TensorFormat::kCHW4:
            {
                os << "chw4";
                break;
            }
            case nvinfer1::TensorFormat::kCHW16:
            {
                os << "chw16";
                break;
            }
            case nvinfer1::TensorFormat::kCHW32:
            {
                os << "chw32";
                break;
            }
            }
        }
    }
    return os;
};

std::ostream& operator<<(std::ostream& os, const nvinfer1::Dims& dims)
{
    for (int i = 0; i < dims.nbDims; ++i)
    {
        os << ( i ? "x" : "" ) << dims.d[i];
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const ShapeRange& dims)
{
    int i = 0;
    for (const auto& d : dims)
    {
        if (!d.nbDims)
        {
            break;
        }
        os << ( i ? "+" : "" ) << d;
        ++i;
    }
    return os;
}

namespace
{

template <typename T>
void printShapes(std::ostream& os, const char* phase, const T& shapes)
{
    if (shapes.empty())
    {
        os << "Input " << phase << " shapes: model" << std::endl;
    }
    else
    {
        for (const auto& s : shapes)
        {
            os << "Input " << phase << " shape: " << s.first << "=" << s.second << std::endl;
        }
    }
}

std::ostream& printBatch(std::ostream& os, int maxBatch)
{
    if (maxBatch)
    {
        os << maxBatch;
    }
    else
    {
        os << "explicit";
    }
    return os;
}

}

std::ostream& operator<<(std::ostream& os, const BuildOptions& options)
{
// clang-format off
    os << "=== Build Options ==="                                                                                       << std::endl <<

          "Max batch: ";        printBatch(os, options.maxBatch)                                                        << std::endl <<
          "Workspace: "      << options.workspace << " MB"                                                              << std::endl <<
          "minTiming: "      << options.minTiming                                                                       << std::endl <<
          "avgTiming: "      << options.avgTiming                                                                       << std::endl <<
          "Precision: "      << (options.fp16 ? "FP16" : (options.int8 ? "INT8" : "FP32"))                              << std::endl <<
          "Calibration: "    << (options.int8 && options.calibration.empty() ? "Dynamic" : options.calibration.c_str()) << std::endl <<
          "Safe mode: "      << boolToEnabled(options.safe)                                                             << std::endl <<
          "Save engine: "    << (options.save ? options.engine : "")                                                    << std::endl <<
          "Load engine: "    << (options.load ? options.engine : "")                                                    << std::endl;
// clang-format on

    auto printIOFormats = [](std::ostream& os, const char* direction, const std::vector<IOFormat> formats)
    {
        if (formats.empty())
        {
            os << direction << "s format: fp32:CHW" << std::endl;
        }
        else
        {
            for(const auto& f : formats)
            {
                os << direction << ": " << f << std::endl;
            }
        }
    };

    printIOFormats(os, "Input", options.inputFormats);
    printIOFormats(os, "Output", options.outputFormats);
    printShapes(os, "build", options.shapes);

    return os;
}

std::ostream& operator<<(std::ostream& os, const SystemOptions& options)
{
// clang-format off
    os << "=== System Options ==="                                                                << std::endl <<

          "Device: "  << options.device                                                           << std::endl <<
          "DLACore: " << (options.DLACore != -1 ? std::to_string(options.DLACore) : "")           <<
                         (options.DLACore != -1 && options.fallback ? "(With GPU fallback)" : "") << std::endl;
// clang-format on
    os << "Plugins:";
    for (const auto p : options.plugins)
    {
        os << " " << p;
    }
    os << std::endl;

    return os;
}

std::ostream& operator<<(std::ostream& os, const InferenceOptions& options)
{
// clang-format off
    os << "=== Inference Options ==="                                        << std::endl <<

          "Batch: ";
    if (options.batch && options.shapes.empty())
    {
                          os << options.batch                                << std::endl;
    }
    else
    {
                          os << "Explicit"                                   << std::endl;
    }
    os << "Iterations: "     << options.iterations << " (" << options.warmup <<
                                                      " ms warm up)"         << std::endl <<
          "Duration: "       << options.duration   << "s"                    << std::endl <<
          "Sleep time: "     << options.sleep      << "ms"                   << std::endl <<
          "Streams: "        << options.streams                              << std::endl <<
          "Spin-wait: "      << boolToEnabled(options.spin)                  << std::endl <<
          "Multithreading: " << boolToEnabled(options.threads)               << std::endl <<
          "CUDA Graph: "     << boolToEnabled(options.graph)                 << std::endl <<
          "Skip inference: " << boolToEnabled(options.skip)                  << std::endl;
// clang-format on
    if (options.batch)
    {
        printShapes(os, "inference", options.shapes);
    }

    return os;
}

std::ostream& operator<<(std::ostream& os, const ReportingOptions& options)
{
// clang-format off
    os << "=== Reporting Options ==="                                       << std::endl <<

          "Verbose: "                     << boolToEnabled(options.verbose) << std::endl <<
          "Averages: "                    << options.avgs << " inferences"  << std::endl <<
          "Percentile: "                  << options.percentile             << std::endl <<
          "Dump output: "                 << boolToEnabled(options.output)  << std::endl <<
          "Profile: "                     << boolToEnabled(options.profile) << std::endl <<
          "Export timing to JSON file: "  << options.exportTimes            << std::endl <<
          "Export profile to JSON file: " << options.exportProfile          << std::endl;
// clang-format on

    return os;
}

std::ostream& operator<<(std::ostream& os, const AllOptions& options)
{
    os << options.model << options.build << options.system << options.inference << options.reporting << std::endl;
    return os;
}

void BaseModelOptions::help(std::ostream& os)
{
// clang-format off
    os << "  --uff=<file>                UFF model"                                             << std::endl <<
          "  --onnx=<file>               ONNX model"                                            << std::endl <<
          "  --model=<file>              Caffe model (default = no model, random weights used)" << std::endl;
// clang-format on
}

void UffInput::help(std::ostream& os)
{
// clang-format off
    os << "  --uffInput=<name>,X,Y,Z     Input blob name and its dimensions (X,Y,Z=C,H,W), it can be specified "
                                                       "multiple times; at least one is required for UFF models" << std::endl <<
          "  --uffNHWC                   Set if inputs are in the NHWC layout instead of NCHW (use "             <<
                                                                    "X,Y,Z=H,W,C order in --uffInput)"           << std::endl;
// clang-format on
}

void ModelOptions::help(std::ostream& os)
{
// clang-format off
    os << "=== Model Options ==="                                                                                 << std::endl;
    BaseModelOptions::help(os);
    os << "  --deploy=<file>             Caffe prototxt file"                                                     << std::endl <<
          "  --output=<name>[,<name>]*   Output names (it can be specified multiple times); at least one output "
                                                                                  "is required for UFF and Caffe" << std::endl;
    UffInput::help(os);
// clang-format on
}

void BuildOptions::help(std::ostream& os)
{
// clang-format off
    os << "=== Build Options ==="                                                                                                     << std::endl <<

          "  --maxBatch                  Set max batch size and build an implicit batch engine (default = " << defaultMaxBatch << ")" << std::endl <<
          "  --explicitBatch             Use explicit batch sizes when building the engine (default = implicit)"                      << std::endl <<
          "  --minShapes=spec            Build with dynamic shapes using a profile with the min shapes provided"                      << std::endl <<
          "  --optShapes=spec            Build with dynamic shapes using a profile with the opt shapes provided"                      << std::endl <<
          "  --maxShapes=spec            Build with dynamic shapes using a profile with the max shapes provided"                      << std::endl <<
          "                              Note: if any of min/max/opt is missing, the profile will be completed using the shapes "     << std::endl <<
          "                                    provided and assuming that opt will be equal to max unless they are both specified;"   << std::endl <<           
          "                                    partially specified shapes are applied starting from the batch size;"                  << std::endl <<           
          "                                    dynamic shapes imply explicit batch"                                                   << std::endl <<           
          "                              Input shapes spec ::= Ishp[\",\"spec]"                                                       << std::endl <<
          "                                           Ishp ::= name\":\"shape"                                                        << std::endl <<
          "                                          shape ::= N[[\"x\"N]*\"*\"]"                                                     << std::endl <<
          "  --inputIOFormats=spec       Type and formats of the input tensors (default = all inputs in fp32:chw)"                    << std::endl <<
          "  --outputIOFormats=spec      Type and formats of the output tensors (default = all outputs in fp32:chw)"                  << std::endl <<
          "                              IO Formats: spec  ::= IOfmt[\",\"spec]"                                                      << std::endl <<
          "                                          IOfmt ::= type:fmt"                                                              << std::endl <<
          "                                          type  ::= \"fp32\"|\"fp16\"|\"int32\"|\"int8\""                                  << std::endl <<
          "                                          fmt   ::= (\"chw\"|\"chw2\"|\"chw4\"|\"hwc8\"|\"chw16\"|\"chw32\")[\"+\"fmt]"    << std::endl <<
          "  --workspace=N               Set workspace size in megabytes (default = "                      << defaultWorkspace << ")" << std::endl <<
          "  --minTiming=M               Set the minimum number of iterations used in kernel selection (default = "
                                                                                                           << defaultMinTiming << ")" << std::endl <<
          "  --avgTiming=M               Set the number of times averaged in each iteration for kernel selection (default = "
                                                                                                           << defaultAvgTiming << ")" << std::endl <<
          "  --fp16                      Enable fp16 mode (default = disabled)"                                                       << std::endl <<
          "  --int8                      Run in int8 mode (default = disabled)"                                                       << std::endl <<
          "  --calib=<file>              Read INT8 calibration cache file"                                                            << std::endl <<
          "  --safe                      Only test the functionality available in safety restricted flows"                            << std::endl <<
          "  --saveEngine=<file>         Save the serialized engine"                                                                  << std::endl <<
          "  --loadEngine=<file>         Load a serialized engine"                                                                    << std::endl;
// clang-format on
}

void SystemOptions::help(std::ostream& os)
{
// clang-format off
    os << "=== System Options ==="                                                                         << std::endl <<
          "  --device=N                  Select cuda device N (default = "         << defaultDevice << ")" << std::endl <<
          "  --useDLACore=N              Select DLA core N for layers that support DLA (default = none)"   << std::endl <<
          "  --allowGPUFallback          When DLA is enabled, allow GPU fallback for unsupported layers "
                                                                                    "(default = disabled)" << std::endl;
    os << "  --plugins                   Plugin library (.so) to load (can be specified multiple times)"   << std::endl;
// clang-format on
}

void InferenceOptions::help(std::ostream& os)
{
// clang-format off
    os << "=== Inference Options ==="                                                                                            << std::endl <<
          "  --batch=N                   Set batch size for implicit batch engines (default = "           << defaultBatch << ")" << std::endl <<
          "  --shapes=spec               Set input shapes for explicit batch and dynamic shapes inputs"                          << std::endl <<
          "                              Input shapes spec ::= Ishp[\",\"spec]"                                                  << std::endl <<
          "                                           Ishp ::= name\":\"shape"                                                   << std::endl <<
          "                                          shape ::= N[[\"x\"N]*\"*\"]"                                                << std::endl <<
          "  --iterations=N              Run at least N inference iterations (default = "            << defaultIterations << ")" << std::endl <<
          "  --warmUp=N                  Run for N milliseconds to warmup before measuring performance (default = "
                                                                                                         << defaultWarmUp << ")" << std::endl <<
          "  --duration=N                Run performance measurements for at least N seconds wallclock time (default = "
                                                                                               << defaultDuration << ")"         << std::endl <<
          "  --sleepTime=N               Delay inference start with a gap of N milliseconds between launch and compute "
                                                                                            "(default = " << defaultSleep << ")" << std::endl <<
          "  --streams=N                 Instantiate N engines to use concurrently (default = "         << defaultStreams << ")" << std::endl <<
          "  --useSpinWait               Actively synchronize on GPU events. This option may decrease synchronization time but "
                                                                                "increase CPU usage and power (default = false)" << std::endl <<
          "  --threads                   Enable multithreading to drive engines with independent threads (default = disabled)"   << std::endl <<
          "  --useCudaGraph              Use cuda graph to capture engine execution and then launch inference (default = false)" << std::endl <<
          "  --buildOnly                 Skip inference perf measurement (default = disabled)"                                   << std::endl;
// clang-format on
}

void ReportingOptions::help(std::ostream& os)
{
// clang-format off
    os << "=== Reporting Options ==="                                                                    << std::endl <<
          "  --verbose                   Use verbose logging (default = false)"                          << std::endl <<
          "  --avgRuns=N                 Report performance measurements averaged over N consecutive "
                                                       "iterations (default = " << defaultAvgRuns << ")" << std::endl <<
          "  --percentile=P              Report performance for the P percentage (0<=P<=100, 0 "
                                        "representing max perf, and 100 representing min perf; (default"
                                                                      " = " << defaultPercentile << "%)" << std::endl <<
          "  --dumpOutput                Print the output tensor(s) of the last inference iteration "
                                                                                  "(default = disabled)" << std::endl <<
          "  --dumpProfile               Print profile information per layer (default = disabled)"       << std::endl <<
          "  --exportTimes=<file>        Write the timing results in a json file (default = disabled)"   << std::endl <<
          "  --exportProfile=<file>      Write the profile information per layer in a json file "
                                                                              "(default = disabled)"     << std::endl;
// clang-format on
}

void helpHelp(std::ostream& os)
{
    os << "=== Help ==="                                     << std::endl <<
          "  --help                      Print this message" << std::endl;
}

void AllOptions::help(std::ostream& os)
{
    ModelOptions::help(os);
    os << std::endl;
    BuildOptions::help(os);
    os << std::endl;
    InferenceOptions::help(os);
    os << std::endl;
// clang-format off
    os << "=== Build and Inference Batch Options ==="                                                                   << std::endl <<
          "                              When using implicit batch, the max batch size of the engine, if not given, "   << std::endl <<
          "                              is set to the inference batch size;"                                           << std::endl <<
          "                              when using explicit batch, if shapes are specified only for inference, they "  << std::endl <<
          "                              will be used also as min/opt/max in the build profile; if shapes are "         << std::endl <<
          "                              specified only for the build, the opt shapes will be used also for inference;" << std::endl <<
          "                              if both are specified, they must be compatible; and if explicit batch is "     << std::endl <<
          "                              enabled but neither is specified, the model must provide complete static"      << std::endl <<
          "                              dimensions, including batch size, for all inputs"                              << std::endl <<
    std::endl;
// clang-format on
    ReportingOptions::help(os);
    os << std::endl;
    SystemOptions::help(os);
    os << std::endl;
    helpHelp(os);
}

} // namespace sample
