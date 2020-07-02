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

#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>
#include <iterator>
#include <string>
#include <map>
#include <cuda.h>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "NvOnnxParser.h"
#include "NvUffParser.h"

#include "logger.h"
#include "sampleUtils.h"
#include "sampleOptions.h"
#include "sampleEngines.h"

using namespace nvinfer1;

namespace sample
{

namespace
{

struct CaffeBufferShutter
{
    ~CaffeBufferShutter() { nvcaffeparser1::shutdownProtobufLibrary(); }
};

struct UffBufferShutter
{
    ~UffBufferShutter() { nvuffparser::shutdownProtobufLibrary(); }
};

}

Parser modelToNetwork(const ModelOptions& model, nvinfer1::INetworkDefinition& network, std::ostream& err)
{
    Parser parser;
    const std::string& modelName = model.baseModel.model;
    switch (model.baseModel.format)
    {
    case ModelFormat::kCAFFE:
    {
        using namespace nvcaffeparser1;
        parser.caffeParser.reset(createCaffeParser());
        CaffeBufferShutter bufferShutter;
        const auto blobNameToTensor = parser.caffeParser->parse(model.prototxt.c_str(), modelName.empty() ? nullptr : modelName.c_str(), network, DataType::kFLOAT);
        if (!blobNameToTensor)
        {
            err << "Failed to parse caffe model or prototxt, tensors blob not found" << std::endl;
            parser.caffeParser.reset();
            break;
        }

        for (const auto& s : model.outputs)
        {
            if (blobNameToTensor->find(s.c_str()) == nullptr)
            {
                err << "Could not find output blob " << s << std::endl;
                parser.caffeParser.reset();
                break;
            }
            network.markOutput(*blobNameToTensor->find(s.c_str()));
        }
        break;
    }
    case ModelFormat::kUFF:
    {
        using namespace nvuffparser;
        parser.uffParser.reset(createUffParser());
        UffBufferShutter bufferShutter;
        for (const auto& s : model.uffInputs.inputs)
        {
            if (!parser.uffParser->registerInput(s.first.c_str(), s.second, model.uffInputs.NHWC ? UffInputOrder::kNHWC : UffInputOrder::kNCHW))
            {
                err << "Failed to register input " << s.first << std::endl;
                parser.uffParser.reset();
                break;
            }
        }

        for (const auto& s : model.outputs)
        {
            if (!parser.uffParser->registerOutput(s.c_str()))
            {
                err << "Failed to register output " << s << std::endl;
                parser.uffParser.reset();
                break;
            }
        }

        if (!parser.uffParser->parse(model.baseModel.model.c_str(), network))
        {
            err << "Failed to parse uff file" << std::endl;
            parser.uffParser.reset();
            break;
        }
        break;
    }
    case ModelFormat::kONNX:
    {
        using namespace nvonnxparser;
        parser.onnxParser.reset(createParser(network, gLogger.getTRTLogger()));
        if (!parser.onnxParser->parseFromFile(model.baseModel.model.c_str(), static_cast<int>(gLogger.getReportableSeverity())))
        {
            err << "Failed to parse onnx file" << std::endl;
            parser.onnxParser.reset();
        }
        break;
    }
    case ModelFormat::kANY:
        break;
    }

    return parser;

}

namespace
{

class RndInt8Calibrator : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    RndInt8Calibrator(int batches, const std::string& cacheFile, const nvinfer1::INetworkDefinition& network, std::ostream& err);

    ~RndInt8Calibrator()
    {
        for (auto& elem : mInputDeviceBuffers)
        {
            cudaCheck(cudaFree(elem.second), mErr);
        }
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) override;

    int getBatchSize() const override { return 1; }

    const void* readCalibrationCache(size_t& length) override;

    virtual void writeCalibrationCache(const void*, size_t) override {}

private:
    int mBatches{};
    int mCurrentBatch{};
    std::string mCacheFile;
    std::map<std::string, void*> mInputDeviceBuffers;
    std::vector<char> mCalibrationCache;
    std::ostream& mErr;
};

RndInt8Calibrator::RndInt8Calibrator(int batches, const std::string& cacheFile, const INetworkDefinition& network, std::ostream& err)
    : mBatches(batches), mCurrentBatch(0), mCacheFile(cacheFile), mErr(err)
{
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    auto gen = [&generator, &distribution]() { return distribution(generator); };

    for (int i = 0; i < network.getNbInputs(); i++)
    {
        auto input = network.getInput(i);
        int elemCount = volume(input->getDimensions());
        std::vector<float> rnd_data(elemCount);
        std::generate_n(rnd_data.begin(), elemCount, gen);

        void* data;
        cudaCheck(cudaMalloc(&data, elemCount * sizeof(float)), mErr);
        cudaCheck(cudaMemcpy(data, rnd_data.data(), elemCount * sizeof(float), cudaMemcpyHostToDevice), mErr);

        mInputDeviceBuffers.insert(std::make_pair(input->getName(), data));
    }
}

bool RndInt8Calibrator::getBatch(void* bindings[], const char* names[], int nbBindings)
{
    if (mCurrentBatch >= mBatches)
    {
        return false;
    }

    for (int i = 0; i < nbBindings; ++i)
    {
        bindings[i] = mInputDeviceBuffers[names[i]];
    }

    ++mCurrentBatch;

    return true;
}

const void* RndInt8Calibrator::readCalibrationCache(size_t& length)
{
    mCalibrationCache.clear();
    std::ifstream input(mCacheFile, std::ios::binary);
    input >> std::noskipws;
    if (input.good())
    {
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
            std::back_inserter(mCalibrationCache));
    }

    return mCalibrationCache.size() ? mCalibrationCache.data() : nullptr;
}

void setTensorScales(const INetworkDefinition& network, float inScales = 2.0f, float outScales = 4.0f)
{
    // Ensure that all layer inputs have a scale.
    for (int l = 0; l < network.getNbLayers(); l++)
    {
        auto layer = network.getLayer(l);
        for (int i = 0; i < layer->getNbInputs(); i++)
        {
            ITensor* input{layer->getInput(i)};
            // Optional inputs are nullptr here and are from RNN layers.
            if (input && !input->dynamicRangeIsSet())
            {
                input->setDynamicRange(-inScales, inScales);
            }
        }
        for (int o = 0; o < layer->getNbOutputs(); o++)
        {
            ITensor* output{layer->getOutput(o)};
            // Optional outputs are nullptr here and are from RNN layers.
            if (output && !output->dynamicRangeIsSet())
            {
                // Pooling must have the same input and output scales.
                if (layer->getType() == LayerType::kPOOLING)
                {
                    output->setDynamicRange(-inScales, inScales);
                }
                else
                {
                    output->setDynamicRange(-outScales, outScales);
                }
            }
        }
    }
}

}

ICudaEngine* networkToEngine(const BuildOptions& build, const SystemOptions& sys, IBuilder& builder, INetworkDefinition& network, std::ostream& err)
{
    unique_ptr<IBuilderConfig> config{builder.createBuilderConfig()};

    IOptimizationProfile* profile{nullptr};
    if (build.maxBatch)
    {
        builder.setMaxBatchSize(build.maxBatch);
    }
    else
    {
        if (!build.shapes.empty())
        {
            profile = builder.createOptimizationProfile();
        }
    }

    for (unsigned int i = 0, n = network.getNbInputs(); i < n; i++)
    {
        // Set formats and data types of inputs
        auto input = network.getInput(i);
        if (!build.inputFormats.empty())
        {
            input->setType(build.inputFormats[i].first);
            input->setAllowedFormats(build.inputFormats[i].second);
        }
        else
        {
            input->setType(DataType::kFLOAT);
            input->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));
        }

        if (profile)
        {
            Dims dims = input->getDimensions();
            if (std::any_of(dims.d + 1, dims.d + dims.nbDims, [](int dim){ return dim == -1; }))
            {
                err << "Only dynamic batch dimension is currently supported, other dimensions must be static" << std::endl;
                return nullptr;
            }
            dims.d[0] = -1;
            Dims profileDims = dims;
            auto shape = build.shapes.find(input->getName());
            if (shape == build.shapes.end())
            {
                err << "Dynamic dimensions required for input " << input->getName() << std::endl;
                return nullptr;
            }
            profileDims.d[0] = shape->second[static_cast<size_t>(OptProfileSelector::kMIN)].d[0];
            profile->setDimensions(input->getName(), OptProfileSelector::kMIN, profileDims);
            profileDims.d[0] = shape->second[static_cast<size_t>(OptProfileSelector::kOPT)].d[0];
            profile->setDimensions(input->getName(), OptProfileSelector::kOPT, profileDims);
            profileDims.d[0] = shape->second[static_cast<size_t>(OptProfileSelector::kMAX)].d[0];
            profile->setDimensions(input->getName(), OptProfileSelector::kMAX, profileDims);

            input->setDimensions(dims);
        }
    }

    if (profile)
    {
        if (!profile->isValid())
        {
            err << "Required optimization profile is invalid" << std::endl;
            return nullptr;
        }
        config->addOptimizationProfile(profile);
    }

    for (unsigned int i = 0, n = network.getNbOutputs(); i < n; i++)
    {
        // Set formats and data types of outputs
        auto output = network.getOutput(i);
        if (!build.outputFormats.empty())
        {
            output->setType(build.outputFormats[i].first);
            output->setAllowedFormats(build.outputFormats[i].second);
        }
        else
        {
            output->setType(DataType::kFLOAT);
            output->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));
        }
    }

    config->setMaxWorkspaceSize(static_cast<size_t>(build.workspace) << 20);

    if (build.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }

    if (build.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
    }

    auto isInt8 = [](const IOFormat& format){ return format.first == DataType::kINT8; };
    auto int8IO = std::count_if(build.inputFormats.begin(), build.inputFormats.end(), isInt8) +
                  std::count_if(build.outputFormats.begin(), build.outputFormats.end(), isInt8);

    if ((build.int8 && build.calibration.empty()) || int8IO) 
    {
        // Explicitly set int8 scales if no calibrator is provided and if I/O tensors use int8,
        // because auto calibration does not support this case.
        setTensorScales(network);
    }
    else if (build.int8)
    {
        config->setInt8Calibrator(new RndInt8Calibrator(1, build.calibration, network, err));
    }

    if (build.safe)
    {
        config->setEngineCapability(sys.DLACore != -1 ? EngineCapability::kSAFE_DLA : EngineCapability::kSAFE_GPU);
    }

    if (sys.DLACore != -1)
    {
        if (sys.DLACore < builder.getNbDLACores())
        {
            config->setDefaultDeviceType(DeviceType::kDLA);
            config->setDLACore(sys.DLACore);
            config->setFlag(BuilderFlag::kSTRICT_TYPES);

            if (sys.fallback)
            {
                config->setFlag(BuilderFlag::kGPU_FALLBACK);
            }
            if (!build.int8)
            {
                config->setFlag(BuilderFlag::kFP16);
            }
        }
        else
        {
            err << "Cannot create DLA engine, " << sys.DLACore << " not available" << std::endl;
            return nullptr;
        }
    }

    return builder.buildEngineWithConfig(network, *config);
}

ICudaEngine* modelToEngine(const ModelOptions& model, const BuildOptions& build, const SystemOptions& sys, std::ostream& err)
{
    unique_ptr<IBuilder> builder{createInferBuilder(gLogger.getTRTLogger())};
    if (builder == nullptr)
    {
        err << "Builder creation failed" << std::endl;
        return nullptr;
    }
    auto batchFlag = (build.maxBatch ? 0U : 1U) << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    unique_ptr<INetworkDefinition> network{builder->createNetworkV2(batchFlag)};
    if (!network)
    {
        err << "Network creation failed" << std::endl;
        return nullptr;
    }
    Parser parser = modelToNetwork(model, *network, err);
    if (!parser)
    {
        err << "Parsing model failed" << std::endl;
        return nullptr;
    }

    return networkToEngine(build, sys, *builder, *network, err);
}

ICudaEngine* loadEngine(const std::string& engine, int DLACore, std::ostream& err)
{
    std::ifstream engineFile(engine, std::ios::binary);
    if (!engineFile)
    {
        err << "Error opening engine file: " << engine << std::endl;
        return nullptr;
    }

    engineFile.seekg(0, engineFile.end);
    long int fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);
    if (!engineFile)
    {
        err << "Error loading engine file: " << engine << std::endl;
        return nullptr;
    }

    unique_ptr<IRuntime> runtime{createInferRuntime(gLogger.getTRTLogger())};
    if (DLACore != -1)
    {
        runtime->setDLACore(DLACore);
    }

    return runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr);
}

bool saveEngine(const ICudaEngine& engine, const std::string& fileName, std::ostream& err)
{
    std::ofstream engineFile(fileName, std::ios::binary);
    if (!engineFile)
    {
        err << "Cannot open engine file: " << fileName << std::endl;
        return false;
    }

    unique_ptr<IHostMemory> serializedEngine{engine.serialize()};
    if (serializedEngine == nullptr)
    {
        err << "Engine serialization failed" << std::endl;
        return false;
    }

    engineFile.write(static_cast<char*>(serializedEngine->data()), serializedEngine->size());
    return !engineFile.fail();
}

} // namespace sample
