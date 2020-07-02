# TensorRT Command-Line Wrapper: trtexec

**Table Of Contents**
- [Description](#description)
- [Building `trtexec`](#building-trtexec)
- [Using `trtexec`](#using-trtexec)
    * [Example 1: Simple MNIST model from Caffe](#example-1-simple-mnist-model-from-caffe)
    * [Example 2: Profiling a custom layer](#example-2-profiling-a-custom-layer)
    * [Example 3: Running a network on DLA](#example-3-running-a-network-on-dla)
- [Tool command line arguments](#tool-command-line-arguments)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

Included in the `samples` directory is a command line wrapper tool, called `trtexec`. `trtexec` is a tool to quickly utilize TensorRT without having to develop your own application. The `trtexec` tool has two main purposes:
-   It’s useful for benchmarking networks on random data.
-   It’s useful for generating serialized engines from models.

**Benchmarking network** - If you have a model saved as a UFF file, ONNX file, or if you have a network description in a Caffe prototxt format, you can use the `trtexec` tool to test the performance of running inference on your network using TensorRT. The `trtexec` tool has many options for specifying inputs and outputs, iterations for performance timing, precision allowed, and other options.

**Serialized engine generation** - If you generate a saved serialized engine file, you can pull it into another application that runs inference. For example, you can use the [TensorRT Laboratory](https://github.com/NVIDIA/tensorrt-laboratory) to run the engine with multiple execution contexts from multiple threads in a fully pipelined asynchronous way to test parallel inference performance. There are some caveats, for example, if you used a Caffe prototxt file and a model is not supplied, random weights are generated. Also, in INT8 mode, random weights are used, meaning trtexec does not provide calibration capability.

## Building `trtexec`

`trtexec` can be used to build engines, using different TensorRT features (see command line arguments), and run inference. `trtexec` also measures and reports execution time and can be used to understand performance and possibly locate bottlenecks.

Compile this sample by running `make` in the `<TensorRT root directory>/samples/trtexec` directory. The binary named `trtexec` will be created in the `<TensorRT root directory>/bin` directory.
```
cd <TensorRT root directory>/samples/trtexec
make
```
Where `<TensorRT root directory>` is where you installed TensorRT.

## Using `trtexec`

`trtexec` can build engines from models in Caffe, UFF, or ONNX format.

### Example 1: Simple MNIST model from Caffe

The example below shows how to load a model description and its weights, build the engine that is optimized for batch size 16, and save it to a file.
`trtexec --deploy=/path/to/mnist.prototxt --model=/path/to/mnist.caffemodel --output=prob --batch=16 --saveEngine=mnist16.trt`

Then, the same engine can be used for benchmarking; the example below shows how to load the engine and run inference on batch 16 inputs (randomly generated).
`trtexec --loadEngine=mnist16.trt --batch=16`

### Example 2: Profiling a custom layer

You can profile a custom layer using the `IPluginRegistry` for the plugins and `trtexec`. You’ll need to first register the plugin with `IPluginRegistry`.

If you are using TensorRT shipped plugins, you should load the `libnvinfer_plugin.so` file, as these plugins are pre-registered.

If you have your own plugin, then it has to be registered explicitly. The following macro can be used to register the plugin creator `YourPluginCreator` with the `IPluginRegistry`.
`REGISTER_TENSORRT_PLUGIN(YourPluginCreator);`

### Example 3: Running a network on DLA

To run the AlexNet network on NVIDIA DLA (Deep Learning Accelerator) using `trtexec` in FP16 mode, issue:
```
./trtexec --deploy=data/AlexNet/AlexNet_N2.prototxt --output=prob --useDLACore=1 --fp16 --allowGPUFallback
```
To run the AlexNet network on DLA using `trtexec` in INT8 mode, issue:
```
./trtexec --deploy=data/AlexNet/AlexNet_N2.prototxt --output=prob --useDLACore=1 --int8 --allowGPUFallback
```
To run the MNIST network on DLA using `trtexec`, issue:
```
./trtexec --deploy=data/mnist/mnist.prototxt --output=prob --useDLACore=0 --fp16 --allowGPUFallback
```

For more information about DLA, see [Working With DLA](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#dla_topic).

## Tool command line arguments

To see the full list of available options and their descriptions, issue the `./trtexec --help` command.

**Note:** Specifying the `--safe` parameter turns the safety mode switch `ON`. By default, the `--safe` parameter is not specified; the safety mode switch is `OFF`. The layers and parameters that are contained within the `--safe` subset are restricted if the switch is set to `ON`. The switch is used for prototyping the safety restricted flows until the TensorRT safety runtime is made available. For more information, see the [Working With Automotive Safety section in the TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#working_auto_safety).

## Additional resources

The following resources provide more details about `trtexec`:

**Documentation**
- [NVIDIA trtexec](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#trtexec)
- [TensorRT Sample Support Guide](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)
documentation.

# Changelog

April 2019
This is the first release of this `README.md` file.

# Known issues

There are no known issues in this sample.
