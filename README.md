# IML: "irregular" machine learning benchmarking toolkit

"Toolkit" of benchmark tools for measuring "irregular" machine learning workloads e.g. Reinforcement Learning (RL).

The current focus is on RL workloads, but these scripts are applicable to any ML TensorFlow script.

Currently, this toolkit is intended to be used as a a python library.
```python
# Inside your_ml_script.py

#
# Create a Profiler instance.
# This includes both nvprof and pyprof profilers.
#
from iml_profiler.profiler import Profiler
profiler = Profiler(...)

for epoch in range(epochs):
    for i in range(steps_per_epoch):

        #
        # Only collect profiling information during inner training loop operations.
        # Don't both collecting "noise" for one-time initialization.
        #
        profiler.enable_profiling()
        
        # Some part of your inner-training loop.
        # For e.g. if its MNIST, this will be both the Forward/Backward passes.
        sess.run(train_op, ...)
        
        profiler.disable_profiling()
        
```

For a full, runnable example of using this toolkit, look at the [end-to-end test](#end-to-end-test).


However in order to collect accurate timing information from the python profiler, I had to make a (extremely small) modification to the python interpreter:
- Make it use CLOCK_MONOTONIC when reporting times
- Make it print the full precision of the `double` when reporting times in pyprof output

The benchmarks are currently focussed on measuring Python/C++/GPU time spent in your ML script, regardless of the framework it uses (but currently I've only tested on TensorFlow).

Below I've documented the current benchmarks you can run.

#Docker

Dependencies:
- nvidia-docker container runtime: https://chunml.github.io/ChunML.github.io/project/Installing-NVIDIA-Docker-On-Ubuntu-16.04/ 

To build and run the container in one command, do this:
```bash
$ ( set -e; set -x; docker build .; docker run --runtime=nvidia -t -i $(docker build . | tail -n 1 | awk '{print $3}'); )
```

#End-to-end test

To test that Python/C++/GPU times reported by this toolkit are correct, I made a simple end-to-end test that measure a script that does the following: 
- Run in Python for 5 seconds
- Run in C++ for 5 seconds
- Run in GPU for 5 seconds

```bash
$ python3 python/test/test_call_c.py
...

#
# The script generates nvprof/pyprof output files in checkpoints/test_call_c
#
$ tree checkpoints/test_call_c
checkpoints/test_call_c
├── gpu_clock_freq.json
├── nvidia.nvprof
├── nvidia.nvprof_logfile.txt
├── python_profile.call_times.json
├── python_profile.prof
└── python_profile.txt

#
# We can generate some informative plots from the nvprof/pyprof output.
#
$ python3 python/run.py process --directory checkpoints/test_call_c


#
# Again, results are generated in checkpoints/test_call_c.
#
$ tree checkpoints/test_call_c
checkpoints/test_call_c
├── ( ... nvprof/pyprof files ... )
├── ( ... additional metrics ... )
# Total "per-iteration/step" time in C++
# *.plot_data.txt shows the corresponding data plot in *.png
├── CppTimeSec.summary.plot_data.txt
├── CppTimeSec.summary.png
# Total "per-iteration/step" time in GPU
├── GPUTimeSec.summary.plot_data.txt
├── GPUTimeSec.summary.png
...
# Total "per-iteration/step" time in Python
├── PythonTimeSec.summary.plot_data.txt
├── PythonTimeSec.summary.png
...
# A stacked bar plot of "per-iteration/step" time spent in Python/C++/GPU
├── time_breakdown.plot_data.txt
└── time_breakdown.png
...
```
Status:
- I have yet to see this script report inconsistent results for any of the metrics, including the "questionable" ones (e.g. [CudaCppTimeSec](#cudacpptimesec)).
- However, I HAVE seen inconsistent results come from when I am benchmarking DQN...I have yet to reproduce such behaviour with this though....

## Metrics

Each one of these metrics has a corresponding plot generated for it from the nvprof/pyprof output.

Metrics have two types:
1. **Raw metric:** \
This metric is measured from pyprof/nvprof output files, or by calling `time.time()` inside python code to measure time spent profiling.
2. **Derived metric:** \
This metric is computed by combining pyprof/nvprof output files; i.e. by adding/subtracting raw metrics.

## `CppAndGPUTimeSec`
- Alias: Time(GPU + C)
- Total profiled execution time from pyprof
- Correctness: Confident
- Raw metric: \
  Measured from: pyprof
  
## `GPUTimeSec` 
- Alias: Time(GPU)
- An underestimate from nvprof, only includes time the GPU is running a kernel / memcpy operation
- Correctness: Questionable
- Raw metric: \
  Measured from: nvprof
  
- Q: Did we sum up the CUDA API calls correctly?
  
## `PythonTimeSec`
- Alias: Time(Python)
- Pyprof profiler information; time spent in TensorFlow C API calls
- Correctness: Confident
- Raw metric: \
  Measured from: pyprof
  
## `CppTimeSec`
- Alias: Time(C)
- Time(GPU + C) - Time(GPU)
- Correctness: depends on Time(GPU) being correct.  If Time(GPU) is underestimated, then Time(C) will be overestimated. ( and vice versa )
- Derived metric: \
  Measured from: Time(C) = Time(GPU + C) - Time(GPU)
  
## `CppAndGPUTimeSec`
- Alias: Time(GPU + CUDA C)
- Sum of API calls times, similar to how Time(GPU) is computed.
- Raw metric: \
  Measured from: nvprof
  
## `CudaCppTimeSec`
- Alias: Time(CUDA C)
- Just keep the API call time; subtract any GPU time.
- If this is negative, then Time(GPU + CUDA C) doesn't include the entire GPU time.
- Correctness: Questionable
- Derived metric: \
  Measured from: Time(CUDA C) = Time(GPU + CUDA C) - Time(GPU)
  
## `FrameworkCppTimeSec`
- Alias: Time(Framework C)
- Derived metric: \
  Measured from: Time(Framework C) = Time(GPU + C) - Time(GPU) - Time(CUDA C)
  
## Total iteration time / `TotalTimeSec`
- Time spent profiling
- Correctness: Confident
- Raw metric: \
  Measured from: python using `time.time()` around the profiled section of code
  
## Sanity check:
- Total iteration time = = Time(GPU) + Time(Python) + Time(GPU + C)

# Other stuff in this repo

### C++ implementation of DQN:
In order to validate my C++ timing results (collected via nvprof/pyprof) and to show the potential benefit of eliminating python code, I was in the process of porting Atari Pong DQN to C++.

This is what all the C++ files in src/ and the CMakeLists.txt file is for.

Status:
- Still not finished
- Still have to get CartPole training... I was able to run a pre-trained model from python.
- After CartPole works, need to port over Atari simulator and Q-network

# Miscellaneous

## Dockerfile details
The Dockerfile handles building the (slightly) modified python3 from source.

If you're developing these scripts (outside a Docker container), I recommend `configure`-ing the cpython build using `--prefix=$HOME/local` to avoid stomping on an existing python install.

Besides that, just build stuff using the same commands present in the Dockerfile.

