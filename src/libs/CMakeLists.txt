add_subdirectory(common_util)
add_subdirectory(range_sampling)
add_subdirectory(rlscope_common)
add_subdirectory(profilerhost_util)
if(RLS_DEBUG_ENABLE_ACHIEVED_OCCUPANCY_EXPERIMENT)
    add_subdirectory(achieved_occupancy_range_profiling)
    add_subdirectory(achieved_occupancy_range_profiling_refactored)
endif()
if(RLS_DEBUG_ENABLE_LEGACY_DQN_CPP)
    add_subdirectory(dqn_atari_test)
endif()
if(RLS_DEBUG_ENABLE_GPU_UTIL_EXPERIMENT)
    add_subdirectory(gpu_util_experiment)
endif()

if(RLS_DEBUG_ENABLE_TENSORRT_EXPERIMENT)
    # Doesn't compile with tensorrt7
    #add_subdirectory(tensorrt_common6)
    #add_subdirectory(trtexec6)
    add_subdirectory(tensorrt_common7)
    add_subdirectory(trtexec7)
endif()

if(RLS_DEBUG_ENABLE_CUPTI_API_EXPERIMENT)
    add_subdirectory(callback_timestamp)
endif()
if(RLS_DEBUG_ENABLE_DEVICE_QUERY)
    add_subdirectory(deviceQuery)
endif()
