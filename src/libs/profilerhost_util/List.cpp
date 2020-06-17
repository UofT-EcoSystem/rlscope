#include "List.h"
#include <iostream>
#include <nvperf_host.h>
#include <nvperf_cuda_host.h>
//#include <ScopeExit.h>
#include "c_util/ScopeExit.h"

#include "common_util.h"

#define RETURN_IF_NVPW_ERROR(retval, apiFuncCall) \
    do { \
        NVPA_Status _status = apiFuncCall; \
        DEBUG_PRINT_API_CALL(apiFuncCall); \
        if (NVPA_STATUS_SUCCESS != _status) { \
            fprintf(stderr, "FAILED: %s\n", #apiFuncCall); \
            return retval; \
        } \
    } while (0)

namespace NV {
    namespace Metric {
        namespace Enum {
            bool ListSupportedChips() {
                NVPW_GetSupportedChipNames_Params getSupportedChipNames = { NVPW_GetSupportedChipNames_Params_STRUCT_SIZE };
                RETURN_IF_NVPW_ERROR(false, NVPW_GetSupportedChipNames(&getSupportedChipNames));
                std::cout << "\n Number of supported chips : " << getSupportedChipNames.numChipNames;
                std::cout << "\n List of supported chips : \n";

                for (size_t i = 0; i < getSupportedChipNames.numChipNames; i++) {
                    std::cout << " " << getSupportedChipNames.ppChipNames[i] << "\n";
                }

                return true;
            }

            bool ListMetrics(const char* chip, bool listSubMetrics) {

                NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = { NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE };
                metricsContextCreateParams.pChipName = chip;
                RETURN_IF_NVPW_ERROR(false, NVPW_CUDA_MetricsContext_Create(&metricsContextCreateParams));

                NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = { NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE };
                metricsContextDestroyParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
                SCOPE_EXIT([&]() { NVPW_MetricsContext_Destroy((NVPW_MetricsContext_Destroy_Params *)&metricsContextDestroyParams); });

                NVPW_MetricsContext_GetMetricNames_Begin_Params getMetricNameBeginParams = { NVPW_MetricsContext_GetMetricNames_Begin_Params_STRUCT_SIZE };
                getMetricNameBeginParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
                getMetricNameBeginParams.hidePeakSubMetrics = !listSubMetrics;
                getMetricNameBeginParams.hidePerCycleSubMetrics = !listSubMetrics;
                getMetricNameBeginParams.hidePctOfPeakSubMetrics = !listSubMetrics;
                RETURN_IF_NVPW_ERROR(false, NVPW_MetricsContext_GetMetricNames_Begin(&getMetricNameBeginParams));

                NVPW_MetricsContext_GetMetricNames_End_Params getMetricNameEndParams = { NVPW_MetricsContext_GetMetricNames_End_Params_STRUCT_SIZE };
                getMetricNameEndParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
                SCOPE_EXIT([&]() { NVPW_MetricsContext_GetMetricNames_End((NVPW_MetricsContext_GetMetricNames_End_Params *)&getMetricNameEndParams); });
                
                std::cout << getMetricNameBeginParams.numMetrics << " metrics in total on the chip\n Metrics List : \n";
                for (size_t i = 0; i < getMetricNameBeginParams.numMetrics; i++) {
                    std::cout << getMetricNameBeginParams.ppMetricNames[i] << "\n";
                }

                return true;
            }

            bool ListCounters(const char* chip) {

              NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = { NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE };
              metricsContextCreateParams.pChipName = chip;
              RETURN_IF_NVPW_ERROR(false, NVPW_CUDA_MetricsContext_Create(&metricsContextCreateParams));

              NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = { NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE };
              metricsContextDestroyParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
              SCOPE_EXIT([&]() { NVPW_MetricsContext_Destroy((NVPW_MetricsContext_Destroy_Params *)&metricsContextDestroyParams); });


              NVPW_MetricsContext_GetCounterNames_Begin_Params getMetricNameBeginParams = { NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE };

              getMetricNameBeginParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
              RETURN_IF_NVPW_ERROR(false, NVPW_MetricsContext_GetCounterNames_Begin(&getMetricNameBeginParams));

              NVPW_MetricsContext_GetCounterNames_End_Params getMetricNameEndParams = { NVPW_MetricsContext_GetCounterNames_End_Params_STRUCT_SIZE };


              getMetricNameEndParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
              SCOPE_EXIT([&]() { NVPW_MetricsContext_GetCounterNames_End((NVPW_MetricsContext_GetCounterNames_End_Params *)&getMetricNameEndParams); });

              std::cout << getMetricNameBeginParams.numCounters << " counters in total on the chip\n Counters List : \n";
              for (size_t i = 0; i < getMetricNameBeginParams.numCounters; i++) {
                std::cout << getMetricNameBeginParams.ppCounterNames[i] << "\n";
              }

              return true;
            }

            bool ListRatios(const char* chip) {

              NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = { NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE };
              metricsContextCreateParams.pChipName = chip;
              RETURN_IF_NVPW_ERROR(false, NVPW_CUDA_MetricsContext_Create(&metricsContextCreateParams));

              NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = { NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE };
              metricsContextDestroyParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
              SCOPE_EXIT([&]() { NVPW_MetricsContext_Destroy((NVPW_MetricsContext_Destroy_Params *)&metricsContextDestroyParams); });


              NVPW_MetricsContext_GetRatioNames_Begin_Params getMetricNameBeginParams = { NVPW_MetricsContext_GetRatioNames_Begin_Params_STRUCT_SIZE };

              getMetricNameBeginParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
              RETURN_IF_NVPW_ERROR(false, NVPW_MetricsContext_GetRatioNames_Begin(&getMetricNameBeginParams));

              NVPW_MetricsContext_GetRatioNames_End_Params getMetricNameEndParams = { NVPW_MetricsContext_GetRatioNames_End_Params_STRUCT_SIZE };


              getMetricNameEndParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
              SCOPE_EXIT([&]() { NVPW_MetricsContext_GetRatioNames_End((NVPW_MetricsContext_GetRatioNames_End_Params *)&getMetricNameEndParams); });

              std::cout << getMetricNameBeginParams.numRatios << " counters in total on the chip\n Ratios List : \n";
              for (size_t i = 0; i < getMetricNameBeginParams.numRatios; i++) {
                std::cout << getMetricNameBeginParams.ppRatioNames[i] << "\n";
              }

              return true;
            }

            bool ListThroughputs(const char* chip) {

              NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = { NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE };
              metricsContextCreateParams.pChipName = chip;
              RETURN_IF_NVPW_ERROR(false, NVPW_CUDA_MetricsContext_Create(&metricsContextCreateParams));

              NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = { NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE };
              metricsContextDestroyParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
              SCOPE_EXIT([&]() { NVPW_MetricsContext_Destroy((NVPW_MetricsContext_Destroy_Params *)&metricsContextDestroyParams); });


              NVPW_MetricsContext_GetThroughputNames_Begin_Params getMetricNameBeginParams = { NVPW_MetricsContext_GetThroughputNames_Begin_Params_STRUCT_SIZE };

              getMetricNameBeginParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
              RETURN_IF_NVPW_ERROR(false, NVPW_MetricsContext_GetThroughputNames_Begin(&getMetricNameBeginParams));

              NVPW_MetricsContext_GetThroughputNames_End_Params getMetricNameEndParams = { NVPW_MetricsContext_GetThroughputNames_End_Params_STRUCT_SIZE };


              getMetricNameEndParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
              SCOPE_EXIT([&]() { NVPW_MetricsContext_GetThroughputNames_End((NVPW_MetricsContext_GetThroughputNames_End_Params *)&getMetricNameEndParams); });

              std::cout << getMetricNameBeginParams.numThroughputs << " counters in total on the chip\n Throughputs List : \n";
              for (size_t i = 0; i < getMetricNameBeginParams.numThroughputs; i++) {
                std::cout << getMetricNameBeginParams.ppThroughputNames[i] << "\n";
              }

              return true;
            }

            bool ListMetricBases(const char* chip) {
              NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = { NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE };
              metricsContextCreateParams.pChipName = chip;
              RETURN_IF_NVPW_ERROR(false, NVPW_CUDA_MetricsContext_Create(&metricsContextCreateParams));

              NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = { NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE };
              metricsContextDestroyParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
              SCOPE_EXIT([&]() { NVPW_MetricsContext_Destroy((NVPW_MetricsContext_Destroy_Params *)&metricsContextDestroyParams); });


              NVPW_MetricsContext_GetMetricBaseNames_Begin_Params getMetricNameBeginParams = { NVPW_MetricsContext_GetMetricBaseNames_Begin_Params_STRUCT_SIZE };

              getMetricNameBeginParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
              RETURN_IF_NVPW_ERROR(false, NVPW_MetricsContext_GetMetricBaseNames_Begin(&getMetricNameBeginParams));

              NVPW_MetricsContext_GetMetricBaseNames_End_Params getMetricNameEndParams = { NVPW_MetricsContext_GetMetricBaseNames_End_Params_STRUCT_SIZE };


              getMetricNameEndParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
              SCOPE_EXIT([&]() { NVPW_MetricsContext_GetMetricBaseNames_End((NVPW_MetricsContext_GetMetricBaseNames_End_Params *)&getMetricNameEndParams); });

              std::cout << getMetricNameBeginParams.numMetricBaseNames << " counters in total on the chip\n MetricBases List : \n";
              for (size_t i = 0; i < getMetricNameBeginParams.numMetricBaseNames; i++) {
                std::cout << getMetricNameBeginParams.ppMetricBaseNames[i] << "\n";
              }

              return true;
            }

        }
    }
}
