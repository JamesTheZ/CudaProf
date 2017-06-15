#ifndef CUDA_HOOK_H_
#define CUDA_HOOK_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <dlfcn.h>
#include <cxxabi.h>
#include <pthread.h>
#include <math.h>
#include <execinfo.h>
#include <cupti.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define METRIC_MAX_NUM 32
#define EVENT_MAX_NUM 128
#define METRIC_NAME_MAX_LEN 64
#define EVENT_NAME_MAX_LEN 64
#define KERNEL_NAME_MAX_LEN 256
#define SHORT_DESC_MAX_LEN 64
#define KERNEL_MAX_RECORD 50

#define DRIVER_API_CALL(apiFuncCall)                   \
	do {                                               \
		CUresult _status = apiFuncCall;                \
		if (_status != CUDA_SUCCESS) {                 \
			fprintf(stderr, "%s:%d: error: function %s \
					failed with error %d.\n",__FILE__, \
					__LINE__, #apiFuncCall, _status);  \
			exit(EXIT_FAILURE);                        \
		}                                              \
	} while (0)

#define RUNTIME_API_CALL(apiFuncCall)                  \
	do {                                               \
		cudaError_t _status = apiFuncCall;             \
		if (_status != cudaSuccess) {                  \
			fprintf(stderr, "%s:%d: error: function %s \
					failed with error %s.\n", __FILE__,\
					__LINE__, #apiFuncCall,            \
					cudaGetErrorString(_status));      \
			void *arr[10];							   \
			size_t size = backtrace(arr, 10);		   \
			char **btStr = backtrace_symbols(arr, size);\
			int i;										\
			for(i = 0; i < 10; i++)						\
			{											\
				fprintf(stderr, "%s\n", btStr[i]);		\
			}											\
			free(btStr);							   \
			exit(EXIT_FAILURE);                        \
		}                                              \
	} while (0)

#define CUPTI_CALL(call)                               \
	do {                                               \
		CUptiResult _status = call;                    \
		if (_status != CUPTI_SUCCESS) {                \
			fprintf(stderr, "err code: %d\n", _status);\
			const char *errstr;                        \
			cuptiGetResultString(_status, &errstr);    \
			fprintf(stderr, "%s:%d: error: function %s \
					failed with error %s.\n", __FILE__,\
					__LINE__, #call, errstr);          \
			void *arr[10];							   \
			size_t size = backtrace(arr, 10);		   \
			char **btStr = backtrace_symbols(arr, size);\
			int i;										\
			for(i = 0; i < 10; i++)						\
			{											\
				fprintf(stderr, "trace: %s\n", btStr[i]);\
			}											\
			free(btStr);							   \
			exit(EXIT_FAILURE);                        \
		}                                              \
	} while (0)

#define PROF_CALL(call)                                \
	do {                                               \
		int _status = call;                            \
		if (_status != 0) {                            \
			fprintf(stderr, "%s:%d: error: function %s \
					failed.\n", __FILE__,              \
					__LINE__, #call);                  \
			void *arr[10];							   \
			size_t size = backtrace(arr, 10);		   \
			char **btStr = backtrace_symbols(arr, size);\
			int i;										\
			for(i = 0; i < 10; i++)						\
			{											\
				fprintf(stderr, "%s\n", btStr[i]);		\
			}											\
			free(btStr);							   \
			exit(EXIT_FAILURE);                        \
		}                                              \
	} while (0)

#define ALIGN_SIZE (8)

#define ALIGN_BUFFER(buffer, align)                    \
	(((uintptr_t) (buffer) & ((align)-1)) ?            \
	 ((buffer) + (align) - ((uintptr_t) (buffer) &     \
		 ((align)-1))) : (buffer)) 

typedef enum memType_t
{
	GLOBAL_MEM,
	PAGELOCKED_MEM,
	MAPPED_MEM_HOST,
	MAPPED_MEM_DEVICE,
	PITCHED_MEM,
	GLOBAL_3D_MEM
}memType;

// User data for event collection callback
typedef struct EGSData_st 
{
	// the set of event groups to collect for a pass
	CUpti_EventGroupSet *groupSet;
	// the number of entries in eventIdArray and eventValueArray
	uint32_t numEvents;
	// array of event ids
	CUpti_EventID *eventIDs;
	// array of event values
	uint64_t *eventValues;
} EGSData_t;

typedef struct eventIDArray_t
{
	CUpti_EventID eventIDs[EVENT_MAX_NUM];
	int numEvents;
}eventIDArray;

typedef struct metricIDEventNeedArray_t
{
	CUpti_MetricID metricIDs[METRIC_MAX_NUM];
	eventIDArray eventsNeed[METRIC_MAX_NUM];
	int numMetrics;
}metricIDEventNeedArray;

typedef struct eventsCollection_t
{
	CUpti_EventID eventIDs[EVENT_MAX_NUM];
	CUpti_EventGroupSets *groupSets;
	int numEventsEachSets[EVENT_MAX_NUM];
	int numEvents;
}eventsCollection;

typedef struct eventData_t
{
	CUpti_EventID eventID;
	uint64_t eventValue;
}eventData;

typedef struct metricData_t
{
	CUpti_MetricID metricID;
	CUpti_MetricValue  metricValue;
}metricData;

/*
typedef union customMetricValue_t
{
	long long valueLongLong;
	long double valueLongDouble;
}customMetricValue;

typedef enum customMetricValueKind_t
{
	CUSTOM_MTR_VALUE_LONGLONG,
	CUSTOM_MTR_VALUE_LONGDOUBLE
}customMetricValueKind;

typedef struct customMetricData_t
{
	char name[METRIC_NAME_MAX_LEN];
	customMetricValue value;
	customMetricValueKind valueKind;
}customMetricData;
*/

typedef struct kernelInfo_t
{
	char kernelName[KERNEL_NAME_MAX_LEN];
	uint32_t deviceID;
	uint32_t streamID;
	int64_t gridID;
	int32_t gridX;
	int32_t gridY;
	int32_t gridZ;
	int32_t blockX;
	int32_t blockY;
	int32_t blockZ;
	struct timespec start;
	struct timespec end;
	int32_t dynamicSharedMem;
	int32_t staticSharedMem;
	uint32_t localMemPerThread;
	uint32_t localMemTotal;
	uint16_t regPerThread;
	uint8_t cacheConfigReq;
	uint8_t cacheConfigUsed;
	uint8_t sharedMemConfigUsed;
}kernelInfo;

typedef enum occupancyLimiter_t
{
	NONE,
	BLOCK_SIZE,
	REG_PER_THREAD,
	SMEM_PER_BLOCK,
	GRID_SIZE
}occupancyLimiter;

typedef struct occupancyData_t
{
	double achieved;
	occupancyLimiter achiLimiter;
	double theory;
	occupancyLimiter theoLimiter;
}occupancyData;

typedef struct kernelProfileData_t
{
	kernelInfo kerInfo;
	eventData eventDatas[EVENT_MAX_NUM];
	int numEvents;
	metricData metricDatas[METRIC_MAX_NUM];
	int numMetrics;
	occupancyData occu;
}kernelProfileData;

typedef struct cudaMem_t
{
	memType type;
	void *addr;
	void *content;
	size_t sizeBytes;
}cudaMem;

typedef struct cudaMemItem_t
{
	cudaMem mem;
	struct cudaMemItem_t *next;
}cudaMemItem;

typedef struct cudaConfigCallArg_t
{
	dim3 gridDim;
	dim3 blockDim;
	size_t sharedMem;
	cudaStream_t stream;
}cudaConfCallArg;

typedef struct cudaSetupArg_t
{
	void *arg;
	size_t size;
	size_t offset;
}cudaSetupArg;

typedef struct cudaSetupArgItem_t
{
	cudaSetupArg arg;
	struct cudaSetupArgItem_t *next;
}cudaSetupArgItem;

typedef struct kernelArgData_t
{
	cudaConfCallArg confArg;
	cudaSetupArgItem *cudaSetupArgHead;
}kernelArgData;

typedef struct cuMemcpyInfo_t
{
	enum cudaMemcpyKind kind;
	size_t count;
}cuMemcpyInfo;

typedef enum traceType_t
{
	RUNTIME_MEMCPY,
	PROF_OVERHEAD
}traceType;

typedef union traceInfo_t
{
	cuMemcpyInfo cpyInfo;
}traceInfo;

typedef struct cuTimeTrace_t
{
	char shortDesc[SHORT_DESC_MAX_LEN];
	struct timespec start;
	struct timespec end;
	traceType type;
	traceInfo info;
}cuTimeTrace;

typedef enum allocGranularity_t
{
	GRANULARITY_BLOCK,
	GRANULARITY_WARP
}allocGranularity;

typedef struct gpuLimit_t
{
	int limitThreadsPerWarp;
	int limitWarpsPerMultpro;
	int limitThreadsPerMultpro;
	int limitBlocksPerMultpro;
	int limitTotalRegs;
	int regAllocaUnitSize;
	allocGranularity allocaGranularity;
	int limitRegsPerThread;
	int limitTotalSMemPerMultpro;
	int SMemAllocaSize;
	int warpAllocaGranularity;
	int maxThreadsPerBlock;
}gpuLimit;

typedef struct kerProfClassRecord_t
{
	kernelInfo kerInfo;
	unsigned long times;
	struct kerProfClassRecord_t *next;
}kerProfClassRecord;



#endif


