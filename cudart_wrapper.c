#include "cudart_wrapper.h"

#define CUDA_HOOK_PROF

const char * cudart_orig_libname = "libcudart.so";
static void *cudart_handle = NULL;

static cudaMemItem *cudaMemHead = NULL;
static kernelArgData keArgData;
static metricIDEventNeedArray *mtrIDEveNdArr = NULL;
static eventIDArray *eveProf = NULL;
static eventsCollection *eveCol = NULL;
static kernelInfo *kerInfo = NULL;
static kerProfClassRecord *kerClassHead = NULL;

static const char *kerProfDataRecordFname = 
"profile_kernel_data_result.csv";
static const char *kernelTraceRecordFname = 
"profile_kernel_trace.csv";
static const char *memcpyTraceRecordFname = 
"profile_memcpy_trace.csv";
static const char *profOverheadRecordFname =
"profile_overhead_trace.csv";
static const char *errorLogFname =
"profile_error.csv";

static int fileOpen = 0;
static FILE *kerProfDataFile = NULL;
static FILE *kerTraceFile = NULL;
static FILE *cpyFile = NULL;
static FILE *overFile = NULL;
static FILE *errFile = NULL;

static const char *eventNamesFilename = 
"./eventNames.ini";
static const char *metricNamesFilename = 
"./metricNames.ini";

static int mutexSet = 0;
static pthread_mutex_t cudaMemMutex;
static pthread_mutex_t mtrEveDataMutex;
static pthread_mutex_t kerProfDataMutex;
static pthread_mutex_t kerTraceMutex;
static pthread_mutex_t cpyTraceMutex;
static pthread_mutex_t profOverheadMutex;
static pthread_mutex_t errLogMutex;
static pthread_mutex_t fopenMutex;
static pthread_mutex_t kerClassMutex;

static void *getSymbol(const char* symbolName)
{
	void *symbol = NULL;
	if (cudart_handle == NULL) 
	{
		cudart_handle = dlopen(cudart_orig_libname, RTLD_NOW); 
	}
	if (cudart_handle == NULL) 
	{ 
		perror("Error opening library in dlopen call"); 
		exit(EXIT_FAILURE);
	}
	else 
	{ 
		symbol = dlsym(cudart_handle,symbolName); 
		if (symbol == NULL) 
		{
			perror("Error obtaining symbol info from dlopen'ed lib"); 
			exit(EXIT_FAILURE);
		}
	}

	return symbol;
}

static cudaError_t origCudaConfigureCall(dim3 a1, dim3 a2, 
		size_t a3, cudaStream_t a4)
{
	typedef cudaError_t (*cudaConfigureCall_p) 
		(dim3, dim3, size_t, cudaStream_t);
	static cudaConfigureCall_p cudaConfigureCall_h = 
		(cudaConfigureCall_p)getSymbol("cudaConfigureCall");

	cudaError_t retval = 
		(*cudaConfigureCall_h)(a1, a2, a3, a4);

	return retval;
}

static cudaError_t origCudaSetupArgument(const void * a1, 
		size_t a2, size_t a3)
{
	typedef cudaError_t (*cudaSetupArgument_p) 
		(const void *, size_t, size_t);
	static cudaSetupArgument_p cudaSetupArgument_h = 
		(cudaSetupArgument_p)getSymbol("cudaSetupArgument");
	cudaError_t retval =  
		(*cudaSetupArgument_h)( a1,  a2,  a3);

	return retval;
}

static cudaError_t origCudaMemcpy(void *dest, const void *src, 
		size_t count, enum cudaMemcpyKind kind)
{
	typedef cudaError_t (*cudaMemcpy_p) 
		(void *, const void *, size_t, enum cudaMemcpyKind);
	static cudaMemcpy_p cudaMemcpy_h = 
		(cudaMemcpy_p)getSymbol("cudaMemcpy");
	cudaError_t retval =  
		(*cudaMemcpy_h)(dest, src, count, kind);

	return retval;
}

static cudaError_t origCudaMemcpyAsync(void *dst, const void *src, 
		size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
	typedef cudaError_t (*cudaMemcpyAsync_p)(void *, 
			const void *, size_t, enum cudaMemcpyKind, cudaStream_t);
	static cudaMemcpyAsync_p cudaMemcpyAsync_h = 
		(cudaMemcpyAsync_p)getSymbol("cudaMemcpyAsync");
	cudaError_t retval = (*cudaMemcpyAsync_h)(dst, 
			src, count, kind, stream);

	return retval;
}

static cudaError_t origCudaMemcpy2D(void *dst, size_t dpitch, 
		const void *src, size_t spitch, size_t width, 
		size_t height, enum cudaMemcpyKind kind)
{
	typedef cudaError_t (*cudaMemcpy2D_p) 
		(void *, size_t, const void *, size_t, 
		 size_t, size_t, enum cudaMemcpyKind);
	static cudaMemcpy2D_p cudaMemcpy2D_h = 
		(cudaMemcpy2D_p)getSymbol("cudaMemcpy2D");
	cudaError_t retval =  
		(*cudaMemcpy2D_h)(dst, dpitch, src, 
				spitch, width, height, kind);

	return retval;
}

cudaError_t origCudaMemcpy3D(const cudaMemcpy3DParms *p)
{
	typedef cudaError_t (*cudaMemcpy3D_p)(
			const cudaMemcpy3DParms *);
	static cudaMemcpy3D_p cudaMemcpy3D_h = 
		(cudaMemcpy3D_p)getSymbol("cudaMemcpy3D");
	cudaError_t retval = (*cudaMemcpy3D_h)(p);

	return retval;
}

cudaError_t origCudaMemset2D(void *devPtr, size_t pitch, 
		int value, size_t width, size_t height)
{
	typedef cudaError_t (*cudaMemset2D_p)
		(void *, size_t, int, size_t, size_t);
	static cudaMemset2D_p cudaMemset2D_h = 
		(cudaMemset2D_p)getSymbol("cudaMemset2D");
	cudaError_t retval = (*cudaMemset2D_h)
		(devPtr, pitch, value, width, height);

	return retval;
}

// TODO: test
cudaError_t origCudaMemset3D(cudaPitchedPtr pitchedDevPtr, 
		int value, cudaExtent extent)
{
	typedef cudaError_t (*cudaMemset3D_p)
		(cudaPitchedPtr, int, cudaExtent);
	static cudaMemset3D_p cudaMemset3D_h = 
		(cudaMemset3D_p)getSymbol("cudaMemset3D");
	cudaError_t retval = (*cudaMemset3D_h)
		(pitchedDevPtr, value, extent);

	return retval;
}

// TODO: test
cudaError_t origCudaMemcpyFromSymbol(void *dst, 
		const void *symbol, size_t count, size_t offset,
		cudaMemcpyKind kind)
{
	typedef cudaError_t (*cudaMemcpyFS_p)(void *, 
			const void *, size_t, size_t, cudaMemcpyKind);
	static cudaMemcpyFS_p cudaMemcpyFS_h = 
		(cudaMemcpyFS_p)getSymbol("cudaMemcpyFromSymbol");
	cudaError_t retval = (*cudaMemcpyFS_h)(
			dst, symbol, count, offset, kind);

	return retval;
}

cudaError_t origCudaMemcpyToSymbol(const void *symbol, 
		const void *src, size_t count, size_t offset,
		cudaMemcpyKind kind)
{
	typedef cudaError_t (*cudaMemcpyTS_p)(const void *, 
			const void *, size_t, size_t, cudaMemcpyKind);
	static cudaMemcpyTS_p cudaMemcpyTS_h = 
		(cudaMemcpyTS_p)getSymbol("cudaMemcpyToSymbol");
	cudaError_t retval = (*cudaMemcpyTS_h)(
			symbol, src, count, offset, kind);

	return retval;
}

// TODO: test
static CUdevice getCurDevice()
{
	CUdevice device;
	int deviceId = 0;
	RUNTIME_API_CALL(cudaGetDevice(&deviceId));
	// 	DRIVER_API_CALL(cuInit(0));
	DRIVER_API_CALL(cuDeviceGet(&device, deviceId));

	//	printf("device %d\n", (int)device);

	return device;
}

// TODO: test
static CUcontext getCurContext()
{

	CUdevice device = getCurDevice();
	CUcontext context;
	CUresult err = cuCtxGetCurrent(&context);
	if(err != CUDA_SUCCESS || context == NULL)
	{
		printf("what here here here?\n");
		DRIVER_API_CALL(cuCtxCreate(&context, 0, device));
	}

	//	printf("some thing wrong?\n");

	if(context == NULL)
	{
		printf("context is null\n");
	}

	return context;
}

static int initMutexes()
{
	if(mutexSet == 1)
	{
		return 1;
	}

	pthread_mutexattr_t attr;
	if(pthread_mutexattr_init(&attr) != 0)
	{
		return 1;
	}

	pthread_mutexattr_settype(
			&attr, PTHREAD_MUTEX_RECURSIVE);
	pthread_mutex_init(&cudaMemMutex, &attr);
	pthread_mutex_init(&mtrEveDataMutex, &attr);
	pthread_mutex_init(&kerProfDataMutex, &attr);
	pthread_mutex_init(&kerTraceMutex, &attr);
	pthread_mutex_init(&cpyTraceMutex, &attr);
	pthread_mutex_init(&profOverheadMutex, &attr);
	pthread_mutex_init(&errLogMutex, &attr);
	pthread_mutex_init(&fopenMutex, &attr);
	pthread_mutex_init(&kerClassMutex, &attr);

	return 0;
}

static int my_mutex_lock(pthread_mutex_t *mutex)
{
	if(mutexSet == 0)
	{
		mutexSet = 1;
		initMutexes();
	}
	if(mutex == NULL)
	{
		return 1;
	}

	return pthread_mutex_lock(mutex);
}

static int my_mutex_unlock(pthread_mutex_t *mutex)
{
	return pthread_mutex_unlock(mutex);
}

static void logError(const char *errstr, int lineno)
{
	my_mutex_lock(&errLogMutex);
	timespec trace;
	clock_gettime(CLOCK_REALTIME, &trace);
	fprintf(errFile, "\n%ld.%09ld,%d,\"%s\"", trace.tv_sec,
			trace.tv_nsec, lineno, errstr);
	my_mutex_unlock(&errLogMutex);
}

static int openFiles()
{
	my_mutex_lock(&fopenMutex);
	kerProfDataFile = fopen(kerProfDataRecordFname, "a");
	kerTraceFile = fopen(kernelTraceRecordFname, "a");
	cpyFile = fopen(memcpyTraceRecordFname, "a");
	overFile = fopen(profOverheadRecordFname, "a");
	errFile = fopen(errorLogFname, "a"); 
	if(kerProfDataFile != NULL && kerTraceFile != NULL
			&& cpyFile != NULL && overFile != NULL 
			&& errFile != NULL)
	{
		fileOpen = 1;
	}
	else
	{
		my_mutex_unlock(&fopenMutex);
		exit(EXIT_FAILURE);
	}

	my_mutex_unlock(&fopenMutex);
}

static void freeKeArgs()
{
	cudaSetupArgItem *p1 = keArgData.cudaSetupArgHead;
	cudaSetupArgItem *p2;
	while(p1 != NULL)
	{
		p2 = p1;
		p1 = p1->next;
		free(p2);
	}
	keArgData.cudaSetupArgHead = NULL;
}

static cudaMemItem *getCudaMemItemPtr(const void *cudaPtr)
{
	my_mutex_lock(&cudaMemMutex);

	cudaMemItem *p = cudaMemHead;

	while(p != NULL)
	{
		if(p->mem.addr == cudaPtr)
		{
			break;
		}
		p = p->next;
	}

	my_mutex_unlock(&cudaMemMutex);

	return p;
}

// backup when direc is 0; restore otherwise
static int someMemcpy(void *dest, const void *source, 
		size_t sizeBytes, memType type, int direc)
{
	//	printf("srcAddr: %lu, dstAddr: %lu\n", source, dest);
	//	printf("size: %lu, type: %d, direc: %d\n", sizeBytes, 
	//	type, direc);
	int ret = 0;
	switch(type)
	{
		case GLOBAL_MEM:
			RUNTIME_API_CALL(origCudaMemcpy(dest, source, 
						sizeBytes, direc == 0 ? 
						cudaMemcpyDeviceToHost : 
						cudaMemcpyHostToDevice));
			break;
		case PAGELOCKED_MEM:
			memcpy(dest, source, sizeBytes);
			break;
		case MAPPED_MEM_HOST:
			memcpy(dest, source, sizeBytes);
			break;
		case MAPPED_MEM_DEVICE:
			RUNTIME_API_CALL(origCudaMemcpy(dest, source, 
						sizeBytes, direc == 0 ? 
						cudaMemcpyDeviceToHost : 
						cudaMemcpyHostToDevice));
			break;
		default:
			ret = 1;
			break;
	}

	return ret;
}

static int addMemItem(memType newType, 
		const void **newAddrP, size_t newSizeBytes)
{
	cudaMemItem *pNew = 
		(cudaMemItem *)malloc(sizeof(cudaMemItem));
	if(pNew == NULL)
	{
		return 1;
	}
	pNew->mem.type = newType;
	pNew->mem.addr = *((void **)newAddrP);
	pNew->mem.content = (void *)malloc(newSizeBytes);
	if(pNew->mem.content == NULL)
	{
		return 1;
	}
	pNew->mem.sizeBytes = newSizeBytes;
	pNew->next = NULL;

	my_mutex_lock(&cudaMemMutex);

	pNew->next = cudaMemHead;
	cudaMemHead = pNew;

	my_mutex_unlock(&cudaMemMutex);

	return 0;
}

static int addMappedMemItem(memType newType, const void **pDevice, 
		const void *pHost)
{
	cudaMemItem *pNew = 
		(cudaMemItem *)malloc(sizeof(cudaMemItem));
	if(pNew == NULL)
	{
		return 1;
	}
	pNew->mem.type = newType;
	pNew->mem.addr = *((void **)pDevice);
	pNew->mem.sizeBytes = 0;
	pNew->mem.content = NULL;
	pNew->next = NULL;

	my_mutex_lock(&cudaMemMutex);

	cudaMemItem *pTmp = cudaMemHead;
	while(pTmp != NULL)
	{
		if(pTmp->mem.addr == pHost)
		{
			pNew->mem.content = pTmp->mem.content;
			memcpy(pNew->mem.content, pHost, pTmp->mem.sizeBytes);
			pNew->mem.sizeBytes = pTmp->mem.sizeBytes;
			break;
		}
	}

	if(pNew->mem.content == NULL)
	{
		free(pNew);
		my_mutex_unlock(&cudaMemMutex);
		return 1;
	}

	pNew->next = cudaMemHead;
	cudaMemHead = pNew;

	my_mutex_unlock(&cudaMemMutex);

	return 0;
}

static int updateBackupMem(const void *ptr)
{
	cudaMemItem *p = getCudaMemItemPtr(ptr);

	if(p == NULL)
	{
		return 1;
	}

	int ret = someMemcpy(p->mem.content, ptr, 
			p->mem.sizeBytes, p->mem.type, 0);

	return ret;
}

// TODO: test
static int updateBackupMemFromSrc(
		const void *dst, const void *src, 
		size_t count, enum cudaMemcpyKind kind)
{
	int ret = 0;
	cudaMemItem *p = getCudaMemItemPtr(dst);

	if(p == NULL)
	{
		return 1;
	}

	if(kind & 1 > 0 )
	{
		kind = (enum cudaMemcpyKind)((int)kind ^ 1);
	}

	RUNTIME_API_CALL(origCudaMemcpy(
				p->mem.content, src, count, kind));

	return ret;
}

// TODO: test
static int updateBackupMemFromSymbol(
		void *dst, const void *symbol, size_t count, 
		size_t offset, cudaMemcpyKind kind)
{
	int ret = 0;
	cudaMemItem *p = getCudaMemItemPtr(dst);

	if(p == NULL)
	{
		return 1;
	}

	if(kind & 1 > 0 )
	{
		kind = (enum cudaMemcpyKind)((int)kind ^ 1);
	}

	RUNTIME_API_CALL(origCudaMemcpyFromSymbol(
				p->mem.content, symbol, count, offset, kind));

	return ret;
}

// TODO: test
static int updateBackupMemToSymbol(const void *symbol, 
		const void *src, size_t count, size_t offset, 
		cudaMemcpyKind kind)
{
	int ret = 0;
	cudaMemItem *p = getCudaMemItemPtr(symbol);

	if(p == NULL)
	{
		return 1;
	}

	if(kind & 1 > 0 )
	{
		kind = (enum cudaMemcpyKind)((int)kind ^ 1);
	}

	RUNTIME_API_CALL(origCudaMemcpyToSymbol(
				p->mem.content, src, count, offset, kind));

	return ret;
}

// TODO: test
static int updateBackupMem2DFromSrc(void *dst, size_t dpitch, 
		const void *src, size_t spitch, size_t width, size_t height, 
		enum cudaMemcpyKind kind)
{
	int ret = 0;
	cudaMemItem *p = getCudaMemItemPtr(dst);

	if(p == NULL)
	{
		return 1;
	}

	if(kind & 1 > 0 )
	{
		kind = (enum cudaMemcpyKind)((int)kind ^ 1);
	}

	RUNTIME_API_CALL(origCudaMemcpy2D(p->mem.content, 
				dpitch, src, spitch, width, height, kind));

	return ret;
}

// TODO: test
static int updateBackupMem3DFromSrc(const cudaMemcpy3DParms *ptr)
{
	if(ptr->srcArray != NULL)
	{
		return 2;
	}

	int ret = 0;
	cudaMemItem *p = getCudaMemItemPtr(ptr->srcPtr.ptr);

	if(p == NULL)
	{
		return 1;
	}

	cudaMemcpy3DParms newParm = *ptr;
	newParm.dstPtr.ptr = p->mem.content;

	if(newParm.kind & 1 > 0 )
	{
		newParm.kind = (enum cudaMemcpyKind)
			((int)(newParm.kind) ^ 1);
	}

	RUNTIME_API_CALL(origCudaMemcpy3D(&newParm));

	return ret;
}

// TODO: test
static int updateBackupMem3DFromSrc(const cudaMemcpy3DPeerParms *ptr)
{
	cudaMemcpy3DParms newParm;
	newParm.dstArray = ptr->dstArray;
	newParm.dstPos = ptr->dstPos;
	newParm.dstPtr = ptr->dstPtr;
	newParm.extent = ptr->extent;
	newParm.srcArray = ptr->srcArray;
	newParm.srcPos = ptr->srcPos;
	newParm.srcPtr = ptr->srcPtr;
	newParm.kind = cudaMemcpyDeviceToHost;

	int ret = updateBackupMem3DFromSrc(&newParm);

	return ret;
}

// TODO: test
static int memsetBackupMem(const void *devPtr, 
		int value, size_t count)
{
	int ret = 0;
	cudaMemItem *p = getCudaMemItemPtr(devPtr);

	if(p == NULL)
	{
		return 1;
	}

	memset(p->mem.content, value, count);

	return ret;
}

// TODO: test
static int memsetBackupMem2D(const void *devPtr, size_t pitch, 
		int value, size_t width, size_t height)
{
	int ret = 0;
	cudaMemItem *p = getCudaMemItemPtr(devPtr);

	if(p == NULL)
	{
		return 1;
	}

	RUNTIME_API_CALL(origCudaMemset2D(
				p->mem.content, pitch, value, width, height));

	return ret;
}

// TODO: test
static int memsetBackupMem3D(cudaPitchedPtr pitchedDevPtr, 
		int value, cudaExtent extent)
{
	int ret = 0;
	cudaMemItem *p = getCudaMemItemPtr(pitchedDevPtr.ptr);

	if(p == NULL)
	{
		return 1;
	}

	cudaPitchedPtr newPtr = pitchedDevPtr;
	newPtr.ptr = p->mem.content;

	RUNTIME_API_CALL(origCudaMemset3D(
				newPtr, value, extent));

	return ret;
}

static int reBackupAddMem()
{
	my_mutex_lock(&cudaMemMutex);

	cudaMemItem *p = cudaMemHead;
	while(p != NULL)
	{
		someMemcpy(p->mem.content, p->mem.addr,
				p->mem.sizeBytes, p->mem.type, 0);
		p = p->next;
	}

	my_mutex_unlock(&cudaMemMutex);

	return 0;
}

static int reBackupMappedMem()
{
	my_mutex_lock(&cudaMemMutex);

	cudaMemItem *p = cudaMemHead;
	while(p != NULL)
	{
		if(p->mem.type == MAPPED_MEM_HOST 
				|| p->mem.type == MAPPED_MEM_DEVICE)
		{
			someMemcpy(p->mem.content, p->mem.addr,
					p->mem.sizeBytes, p->mem.type, 0);
		}
		p = p->next;
	}

	my_mutex_unlock(&cudaMemMutex);

	return 0;
}

static int restoreAllMem()
{
	my_mutex_lock(&cudaMemMutex);

	cudaMemItem *p = cudaMemHead;
	while(p != NULL)
	{
		someMemcpy(p->mem.addr, p->mem.content,
				p->mem.sizeBytes, p->mem.type, 1);
		p = p->next;
	}

	my_mutex_unlock(&cudaMemMutex);

	return 0;
}

static int freeMemItem(void *devPtr)
{
	cudaMemItem *p, *pPre;

	my_mutex_lock(&cudaMemMutex);

	p = cudaMemHead;
	pPre = NULL;
	while(p != NULL)
	{
		if(p->mem.addr == devPtr)
		{
			if(p == cudaMemHead)
			{
				cudaMemHead = p->next;
			}
			else
			{
				pPre->next = p->next;
			}
			if(p->mem.type != MAPPED_MEM_DEVICE)
			{
				free(p->mem.content);
			}
			free(p);
			break;
		}
		pPre = p;
		p = p->next;
	}

	my_mutex_unlock(&cudaMemMutex);

	return 0;
}

static int initEveData(int invalidEveNo[])
{
	if(eveProf == NULL)
	{
		eveProf = (eventIDArray *)malloc(sizeof(eventIDArray));
		if(eveProf == NULL)
		{
			return 1;
		}
	}

	CUdevice device = getCurDevice();
	char eventNames[EVENT_MAX_NUM][EVENT_NAME_MAX_LEN];
	int numEvents = 0;
	FILE *fp = fopen(eventNamesFilename, "r");
	char buf[EVENT_NAME_MAX_LEN];
	int pos = 0, numInvalid = 0;
	while(fgets(buf, EVENT_NAME_MAX_LEN, fp) != NULL)
	{
		int bufLen = strlen(buf) - 1;
		while(buf[bufLen] == '\n')
		{
			buf[bufLen] = '\0';
			bufLen--;
		}
		strncpy(eventNames[numEvents], buf, EVENT_NAME_MAX_LEN);
		eventNames[numEvents][EVENT_NAME_MAX_LEN - 1] = '\0';

		CUptiResult res = cuptiEventGetIdFromName(
				device, eventNames[numEvents], 
				eveProf->eventIDs + numEvents);
		if(res == CUPTI_SUCCESS)
		{
			numEvents++;
		}
		else
		{
			char *errstr;
			cuptiGetResultString(res, (const char **)&errstr);
			int errLen = strlen(errstr) + 
				strlen(eventNames[numEvents]);
			errLen += 5;
			char *errMsg = (char *)malloc(errLen);
			if(errMsg == NULL)
			{
				fprintf(stderr, "Error in line %d, fail to "\
						"malloc for profiling.\n", __LINE__);
				exit(EXIT_FAILURE);
			}
			snprintf(errMsg, errLen, "%s(%s)", 
					errstr, eventNames[numEvents]);
			logError(errMsg, __LINE__);
			free(errMsg);

			invalidEveNo[numInvalid++] = pos;
		}
		pos++;
	}
	invalidEveNo[numInvalid] = -1;

	fclose(fp);

	eveProf->numEvents = numEvents;

	return 0;
}

// invalidMtrNo has to be free.
static int initMtrData(
		CUcontext *context, int invalidMtrNo[])
{
	if(mtrIDEveNdArr == NULL)
	{
		mtrIDEveNdArr  = (metricIDEventNeedArray *)
			malloc(sizeof(metricIDEventNeedArray));
		if(mtrIDEveNdArr == NULL)
		{
			fprintf(stderr, "Error in line %d, fail to"\
					" malloc for profiling.\n", __LINE__);
			exit(EXIT_FAILURE);
		}
	}

	FILE *fp = fopen(metricNamesFilename, "r");
	char buf[METRIC_NAME_MAX_LEN];
	int numMetrics = 0;
	char metricNames[METRIC_MAX_NUM][METRIC_NAME_MAX_LEN]; 
	memset(metricNames, 0, sizeof(char)*METRIC_MAX_NUM*METRIC_NAME_MAX_LEN);
	CUdevice device = getCurDevice();
	int pos = 0, numInvalid = 0;
	while(fgets(buf, METRIC_NAME_MAX_LEN, fp) != NULL)
	{
		int bufLen = strlen(buf) - 1;
		while(buf[bufLen] == '\n')
		{
			buf[bufLen] = '\0';
			bufLen--;
		}
		strncpy(metricNames[numMetrics], buf, METRIC_NAME_MAX_LEN);
		metricNames[numMetrics][METRIC_NAME_MAX_LEN - 1] = '\0';

		CUptiResult res = cuptiMetricGetIdFromName(
				device, metricNames[numMetrics], 
				mtrIDEveNdArr->metricIDs + numMetrics);
		if(res == CUPTI_SUCCESS)
		{
//			printf("Metric: %s, ID: %d\n", metricNames[numMetrics], mtrIDEveNdArr->metricIDs[numMetrics]);

			numMetrics++;
		}
		else
		{
			printf("Failed to get metric ID for %s\n", buf);
			char *errstr;
			cuptiGetResultString(res, (const char **)&errstr);
			int errLen = strlen(errstr) + 
				strlen(metricNames[numMetrics]);
			errLen += 5;
			char *errMsg = (char *)malloc(errLen);
			if(errMsg == NULL)
			{
				fprintf(stderr, "Error in line %d, fail to "\
						"malloc for profiling.\n", __LINE__);
				exit(EXIT_FAILURE);
			}
			snprintf(errMsg, errLen, "%s(%s)", 
					errstr, metricNames[numMetrics]);
			logError(errMsg, __LINE__);
			free(errMsg);

			invalidMtrNo[numInvalid++] = pos;
		}

		// trace the pos of metricName
		pos++;
	}
	invalidMtrNo[numInvalid] = -1;

	fclose(fp);
	mtrIDEveNdArr->numMetrics = numMetrics;

	// initialize eventsNeed
	CUpti_MetricID *metricIDs = mtrIDEveNdArr->metricIDs;
	eventIDArray *eventsNeed = mtrIDEveNdArr->eventsNeed;
	int i;
	for(i = 0; i < numMetrics; i++)
	{
		//printf("Iteration %d\n", i);
		//printf("metric id: %d\n", metricIDs[i]);
		CUpti_EventGroupSets *sets;
		CUPTI_CALL(cuptiMetricCreateEventGroupSets(*context, 
					sizeof(CUpti_MetricID), metricIDs + i,
					&sets));

		eventsNeed[i].numEvents = 0;
		int j;
		// printf("\nMetric: %d, Record numEvents: ", p->metricID);
		for(j = 0; j < sets->numSets; j++)
		{
			CUpti_EventGroupSet *set = sets->sets + j;
			int k;
			for(k = 0; k < set->numEventGroups; k++)
			{
				// printf("\norigNum: %d\t", p->numEvents);
				uint32_t tmpNumEvents;
				size_t numEventsSize = sizeof(tmpNumEvents);
				CUpti_EventGroup *group = set->eventGroups + k;
				CUPTI_CALL(cuptiEventGroupGetAttribute(*group, 
							CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS,
							&numEventsSize, &tmpNumEvents));
				size_t eventIdsSize = 
					tmpNumEvents * sizeof(CUpti_EventID);
				CUPTI_CALL(cuptiEventGroupGetAttribute(*group, 
							CUPTI_EVENT_GROUP_ATTR_EVENTS, 
							&eventIdsSize, eventsNeed[i].eventIDs 
							+ eventsNeed[i].numEvents));
				eventsNeed[i].numEvents += tmpNumEvents;
			}
		}
	}

	return 0;
}

static int initEveMtrEGSData(CUcontext *context, 
		int invalidEveNo[], int invalidMtrNo[])
{
	my_mutex_lock(&mtrEveDataMutex);

	PROF_CALL(initEveData(invalidEveNo));
	PROF_CALL(initMtrData(context, invalidMtrNo));

	if(eveCol == NULL)
	{
		eveCol = (eventsCollection *)malloc(
				sizeof(eventsCollection));
		if(eveCol == NULL)
		{
			my_mutex_unlock(&mtrEveDataMutex);
			return 1;
		}
	}

	// initialize numEvents and eventIDs
	eveCol->numEvents = eveProf->numEvents;
	memcpy(eveCol->eventIDs, eveProf->eventIDs, 
			sizeof(CUpti_EventID)*eveCol->numEvents);

	int index = eveCol->numEvents;
	int i;
	for(i = 0; i < mtrIDEveNdArr->numMetrics; i++)
	{
		eventIDArray *eNeed = mtrIDEveNdArr->eventsNeed + i;
		int j;
		for(j = 0; j < eNeed->numEvents; j++)
		{
			int k;
			for(k = 0; k < index; k++)
			{
				if(eNeed->eventIDs[j] 
						== eveCol->eventIDs[k])
				{
					break;
				}
			}
			if(k == index)
			{
				eveCol->eventIDs[index++] = 
					eNeed->eventIDs[j];
			}
		}
	}
	eveCol->numEvents = index;

	// initialize groupSets
	size_t eveIdSizeBytes = 
		sizeof(CUpti_EventID) * eveCol->numEvents;
	CUPTI_CALL(cuptiEventGroupSetsCreate(*context, eveIdSizeBytes,
				eveCol->eventIDs, &(eveCol->groupSets)));

	uint32_t all = 1;

	// initialize numEventsEachSets, set group attribute
	for(i = 0; i < eveCol->groupSets->numSets; i++)
	{
		CUpti_EventGroupSet *set = 
			eveCol->groupSets->sets + i;
		int j;
		int numEveEachSets = 0;
		for(j = 0; j < set->numEventGroups; j++)
		{
			uint32_t tmpNumEvents;
			size_t numEventsSize = sizeof(tmpNumEvents);
			CUpti_EventGroup *group = set->eventGroups + j;
			CUPTI_CALL(cuptiEventGroupGetAttribute(*group, 
						CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS,
						&numEventsSize, &tmpNumEvents));
			numEveEachSets += tmpNumEvents;

			CUPTI_CALL(cuptiEventGroupSetAttribute(*group, 
						CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
						sizeof(all), &all));
		}
		eveCol->numEventsEachSets[i] = numEveEachSets;
	}

	my_mutex_unlock(&mtrEveDataMutex);

	return 0;
}

static int startColEveVals(
		CUcontext *context, EGSData_t *egsData)
{
	cudaDeviceSynchronize();
	CUPTI_CALL(cuptiSetEventCollectionMode(*context,
				CUPTI_EVENT_COLLECTION_MODE_KERNEL));
	CUPTI_CALL(cuptiEventGroupSetEnable(egsData->groupSet));

	return 0;
}

static int endColEveVals(CUcontext *context, 
		CUdevice *device, EGSData_t *egsData)
{
	cudaDeviceSynchronize();
	int eventIdx = 0;

	int i;
	for (i = 0; i < egsData->groupSet->numEventGroups; i++) 
	{
		CUpti_EventGroup group = 
			egsData->groupSet->eventGroups[i];
		CUpti_EventDomainID groupDomain;
		uint32_t numEvents, numInstances, numTotalInstances;
		CUpti_EventID *eventIds;
		size_t groupDomainSize = sizeof(groupDomain);
		size_t numEventsSize = sizeof(numEvents);
		size_t numInstancesSize = sizeof(numInstances);
		size_t numTotalInstancesSize = sizeof(numTotalInstances);
		uint64_t *values, normalized, sum;
		size_t valuesSize, eventIdsSize;

		CUPTI_CALL(cuptiEventGroupGetAttribute(group, 
					CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID,
					&groupDomainSize, &groupDomain));
		CUPTI_CALL(cuptiDeviceGetEventDomainAttribute(
					*device, groupDomain, 
					CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT,
					&numTotalInstancesSize, &numTotalInstances));
		CUPTI_CALL(cuptiEventGroupGetAttribute(group, 
					CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
					&numInstancesSize, &numInstances));
		CUPTI_CALL(cuptiEventGroupGetAttribute(group, 
					CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS,
					&numEventsSize, &numEvents));
		eventIdsSize = numEvents * sizeof(CUpti_EventID);
		eventIds = (CUpti_EventID *)malloc(eventIdsSize);
		if(eventIds == NULL)
		{
			return 1;
		}
		CUPTI_CALL(cuptiEventGroupGetAttribute(group, 
					CUPTI_EVENT_GROUP_ATTR_EVENTS,
					&eventIdsSize, eventIds));

		valuesSize = sizeof(uint64_t) * numInstances * numEvents;
		values = (uint64_t *)malloc(valuesSize);
		if(values == NULL)
		{
			return 1;
		}

		// TODO: 
		size_t numEventsIdsRead = -1;
		CUPTI_CALL(cuptiEventGroupReadAllEvents(
					group, CUPTI_EVENT_READ_FLAG_NONE, 
					&valuesSize, values, &eventIdsSize, 
					eventIds, &numEventsIdsRead));
		if(numEventsIdsRead != numEvents)
		{
			fprintf(stderr, "Error: %s, %d, %d events need, "\
					"but %d events read\n", __FILE__, __LINE__,
					numEvents, numEventsIdsRead);
			exit(EXIT_FAILURE);
		}

		int j = 0;
		uint64_t *accuVals = (uint64_t *)malloc(
				numEvents * sizeof(uint64_t));
		if(accuVals == NULL)
		{
			return 1;
		}
		memset(accuVals, 0, numEvents * sizeof(uint64_t));
		// printf("origVals:");
		for(j = 0; j < numEvents * numInstances; j++)
		{
			/*
			   if(j % numEvents == 0)
			   {
			   printf("\n");
			   }
			   printf("%ld\t", values[j]);
			   */
			accuVals[j % numEvents] += values[j];
		}
		// printf("\n");
		// printf("accuVals:");
		/*
		   for(j = 0; j < numEvents; j++)
		   {
		   if(j % numEvents == 0)
		   {
		   printf("\n");
		   }
		   printf("%ld\t", accuVals[j]);
		   }
		   printf("\n");
		   */

			for(j = 0; j < numEvents; j++)
			{
				accuVals[j] *= numTotalInstances / numInstances;
			}
		memcpy(egsData->eventIDs + eventIdx, 
				eventIds, numEvents * sizeof(CUpti_EventID));
		memcpy(egsData->eventValues + eventIdx,
				accuVals, numEvents * sizeof(uint64_t));
		eventIdx += numEvents;

		// time consuming 0.06s
		/*
		   int j;
		   for (j = 0; j < numEvents; j++) 
		   {
		   CUPTI_CALL(cuptiEventGroupReadEvent(
		   group, CUPTI_EVENT_READ_FLAG_NONE, 
		   eventIds[j], &valuesSize, values));
		   if (eventIdx >= egsData->numEvents) 
		   {
		   fprintf(stderr, "error: too many events "\
		   "collected, metric expects only %d\n",
		   (int)egsData->numEvents);
		   exit(EXIT_FAILURE);
		   }

			// sum collect event values from all instances
			sum = 0;
			int k;
			printf("id: %lu,  ", eventIds[j]);
			for (k = 0; k < numInstances; k++)
			{
			printf("%llu, ", values[k]);
			sum += values[k];
			}
			printf(",  sum %lld\n", sum);

		// normalize the event value to represent the total number of
		// domain instances on the device
		normalized = (sum * numTotalInstances) / numInstances;

		egsData->eventIDs[eventIdx] 
		= eventIds[j];
		egsData->eventValues[eventIdx] 
		= normalized;
		eventIdx++;
		}
		*/

		free(eventIds);
		free(values);
		free(accuVals);
	}

	// disable all eventGroups
	CUPTI_CALL(cuptiEventGroupSetDisable(
				egsData->groupSet));

	if(eventIdx != egsData->numEvents)
	{
		fprintf(stderr, "Error: %s, %d, %d events need, "\
				"%d events collected\n", __FILE__, __LINE__,
				egsData->numEvents, eventIdx);
		exit(EXIT_FAILURE);
	}

	return 0;
}

static int getMetricsValue(CUdevice *device, 
		unsigned long long duration, CUpti_EventID *eventIDs, 
		uint64_t *eveValues, metricData *mtrData, int *numMtr)
{
	*numMtr = mtrIDEveNdArr->numMetrics;
	int i;
	for(i = 0; i < mtrIDEveNdArr->numMetrics; i++)
	{
		CUpti_MetricID metricID = mtrIDEveNdArr->metricIDs[i];
		uint32_t numEvents = mtrIDEveNdArr->eventsNeed[i].numEvents;
		CUpti_EventID *curEventIDs = 
			mtrIDEveNdArr->eventsNeed[i].eventIDs;
		uint64_t *curEventVals = (uint64_t *)malloc(
				numEvents * sizeof(uint64_t));
		if(curEventVals == NULL)
		{
			return 1;
		}

		int j;
		for(j = 0; j < numEvents; j++)
		{
			int k;
			for(k = 0; k < eveCol->numEvents; k++)
			{
				if(eventIDs[k] == curEventIDs[j])
				{
					curEventVals[j] = eveValues[k];
					break;
				}
			}
		}

		// use all the collected events to calculate the metric value
		CUpti_MetricValue metricValue;
		/*
		   printf("dev: %d, mtrID: %d, numEvents: %d, duration: %ld\n",
		   (int)*device, (int)metricID, numEvents, duration);
		   fflush(NULL);
		   */
		CUPTI_CALL(cuptiMetricGetValue(*device, metricID,
					numEvents * sizeof(CUpti_EventID), 
					curEventIDs, numEvents * sizeof(uint64_t),
					curEventVals, duration, &metricValue));

		mtrData[i].metricID = metricID;
		mtrData[i].metricValue = metricValue;

		free(curEventVals);
	}

	return 0;
}

static int initEGSData(EGSData_t *egsData, 
		CUcontext *context, CUdevice *device, 
		CUpti_EventGroupSet *curSet, int numEvents)
{
	egsData->groupSet = curSet;
	egsData->numEvents = numEvents;
	egsData->eventIDs = (CUpti_EventID *)malloc(
			numEvents * sizeof(CUpti_EventID));
	if(egsData->eventIDs == NULL)
	{
		return 1;
	}
	egsData->eventValues = (uint64_t *)malloc(
			numEvents * sizeof(uint64_t));
	if(egsData->eventValues == NULL)
	{
		return 1;
	}

	return 0;
}

static void freeEGSData(EGSData_t *egsData)
{
	free(egsData->eventIDs);
	free(egsData->eventValues);
}

static void kernelTimeTrace(
		const char *kerName, timespec *start, timespec *end)
{
	my_mutex_lock(&kerTraceMutex);

	fprintf(kerTraceFile, "\n\"%s\",%llu,%llu,%llu,%llu", 
			kerName, start->tv_sec, start->tv_nsec, 
			end->tv_sec, end->tv_nsec);

	my_mutex_unlock(&kerTraceMutex);
}

static void CUPTIAPI bufferRequested(
		uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
	uint8_t *rawBuffer;

	*size = 16 * 1024;
	rawBuffer = (uint8_t *)malloc(*size + ALIGN_SIZE);

	*buffer = ALIGN_BUFFER(rawBuffer, ALIGN_SIZE);
	*maxNumRecords = 0;

	if (*buffer == NULL) 
	{
		printf("Error: out of memory\n");
		exit(EXIT_FAILURE);
	}
}

static void CUPTIAPI bufferCompleted(
		CUcontext ctx, uint32_t streamId, 
		uint8_t *buffer, size_t size, size_t validSize)
{
	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &end);

	CUpti_Activity *record = NULL;
	CUpti_ActivityKernel2 *kernel;

	CUPTI_CALL(cuptiActivityGetNextRecord(buffer, validSize, &record));

	kernel = (CUpti_ActivityKernel2 *)record;
	if (kernel->kind != CUPTI_ACTIVITY_KIND_KERNEL) 
	{
		fprintf(stderr, "Error: expected kernel activity "\
				"record, got %d\n", (int)kernel->kind);
		exit(EXIT_FAILURE);
	}

	if(kerInfo == NULL)
	{
		kerInfo = (kernelInfo *)malloc(sizeof(kernelInfo));
		if(kerInfo ==  NULL)
		{
			fprintf(stderr, "Error: failed to record kernel info\n");
			exit(EXIT_FAILURE);
		}
	}

	start = end;
	start.tv_nsec -= kernel->end - kernel->start;
	while (start.tv_nsec < 0)
	{
		start.tv_nsec += 1000000000;
		start.tv_sec -= 1;
	}

	memset(kerInfo->kernelName, 0, KERNEL_NAME_MAX_LEN);
	int status;
	char *kerName = abi::__cxa_demangle(kernel->name, 
			NULL, NULL, &status);
	if(kerName == NULL)
	{
		strncpy(kerInfo->kernelName, 
				kernel->name, strlen(kernel->name));
	}
	else
	{
		strncpy(kerInfo->kernelName, kerName, KERNEL_NAME_MAX_LEN);
		free(kerName);
	}
	kerInfo->deviceID = kernel->deviceId;
	kerInfo->streamID = kernel->streamId;
	kerInfo->gridID = kernel->gridId;
	kerInfo->gridX = kernel->gridX;
	kerInfo->gridY = kernel->gridY;
	kerInfo->gridZ = kernel->gridZ;
	kerInfo->blockX = kernel->blockX;
	kerInfo->blockY = kernel->blockY;
	kerInfo->blockZ = kernel->blockZ;
	kerInfo->start = start;
	kerInfo->end = end;
	kerInfo->dynamicSharedMem = kernel->dynamicSharedMemory;
	kerInfo->staticSharedMem = kernel->staticSharedMemory;
	kerInfo->localMemPerThread = kernel->localMemoryPerThread;
	kerInfo->localMemTotal = kernel->localMemoryTotal;
	kerInfo->regPerThread = kernel->registersPerThread;
	kerInfo->cacheConfigReq = 
		kernel->cacheConfig.config.requested;
	kerInfo->cacheConfigUsed = 
		kernel->cacheConfig.config.executed;
	kerInfo->sharedMemConfigUsed = kernel->sharedMemoryConfig;

	kernelTimeTrace(kerInfo->kernelName, &start, &end);

	free(buffer);
}

static int recordKernelInfo(kernelInfo *ker)
{
	if(fileOpen == 0)
	{
		openFiles();
	}
	fprintf(kerProfDataFile, "\"%s\",%u,%u,%lld,%d,%d,%d,%d,%d,%d,"\
			"%llu,%llu,%llu,%llu,%d,%d,%u,%u,%u,%u,%u,%u,", 
			ker->kernelName, ker->deviceID, ker->streamID, 
			ker->gridID, ker->gridX, ker->gridY, ker->gridZ, 
			ker->blockX, ker->blockY, ker->blockZ,
			ker->start, ker->end, ker->dynamicSharedMem,
			ker->staticSharedMem, ker->localMemPerThread,
			ker->localMemTotal, ker->regPerThread,
			ker->cacheConfigReq, ker->cacheConfigUsed,
			ker->sharedMemConfigUsed);
	return 0;
}

static int recordOccupancy(occupancyData *occuData)
{
	if(fileOpen == 0)
	{
		openFiles();
	}
	fprintf(kerProfDataFile, "%u,%u,%.2f,", occuData->theoLimiter, 
			occuData->achiLimiter, occuData->theory);

	return 0;
}

static int getMetricValueStr(
		metricData *mtrData, char mtrValStr[], int len)
{
	CUpti_MetricValue metricValue = mtrData->metricValue;
	char metricName[METRIC_NAME_MAX_LEN];
	size_t nameSize = sizeof(char) * METRIC_NAME_MAX_LEN;
	CUPTI_CALL(cuptiMetricGetAttribute(mtrData->metricID, 
				CUPTI_METRIC_ATTR_NAME, &nameSize, &metricName));
	CUpti_MetricValueKind valueKind;
	size_t valueKindSize = sizeof(valueKind);
	CUPTI_CALL(cuptiMetricGetAttribute(mtrData->metricID, 
				CUPTI_METRIC_ATTR_VALUE_KIND, 
				&valueKindSize, &valueKind));

	switch (valueKind) 
	{
		case CUPTI_METRIC_VALUE_KIND_DOUBLE:
			snprintf(mtrValStr, len, "%.2f", 
					metricValue.metricValueDouble);
			break;
		case CUPTI_METRIC_VALUE_KIND_UINT64:
			snprintf(mtrValStr, len, "%llu", 
					(unsigned long long)metricValue.
					metricValueUint64);
			break;
		case CUPTI_METRIC_VALUE_KIND_INT64:
			snprintf(mtrValStr, len, "%lld", 
					(long long)metricValue.metricValueInt64);
			break;
		case CUPTI_METRIC_VALUE_KIND_PERCENT:
			snprintf(mtrValStr, len, "%.2f", 
					metricValue.metricValuePercent);
			break;
		case CUPTI_METRIC_VALUE_KIND_THROUGHPUT:
			snprintf(mtrValStr, len, "%llu", 
					(unsigned long long)metricValue.
					metricValueThroughput);
			break;
		case CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL:
			snprintf(mtrValStr, len, "%u", 
					(unsigned int)metricValue.
					metricValueUtilizationLevel);
			break;
		default:
			fprintf(stderr, "error: unknown metric value kind\n");
			exit(EXIT_FAILURE);
	}

	return 0;
}

static int recordMetrics(metricData *mtrDatas, 
		int numMetrics, int invalidMtrNo[], CUdevice *device)
{
	char mtrValStrs[2048];
	int len = 0;
	memset(mtrValStrs, 0, 2048);
	int i, invalidI;
	for (i = 0, invalidI = 0; i < numMetrics || 
			invalidMtrNo[invalidI] >= 0;)
	{
		char buf[256];
		if(invalidMtrNo[invalidI] == i + invalidI)
		{
			strncpy(buf, "\"null\"", 8);
			invalidI++;
		}
		else
		{
			getMetricValueStr(mtrDatas + i, buf, 254);
			i++;
		}

		int bufLen = strlen(buf);
		memcpy(mtrValStrs + len, buf, bufLen);
		len += bufLen;
		mtrValStrs[len++] = ',';
	}
	mtrValStrs[len] = '\0';

	if(fileOpen == 0)
	{
		openFiles();
	}
	fprintf(kerProfDataFile, "%s", mtrValStrs);

	return 0;
}

static int recordEvents(eventData *eveDatas, 
		int numEvents, int invalidEveNo[])
{
	char eveValStrs[2048];
	int len = 0;
	memset(eveValStrs, 0, 2048);
	int i, invalidI;
	for (i = 0, invalidI = 0; i < numEvents || 
			invalidEveNo[invalidI] >= 0;)
	{
		char buf[256];
		if(invalidEveNo[invalidI] == i + invalidI)
		{
			strncpy(buf, "\"null\"", 8);
			invalidI++;
		}
		else
		{
			snprintf(buf, 256, "%llu", 
					(eveDatas+i)->eventValue);
			i++;
		}

		int lenBuf = strlen(buf);
		memcpy(eveValStrs + len, buf, lenBuf);
		len += lenBuf;
		eveValStrs[len++] = ',';
	}
	/*
	   eveValStrs[len-1] = '\n';
	   eveValStrs[len] = '\0';
	   */
	eveValStrs[len - 1] = '\0';

	if(fileOpen == 0)
	{
		openFiles();
	}
	fprintf(kerProfDataFile, "%s", eveValStrs);

	return 0;
}

static int recordKerProfData(kernelProfileData *kerProfData, 
		int invalidEveNo[], int invalidMtrNo[], CUdevice *device)
{
	my_mutex_lock(&kerProfDataMutex);

	if(fileOpen == 0)
	{
		openFiles();
	}

	fprintf(kerProfDataFile, "\n");
	PROF_CALL(recordKernelInfo(&kerProfData->kerInfo));
	PROF_CALL(recordOccupancy(&kerProfData->occu));
	PROF_CALL(recordMetrics(kerProfData->metricDatas, 
				kerProfData->numMetrics, invalidMtrNo, device));
	PROF_CALL(recordEvents(kerProfData->eventDatas, 
				kerProfData->numEvents, invalidEveNo));

	fflush(kerProfDataFile);

	my_mutex_unlock(&kerProfDataMutex);

	return 0;
}


static cuTimeTrace getBaseTraceNoInfo(
		const char *shortDesc, struct timespec start, 
		struct timespec end, traceType type)
{
	cuTimeTrace ret;
	strncpy(ret.shortDesc, shortDesc, 63);
	ret.shortDesc[63] = '\0';
	ret.start = start;
	ret.end = end;
	ret.type = type;

	return ret;
}

static cuTimeTrace getCuMemcpyTrace(
		const char *shortDesc, struct timespec start, 
		struct timespec end, enum cudaMemcpyKind kind, 
		size_t count)
{
	cuTimeTrace ret = getBaseTraceNoInfo(
			shortDesc, start, end, RUNTIME_MEMCPY);
	ret.info.cpyInfo.kind = kind;
	ret.info.cpyInfo.count = count;

	return ret;
}

static void recordMemcpyTimeTrace(cuTimeTrace *trace)
{
	/*
	   char typeStr[4][8] = 
	   {
	   "HToH", "HToD", "DToH", "DToD"
	   };
	   */

	my_mutex_lock(&cpyTraceMutex);

	if(fileOpen == 0)
	{
		openFiles();
	}
	fprintf(cpyFile, "\n\"%s\",%d,%llu,%llu,%llu,%llu,%lu", 
			trace->shortDesc, (int)trace->info.cpyInfo.kind,
			trace->start.tv_sec, trace->start.tv_nsec,
			trace->end.tv_sec, trace->end.tv_nsec, 
			trace->info.cpyInfo.count);
	fflush(cpyFile);

	my_mutex_unlock(&cpyTraceMutex);
}

static void recordProfOverhead(cuTimeTrace *trace)
{
	my_mutex_lock(&profOverheadMutex);

	if(fileOpen == 0)
	{
		openFiles();
	}
	fprintf(overFile, "\n\"%s\",%llu,%llu,%llu,%llu",
			trace->shortDesc, trace->start.tv_sec, 
			trace->start.tv_nsec, trace->end.tv_sec, 
			trace->end.tv_nsec);
	fflush(overFile);

	my_mutex_unlock(&profOverheadMutex);
}

static void recordTimeTrace(cuTimeTrace *trace)
{
	switch(trace->type)
	{
		case RUNTIME_MEMCPY:
			recordMemcpyTimeTrace(trace);
			break;
		case PROF_OVERHEAD:
			recordProfOverhead(trace);
			break;
		default:
			perror("Error trace type\n");
			exit(EXIT_FAILURE);
	}
}

long ceilingMod(double number, long significance)
{
	long retval = (long)(ceil(number) + 0.1);
	retval += retval % significance;

	return retval;
}

long floorMod(double number, long significance)
{
	long retval = (long)(floor(number) + 0.1);
	retval -= retval % significance;

	return retval;
}

long min2(long num1, long num2)
{
	return num1 < num2 ? num1: num2;
}

long min3(long num1, long num2, long num3)
{
	long tmpMin = min2(num1, num2);
	return min2(tmpMin, num3);
}

gpuLimit getGPULimit()
{
	gpuLimit gLimit;

	cudaDeviceProp prop;

	int capabMajor=0;
	int capabMinor=0;
	int deviceId = 0;
	cudaGetDevice(&deviceId);
	RUNTIME_API_CALL(
			cudaGetDeviceProperties(&prop, deviceId));
	capabMajor = prop.major;
	capabMinor = prop.minor;
	int capabMul10 = capabMajor * 10 + capabMinor;

	gLimit.limitThreadsPerWarp = prop.warpSize;
	gLimit.limitWarpsPerMultpro = 
		prop.maxThreadsPerMultiProcessor/prop.warpSize;
	gLimit.limitThreadsPerMultpro = 
		prop.maxThreadsPerMultiProcessor;
	gLimit.limitBlocksPerMultpro = capabMul10 < 30 ? 8 : 16;
	// gLimit.limitTotalRegs = prop.regsPerMultiprocessor;
	gLimit.limitTotalRegs = capabMul10< 12 ? 8192 : 
		capabMul10 < 20 ? 16384: capabMul10 < 30 ? 32768 : 65536;
	gLimit.regAllocaUnitSize = capabMul10 < 12 ? 256 : 
		capabMul10 < 20 ? 512: capabMul10 < 30 ? 64 : 256;
	gLimit.allocaGranularity = capabMul10 < 20 ? 
		GRANULARITY_BLOCK : GRANULARITY_WARP;
	gLimit.limitRegsPerThread = capabMul10 < 20 ? 124 :
		capabMul10 < 35 ? 63: 255;
	// gLimit.limitTotalSMemPerMultpro = 
	// 	prop.sharedMemPerMultiprocessor;
	gLimit.limitTotalSMemPerMultpro = capabMul10 < 20 ? 
		16384: 49152;
	gLimit.SMemAllocaSize = capabMul10 < 20 ? 512 : 
		capabMul10 < 30 ? 128 : 256;
	gLimit.warpAllocaGranularity = capabMul10 < 30 ? 2 : 4;
	gLimit.maxThreadsPerBlock = prop.maxThreadsPerBlock;

	// gLimit.limitThreadsPerWarp = 32;
	// gLimit.limitWarpsPerMultpro = capabMul10 < 12 ? 24 : 
	//  	capabMul10 < 20 ? 32: capabMul10 < 30 ? 48 : 64;
	// gLimit.limitThreadsPerMultpro = 
	// 	gLimit.limitThreadsPerWarp * gLimit.limitWarpsPerMultpro;
	// gLimit.limitBlocksPerMultpro = capabMul10 < 30 ? 8 : 16;
	// gLimit.limitTotalRegs = capabMul10< 12 ? 8192 : 
	// 	capabMul10 < 20 ? 16384: capabMul10 < 30 ? 32768 : 65536;
	// gLimit.regAllocaUnitSize = capabMul10 < 12 ? 256 : 
	// 	capabMul10 < 20 ? 512: capabMul10 < 30 ? 64 : 256;
	// gLimit.allocaGranularity = capabMul10 < 20 ? 
	// 	GRANULARITY_BLOCK : GRANULARITY_WARP;
	// gLimit.limitRegsPerThread = capabMul10 < 20 ? 124 :
	// 	capabMul10 < 35 ? 63: 255;
	// gLimit.limitTotalSMemPerMultpro = capabMul10 < 20 ? 
	//	16384: 49152;
	// gLimit.SMemAllocaSize = capabMul10 < 20 ? 512 : 
	//	capabMul10 < 30 ? 128 : 256;
	// gLimit.warpAllocaGranularity = capabMul10 < 30 ? 2 : 4;
	// gLimit.maxThreadsPerBlock = capabMul10 < 20 ? 512 : 1024;

	return gLimit;
}

double theoryOccupancy(
		int threadCount, int regCount, int sharedMemory, 
		size_t sharedSize, occupancyLimiter *limiter)
{
	gpuLimit gLimit = getGPULimit();
	if(sharedSize != -1)
	{
		gLimit.limitTotalSMemPerMultpro = min2(
				gLimit.limitTotalSMemPerMultpro, sharedSize);
	}

	int warpsPerBlock = ceilingMod(1.0 * threadCount 
			/ gLimit.limitThreadsPerWarp, 1);
	if(warpsPerBlock == 0)
	{
		return 0;
	}

	int limitBlocksDueToWarps = min2(
			gLimit.limitBlocksPerMultpro, floorMod(1.0 *
				gLimit.limitWarpsPerMultpro / warpsPerBlock, 1));

	int regsLimitPerSM;
	if(gLimit.allocaGranularity == GRANULARITY_BLOCK)
	{
		regsLimitPerSM = gLimit.limitTotalRegs;
	}
	else
	{
		regsLimitPerSM = floorMod(1.0 * gLimit.limitTotalRegs / 
				ceilingMod(1.0 * regCount * 
					gLimit.limitThreadsPerWarp, 
					gLimit.regAllocaUnitSize),
				gLimit.warpAllocaGranularity);
	}

	int regsPerBlock;
	if(gLimit.allocaGranularity == GRANULARITY_BLOCK)
	{
		regsPerBlock = ceilingMod(ceilingMod(1.0 * warpsPerBlock,
					gLimit.warpAllocaGranularity) * regCount *
				gLimit.limitThreadsPerWarp, 
				gLimit.regAllocaUnitSize);
	}
	else
	{
		regsPerBlock = warpsPerBlock;
	}

	int limitBlocksDueToRegs;
	if(regCount>gLimit.limitRegsPerThread)
	{
		limitBlocksDueToRegs = 0;
	}
	else
	{
		if(regCount > 0)
		{
			limitBlocksDueToRegs = floorMod(1.0 * 
					regsLimitPerSM / regsPerBlock, 1);
		}
		else
		{
			limitBlocksDueToRegs = gLimit.limitBlocksPerMultpro;
		}
	}

	int sharedMemPerBlock = ceilingMod(1.0 * sharedMemory, 
			gLimit.SMemAllocaSize);

	int limitBlocksDueToSMem;
	if(sharedMemPerBlock>0)
	{
		limitBlocksDueToSMem = floorMod(1.0 *
				gLimit.limitTotalSMemPerMultpro / 
				sharedMemPerBlock, 1);
	}
	else
	{
		limitBlocksDueToSMem = gLimit.limitBlocksPerMultpro;
	}

	double minLimit = min3(limitBlocksDueToWarps, 
			limitBlocksDueToRegs, limitBlocksDueToSMem);

	double occupancyOfEachMultip = 1.0 * ceilingMod(1.0 * 
			threadCount / gLimit.limitThreadsPerWarp, 1) * 
		minLimit / gLimit.limitWarpsPerMultpro;

	if(minLimit == limitBlocksDueToWarps)
	{
		*limiter = BLOCK_SIZE;
	}
	else if(minLimit == limitBlocksDueToSMem)
	{
		*limiter = SMEM_PER_BLOCK;
	}
	else if(minLimit == limitBlocksDueToRegs)
	{
		*limiter = REG_PER_THREAD;
	}

	return occupancyOfEachMultip;
}

static int getSMemSizeConfig()
{
	cudaFuncCache cacheConfig;
	int sharedSize;
	RUNTIME_API_CALL(cudaDeviceGetCacheConfig(&cacheConfig));
	switch(cacheConfig)
	{
		case cudaFuncCachePreferNone:
			sharedSize = -1;
			break;
		case cudaFuncCachePreferShared:
			sharedSize = 48 * 1024;
			break;
		case cudaFuncCachePreferL1:
			sharedSize = 16 * 1024;
			break;
		case cudaFuncCachePreferEqual:
			sharedSize = 32 * 1024;
			break;
		default:
			perror("Error get cache config\n");
			exit(EXIT_FAILURE);
	}

	return sharedSize;
}

static int kerInfoIsSameClass(
		const kernelInfo *kerA, const kernelInfo *kerB)
{
	return strcmp(kerA->kernelName, kerB->kernelName) == 0
		&& kerA->deviceID == kerB->deviceID
		&& kerA->gridX == kerB->gridX 
		&& kerA->gridY == kerB->gridY
		&& kerA->gridZ == kerB->gridZ
		&& kerA->blockX == kerB->blockX 
		&& kerA->blockY == kerB->blockY
		&& kerA->blockZ == kerB->blockZ
		&& kerA->dynamicSharedMem == kerB->dynamicSharedMem
		&& kerA->staticSharedMem == kerB->staticSharedMem
		&& kerA->localMemPerThread == kerB->localMemPerThread
		&& kerA->localMemTotal == kerB->localMemTotal
		&& kerA->regPerThread == kerB->regPerThread
		&& kerA->cacheConfigReq == kerB->cacheConfigReq
		&& kerA->cacheConfigUsed == kerB->cacheConfigUsed
		&& kerA->sharedMemConfigUsed == kerB->sharedMemConfigUsed;
}

static int reachMaxProfRecord(kernelInfo * theKerInfo)
{
	if(theKerInfo == NULL)
	{
		fprintf(stderr, "Error in line %d, theKerInfo "\
				"is NULL.\n", __LINE__);
		exit(EXIT_FAILURE);
	}

	int reachMax = 0;
	my_mutex_lock(&kerClassMutex);

	if(kerClassHead == NULL)
	{
		kerClassHead = (kerProfClassRecord*)
			malloc(sizeof(kerProfClassRecord));
		if(kerClassHead == NULL)
		{
			fprintf(stderr, "Error in line %d, fail to "\
					"malloc for profiling.\n", __LINE__);
			exit(EXIT_FAILURE);
		}

		kerClassHead->kerInfo = *theKerInfo;

		kerClassHead->times = 1;
		kerClassHead->next = NULL;


		my_mutex_unlock(&kerClassMutex);
		return 0;
	}

	kerProfClassRecord *classP = kerClassHead;
	kerProfClassRecord *postP = kerClassHead->next;
	while(postP != NULL)
	{
		if(kerInfoIsSameClass(&classP->kerInfo, theKerInfo))
		{
			classP->times++;
			int retval = 0;
			if(classP->times > KERNEL_MAX_RECORD)
			{
				retval = 1;
			}
			my_mutex_unlock(&kerClassMutex);
			return retval;
		}
		classP = classP->next;
		postP = postP->next;
	}

	kerProfClassRecord *newRecord = (kerProfClassRecord*)
		malloc(sizeof(kerProfClassRecord));
	if(newRecord == NULL)
	{
		fprintf(stderr, "Error in line %d, fail to "\
				"malloc for profiling.\n", __LINE__);
		exit(EXIT_FAILURE);
	}
	newRecord->kerInfo = *theKerInfo;
	newRecord->times = 1;
	newRecord->next = NULL;
	classP->next = newRecord;

	my_mutex_unlock(&kerClassMutex);

	return 0;
}

cudaError_t cudaLaunch(const void * a1) 
{
#ifdef CUDA_HOOK_PROF
	CUPTI_CALL(cuptiActivityEnable(
				CUPTI_ACTIVITY_KIND_KERNEL));
	CUPTI_CALL(cuptiActivityRegisterCallbacks(
				bufferRequested, bufferCompleted));

	// backup mapped memory before launch kernel
	PROF_CALL(reBackupMappedMem());
#endif

	typedef cudaError_t (*cudaLaunch_p_h) (const void *);
	static cudaLaunch_p_h cudaLaunch_h = 
		(cudaLaunch_p_h)getSymbol("cudaLaunch");
	cudaError_t retval = (*cudaLaunch_h)(a1);

#ifdef CUDA_HOOK_PROF
	cudaDeviceSynchronize();

	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	CUPTI_CALL(cuptiActivityDisable(
				CUPTI_ACTIVITY_KIND_KERNEL));
	CUPTI_CALL(cuptiActivityFlushAll(0));

	if(reachMaxProfRecord(kerInfo))
	{
		clock_gettime(CLOCK_REALTIME, &end);
		cuTimeTrace timeTrace = getBaseTraceNoInfo(
				"profileKernelLoop", start, end, PROF_OVERHEAD);
		recordTimeTrace(&timeTrace);
		return retval;
	}


	// start collecting events

	CUcontext context = getCurContext();
	CUdevice device = getCurDevice();
	int invalidEveNo[EVENT_MAX_NUM];
	int invalidMtrNo[METRIC_MAX_NUM];
	invalidEveNo[0] = -1;
	invalidMtrNo[0] = -1;
	PROF_CALL(initEveMtrEGSData(&context, 
				invalidEveNo, invalidMtrNo));

	int numPass = eveCol->groupSets->numSets;
	uint64_t *eveValues = (uint64_t *)malloc(
			eveCol->numEvents * sizeof(uint64_t));
	if(eveValues ==  NULL)
	{
		fprintf(stderr, "Error: failed to malloc while profiling.\n");
		exit(EXIT_FAILURE);
	}
	CUpti_EventGroupSet *setHead = 
		eveCol->groupSets->sets;
	EGSData_t egsData;
	int evesCollect = 0;
	int i;
	for(i = 0; i < numPass; i++)
	{
		int numEventsCurSet = 
			eveCol->numEventsEachSets[i];
		PROF_CALL(initEGSData(&egsData, &context, &device, 
					setHead+i, numEventsCurSet));
		// time consuming 0.01+s
		PROF_CALL(startColEveVals(&context, &egsData));

		RUNTIME_API_CALL(origCudaConfigureCall(
					keArgData.confArg.gridDim, 
					keArgData.confArg.blockDim, 
					keArgData.confArg.sharedMem, 
					keArgData.confArg.stream));
		cudaSetupArgItem *p = keArgData.cudaSetupArgHead;
		while(p != NULL)
		{
			RUNTIME_API_CALL(origCudaSetupArgument(
						p->arg.arg, p->arg.size, p->arg.offset));
			p = p->next;
		}
		(*cudaLaunch_h)(a1);

		if(i < numPass - 1)
		{
			PROF_CALL(restoreAllMem()); // TODO debug
		}

		// time consuming 0.01s
		PROF_CALL(endColEveVals(&context, &device, &egsData));

		int j;
		for(j = 0; j < egsData.numEvents; j++)
		{
			CUpti_EventID curEveID = egsData.eventIDs[j];
			int k;
			for(k = 0; k < eveCol->numEvents; k++)
			{
				if(eveCol->eventIDs[k] == curEveID)
				{
					eveValues[k] = egsData.eventValues[j];
					break;
				}
			}
		}

		evesCollect += egsData.numEvents;

		freeEGSData(&egsData);
	}

	kernelProfileData kerProfData;
	kerProfData.kerInfo = *kerInfo;
	kerProfData.numEvents = eveProf->numEvents;
	eventData *eveData = kerProfData.eventDatas;
	for(i = 0; i < eveProf->numEvents; i++)
	{
		eveData[i].eventID = eveProf->eventIDs[i];
		int j;
		for(j = 0; j < eveCol->numEvents; j++)
		{
			if(eveProf->eventIDs[i] == eveCol->eventIDs[j])
			{
				eveData[i].eventValue = eveValues[j];
				break;
			}
		}
	}
	unsigned long long duration = (kerInfo->end.tv_sec - 
			kerInfo->start.tv_sec) * 1000000000 + 
		kerInfo->end.tv_nsec - kerInfo->start.tv_nsec; 

	PROF_CALL(getMetricsValue(&device, duration, 
				eveCol->eventIDs, eveValues, 
				kerProfData.metricDatas, 
				&kerProfData.numMetrics));

	// occupancy
	{
		// achieved
		metricData *mtrDatas = kerProfData.metricDatas;
		CUpti_MetricID mtrID;
		CUPTI_CALL(cuptiMetricGetIdFromName(device, 
					"achieved_occupancy", &mtrID));
		for(i = 0; i < kerProfData.numMetrics; i++)
		{
			if(mtrDatas[i].metricID == mtrID)
			{
				kerProfData.occu.achieved = 
					mtrDatas[i].metricValue.
					metricValuePercent;
				/*
				   mtrDatas[i] = 
				   mtrDatas[--kerProfData.numMetrics];
				   */
				break;
			}
		}
		kerProfData.occu.achiLimiter = NONE;
		if(kerProfData.occu.achieved < 0.7)
		{
			CUpti_EventID eveID;
			CUPTI_CALL(cuptiEventGetIdFromName(device, 
						"threads_launched", &eveID));
			unsigned long numThreads = -1;

			for(i = 0; i < kerProfData.numEvents; i++)
			{
				if(eveData[i].eventID == eveID)
				{
					numThreads = eveData[i].eventValue;
					/*
					   eveData[i] = 
					   eveData[--kerProfData.numEvents];
					   */
					break;
				}
			}

			if(kerInfo->gridX * kerInfo->gridY * 
					kerInfo->gridZ < numThreads / 
					(kerInfo->blockX * kerInfo->blockY * 
					 kerInfo->blockZ))
			{
				kerProfData.occu.achiLimiter = GRID_SIZE;
			}
		}

		// theory
		int threadCount = kerInfo->blockX * 
			kerInfo->blockY * kerInfo->blockZ;
		int regCount = kerInfo->regPerThread;
		int sharedMemory = kerInfo->dynamicSharedMem + 
			kerInfo->staticSharedMem;
		cudaFuncCache cacheConfig;
		int sharedSize = getSMemSizeConfig();
		occupancyLimiter theoLimiter;
		double occu = theoryOccupancy(
				threadCount, regCount, sharedMemory, 
				sharedSize, &theoLimiter);
		kerProfData.occu.theory = occu;
		kerProfData.occu.theoLimiter = NONE;
		if(occu < 0.85)
		{
			kerProfData.occu.theoLimiter = theoLimiter;
		}
	}

	PROF_CALL(recordKerProfData(&kerProfData, 
				invalidEveNo, invalidMtrNo, &device));

	free(eveValues);
	freeKeArgs();
	PROF_CALL(reBackupAddMem());

	clock_gettime(CLOCK_REALTIME, &end);
	cuTimeTrace timeTrace = getBaseTraceNoInfo(
			"profileKernelLoop", start, end, PROF_OVERHEAD);
	recordTimeTrace(&timeTrace);

#endif

	return retval;
}

cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, 
		size_t sharedMem, cudaStream_t stream)
{
#ifdef CUDA_HOOK_PROF
	keArgData.confArg.gridDim = gridDim;
	keArgData.confArg.blockDim = blockDim;
	keArgData.confArg.sharedMem = sharedMem;
	keArgData.confArg.stream = stream;
	keArgData.cudaSetupArgHead = NULL;
#endif

	cudaError_t retval = origCudaConfigureCall(
			gridDim, blockDim, sharedMem, stream);

	return retval;
}

cudaError_t cudaSetupArgument(const void * arg, 
		size_t size, size_t offset) 
{
#ifdef CUDA_HOOK_PROF
	/*
	   struct timespec start, end;
	   clock_gettime(CLOCK_REALTIME, &start);
	   */

	cudaSetupArgItem *p = (cudaSetupArgItem *)
		malloc(sizeof(cudaSetupArgItem));
	if(p ==  NULL)
	{
		fprintf(stderr, "Error: failed to malloc while profiling.\n");
		exit(EXIT_FAILURE);
	}
	p->arg.arg = (void *)arg;
	p->arg.size = size;
	p->arg.offset = offset;
	p->next = NULL;
	if(keArgData.cudaSetupArgHead == NULL)
	{
		keArgData.cudaSetupArgHead = p;
	}
	else
	{
		cudaSetupArgItem *pTmp = keArgData.cudaSetupArgHead;
		while(pTmp->next != NULL)
		{
			pTmp = pTmp->next;
		}
		pTmp->next = p;
	}

	/*
	   clock_gettime(CLOCK_REALTIME, &end);
	   cuTimeTrace timeTrace = getBaseTraceNoInfo(
	   "cudaSetupArgument", start, end, PROF_OVERHEAD);
	   recordTimeTrace(&timeTrace);
	   */
#endif

	cudaError_t retval = origCudaSetupArgument(
			arg, size, offset);

	return retval;
}

cudaError_t cudaMalloc(void **devPtr, size_t size)
{
	typedef cudaError_t (*cudaMalloc_p) 
		(void **, size_t);
	static cudaMalloc_p cudaMalloc_h = 
		(cudaMalloc_p)getSymbol("cudaMalloc");
	cudaError_t retval =  
		(*cudaMalloc_h)(devPtr, size);

#ifdef CUDA_HOOK_PROF
	/*
	   struct timespec start, end;
	   clock_gettime(CLOCK_REALTIME, &start);
	   */

	PROF_CALL(addMemItem(GLOBAL_MEM, (const void **)devPtr, size));

	/*
	   clock_gettime(CLOCK_REALTIME, &end);
	   cuTimeTrace timeTrace = getBaseTraceNoInfo(
	   "cudaMalloc", start, end, PROF_OVERHEAD);
	   recordTimeTrace(&timeTrace);
	   */
#endif

	return retval;
}

// TODO: test
cudaError_t cudaMallocPitch(void **devPtr, 
		size_t *pitch, size_t width, size_t height)
{
	typedef cudaError_t (*cudaMallocPitch_p) 
		(void **, size_t *, size_t, size_t);
	static cudaMallocPitch_p cudaMallocPitch_h = 
		(cudaMallocPitch_p)getSymbol("cudaMallocPitch");
	cudaError_t retval =  
		(*cudaMallocPitch_h)(devPtr, pitch, width, height);

#ifdef CUDA_HOOK_PROF
	/*
	   struct timespec start, end;
	   clock_gettime(CLOCK_REALTIME, &start);
	   */

	PROF_CALL(addMemItem(PITCHED_MEM, 
				(const void **)devPtr, (*pitch) * height));

	/*
	   clock_gettime(CLOCK_REALTIME, &end);
	   cuTimeTrace timeTrace = getBaseTraceNoInfo(
	   "cudaMallocPitch", start, end, PROF_OVERHEAD);
	   recordTimeTrace(&timeTrace);
	   */
#endif

	return retval;
}

// TODO: test
cudaError_t cudaMalloc3D(cudaPitchedPtr *pitchedDevPtr, 
		cudaExtent extent)
{
	typedef cudaError_t (*cudaMalloc3D_p) 
		(cudaPitchedPtr *, cudaExtent);
	static cudaMalloc3D_p cudaMalloc3D_h = 
		(cudaMalloc3D_p)getSymbol("cudaMalloc3D");
	cudaError_t retval =  
		(*cudaMalloc3D_h)(pitchedDevPtr, extent);

#ifdef CUDA_HOOK_PROF
	/*
	   struct timespec start, end;
	   clock_gettime(CLOCK_REALTIME, &start);
	   */

	PROF_CALL(addMemItem(GLOBAL_3D_MEM, 
				(const void **)&(pitchedDevPtr->ptr), 
				(pitchedDevPtr->pitch) * 
				extent.height * extent.depth));

	/*
	   clock_gettime(CLOCK_REALTIME, &end);
	   cuTimeTrace timeTrace = getBaseTraceNoInfo(
	   "cudaMalloc3D", start, end, PROF_OVERHEAD);
	   recordTimeTrace(&timeTrace);
	   */
#endif

	return retval;
}

cudaError_t cudaMallocHost(void **ptr, size_t size)
{
	typedef cudaError_t (*cudaMallocHost_p) 
		(void **, size_t);
	static cudaMallocHost_p cudaMallocHost_h = 
		(cudaMallocHost_p)getSymbol("cudaMallocHost");
	cudaError_t retval =  
		(*cudaMallocHost_h)(ptr, size);

#ifdef CUDA_HOOK_PROF
	/*
	   struct timespec start, end;
	   clock_gettime(CLOCK_REALTIME, &start);
	   */

	PROF_CALL(addMemItem(PAGELOCKED_MEM, (const void **)ptr, size));

	/*
	   clock_gettime(CLOCK_REALTIME, &end);
	   cuTimeTrace timeTrace = getBaseTraceNoInfo(
	   "cudaMallocHost", start, end, PROF_OVERHEAD);
	   recordTimeTrace(&timeTrace);
	   */
#endif

	return retval;
}

cudaError_t cudaHostAlloc(void **pHost, 
		size_t size, unsigned int flags)
{
	typedef cudaError_t (*cudaHostAlloc_p) 
		(void **, size_t, unsigned int);
	static cudaHostAlloc_p cudaHostAlloc_h = 
		(cudaHostAlloc_p)getSymbol("cudaHostAlloc");
	cudaError_t retval =  
		(*cudaHostAlloc_h)(pHost, size, flags);

#ifdef CUDA_HOOK_PROF
	/*
	   struct timespec start, end;
	   clock_gettime(CLOCK_REALTIME, &start);
	   */

	if(flags & cudaHostRegisterMapped) // Mapped page-locked memory
	{
		PROF_CALL(addMemItem(MAPPED_MEM_HOST, 
					(const void **)pHost, size));
	}
	else
	{
		PROF_CALL(addMemItem(PAGELOCKED_MEM, 
					(const void **)pHost, size));
	}

	/*
	   clock_gettime(CLOCK_REALTIME, &end);
	   cuTimeTrace timeTrace = getBaseTraceNoInfo(
	   "cudaHostAlloc", start, end, PROF_OVERHEAD);
	   recordTimeTrace(&timeTrace);
	   */
#endif

	return retval;
}

// TODO: test
cudaError_t cudaHostRegister(void *ptr, 
		size_t size, unsigned int flags)
{
	typedef cudaError_t (*cudaHostRegister_p) 
		(void *, size_t, unsigned int);
	static cudaHostRegister_p cudaHostRegister_h = 
		(cudaHostRegister_p)getSymbol("cudaHostRegister");
	cudaError_t retval =  
		(*cudaHostRegister_h)(ptr, size, flags);

#ifdef CUDA_HOOK_PROF
	/*
	   struct timespec start, end;
	   clock_gettime(CLOCK_REALTIME, &start);
	   */

	if(flags & cudaHostRegisterMapped) // Mapped page-locked memory
	{
		PROF_CALL(addMemItem(MAPPED_MEM_HOST, 
					(const void **)&ptr, size));
	}
	else
	{
		PROF_CALL(addMemItem(PAGELOCKED_MEM, 
					(const void **)&ptr, size));
	}

	/*
	   clock_gettime(CLOCK_REALTIME, &end);
	   cuTimeTrace timeTrace = getBaseTraceNoInfo(
	   "cudaHostRegister", start, end, PROF_OVERHEAD);
	   recordTimeTrace(&timeTrace);
	   */
#endif

	return retval;
}

// TODO: test
cudaError_t cudaHostUnregister(void *ptr)
{
	typedef cudaError_t (*cudaHostUnregister_p)(void *);
	static cudaHostUnregister_p cudaHostUnregister_h = 
		(cudaHostUnregister_p)getSymbol("cudaHostUnregister");
	cudaError_t retval = (*cudaHostUnregister_h)(ptr);

#ifdef CUDA_HOOK_PROF
	/*
	   struct timespec start, end;
	   clock_gettime(CLOCK_REALTIME, &start);
	   */

	freeMemItem(ptr);

	/*
	   clock_gettime(CLOCK_REALTIME, &end);
	   cuTimeTrace timeTrace = getBaseTraceNoInfo(
	   "cudaHostUnregister", start, end, PROF_OVERHEAD);
	   recordTimeTrace(&timeTrace);
	   */
#endif

	return retval;
}

cudaError_t cudaHostGetDevicePointer(void **pDevice, 
		void *pHost, unsigned int flags)
{
	typedef cudaError_t (*cudaHostGetDP_p) 
		(void **, void *, unsigned int);
	static cudaHostGetDP_p cudaHostGetDP_h = 
		(cudaHostGetDP_p)getSymbol("cudaHostGetDevicePointer");
	cudaError_t retval =  
		(*cudaHostGetDP_h)(pDevice, pHost, flags);

#ifdef CUDA_HOOK_PROF
	/*
	   struct timespec start, end;
	   clock_gettime(CLOCK_REALTIME, &start);
	   */

	int result = addMappedMemItem(MAPPED_MEM_DEVICE, 
			(const void **)pDevice, (const void *)pHost);
	if(result == 1)
	{
		printf("ERROR in %s: %d\n", __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	/*
	   clock_gettime(CLOCK_REALTIME, &end);
	   cuTimeTrace timeTrace = getBaseTraceNoInfo(
	   "cudaHostGetDevicePointer", start, end, PROF_OVERHEAD);
	   recordTimeTrace(&timeTrace);
	   */
#endif

	return retval;
}

cudaError_t cudaFree(void *devPtr)
{
	typedef cudaError_t (*cudaFree_p) 
		(void *);
	static cudaFree_p cudaFree_h = 
		(cudaFree_p)getSymbol("cudaFree");
	cudaError_t retval =  
		(*cudaFree_h)(devPtr);

#ifdef CUDA_HOOK_PROF
	/*
	   struct timespec start, end;
	   clock_gettime(CLOCK_REALTIME, &start);
	   */

	freeMemItem(devPtr);

	/*
	   clock_gettime(CLOCK_REALTIME, &end);
	   cuTimeTrace timeTrace = getBaseTraceNoInfo(
	   "cudaFree", start, end, PROF_OVERHEAD);
	   recordTimeTrace(&timeTrace);
	   */
#endif

	return retval;
}

cudaError_t cudaFreeHost(void *devPtr)
{
	typedef cudaError_t (*cudaFreeHost_p) 
		(void *);
	static cudaFreeHost_p cudaFreeHost_h = 
		(cudaFreeHost_p)getSymbol("cudaFreeHost");
	cudaError_t retval =  
		(*cudaFreeHost_h)(devPtr);

#ifdef CUDA_HOOK_PROF
	/*
	   struct timespec start, end;
	   clock_gettime(CLOCK_REALTIME, &start);
	   */

	freeMemItem(devPtr);

	/*
	   clock_gettime(CLOCK_REALTIME, &end);
	   cuTimeTrace timeTrace = getBaseTraceNoInfo(
	   "cudaFreeHost", start, end, PROF_OVERHEAD);
	   recordTimeTrace(&timeTrace);
	   */
#endif

	return retval;
}

cudaError_t cudaMemcpy(void *dest, const void *src, 
		size_t count, enum cudaMemcpyKind kind)
{
	typedef cudaError_t (*cudaMemcpy_p) 
		(void *, const void *, size_t, enum cudaMemcpyKind);
	static cudaMemcpy_p cudaMemcpy_h = 
		(cudaMemcpy_p)getSymbol("cudaMemcpy");

	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	cudaError_t retval =  
		(*cudaMemcpy_h)(dest, src, count, kind);

	clock_gettime(CLOCK_REALTIME, &end);
	cuTimeTrace timeTrace = getCuMemcpyTrace(
			"cudaMemcpy", start, end, kind, count);
	recordTimeTrace(&timeTrace);

#ifdef CUDA_HOOK_PROF
	/*
	   clock_gettime(CLOCK_REALTIME, &start);
	   */

	updateBackupMem(dest);

	/*
	   clock_gettime(CLOCK_REALTIME, &end);
	   timeTrace = getBaseTraceNoInfo(
	   "cudaMemcpy", start, end, PROF_OVERHEAD);
	   recordTimeTrace(&timeTrace);
	   */
#endif

	return retval;
}

// TODO: test
cudaError_t cudaMemcpyAsync(void *dst, 
		const void *src, size_t count, 
		enum cudaMemcpyKind kind, cudaStream_t stream)
{
	typedef cudaError_t (*cudaMemcpyAsync_p)(void *, 
			const void *, size_t, enum cudaMemcpyKind, cudaStream_t);
	static cudaMemcpyAsync_p cudaMemcpyAsync_h = 
		(cudaMemcpyAsync_p)getSymbol("cudaMemcpyAsync");

	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	cudaError_t retval = (*cudaMemcpyAsync_h)(dst, 
			src, count, kind, stream);

	clock_gettime(CLOCK_REALTIME, &end);
	cuTimeTrace timeTrace = getCuMemcpyTrace(
			"cudaMemcpyAsync", start, end, kind, count);
	recordTimeTrace(&timeTrace);

#ifdef CUDA_HOOK_PROF
	/*
	   clock_gettime(CLOCK_REALTIME, &start);
	   */

	updateBackupMemFromSrc(dst, src, count, kind);

	/*
	   clock_gettime(CLOCK_REALTIME, &end);
	   timeTrace = getBaseTraceNoInfo(
	   "cudaMemcpyAsync", start, end, PROF_OVERHEAD);
	   recordTimeTrace(&timeTrace);
	   */
#endif

	return retval;
}

// TODO: test
cudaError_t cudaMemcpyPeer(void *dst, int dstDevice, 
		const void *src, int srcDevice, size_t count)
{
	typedef cudaError_t (*cudaMemcpyPeer_p)(void *, 
			int, const void *, int, size_t);
	static cudaMemcpyPeer_p cudaMemcpyPeer_h = 
		(cudaMemcpyPeer_p)getSymbol("cudaMemcpyPeer");

	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	cudaError_t retval = (*cudaMemcpyPeer_h)(
			dst, dstDevice, src, srcDevice, count);

	clock_gettime(CLOCK_REALTIME, &end);
	cuTimeTrace timeTrace = getCuMemcpyTrace(
			"cudaMemcpyPeer", start, end, 
			cudaMemcpyDeviceToDevice, count);
	recordTimeTrace(&timeTrace);

#ifdef CUDA_HOOK_PROF
	/*
	   clock_gettime(CLOCK_REALTIME, &start);
	   */

	updateBackupMem(dst);

	/*
	   clock_gettime(CLOCK_REALTIME, &end);
	   timeTrace = getBaseTraceNoInfo(
	   "cudaMemcpyPeer", start, end, PROF_OVERHEAD);
	   recordTimeTrace(&timeTrace);
	   */
#endif

	return retval;
}

// TODO: test
cudaError_t cudaMemcpyPeerAsync(void *dst, 
		int dstDevice, const void *src, int srcDevice, 
		size_t count, cudaStream_t stream)
{
	typedef cudaError_t (*cudaMemcpyPAsy_p)(void *, 
			int, const void *, int, size_t, cudaStream_t);
	static cudaMemcpyPAsy_p cudaMemcpyPAsy_h = 
		(cudaMemcpyPAsy_p)getSymbol("cudaMemcpyPeerAsync");

	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	cudaError_t retval = (*cudaMemcpyPAsy_h)(dst, 
			dstDevice, src, srcDevice, count, stream);

	clock_gettime(CLOCK_REALTIME, &end);
	cuTimeTrace timeTrace = getCuMemcpyTrace(
			"cudaMemcpyPeerAsync", start, end, 
			cudaMemcpyDeviceToDevice, count);
	recordTimeTrace(&timeTrace);

#ifdef CUDA_HOOK_PROF
	/*
	   clock_gettime(CLOCK_REALTIME, &start);
	   */

	updateBackupMemFromSrc(dst, src, count, 
			cudaMemcpyDeviceToHost);

	/*
	   clock_gettime(CLOCK_REALTIME, &end);
	   timeTrace = getBaseTraceNoInfo(
	   "cudaMemcpyPeerAsyn", start, end, PROF_OVERHEAD);
	   recordTimeTrace(&timeTrace);
	   */
#endif

	return retval;
}

// TODO: test
cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, 
		const void *src, size_t spitch, size_t width, 
		size_t height, enum cudaMemcpyKind kind)
{
	typedef cudaError_t (*cudaMemcpy2D_p) 
		(void *, size_t, const void *, size_t, 
		 size_t, size_t, enum cudaMemcpyKind);
	static cudaMemcpy2D_p cudaMemcpy2D_h = 
		(cudaMemcpy2D_p)getSymbol("cudaMemcpy2D");

	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	cudaError_t retval =  
		(*cudaMemcpy2D_h)(dst, dpitch, src, 
				spitch, width, height, kind);

	clock_gettime(CLOCK_REALTIME, &end);
	cuTimeTrace timeTrace = getCuMemcpyTrace(
			"cudaMemcpy2D", start, end, kind, height * width);
	recordTimeTrace(&timeTrace);

#ifdef CUDA_HOOK_PROF
	/*
	   clock_gettime(CLOCK_REALTIME, &start);
	   */

	updateBackupMem(dst);

	/*
	   clock_gettime(CLOCK_REALTIME, &end);
	   timeTrace = getBaseTraceNoInfo(
	   "cudaMemcpy2D", start, end, PROF_OVERHEAD);
	   recordTimeTrace(&timeTrace);
	   */
#endif

	return retval;
}

// TODO: test
cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t dst, 
		size_t wOffsetDst, size_t hOffsetDst, 
		cudaArray_const_t src, size_t wOffsetSrc,
		size_t hOffsetSrc, size_t width, 
		size_t height, cudaMemcpyKind kind)
{
	typedef cudaError_t (*cudaMemcpy2DA2A_p)(cudaArray_t, 
			size_t, size_t, cudaArray_const_t, size_t,
			size_t, size_t, size_t, cudaMemcpyKind);
	static cudaMemcpy2DA2A_p cudaMemcpy2DA2A_h = 
		(cudaMemcpy2DA2A_p)getSymbol(
				"cudaMemcpy2DArrayToArray");

	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	cudaError_t retval = (*cudaMemcpy2DA2A_h)(
			dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, 
			hOffsetSrc, width, height, kind);

	clock_gettime(CLOCK_REALTIME, &end);
	cuTimeTrace timeTrace = getCuMemcpyTrace(
			"cudaMemcpy2DArrayToArray", start, 
			end, kind, width * height);
	recordTimeTrace(&timeTrace);

	return retval;
}

// TODO: test
cudaError_t cudaMemcpy2DAsync(void *dst,
		size_t dpitch, const void *src, size_t spitch, 
		size_t width, size_t height, 
		enum cudaMemcpyKind kind, cudaStream_t stream)
{
	typedef cudaError_t (*cudaMemcpy2DAsync_p) 
		(void *, size_t, const void *, size_t, size_t, 
		 size_t, enum cudaMemcpyKind, cudaStream_t);
	static cudaMemcpy2DAsync_p cudaMemcpy2DAsync_h = 
		(cudaMemcpy2DAsync_p)getSymbol("cudaMemcpy2DAsync");

	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	cudaError_t retval = (*cudaMemcpy2DAsync_h)(dst, 
			dpitch, src, spitch, width, height, kind, stream);

	clock_gettime(CLOCK_REALTIME, &end);
	cuTimeTrace timeTrace = getCuMemcpyTrace(
			"cudaMemcpy2DAsync", start, end, 
			kind, width * height);
	recordTimeTrace(&timeTrace);

#ifdef CUDA_HOOK_PROF
	/*
	   clock_gettime(CLOCK_REALTIME, &start);
	   */

	updateBackupMem2DFromSrc(dst, dpitch, src, 
			spitch, width, height, kind);

	/*
	   clock_gettime(CLOCK_REALTIME, &end);
	   timeTrace = getBaseTraceNoInfo(
	   "cudaMemcpy2DAsync", start, end, PROF_OVERHEAD);
	   recordTimeTrace(&timeTrace);
	   */
#endif

	return retval;
}

cudaError_t cudaMemcpy2DFromArray(void * dst, 
		size_t dpitch, cudaArray_const_t src, 
		size_t wOffset, size_t hOffset, size_t width, 
		size_t height, enum cudaMemcpyKind kind) 
{
	typedef cudaError_t (*cudaMemcpy2DFromArray_p) 
		(void *, size_t, cudaArray_const_t, 
		 size_t, size_t, size_t, size_t, enum cudaMemcpyKind);
	static cudaMemcpy2DFromArray_p cudaMemcpy2DFromArray_h = 
		(cudaMemcpy2DFromArray_p)getSymbol(
				"cudaMemcpy2DFromArray");

	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	cudaError_t retval = (*cudaMemcpy2DFromArray_h)(
			dst, dpitch, src, wOffset, hOffset, 
			width, height, kind);

	clock_gettime(CLOCK_REALTIME, &end);
	cuTimeTrace timeTrace = getCuMemcpyTrace(
			"cudaMemcpy2DFromArray", start, end, 
			kind, width * height);
	recordTimeTrace(&timeTrace);

	return retval;
}

cudaError_t cudaMemcpy2DFromArrayAsync(void * dst, 
		size_t dpitch, cudaArray_const_t src, 
		size_t wOffset, size_t hOffset, size_t width, 
		size_t height, enum cudaMemcpyKind kind, 
		cudaStream_t stream) 
{
	typedef cudaError_t (*cudaMemcpy2DFroArrAsy_p) 
		(void *, size_t, cudaArray_const_t, 
		 size_t, size_t, size_t, size_t, 
		 enum cudaMemcpyKind, cudaStream_t);
	static cudaMemcpy2DFroArrAsy_p cudaMemcpy2DFroArrAsy_h = 
		(cudaMemcpy2DFroArrAsy_p)getSymbol(
				"cudaMemcpy2DFromArrayAsync");

	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	cudaError_t retval = (*cudaMemcpy2DFroArrAsy_h)(
			dst, dpitch, src, wOffset, hOffset, 
			width, height, kind, stream);

	clock_gettime(CLOCK_REALTIME, &end);
	cuTimeTrace timeTrace = getCuMemcpyTrace(
			"cudaMemcpy2DFromArrayAsync", start, 
			end, kind, width * height);
	recordTimeTrace(&timeTrace);

	return retval;
}

cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, 
		size_t wOffset, size_t hOffset, const void *src,
		size_t spitch, size_t width, size_t height, 
		enum cudaMemcpyKind kind) 
{
	typedef cudaError_t (*cudaMemcpy2DToArray_p)
		(cudaArray_t, size_t, size_t, const void *, 
		 size_t, size_t, size_t, enum cudaMemcpyKind); 
	static cudaMemcpy2DToArray_p cudaMemcpy2DToArray_h = 
		(cudaMemcpy2DToArray_p)getSymbol(
				"cudaMemcpy2DToArray");

	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	cudaError_t retval = (*cudaMemcpy2DToArray_h)(
			dst, wOffset, hOffset, src, spitch, 
			width, height, kind);

	clock_gettime(CLOCK_REALTIME, &end);
	cuTimeTrace timeTrace = getCuMemcpyTrace(
			"cudaMemcpy2DToArray", start, end, 
			kind, width * height);
	recordTimeTrace(&timeTrace);

	return retval;
}

cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, 
		size_t wOffset, size_t hOffset, const void *src,
		size_t spitch, size_t width, size_t height, 
		enum cudaMemcpyKind kind, cudaStream_t stream) 
{
	typedef cudaError_t (*cudaMemcpy2DToArrAsy_p)
		(cudaArray_t, size_t, size_t, const void *, size_t, 
		 size_t, size_t, enum cudaMemcpyKind, cudaStream_t); 
	static cudaMemcpy2DToArrAsy_p cudaMemcpy2DToArrAsy_h = 
		(cudaMemcpy2DToArrAsy_p)getSymbol(
				"cudaMemcpy2DToArrayAsync");

	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	cudaError_t retval = (*cudaMemcpy2DToArrAsy_h)(
			dst, wOffset, hOffset, src, spitch, 
			width, height, kind, stream); 

	clock_gettime(CLOCK_REALTIME, &end);
	cuTimeTrace timeTrace = getCuMemcpyTrace(
			"cudaMemcpy2DToArrayAsync", start, 
			end, kind, width * height);
	recordTimeTrace(&timeTrace);

	return retval;
}

// TODO: test
cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms *p)
{
	typedef cudaError_t (*cudaMemcpy3D_p)(
			const cudaMemcpy3DParms *);
	static cudaMemcpy3D_p cudaMemcpy3D_h = 
		(cudaMemcpy3D_p)getSymbol("cudaMemcpy3D");

	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	cudaError_t retval = (*cudaMemcpy3D_h)(p);

	clock_gettime(CLOCK_REALTIME, &end);
	cuTimeTrace timeTrace = getCuMemcpyTrace(
			"cudaMemcpy3D", start, end, p->kind, 
			p->extent.width*p->extent.height*p->extent.depth);
	recordTimeTrace(&timeTrace);

#ifdef CUDA_HOOK_PROF
	/*
	   clock_gettime(CLOCK_REALTIME, &start);
	   */

	if(p->dstArray != NULL)
	{
		updateBackupMem(p->dstPtr.ptr);
	}

	/*
	   clock_gettime(CLOCK_REALTIME, &end);
	   timeTrace = getBaseTraceNoInfo(
	   "cudaMemcpy3D", start, end, PROF_OVERHEAD);
	   recordTimeTrace(&timeTrace);
	   */
#endif

	return retval;
}

// TODO: test
cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms *p, 
		cudaStream_t stream)
{
	typedef cudaError_t (*cudaMemcpy3DAsync_p)
		(const cudaMemcpy3DParms *, cudaStream_t);
	static cudaMemcpy3DAsync_p cudaMemcpy3DAsync_h = 
		(cudaMemcpy3DAsync_p)getSymbol("cudaMemcpy3DAsync");

	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	cudaError_t retval = (*cudaMemcpy3DAsync_h)(p, stream);

	clock_gettime(CLOCK_REALTIME, &end);
	cuTimeTrace timeTrace = getCuMemcpyTrace(
			"cudaMemcpy3DAsync", start, end, p->kind, 
			p->extent.width*p->extent.height*p->extent.depth);
	recordTimeTrace(&timeTrace);

#ifdef CUDA_HOOK_PROF
	/*
	   clock_gettime(CLOCK_REALTIME, &start);
	   */

	updateBackupMem3DFromSrc(p);

	/*
	   clock_gettime(CLOCK_REALTIME, &end);
	   timeTrace = getBaseTraceNoInfo(
	   "cudaMemcpy3DAsync", start, end, PROF_OVERHEAD);
	   recordTimeTrace(&timeTrace);
	   */
#endif

	return retval;
}

// TODO: test
cudaError_t cudaMemcpy3DPeer(const cudaMemcpy3DPeerParms *p)
{
	typedef cudaError_t (*cudaMemcpy3DPeer_p)
		(const cudaMemcpy3DPeerParms *);
	static cudaMemcpy3DPeer_p cudaMemcpy3DPeer_h = 
		(cudaMemcpy3DPeer_p)getSymbol("cudaMemcpy3DPeer");

	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	cudaError_t retval = (*cudaMemcpy3DPeer_h)(p);

	clock_gettime(CLOCK_REALTIME, &end);
	cuTimeTrace timeTrace = getCuMemcpyTrace(
			"cudaMemcpy3DPeer", start, end, 
			cudaMemcpyDeviceToDevice, 
			p->extent.width*p->extent.height*p->extent.depth);
	recordTimeTrace(&timeTrace);

#ifdef CUDA_HOOK_PROF
	/*
	   clock_gettime(CLOCK_REALTIME, &start);
	   */

	if(p->dstArray != NULL)
	{
		updateBackupMem(p->dstPtr.ptr);
	}

	/*
	   clock_gettime(CLOCK_REALTIME, &end);
	   timeTrace = getBaseTraceNoInfo(
	   "cudaMemcpy3DPeer", start, end, PROF_OVERHEAD);
	   recordTimeTrace(&timeTrace);
	   */
#endif

	return retval;
}

// TODO: test
cudaError_t cudaMemcpy3DPeerAsync(
		const cudaMemcpy3DPeerParms *p, cudaStream_t stream)
{
	typedef cudaError_t (*cudaMemcpy3DPAsy_p)
		(const cudaMemcpy3DPeerParms *, cudaStream_t);
	static cudaMemcpy3DPAsy_p cudaMemcpy3DPAsy_h = 
		(cudaMemcpy3DPAsy_p)getSymbol("cudaMemcpy3DPeerAsync");

	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	cudaError_t retval = (*cudaMemcpy3DPAsy_h)(p, stream);

	clock_gettime(CLOCK_REALTIME, &end);
	cuTimeTrace timeTrace = getCuMemcpyTrace(
			"cudaMemcpy3DPeerAsync", start, end, 
			cudaMemcpyDeviceToDevice, 
			p->extent.width*p->extent.height*p->extent.depth);
	recordTimeTrace(&timeTrace);

#ifdef CUDA_HOOK_PROF
	/*
	   clock_gettime(CLOCK_REALTIME, &start);
	   */

	updateBackupMem3DFromSrc(p);

	/*
	   clock_gettime(CLOCK_REALTIME, &end);
	   timeTrace = getBaseTraceNoInfo(
	   "cudaMemcpy3DPeerAsync", start, end, PROF_OVERHEAD);
	   recordTimeTrace(&timeTrace);
	   */
#endif

	return retval;
}

cudaError_t cudaMemcpyArrayToArray(cudaArray_t dst, 
		size_t wOffsetDst, size_t hOffsetDst, 
		cudaArray_const_t src, size_t wOffsetSrc, 
		size_t hOffsetSrc, size_t count, cudaMemcpyKind kind)
{
	typedef cudaError_t (*cudaMemcpyA2A_p)
		(cudaArray_t, size_t, size_t, cudaArray_const_t, 
		 size_t, size_t, size_t, cudaMemcpyKind);
	static cudaMemcpyA2A_p cudaMemcpyA2A_h = 
		(cudaMemcpyA2A_p)getSymbol("cudaMemcpyArrayToArray");

	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	cudaError_t retval = (*cudaMemcpyA2A_h)(
			dst, wOffsetDst, hOffsetDst, src, 
			wOffsetSrc, hOffsetSrc, count, kind);

	clock_gettime(CLOCK_REALTIME, &end);
	cuTimeTrace timeTrace = getCuMemcpyTrace(
			"cudaMemcpyArrayToArray", start, end, kind, count);
	recordTimeTrace(&timeTrace);

	return retval;
}

cudaError_t cudaMemcpyFromArray(void *dst, 
		cudaArray_const_t src, size_t wOffset, size_t hOffset, 
		size_t count, cudaMemcpyKind kind)
{
	typedef cudaError_t (*cudaMemcpyFromArr_p)
		(void *, cudaArray_const_t, size_t, size_t, 
		 size_t, cudaMemcpyKind);
	static cudaMemcpyFromArr_p cudaMemcpyFromArr_h = 
		(cudaMemcpyFromArr_p)getSymbol("cudaMemcpyFromArray");

	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	cudaError_t retval = (*cudaMemcpyFromArr_h)(
			dst, src, wOffset, hOffset, count, kind);

	clock_gettime(CLOCK_REALTIME, &end);
	cuTimeTrace timeTrace = getCuMemcpyTrace(
			"cudaMemcpyFromArray", start, end, kind, count);
	recordTimeTrace(&timeTrace);

	return retval;
}

cudaError_t cudaMemcpyFromArrayAsync(void *dst, 
		cudaArray_const_t src, size_t wOffset, size_t hOffset, 
		size_t count, cudaMemcpyKind kind, cudaStream_t stream)
{
	typedef cudaError_t (*cudaMemcpyFroArrAsy_p)
		(void *, cudaArray_const_t, size_t, size_t, 
		 size_t, cudaMemcpyKind, cudaStream_t);
	static cudaMemcpyFroArrAsy_p cudaMemcpyFroArrAsy_h = 
		(cudaMemcpyFroArrAsy_p)getSymbol(
				"cudaMemcpyFromArrayAsync");

	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	cudaError_t retval = (*cudaMemcpyFroArrAsy_h)(dst, 
			src, wOffset, hOffset, count, kind, stream);

	clock_gettime(CLOCK_REALTIME, &end);
	cuTimeTrace timeTrace = getCuMemcpyTrace(
			"cudaMemcpyFromArrayAsync", start, end, kind, count);
	recordTimeTrace(&timeTrace);

	return retval;
}

cudaError_t cudaMemcpyToArray(cudaArray_t dst, 
		size_t wOffset, size_t hOffset, const void *src,
		size_t count, cudaMemcpyKind kind)
{
	typedef cudaError_t (*cudaMemcpyToArr_p)
		(cudaArray_t, size_t, size_t, const void *,
		 size_t, cudaMemcpyKind);
	static cudaMemcpyToArr_p cudaMemcpyToArr_h = 
		(cudaMemcpyToArr_p)getSymbol("cudaMemcpyToArray");

	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	cudaError_t retval = (*cudaMemcpyToArr_h)(
			dst, wOffset, hOffset, src, count, kind);

	clock_gettime(CLOCK_REALTIME, &end);
	cuTimeTrace timeTrace = getCuMemcpyTrace(
			"cudaMemcpyToArray", start, end, kind, count);
	recordTimeTrace(&timeTrace);

	return retval;
}

cudaError_t cudaMemcpyToArrayAsync(
		cudaArray_t dst, size_t wOffset, size_t hOffset, 
		const void *src, size_t count, cudaMemcpyKind kind, 
		cudaStream_t stream)
{
	typedef cudaError_t (*cudaMemcpyToArrAsy_p)
		(cudaArray_t, size_t, size_t, const void *,
		 size_t, cudaMemcpyKind, cudaStream_t);
	static cudaMemcpyToArrAsy_p cudaMemcpyToArrAsy_h = 
		(cudaMemcpyToArrAsy_p)getSymbol(
				"cudaMemcpyToArrayAsync");

	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	cudaError_t retval = (*cudaMemcpyToArrAsy_h)(dst, 
			wOffset, hOffset, src, count, kind, stream);

	clock_gettime(CLOCK_REALTIME, &end);
	cuTimeTrace timeTrace = getCuMemcpyTrace(
			"cudaMemcpyToArrayAsync", start, end, kind, count);
	recordTimeTrace(&timeTrace);

	return retval;
}


// TODO: test
cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol,
		size_t count, size_t offset, cudaMemcpyKind kind)
{
	typedef cudaError_t (*cudaMemcpyFS_p)(void *, 
			const void *, size_t, size_t, cudaMemcpyKind);
	static cudaMemcpyFS_p cudaMemcpyFS_h = 
		(cudaMemcpyFS_p)getSymbol("cudaMemcpyFromSymbol");

	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	cudaError_t retval = (*cudaMemcpyFS_h)(
			dst, symbol, count, offset, kind);

	clock_gettime(CLOCK_REALTIME, &end);
	cuTimeTrace timeTrace = getCuMemcpyTrace(
			"cudaMemcpyFromSymbol", start, end, kind, count);
	recordTimeTrace(&timeTrace);

#ifdef CUDA_HOOK_PROF
	/*
	   clock_gettime(CLOCK_REALTIME, &start);
	   */

	updateBackupMem(dst);

	/*
	   clock_gettime(CLOCK_REALTIME, &end);
	   timeTrace = getBaseTraceNoInfo(
	   "cudaMemcpyFromSynbol", start, end, PROF_OVERHEAD);
	   recordTimeTrace(&timeTrace);
	   */
#endif

	return retval;
}

// TODO: test
cudaError_t cudaMemcpyFromSymbolAsync(void *dst, 
		const void *symbol, size_t count, size_t offset, 
		cudaMemcpyKind kind, cudaStream_t stream)
{
	typedef cudaError_t (*cuMemcpyFSA_p)(void *, 
			const void *, size_t, size_t, 
			cudaMemcpyKind, cudaStream_t);
	static cuMemcpyFSA_p cuMemcpyFSA_h = 
		(cuMemcpyFSA_p)getSymbol("cudaMemcpyFromSymbolAsync");

	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	cudaError_t retval = (*cuMemcpyFSA_h)(dst, symbol, 
			count, offset, kind, stream);

	clock_gettime(CLOCK_REALTIME, &end);
	cuTimeTrace timeTrace = getCuMemcpyTrace(
			"cudaMemcpyFromSymbolAsync", start, 
			end, kind, count);
	recordTimeTrace(&timeTrace);

#ifdef CUDA_HOOK_PROF
	/*
	   clock_gettime(CLOCK_REALTIME, &start);
	   */

	updateBackupMemFromSymbol(dst, symbol, count, 
			offset, kind);

	/*
	   clock_gettime(CLOCK_REALTIME, &end);
	   timeTrace = getBaseTraceNoInfo(
	   "cudaMemcpyFromSymbolAsync", start, end, PROF_OVERHEAD);
	   recordTimeTrace(&timeTrace);
	   */
#endif

	return retval;
}

// TODO: test
cudaError_t cudaMemcpyToSymbol(const void *symbol, 
		const void *src, size_t count, size_t offset,
		cudaMemcpyKind kind)
{
	typedef cudaError_t (*cudaMemcpyTS_p)(const void *, 
			const void *, size_t, size_t, cudaMemcpyKind);
	static cudaMemcpyTS_p cudaMemcpyTS_h = 
		(cudaMemcpyTS_p)getSymbol("cudaMemcpyToSymbol");

	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	cudaError_t retval = (*cudaMemcpyTS_h)(
			symbol, src, count, offset, kind);

	clock_gettime(CLOCK_REALTIME, &end);
	cuTimeTrace timeTrace = getCuMemcpyTrace(
			"cudaMemcpyToSymbol", start, end, kind, count);
	recordTimeTrace(&timeTrace);

#ifdef CUDA_HOOK_PROF
	/*
	   clock_gettime(CLOCK_REALTIME, &start);
	   */

	updateBackupMem(symbol);

	/*
	   clock_gettime(CLOCK_REALTIME, &end);
	   timeTrace = getBaseTraceNoInfo(
	   "cudaMemcpyToSymbol", start, end, PROF_OVERHEAD);
	   recordTimeTrace(&timeTrace);
	   */
#endif

	return retval;
}

// TODO: test
cudaError_t cudaMemcpyToSymbolAsync(const void *symbol, 
		const void *src, size_t count, size_t offset,
		cudaMemcpyKind kind, cudaStream_t stream)
{
	typedef cudaError_t (*cuMemcpyTSA_p)(const void *, 
			const void *, size_t, size_t, 
			cudaMemcpyKind, cudaStream_t);
	static cuMemcpyTSA_p cuMemcpyTSA_h = 
		(cuMemcpyTSA_p)getSymbol("cudaMemcpyToSymbolAsync");

	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	cudaError_t retval = (*cuMemcpyTSA_h)(
			symbol, src, count, offset, kind, stream);

	clock_gettime(CLOCK_REALTIME, &end);
	cuTimeTrace timeTrace = getCuMemcpyTrace(
			"cudaMemcpyToSymbolAsync", start, 
			end, kind, count);
	recordTimeTrace(&timeTrace);

#ifdef CUDA_HOOK_PROF
	/*
	   clock_gettime(CLOCK_REALTIME, &start);
	   */

	updateBackupMemToSymbol(symbol, src, 
			count, offset, kind);

	/*
	   clock_gettime(CLOCK_REALTIME, &end);
	   timeTrace = getBaseTraceNoInfo(
	   "cudaMemcpyToSymbolAsync", start, end, PROF_OVERHEAD);
	   recordTimeTrace(&timeTrace);
	   */
#endif

	return retval;
}

// TODO: test
cudaError_t cudaMemset(void *devPtr, int value, size_t count)
{
	typedef cudaError_t (*cudaMemset_p) 
		(void *, int, size_t);
	static cudaMemset_p cudaMemset_h = 
		(cudaMemset_p)getSymbol("cudaMemset");
	cudaError_t retval =  
		(*cudaMemset_h)(devPtr, value, count);

#ifdef CUDA_HOOK_PROF
	/*
	   struct timespec start, end;
	   clock_gettime(CLOCK_REALTIME, &start);
	   */

	updateBackupMem(devPtr);

	/*
	   clock_gettime(CLOCK_REALTIME, &end);
	   cuTimeTrace timeTrace = getBaseTraceNoInfo(
	   "cudaMemset", start, end, PROF_OVERHEAD);
	   recordTimeTrace(&timeTrace);
	   */
#endif

	return retval;
}

// TODO: test
cudaError_t cudaMemsetAsync(void *devPtr, int value, 
		size_t count, cudaStream_t stream)
{
	typedef cudaError_t (*cudaMemsetAsync_p) 
		(void *, int, size_t, cudaStream_t);
	static cudaMemsetAsync_p cudaMemsetAsync_h = 
		(cudaMemsetAsync_p)getSymbol("cudaMemsetAsync");
	cudaError_t retval =  
		(*cudaMemsetAsync_h)(devPtr, value, count, stream);

#ifdef CUDA_HOOK_PROF
	/*
	   struct timespec start, end;
	   clock_gettime(CLOCK_REALTIME, &start);
	   */

	memsetBackupMem((const void *)devPtr, value, count);

	/*
	   clock_gettime(CLOCK_REALTIME, &end);
	   cuTimeTrace timeTrace = getBaseTraceNoInfo(
	   "cudaMemsetAsync", start, end, PROF_OVERHEAD);
	   recordTimeTrace(&timeTrace);
	   */
#endif

	return retval;
}

// TODO: test
cudaError_t cudaMemset2D(void *devPtr, size_t pitch, 
		int value, size_t width, size_t height)
{
	cudaError_t retval = origCudaMemset2D(devPtr, 
			pitch, value, width, height);

#ifdef CUDA_HOOK_PROF
	/*
	   struct timespec start, end;
	   clock_gettime(CLOCK_REALTIME, &start);
	   */

	updateBackupMem(devPtr);

	/*
	   clock_gettime(CLOCK_REALTIME, &end);
	   cuTimeTrace timeTrace = getBaseTraceNoInfo(
	   "cudaMemset2D", start, end, PROF_OVERHEAD);
	   recordTimeTrace(&timeTrace);
	   */
#endif

	return retval;
}

// TODO: test
cudaError_t cudaMemset2DAsync(void *devPtr, 
		size_t pitch, int value, size_t width, 
		size_t height, cudaStream_t stream)
{
	typedef cudaError_t (*cudaMemset2DAsync_p)
		(void *, size_t, int, size_t, size_t, cudaStream_t);
	static cudaMemset2DAsync_p cudaMemset2DAsync_h = 
		(cudaMemset2DAsync_p)getSymbol("cudaMemset2DAsync");
	cudaError_t retval = (*cudaMemset2DAsync_h)
		(devPtr, pitch, value, width, height, stream);

#ifdef CUDA_HOOK_PROF
	/*
	   struct timespec start, end;
	   clock_gettime(CLOCK_REALTIME, &start);
	   */

	memsetBackupMem2D((const void *)devPtr, pitch, 
			value, width, height);

	/*
	   clock_gettime(CLOCK_REALTIME, &end);
	   cuTimeTrace timeTrace = getBaseTraceNoInfo(
	   "cudaMemset2DAsync", start, end, PROF_OVERHEAD);
	   recordTimeTrace(&timeTrace);
	   */
#endif

	return retval;
}

// TODO: test
cudaError_t cudaMemset3D(cudaPitchedPtr pitchedDevPtr, 
		int value, cudaExtent extent)
{
	cudaError_t retval = origCudaMemset3D(
			pitchedDevPtr, value, extent);

#ifdef CUDA_HOOK_PROF
	/*
	   struct timespec start, end;
	   clock_gettime(CLOCK_REALTIME, &start);
	   */

	updateBackupMem(pitchedDevPtr.ptr);

	/*
	   clock_gettime(CLOCK_REALTIME, &end);
	   cuTimeTrace timeTrace = getBaseTraceNoInfo(
	   "cudaMemset3D", start, end, PROF_OVERHEAD);
	   recordTimeTrace(&timeTrace);
	   */
#endif

	return retval;
}

// TODO: test
cudaError_t cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, 
		int value, cudaExtent extent, cudaStream_t stream)
{
	typedef cudaError_t (*cudaMemset3DAsync_p)
		(cudaPitchedPtr, int, cudaExtent, cudaStream_t);
	static cudaMemset3DAsync_p cudaMemset3DAsync_h = 
		(cudaMemset3DAsync_p)getSymbol("cudaMemset3DAsync");
	cudaError_t retval = (*cudaMemset3DAsync_h)(
			pitchedDevPtr, value, extent, stream);

#ifdef CUDA_HOOK_PROF
	/*
	   struct timespec start, end;
	   clock_gettime(CLOCK_REALTIME, &start);
	   */

	memsetBackupMem3D(pitchedDevPtr, value, extent);

	/*
	   clock_gettime(CLOCK_REALTIME, &end);
	   cuTimeTrace timeTrace = getBaseTraceNoInfo(
	   "cudaMemset3DAsync", start, end, PROF_OVERHEAD);
	   recordTimeTrace(&timeTrace);
	   */
#endif

	return retval;
}


