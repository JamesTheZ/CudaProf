#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define COMMAND_MAX_LEN 1024
#define METRIC_NAME_MAX_LEN 64
#define EVENT_NAME_MAX_LEN 64
#define LINE_MAX_LEN 4096
#define SHORT_DESC_MAX_LEN 64
#define LINE_FIELD_MAX_NUM 100
#define KERNEL_NAME_MAX_LEN 256
#define MTR_MAX_NUM 128

typedef struct commandData_t
{
	char command[COMMAND_MAX_LEN];
	int len;
}commandData;

typedef struct csv_t
{
	char line[LINE_MAX_LEN];
	char *fields[LINE_FIELD_MAX_NUM];
	int numField;
}csv;

typedef struct kerInfoClass_t
{
	char kernelName[KERNEL_NAME_MAX_LEN];
	long double mtrItems[MTR_MAX_NUM];
	int mtrItemTypes[MTR_MAX_NUM];	// 0 means (ll)d, 1 means f, 2 means null
	size_t numMtrItems;		// number of mtr items
	size_t numClassItems;	// number of items in this class
	// struct timespec duration;
	struct kerInfoClass_t *next;
}kerInfoClass;

typedef struct kernelTrace_t
{
	char kernelName[KERNEL_NAME_MAX_LEN];
	size_t numExeTimes;
	struct timespec duration;
	struct kernelTrace_t *next;
}kernelTrace;

static int plot = 0;
static commandData command;

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

static const char *kerClassRecordFname = 
"profile_kernel_data_result_summary.csv";
static const char *profTraceSumFname = 
"profile_trace_summary.csv";

static const char *eventNamesFilename = 
"eventNames.ini";
static const char *metricNamesFilename = 
"metricNames.ini";

static kerInfoClass *kerClassHead = NULL;

static void printUsage()
{
	printf("usage: cuptiQuery targetexecfile [args]\n");
	printf("  -help:\t\tdisplay help message\n");
	printf("  -plot:\t\tplot profiling result or not, "\
			"1 plot and 0 otherwise\n");
	printf("Note: in default plot is 0\n");
}

static void parseCommandLineArgs(int argc, char *argv[])
{
	char const *preload = "LD_PRELOAD=\"./cudart_wrapper.so\" " ;
	memset(&command, 0, sizeof(command));
	memcpy(command.command, preload, strlen(preload));
	command.len = strlen(preload);
	int isFirstArg = 1;
	for(int k=1; k<argc; k++) 
	{
		if ((k+1 < argc) && strcasecmp(argv[k], "-plot") == 0) 
		{
			plot = atoi(argv[k+1]);
			// setOptionsFlag(FLAG_DEVICE_ID);
			k++;
		}
		else if ((strcasecmp(argv[k], "--help") == 0) ||
				(strcasecmp(argv[k], "-help") == 0) ||
				(strcasecmp(argv[k], "-h") == 0)) 
		{
			printUsage();
			exit(-1);
		}
		else 
		{
			if(isFirstArg)
			{
				isFirstArg = 0;
				if(argv[k][0] != '/')
				{
					memcpy(command.command + command.len, "./", 2);
					command.len += 2;
				}
			}
			int len = strlen(argv[k]);
			memcpy(command.command + command.len, argv[k], len);
			command.len += len;
			command.command[command.len++] = ' ';
		}
	}
}

static int getMetricNamesStr(char *metricNames)
{
	if(metricNames == NULL)
	{
		return 1;
	}
	FILE *fp = fopen(metricNamesFilename, "r");
	char buf[METRIC_NAME_MAX_LEN];
	int len = 0;
	while(fgets(buf, METRIC_NAME_MAX_LEN, fp) != NULL)
	{
		int bufLen = strlen(buf);
		while(buf[bufLen-1] == '\n')
		{
			bufLen--;
		}
		buf[bufLen] = ',';
		buf[++bufLen] = '\0';
		strncpy(metricNames + len, buf, bufLen);
		len += bufLen;
	}
	metricNames[len-1] = '\0';
	fclose(fp);

	return 0;
}

static int getEventNamesStr(char *eventNames)
{
	if(eventNames == NULL)
	{
		return 1;
	}
	FILE *fp = fopen(eventNamesFilename, "r");
	char buf[EVENT_NAME_MAX_LEN];
	int len = 0;
	while(fgets(buf, EVENT_NAME_MAX_LEN, fp) != NULL)
	{
		int bufLen = strlen(buf);
		while(buf[bufLen-1] == '\n')
		{
			bufLen--;
		}
		buf[bufLen] = ',';
		buf[++bufLen] = '\0';
		strncpy(eventNames + len, buf, bufLen);
		len += bufLen;
	}
	eventNames[len-1] = '\0';
	fclose(fp);

	return 0;
}

static void initProfFiles()
{
	const char *kerProfDataStr = 
		"name,device,stream,"\
		"grid,gridX,gridY,gridZ,blockX,blockY,"\
		"blockZ,start_sec,start_nsec,end_sec,"\
		"end_nsec,dynamicSharedMem,staticSharedMem,"\
		"localMemPerThread,localMemTotal,regPerThread,"\
		"cacheConfigReq,cacheConfigUsed,"\
		"sharedMemConfigUsed,theory_limiter,"\
		"achieved_limiter,theory_occupancy,%s,%s";
	const char *kerTraceStr = "name,start_sec,"\
							   "start_nsec,end_sec,end_nsec";
	const char *cpyTraceStr = 
		"func_name,cpy_type,"\
		"start_sec,start_nsec,end_sec,end_nsec,count";
	const char *overheadStr = "desc,start_sec,start_nsec,"\
							   "end_sec,end_nsec";

	char metricNames[2048], eventNames[3072];
	getMetricNamesStr(metricNames);
	getEventNamesStr(eventNames);
	FILE *kerDataFile = 
		fopen(kerProfDataRecordFname, "w");
	fprintf(kerDataFile, kerProfDataStr, 
			metricNames, eventNames);
	fclose(kerDataFile);

	FILE *kerTraceFile = 
		fopen(kernelTraceRecordFname, "w");
	fprintf(kerTraceFile, kerTraceStr);
	fclose(kerTraceFile);

	FILE *cpyTraceFile = 
		fopen(memcpyTraceRecordFname , "w");
	fprintf(cpyTraceFile, cpyTraceStr);
	fclose(cpyTraceFile);

	FILE *overheadFile = 
		fopen(profOverheadRecordFname , "w");
	fprintf(overheadFile, overheadStr);
	fclose(overheadFile);

	FILE *errLogFile = fopen(errorLogFname, "w");
	fprintf(errLogFile, "time,line,errorMsg");
	fclose(errLogFile);
}

static void diff(timespec *diff, 
		const timespec *start, const timespec *end)
{
	if(end->tv_nsec - start->tv_nsec < 0)
	{
		diff->tv_sec = end->tv_sec - start->tv_sec - 1;
		diff->tv_nsec = 1000000000 + 
			end->tv_nsec - start->tv_nsec;
	}
	else
	{
		diff->tv_sec = end->tv_sec - start->tv_sec;
		diff->tv_nsec = end->tv_nsec - start->tv_nsec;
	}
}

void printTimespec(const timespec *ts)
{
	printf("%ld.%09ld sec\n", ts->tv_sec, ts->tv_nsec);
}

static void sumKernelTrace(
		FILE *wbFp, struct timespec *kerTime)
{
	kernelTrace *kerTrcHead = NULL;

	FILE *rdFp = fopen(kernelTraceRecordFname, "r");
	char buf[LINE_MAX_LEN];
	int headLines = 1;
	while(fgets(buf, LINE_MAX_LEN, rdFp) != NULL)
	{
		if(headLines)
		{
			headLines--;
			continue;
		}
		char curKernelName[KERNEL_NAME_MAX_LEN];
		long start_sec, start_nsec;
		long end_sec, end_nsec;
		long dura_sec, dura_nsec;

		sscanf(buf, "\"%[^\"]\",%llu,%llu,%llu,%llu",
				curKernelName, &start_sec, &start_nsec, 
				&end_sec, &end_nsec);
		dura_nsec = end_nsec - start_nsec;
		dura_sec = end_sec - start_sec;
		if(dura_nsec < 0)
		{
			dura_nsec += 1000000000;
			dura_sec--;
		}

		// summarize kernel executions of each class
		kernelTrace *kerTrcP = kerTrcHead;
		while(kerTrcP != NULL)
		{
			if(strcmp(kerTrcP->kernelName, 
						curKernelName) == 0)
			{
				long origNsec = 
					kerTrcP->duration.tv_nsec + dura_nsec;
				kerTrcP->duration.tv_nsec = origNsec % 1000000000;
				kerTrcP->duration.tv_sec += 
					origNsec / 1000000000 + dura_sec;
				break;
			}
			kerTrcP = kerTrcP->next;
		}

		if(kerTrcP == NULL)
		{
			kerTrcP = (kernelTrace *)malloc(sizeof(kernelTrace));
			if(kerTrcP == NULL)
			{
				fprintf(stderr, "Error in line %d, fail to "\
						"malloc for profiling.\n", __LINE__);
				exit(EXIT_FAILURE);
			}
			strncpy(kerTrcP->kernelName, 
					curKernelName, KERNEL_NAME_MAX_LEN);
			kerTrcP->numExeTimes = 0;
			kerTrcP->duration.tv_nsec = dura_nsec;
			kerTrcP->duration.tv_sec = dura_sec;
			kerTrcP->next = kerTrcHead;
			kerTrcHead = kerTrcP;
		}

		kerTrcP->numExeTimes++;
	}
	fclose(rdFp);

	kerTime->tv_sec = 0;
	kerTime->tv_nsec = 0;
	kernelTrace *kerTrcP = kerTrcHead;
	while(kerTrcP != NULL)
	{
		// summarize all kernel executions, calculate kerTime
		long origNsec = kerTime->tv_nsec + 
			kerTrcP->duration.tv_nsec;
		kerTime->tv_nsec = origNsec % 1000000000;
		kerTime->tv_sec += origNsec / 1000000000 + 
			kerTrcP->duration.tv_sec;

		fprintf(wbFp, "\"%s\",%ld,%ld,\"%s(%d times)\"\n", 
				"kernel", kerTrcP->duration.tv_sec, 
				kerTrcP->duration.tv_nsec, kerTrcP->kernelName,
				kerTrcP->numExeTimes);
		kerTrcP = kerTrcP->next;
	}
}

static void sumMemcpyTrace(
		FILE *wtFp, struct timespec *cpyTime)
{
	timespec timeSpent[5]; 
	unsigned long long  cpyCount[5];
	int i;
	memset(timeSpent, 0, sizeof(timeSpent));
	memset(cpyCount, 0, sizeof(cpyCount));

	char funcName[SHORT_DESC_MAX_LEN];
	int cpyType;
	long startSec, startNsec, endSec, endNsec;
	size_t count;

	char buf[LINE_MAX_LEN];
	int len = 0;
	int headLines = 1;
	FILE *rdFp = fopen(memcpyTraceRecordFname, "r");
	while(fgets(buf, LINE_MAX_LEN, rdFp) != NULL)
	{
		if(headLines)
		{
			headLines--;
			continue;
		}
		sscanf(buf, "\"%[^\"]\",%d,%ld,%ld,%ld,%ld,%u",
				funcName, &cpyType, &startSec, &startNsec,
				&endSec, &endNsec, &count);

		if(endNsec < startNsec)
		{
			endNsec += 1000000000;
			endSec--;
		}

		timeSpent[cpyType].tv_nsec += endNsec - startNsec;
		timeSpent[cpyType].tv_sec += endSec - startSec;
		long carrySec = timeSpent[cpyType].tv_nsec / 1000000000;
		timeSpent[cpyType].tv_nsec %= 1000000000;
		timeSpent[cpyType].tv_sec += carrySec;		
		cpyCount[cpyType] += count;
	}
	fclose(rdFp);

	cpyTime->tv_sec = 0;
	cpyTime->tv_nsec = 0;

	char cpyTypeStr[5][20] = 
	{
		"HToH", "HToD", "DToH", "DToD", "UnifiedVisualAddr"
	};
	for(i = 0; i < 5; i++)
	{
		// summarize all kernel executions, calculate kerTime
		long origNsec = 
			cpyTime->tv_nsec + timeSpent[i].tv_nsec;
		cpyTime->tv_nsec = origNsec % 1000000000;
		cpyTime->tv_sec += 
			origNsec / 1000000000 + timeSpent[i].tv_sec;

		fprintf(wtFp, "\"%s\",%ld,%ld,\"%s(%d Bytes)\"\n",
				"memcpy", timeSpent[i].tv_sec, 
				timeSpent[i].tv_nsec, cpyTypeStr[i], 
				cpyCount[i]);
	}

}

static void sumOverheadTrace(
		FILE *wbFp, struct timespec *overhead)
{
	char funcName[SHORT_DESC_MAX_LEN];
	long startSec, startNsec, endSec, endNsec;
	overhead->tv_sec = 0;
	overhead->tv_nsec = 0;

	char buf[LINE_MAX_LEN];
	int len = 0;
	int headLines = 1;
	FILE *rdFp = fopen(profOverheadRecordFname , "r");
	while(fgets(buf, LINE_MAX_LEN, rdFp) != NULL)
	{
		if(headLines)
		{
			headLines--;
			continue;
		}
		sscanf(buf, "\"%[^\"]\",%ld,%ld,%ld,%ld", funcName, 
				&startSec, &startNsec, &endSec, &endNsec);

		if(endNsec < startNsec)
		{
			endNsec += 1000000000;
			endSec--;
		}
		overhead->tv_nsec += endNsec - startNsec;
		overhead->tv_sec += endSec - startSec;
		long carrySec = overhead->tv_nsec / 1000000000;
		overhead->tv_nsec %= 1000000000;
		overhead->tv_sec += carrySec;
	}
	fclose(rdFp);

	fprintf(wbFp, "\"%s\",%ld,%ld,\"\"\n", "profileOverhead", 
			overhead->tv_sec, overhead->tv_nsec, "");
}

static void timespecSub(struct timespec *minuend,
		const struct timespec *subtrahend)
{
	minuend->tv_nsec -= subtrahend->tv_nsec;
	minuend->tv_sec -= subtrahend->tv_sec;
	if(minuend->tv_nsec < 0)
	{
		minuend->tv_nsec += 1000000000;
		minuend->tv_sec--;
	}
}

static void sumTraceInfo(struct timespec *exeTime)
{
	struct timespec kernelTime, memcpyTime, overhead;

	FILE *fp = fopen(profTraceSumFname, "w");
	fprintf(fp, "\"type\",\"overall_sec\","\
			"\"overall_nsec\",\"info\"\n");

	sumOverheadTrace(fp, &overhead);
	sumKernelTrace(fp, &kernelTime);
	sumMemcpyTrace(fp, &memcpyTime);

	struct timespec cpuTime = *exeTime;
	timespecSub(&cpuTime, &overhead);
	timespecSub(&cpuTime, &kernelTime);
	timespecSub(&cpuTime, &memcpyTime);
	fprintf(fp, "\"%s\",%ld,%ld,\"\"\n",
			"cpu", cpuTime.tv_sec, cpuTime.tv_nsec);

	fclose(fp);
}

/* advquoted: quoted field; return pointer to next separator */
static char *advquoted(char *p)
{
	int i, j;

	for (i = j = 0; p[j] != '\0'; i++, j++)
	{
		if (p[j] == '"' && p[++j] != '"')
		{
			/* copy up to next separator or \0 */
			int k = strcspn(p+j, ",");
			memmove(p+i, p+j, k);
			i += k;
			j += k;
			break;
		}
		p[i] = p[j];
	}
	p[i] = '\0';

	return p + j;
}

/* split: split line into fields, 
 * delete '.' and replace "null" with "0" 
 * */
static int split(csv *csvLine)
{
	char *p;
	char *sepp; /* pointer to temporary separator character */
	int sepc;   /* temporary separator character */

	csvLine->numField = 0;
	if (csvLine->line[0] == '\0')
	{
		return 0;
	}
	p = csvLine->line;

	do
	{
		if (*p == '"')
		{
			sepp = advquoted(++p);  /* skip initial quote */
		}
		else
		{
			sepp = p + strcspn(p, ",");
		}
		sepc = sepp[0];
		sepp[0] = '\0';             /* terminate field */
		csvLine->fields[csvLine->numField++] = p;
		p = sepp + 1;
	}
	while (sepc == ',');

	return csvLine->numField;
}

static void recordKerClassData(kerInfoClass *kerClsP)
{
	int i;
	int numMtrs = 57;
	char mtrNames[57][64] =
	{
		"name", "device", "gridX", "gridY", "gridZ", 
		"blockX", "blockY", "blockZ", "dynamicSharedMem", 
		"staticSharedMem", "localMemPerThread", 
		"localMemTotal", "regPerThread", "cacheConfigReq", 
		"cacheConfigUsed", "sharedMemConfigUsed", 
		"theory_limiter", "achieved_limiter", "theory_occupancy", 
		"achieved_occupancy", "sm_efficiency", "gld_efficiency", 
		"gst_efficiency", "global_replay_overhead", 
		"gld_throughput", "gst_throughput",
		"local_memory_overhead", "warp_execution_efficiency", 
		"warp_nonpred_execution_efficiency", "shared_efficiency", 
		"shared_replay_overhead", "dram_read_throughput", 
		"dram_write_throughput", "dram_utilization", 
		"ldst_executed", "inst_fp_32", "inst_fp_64", 
		"inst_integer", "ipc", "l1_cache_global_hit_rate", 
		"l1_shared_utilization", "l2_utilization", 
		"ldst_fu_utilization", "alu_fu_utilization", 
		"shared_load_replay", "shared_store_replay", 
		"l2_total_misses", "l2_total_hit", "l1_global_load_miss", 
		"l1_global_load_hit", "inst_executed", "shared_load", 
		"shared_store", "local_load", "local_store", 
		"gld_request", "gst_request"
	};

	// remember to change the pos when mtrNames are changed
	int l2MissDataStartPos = 45;
	int l2HitDataStartPos = 53;

	FILE *fp = fopen(kerClassRecordFname, "w");
	if(fp == NULL)
	{
		fprintf(stderr, "Error, unable to open file %s, "\
				"in line %d\n", kerClassRecordFname, __LINE__);
	}
	for(i = 0; i < numMtrs; i++)
	{
		fprintf(fp, "\"%s\"", mtrNames[i]);
		if(i == numMtrs - 1)
		{
			fprintf(fp, "\n");
		}
		else
		{
			fprintf(fp, ",");
		}
	}

	while(kerClsP != NULL)
	{
		char line[2048];
		int len;
		snprintf(line, KERNEL_NAME_MAX_LEN, 
				"\"%s\",", kerClsP->kernelName);
		for(i = 0; i < kerClsP->numMtrItems; i++)
		{
			len = strlen(line);
			if(i == l2MissDataStartPos)
			{
				long double totalMiss = 
					kerClsP->mtrItems[i] + 
					kerClsP->mtrItems[i + 1] + 
					kerClsP->mtrItems[i + 2] + 
					kerClsP->mtrItems[i + 3] + 
					kerClsP->mtrItems[i + 4] + 
					kerClsP->mtrItems[i + 5] + 
					kerClsP->mtrItems[i + 6] + 
					kerClsP->mtrItems[i + 7];
				i += 7;
				snprintf(line + len, 64, "%llu,", 
						(unsigned long long)(totalMiss + 1e-5));
				continue;
			}
			if(i == l2HitDataStartPos)
			{
				long double totalHit = 
					kerClsP->mtrItems[i] + 
					kerClsP->mtrItems[i + 1] + 
					kerClsP->mtrItems[i + 2] + 
					kerClsP->mtrItems[i + 3] + 
					kerClsP->mtrItems[i + 4] + 
					kerClsP->mtrItems[i + 5] + 
					kerClsP->mtrItems[i + 6] + 
					kerClsP->mtrItems[i + 7];
				i += 7;
				snprintf(line + len, 64, "%llu,", 
						(unsigned long long)(totalHit + 1e-5));
				continue;
			}
			if(kerClsP->mtrItemTypes[i] == 0)
			{
				snprintf(line + len, 64, "%llu,", 
						(unsigned long long)(
							kerClsP->mtrItems[i] + 1e-5));
			}
			else if(kerClsP->mtrItemTypes[i] == 1)
			{
				snprintf(line + len, 64, "%.2llf,", 
						kerClsP->mtrItems[i]);
			}
			else if(kerClsP->mtrItemTypes[i] == 2)
			{
				snprintf(line + len, 8, "\"null\",");
			}
			else
			{
				fprintf(stderr, "Error, unknow mtrItem type in"\
						" kerInfoClass, line %d\n", __LINE__);
				exit(EXIT_FAILURE);
			}
		}
		line[strlen(line) - 1] = '\0';
		fprintf(fp, "%s\n", line);

		kerClsP = kerClsP->next;
	}
}

static void sumKerProfRes()
{
	char buf[LINE_MAX_LEN];
	FILE *fp = fopen(kerProfDataRecordFname, "r");
	if(fp == NULL)
	{
		fprintf(stderr, "Error in line %d, fail to open file "\
				"%s.\n", __LINE__, kerProfDataRecordFname);
		exit(EXIT_FAILURE);
	}
	int kerInfoMask[MTR_MAX_NUM]; // do not include kernelName
	memset(kerInfoMask, 255, sizeof(kerInfoMask));
	kerInfoMask[1] = 0;
	kerInfoMask[2] = 0;
	kerInfoMask[9] = 0;
	kerInfoMask[10] = 0;
	kerInfoMask[11] = 0;
	kerInfoMask[12] = 0;

	int isHead = 1;
	while(fgets(buf, LINE_MAX_LEN, fp) != NULL)
	{
		if(isHead)
		{
			isHead = 0;
			continue;
		}
		csv aCsv;
		strncpy(aCsv.line, buf, LINE_MAX_LEN);
		split(&aCsv);

		kerInfoClass *kerCls = kerClassHead;
		while(kerCls != NULL)
		{
			if(strcmp(kerCls->kernelName, 
						aCsv.fields[0]) == 0)
			{
				break;
			}
			kerCls = kerCls->next;
		}

		if(kerCls == NULL)
		{
			kerCls = (kerInfoClass *)
				malloc(sizeof(kerInfoClass));
			if(kerCls == NULL)
			{
				fprintf(stderr, "Error in line %d, fail to "\
						"malloc for profiling.\n", __LINE__);
				exit(EXIT_FAILURE);
			}
			strncpy(kerCls->kernelName, 
					aCsv.fields[0], KERNEL_NAME_MAX_LEN);
			memset(kerCls->mtrItemTypes, 0, 
					sizeof(kerCls->mtrItemTypes));
			memset(kerCls->mtrItems, 0, 
					sizeof(kerCls->mtrItems));

			// exclude name and other seven mtr items
			kerCls->numMtrItems = aCsv.numField - 7; 
			kerCls->numClassItems = 0;
			kerCls->next = kerClassHead;
			kerClassHead = kerCls;
		}

		int i;
		int mtrPos = 0;

		// set mtrItemTypes
		for(i = 1; i < aCsv.numField; i++)
		{
			if(kerInfoMask[i - 1] == 0)
			{
				continue;
			}

			int len = strlen(aCsv.fields[i]);
			while(aCsv.fields[i][len-1] == '\n')
			{
				aCsv.fields[i][--len] = '\0';
			}

			int dotPos = strcspn(aCsv.fields[i], ".");
			if(dotPos != len)
			{
				/*
				memmove(aCsv.fields[i] + dotPos, 
						aCsv.fields[i] + dotPos + 1, len-dotPos);
						*/
				kerCls->mtrItemTypes[mtrPos] = 1;
			}
			if(strcmp(aCsv.fields[i], "null") == 0)
			{
				strncpy(aCsv.fields[i], "0", 2);
				kerCls->mtrItemTypes[mtrPos] = 2;
			}
			mtrPos++;
		}

		mtrPos = 0;
		for(i = 1; i < aCsv.numField; i++)
		{
			if(kerInfoMask[i - 1] == 0)
			{
				continue;
			}
			long double tmp;
			sscanf(aCsv.fields[i], "%llf", &tmp);
			if(kerCls->mtrItems[mtrPos] != tmp)
			{
				kerCls->mtrItems[mtrPos] +=
					(tmp - kerCls->mtrItems[mtrPos]) / 
					(kerCls->numClassItems + 1);
			}

			mtrPos++;
		}
		kerCls->numClassItems++;
	}

	recordKerClassData(kerClassHead);
}

int main(int argc, char *argv[])
{
	parseCommandLineArgs(argc, argv);

	initProfFiles();

	struct timespec tsStart, tsEnd, exeTime;

	clock_gettime(CLOCK_MONOTONIC, &tsStart);

	printf("command: %s\n", command.command);
	int result = system(command.command);
	if(result == 1)
	{
		fprintf(stderr, "Error while profiling. exit.");
		exit(EXIT_FAILURE);
	}

	clock_gettime(CLOCK_MONOTONIC, &tsEnd);
	diff(&exeTime, &tsStart, &tsEnd);

	sumTraceInfo(&exeTime);

	sumKerProfRes();

	// printTimespec(&tsStart);
	// printTimespec(&tsEnd);
	printf("\nTotal execution time: ");
	printTimespec(&exeTime);

	return 0;
}


