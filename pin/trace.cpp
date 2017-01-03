#include "pin.H"
#include "instlib.H"
#include "portability.H"
#include <assert.h>
#include <stdio.h>
#include <map>
#include <set>
#include <iostream>
#include <fstream>

/************************************************/

#include "bloom.h"
#include "eventlist_common.h"

/************************************************/

#include "pin_base.cpp" // code that does instrumentation

/************************************************/

/* ================================================================== */
/*
  Convert an unsigned integer (representing bytes) into a string 
  that uses KB or MB as necessary
*/
string BytesToString(UINT32 numBytes)
{
    if (numBytes < 10240)
        return decstr(numBytes) + " bytes"; 
    else if (numBytes < (1024*10240))
        return decstr(numBytes>>10) + " KB"; 
    else 
        return decstr(numBytes>>20) + " MB"; 
}

/* ================================================================== */
/*
  Print command-line switches on error
*/
INT32 Usage()
{
    cerr << KNOB_BASE::StringKnobSummary() << endl;
    return -1;
}

static void PIN_FAST_ANALYSIS_CALL ThreadCreate(VOID* addr, THREADID tid, ADDRINT pc) {
#if EVENT_ENABLED
	RecordEvent_Fork(tid, totalThreads, pc);

	ADDRINT value;
	PIN_SafeCopy(&value, addr, sizeof(ADDRINT)); // take value from memory
	pthreadValMap[value] = totalThreads;
	totalThreads++;
#ifdef PAPI
	if (PAPI_thread_init(pthread_self) != PAPI_OK)
	{
       cout << "Error in thread initialization!\n";
       exit(4);
	}
#endif
#endif
}

static void PIN_FAST_ANALYSIS_CALL ThreadJoin(ADDRINT addr, THREADID tid, ADDRINT pc) {
#if EVENT_ENABLED
	RecordEvent_Join(tid, pthreadValMap[addr], pc);
#endif
}

static void PIN_FAST_ANALYSIS_CALL RecordMEMREFInBuffer_Read(ADDRINT pc, ADDRINT ea, THREADID tid){
#if EVENT_ENABLED
	RecordEvent_SharedRead(tid, ea, pc);
#endif
}

static void PIN_FAST_ANALYSIS_CALL RecordMEMREFInBuffer_Write(ADDRINT pc, ADDRINT ea, THREADID tid){
#if EVENT_ENABLED
	RecordEvent_SharedWrite(tid, ea, pc);
#endif
}

static void PIN_FAST_ANALYSIS_CALL LockForShared(THREADID tid){
#if EVENT_ENABLED && defined(LOCK_FOR_SHARED)
	lockForShared(tid);
#endif
}

static void PIN_FAST_ANALYSIS_CALL UnlockForShared(THREADID tid){
#if EVENT_ENABLED && defined(LOCK_FOR_SHARED)
	unlockForShared(tid);
#endif
}

static void PIN_FAST_ANALYSIS_CALL UNLOCKING(ADDRINT value, THREADID tid, ADDRINT pc) {
#if EVENT_ENABLED
	if ((int) value > 0)
		RecordEvent_Unlock(tid, value, pc);
#endif
}

static void PIN_FAST_ANALYSIS_CALL LOCKING(ADDRINT value, THREADID tid, ADDRINT pc) {
#if EVENT_ENABLED
	if ((int) value > 0)
		RecordEvent_Lock(tid, value, pc);
#endif
}

static void PIN_FAST_ANALYSIS_CALL RWUNLOCKING(ADDRINT value, THREADID tid, ADDRINT pc) {
#if EVENT_ENABLED
	if ((int) value > 0)
		RecordEvent_RWUnlock(tid, value, pc);
#endif
}

static void PIN_FAST_ANALYSIS_CALL RLOCKING(ADDRINT value, THREADID tid, ADDRINT pc) {
#if EVENT_ENABLED
	if ((int) value > 0)
		RecordEvent_RLock(tid, value, pc);
#endif
}

static void PIN_FAST_ANALYSIS_CALL WLOCKING(ADDRINT value, THREADID tid, ADDRINT pc) {
#if EVENT_ENABLED
	if ((int) value > 0)
		RecordEvent_WLock(tid, value, pc);
#endif
}

static void PIN_FAST_ANALYSIS_CALL ProcedureCall(ADDRINT pc, ADDRINT addr, THREADID tid){
#if EVENT_ENABLED
	RecordEvent_Call(tid, addr, pc);
#endif
}

static void PIN_FAST_ANALYSIS_CALL ProcedureReturn(ADDRINT pc, ADDRINT addr, THREADID tid){
#if EVENT_ENABLED
	RecordEvent_Return(tid, addr, pc);
#endif
}
/* ================================================================== */
/* Global Variables                                                   */
/* ================================================================== */

ofstream TraceFile;

/* ================================================================== */
/* Command-Line Switches                                              */
/* ================================================================== */

#ifdef PAPI
KNOB<BOOL>   KnobPid(KNOB_MODE_WRITEONCE, "pintool",
    "p", "1", "append pid to output");
KNOB<string> KnobOutputFile(KNOB_MODE_WRITEONCE, "pintool",
    "o", "papiDemo.out", "specify trace file name");

int EventSet;
long long values[10];
/* ================================================================== */
/*
 Use PAPI to start the counters
*/
VOID StartPapiCounters()
{
   EventSet = PAPI_NULL;

   if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
   {
       cout << "Error in PAPI_library_init!\n";
       exit(1);
   }
   if (PAPI_create_eventset(&EventSet) != PAPI_OK)
   {
       cout << "Error in PAPI_create_eventset!\n";
       exit(1);
   }
   if (PAPI_add_event(EventSet, PAPI_TOT_INS) != PAPI_OK)
   {
       cout << "Error in PAPI_add_event PAPI_TOT_INS!\n";
       exit(2);
   }
   if (PAPI_add_event(EventSet, PAPI_TOT_CYC) != PAPI_OK)
   {
       cout << "Error in PAPI_add_event PAPI_TOT_CYC!\n";
       exit(3);
   }
   if (PAPI_start(EventSet) != PAPI_OK)
   {
       cout << "Error in PAPI_start!\n";
       exit(4);
   }
}

/* ================================================================== */
/*
 Use PAPI to stop the counters.  Print details at the end of the run
*/
VOID StopAndPrintPapiCounters(INT32 code, VOID *v)
{
    string logFileName = KnobOutputFile.Value();
    if( KnobPid )
    {
        logFileName += "." + decstr( PIN_GetPid() );
    }
    TraceFile.open(logFileName.c_str());
    
    TraceFile << dec << "\n--------------------------------------------\n";
    if (PAPI_stop(EventSet, values) != PAPI_OK)
    {
        TraceFile << "Error in PAPI_stop!\n";
    }
    TraceFile << "Total Instructions " << values[0] << " Total Cycles " << values[1] << endl;

    TraceFile << dec << "--------------------------------------------\n";
    TraceFile << "Code cache size: ";
    int cacheSizeLimit = CODECACHE_CacheSizeLimit();
    if (cacheSizeLimit)
        TraceFile << BytesToString(cacheSizeLimit) << endl;
    else
        TraceFile << "UNLIMITED" << endl;
    TraceFile << BytesToString(CODECACHE_CodeMemUsed()) << " used" << endl;
    TraceFile << "#eof" << endl;
    TraceFile.close();
}
#endif
VOID Fini(INT32 code, VOID *v) {
	// this is the final point in the entire execution
	// thus, let us print the statistics
	IF_STATS(printStatistics(stdout));
	#ifdef PAPI
	StopAndPrintPapiCounters(code, v);
	#endif
}

static PIN_THREAD_UID EventThread1;
static THREADID EventThreadID1 = 0;

#ifdef SEPARATE_WORKER_THREADS
static PIN_THREAD_UID EventThread2;
static THREADID EventThreadID2 = 0;
#endif

VOID FiniUnlocked(INT32 code, VOID *v) {
// Before Fini function
#if EVENT_ENABLED
	// finalize the event list
	finalizeEventList();

	if(!PIN_WaitForThreadTermination(EventThread1, PIN_INFINITE_TIMEOUT, NULL)) {
		printf("Failed to join the event thread 1!");
		exit(-1);
	}

//#ifdef SEPARATE_WORKER_THREADS
//	if(!PIN_WaitForThreadTermination(EventThread2, PIN_INFINITE_TIMEOUT, NULL)) {
//		printf("Failed to join the event thread 2!");
//		exit(-1);
//	}
//#endif
#endif
}

int main(int argc, char * argv[]) {

	if (PIN_Init(argc, argv)) return Usage();
	#ifdef PAPI 
	CODECACHE_AddCacheInitFunction(StartPapiCounters, 0);
	#endif
	PIN_InitSymbols();

	PIN_AddFiniFunction(Fini, 0);

	// init statistics
	IF_STATS(initStatistics());


#if INSTR_ENABLED
		TRACE_AddInstrumentFunction(TraceAnalysisCalls, 0);
		IMG_AddInstrumentFunction(ImageLoad, 0);
		IMG_AddUnloadFunction(ImageUnload, 0);

	#if EVENT_ENABLED
		// initialize global event list
		initEventList();
		setGetSourceLocationFunc(PINGetSourceLocationFunc);
		#ifndef USE_PTHREAD
		EventThreadID1 = PIN_SpawnInternalThread(EventThreadProc1, &EventThreadID1, 0, &EventThread1);
		if(EventThreadID1 == INVALID_THREADID) {
			printf("Failed to spawn the worker thread!");
			exit(-1);
		}
		#endif
		PIN_AddFiniUnlockedFunction(FiniUnlocked, 0);

	#endif

		filter.Activate();

#endif

	PIN_StartProgram();

	return 0;
}
