#include "pin.H"
#include "instlib.H"
#include "portability.H"
#include <assert.h>
#include <stdio.h>
#include <map>
#include <set>
#include <iostream>
#include <fstream>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <sched.h>
#include <assert.h>
#include <time.h>
#include <sys/unistd.h>
#include <sys/syscall.h>
#include <errno.h>
#include <stdint.h>
#include <sys/time.h>

/************************************************/

#include "lockset.h"
#include "eventlist_common.h"
#include "synch.h"

#undef CALLS_ENABLED

/************************************************/

#include "pin_base.cpp" // code that does instrumentation

/************************************************/

static void PIN_FAST_ANALYSIS_CALL ThreadCreate(VOID* addr, THREADID tid, ADDRINT pc) {
#if EVENT_ENABLED
//	RecordEvent_Fork(tid, totalThreads, pc);

	ADDRINT value;
	PIN_SafeCopy(&value, addr, sizeof(ADDRINT)); // take value from memory
	pthreadValMap[value] = totalThreads;
	totalThreads++;

#endif
}

static void PIN_FAST_ANALYSIS_CALL ThreadJoin(ADDRINT addr, THREADID tid, ADDRINT pc) {
#if EVENT_ENABLED
//	RecordEvent_Join(tid, pthreadValMap[addr], pc);

#endif
}

static void PIN_FAST_ANALYSIS_CALL RecordMEMREFInBuffer_Read(ADDRINT pc, ADDRINT ea, THREADID tid)
{
#if EVENT_ENABLED
//	RecordEvent_SharedRead(tid, ea, pc);

	GLB_LOCK();

	int var = ea;
	if(!isRacy(var)) {
		lockset_t LH = lockset_get_for_tid(tid);
		if(lockset_is_var_initialized(var)) {
			lockset_t LS = lockset_get_for_var(var);
			lockset_t LS2 = lockset_intersect(LS, LH);
			if(LS2.size() == 0) {
				setRacy(var);
				// printf("Race!\n");
			}
			lockset_set_for_var(var, LS2);
		} else {
			lockset_t LS = LH;
			lockset_set_for_var(var, LS);
		}
	}

	GLB_UNLOCK();

#endif
}

static void PIN_FAST_ANALYSIS_CALL RecordMEMREFInBuffer_Write(ADDRINT pc, ADDRINT ea, THREADID tid)
{
#if EVENT_ENABLED
//	RecordEvent_SharedWrite(tid, ea, pc);

	GLB_LOCK();

	int var = ea;
	if(!isRacy(var)) {
		lockset_t LH = lockset_getw_for_tid(tid);
		if(lockset_is_var_initialized(var)) {
			lockset_t LS = lockset_get_for_var(var);
			lockset_t LS2 = lockset_intersect(LS, LH);
			if(LS2.size() == 0) {
				setRacy(var);
				// printf("Race!\n");
			}
			lockset_set_for_var(var, LS2);
		} else {
			lockset_t LS = LH;
			lockset_set_for_var(var, LS);
		}
	}

	GLB_UNLOCK();

#endif
}

static void PIN_FAST_ANALYSIS_CALL LockForShared(THREADID tid)
{
#if EVENT_ENABLED && defined(LOCK_FOR_SHARED)
//	lockForShared(tid);
#endif
}

static void PIN_FAST_ANALYSIS_CALL UnlockForShared(THREADID tid)
{
#if EVENT_ENABLED && defined(LOCK_FOR_SHARED)
//	unlockForShared(tid);
#endif
}

static void PIN_FAST_ANALYSIS_CALL UNLOCKING(ADDRINT value, THREADID tid, ADDRINT pc) {
#if EVENT_ENABLED
	if ((int) value > 0) {

		GLB_LOCK();

		lockset_t LH = lockset_get_for_tid(tid);
		LH.erase(value);
		lockset_set_for_tid(tid, LH);

		lockset_t WLH = lockset_getw_for_tid(tid);
		WLH.erase(value);
		lockset_setw_for_tid(tid, WLH);

		GLB_UNLOCK();
	}
#endif

}
static void PIN_FAST_ANALYSIS_CALL LOCKING(ADDRINT value, THREADID tid, ADDRINT pc) {
#if EVENT_ENABLED
	if ((int) value > 0) {

		GLB_LOCK();

		lockset_t LH = lockset_get_for_tid(tid);
		LH.insert(value);
		lockset_set_for_tid(tid, LH);

		lockset_t WLH = lockset_getw_for_tid(tid);
		WLH.insert(value);
		lockset_setw_for_tid(tid, WLH);

		GLB_UNLOCK();
	}
#endif
}

static void PIN_FAST_ANALYSIS_CALL RWUNLOCKING(ADDRINT value, THREADID tid, ADDRINT pc) {
#if EVENT_ENABLED
	if ((int) value > 0) {

		GLB_LOCK();

		lockset_t LH = lockset_get_for_tid(tid);
		LH.erase(value);
		lockset_set_for_tid(tid, LH);

		lockset_t WLH = lockset_getw_for_tid(tid);
		WLH.erase(value);
		lockset_setw_for_tid(tid, WLH);

		GLB_UNLOCK();
	}
#endif
}

static void PIN_FAST_ANALYSIS_CALL RLOCKING(ADDRINT value, THREADID tid, ADDRINT pc) {
#if EVENT_ENABLED
	if ((int) value > 0) {

		GLB_LOCK();

		lockset_t LH = lockset_get_for_tid(tid);
		LH.insert(value);
		lockset_set_for_tid(tid, LH);

		GLB_UNLOCK();
	}
#endif
}

static void PIN_FAST_ANALYSIS_CALL WLOCKING(ADDRINT value, THREADID tid, ADDRINT pc) {
#if EVENT_ENABLED
	if ((int) value > 0) {

		GLB_LOCK();

		lockset_t LH = lockset_get_for_tid(tid);
		LH.insert(value);
		lockset_set_for_tid(tid, LH);

		lockset_t WLH = lockset_getw_for_tid(tid);
		WLH.insert(value);
		lockset_setw_for_tid(tid, WLH);

		GLB_UNLOCK();
	}
#endif
}

static void PIN_FAST_ANALYSIS_CALL ProcedureCall(ADDRINT pc, ADDRINT addr, THREADID tid)
{
#if EVENT_ENABLED
//	RecordEvent_Call(tid, addr, pc);
#endif
}

static void PIN_FAST_ANALYSIS_CALL ProcedureReturn(ADDRINT pc, ADDRINT addr, THREADID tid)
{
#if EVENT_ENABLED
//	RecordEvent_Return(tid, addr, pc);
#endif
}

VOID Fini(INT32 code, VOID *v) {
	if(gettimeofday(&endTime, NULL)) {
		printf("Failed to get the end time!");
		exit(-1);
	}

	timeval_subtract(&runningTime, &endTime, &startTime);

	printf("\nTIME for program:\t%lu seconds, %lu microseconds\n", runningTime.tv_sec, runningTime.tv_usec);
}

VOID FiniUnlocked(INT32 code, VOID *v) {
// Before Fini function

	freeAll();
	
}

int main(int argc, char * argv[]) {

	if (PIN_Init(argc, argv)) return -1;

	PIN_InitSymbols();

	PIN_AddFiniFunction(Fini, 0);

	TRACE_AddInstrumentFunction(TraceAnalysisCalls, 0);
	IMG_AddInstrumentFunction(ImageLoad, 0);
	IMG_AddUnloadFunction(ImageUnload, 0);

	filter.Activate();

	bloom_clear(&glbRacyVars);

	GLB_LOCK_INIT();

	if(gettimeofday(&startTime, NULL)) {
		printf("Failed to get the start time!");
		exit(-1);
	}
	//PIN_AddFiniUnlockedFunction(FiniUnlocked, 0);
	
	PIN_StartProgram();

	return 0;
}
