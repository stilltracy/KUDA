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

#include "vc.h"
#include "eventlist_common.h"
#include "synch.h"

#undef CALLS_ENABLED

/************************************************/

#include "pin_base.cpp" // code that does instrumentation

/************************************************/

static void PIN_FAST_ANALYSIS_CALL ThreadCreate(VOID* addr, THREADID tid, ADDRINT pc) {
#if EVENT_ENABLED
//	RecordEvent_Fork(tid, totalThreads, pc);

	GLB_LOCK();

	int t = tid;
	int u = totalThreads;
	VC Ct = vc_get_vc(glbC, t);
	VC Cu = vc_get_vc(glbC, u);
	vc_set_vc(glbC, u, vc_cup(Ct, Cu));
	vc_set_vc(glbC, t, vc_inc(Ct, t));

	ADDRINT value;
	PIN_SafeCopy(&value, addr, sizeof(ADDRINT)); // take value from memory
	pthreadValMap[value] = totalThreads;
	totalThreads++;

	GLB_UNLOCK();

#endif
}

static void PIN_FAST_ANALYSIS_CALL ThreadJoin(ADDRINT addr, THREADID tid, ADDRINT pc) {
#if EVENT_ENABLED
//	RecordEvent_Join(tid, pthreadValMap[addr], pc);

	GLB_LOCK();

	int t = tid;
	int u = pthreadValMap[addr];
	VC Ct = vc_get_vc(glbC, t);
	VC Cu = vc_get_vc(glbC, u);
	vc_set_vc(glbC, t, vc_cup(Ct, Cu));
	vc_set_vc(glbC, u, vc_inc(Cu, u));

	GLB_UNLOCK();

#endif
}

static void PIN_FAST_ANALYSIS_CALL RecordMEMREFInBuffer_Read(ADDRINT pc, ADDRINT ea, THREADID tid)
{
#if EVENT_ENABLED
//	RecordEvent_SharedRead(tid, ea, pc);

	GLB_LOCK();

	int x = ea;
	if(!isRacy(x)) {
		int t = tid;
		VC Rx = vc_get_vc(glbR, x);
		VC Ct = vc_get_vc(glbC, t);
		if(vc_get(Rx, t) != vc_get(Ct, t)) {
			VC Wx = vc_get_vc(glbW, x);
			if(!vc_leq_all(Wx, Ct)) {
				setRacy(x);
				printf("Race!\n");
			}
			vc_set_vc(glbR, x, vc_set(Rx, t, vc_get(Ct, t)));
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

	int x = ea;
	if(!isRacy(x)) {
		int t = tid;
		VC Wx = vc_get_vc(glbW, x);
		VC Ct = vc_get_vc(glbC, t);
		if(vc_get(Wx, t) != vc_get(Ct, t)) {
			VC Rx = vc_get_vc(glbR, x);
			if(!vc_leq_all(Wx, Ct) || !vc_leq_all(Rx, Ct)) {
				setRacy(x);
				printf("Race!\n");
			}
			vc_set_vc(glbW, x, vc_set(Wx, t, vc_get(Ct, t)));

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

		VC Ct = vc_get_vc(glbC, tid);
		vc_set_vc(glbL, value, Ct);
		vc_set_vc(glbC, tid, vc_inc(Ct, tid));

		GLB_UNLOCK();
	}
#endif

}
static void PIN_FAST_ANALYSIS_CALL LOCKING(ADDRINT value, THREADID tid, ADDRINT pc) {
#if EVENT_ENABLED
	if ((int) value > 0) {

		GLB_LOCK();

		VC Ct = vc_get_vc(glbC, tid);
		VC Lm = vc_get_vc(glbL, value);
		vc_set_vc(glbC, tid, vc_cup(Ct, Lm));

		GLB_UNLOCK();
	}
#endif
}

static void PIN_FAST_ANALYSIS_CALL RWUNLOCKING(ADDRINT value, THREADID tid, ADDRINT pc) {
#if EVENT_ENABLED
//	if ((int) value > 0)
//		RecordEvent_RWUnlock(tid, value, pc);
#endif
}

static void PIN_FAST_ANALYSIS_CALL RLOCKING(ADDRINT value, THREADID tid, ADDRINT pc) {
#if EVENT_ENABLED
//	if ((int) value > 0)
//		RecordEvent_RLock(tid, value, pc);
#endif
}

static void PIN_FAST_ANALYSIS_CALL WLOCKING(ADDRINT value, THREADID tid, ADDRINT pc) {
#if EVENT_ENABLED
//	if ((int) value > 0)
//		RecordEvent_WLock(tid, value, pc);
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

	PIN_StartProgram();

	return 0;
}
