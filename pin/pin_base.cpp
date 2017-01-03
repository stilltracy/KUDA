
//static MutexLock lock;
//#define GLB_LOCK_INIT()		initMutex(&lock)
//#define GLB_LOCK()			lockMutex(&lock, 10)
//#define GLB_UNLOCK()		unlockMutex(&lock)
//  #!# check for actual (both Pin and Eventlist) speedups!!!
// #define MAINEXEC

static PIN_LOCK lock;
#define GLB_LOCK_INIT()		InitLock(&lock)
#define GLB_LOCK()			GetLock(&lock, 1);
#define GLB_UNLOCK()		ReleaseLock(&lock);

/************************************************/

static void PIN_FAST_ANALYSIS_CALL ThreadCreate(VOID* addr, THREADID tid, ADDRINT pc);
static void PIN_FAST_ANALYSIS_CALL ThreadJoin(ADDRINT addr, THREADID tid, ADDRINT pc);
static void PIN_FAST_ANALYSIS_CALL RecordMEMREFInBuffer_Read(ADDRINT pc, ADDRINT ea, THREADID tid);
static void PIN_FAST_ANALYSIS_CALL RecordMEMREFInBuffer_Write(ADDRINT pc, ADDRINT ea, THREADID tid);
static void PIN_FAST_ANALYSIS_CALL LockForShared(THREADID tid);
static void PIN_FAST_ANALYSIS_CALL UnlockForShared(THREADID tid);
static void PIN_FAST_ANALYSIS_CALL UNLOCKING(ADDRINT value, THREADID tid, ADDRINT pc);
static void PIN_FAST_ANALYSIS_CALL LOCKING(ADDRINT value, THREADID tid, ADDRINT pc);
static void PIN_FAST_ANALYSIS_CALL ProcedureCall(ADDRINT pc, ADDRINT addr, THREADID tid);
static void PIN_FAST_ANALYSIS_CALL ProcedureReturn(ADDRINT pc, ADDRINT addr, THREADID tid);

static void PIN_FAST_ANALYSIS_CALL RWUNLOCKING(ADDRINT value, THREADID tid, ADDRINT pc);
static void PIN_FAST_ANALYSIS_CALL RLOCKING(ADDRINT value, THREADID tid, ADDRINT pc);
static void PIN_FAST_ANALYSIS_CALL WLOCKING(ADDRINT value, THREADID tid, ADDRINT pc);

/************************************************/

INSTLIB::FILTER_LIB filter;

#if EVENT_ENABLED
	map<ADDRINT, int> pthreadValMap;
	THREADID totalThreads = 1;
#endif

/************************************************/

void TraceAnalysisCalls(TRACE trace, void *) {

	if (!filter.SelectTrace(trace))
		return;

	for (BBL bbl = TRACE_BblHead(trace); BBL_Valid(bbl); bbl = BBL_Next(bbl)) {

		for (INS ins = BBL_InsHead(bbl); INS_Valid(ins); ins = INS_Next(ins)) {

			if (INS_IsOriginal(ins)) {
				//  #!# check for actual (both Pin and Eventlist) speedups!!!
		#ifdef MAINEXEC
				RTN rtn = TRACE_Rtn(trace));
				if(RTN_Valid(rtn)) {
					if(IMG_IsMainExecutable(SEC_Img(RTN_Sec(rtn)))) {
		#endif
						// #!#
						if(!INS_IsStackRead(ins)
							&& !INS_IsStackWrite(ins) && (INS_IsMemoryRead(ins)
							|| INS_IsMemoryWrite(ins) || INS_HasMemoryRead2(ins))) {

		#ifdef LOCK_FOR_SHARED
							// acquire the global lock to serialize the access
							INS_InsertCall(ins, IPOINT_BEFORE, AFUNPTR(LockForShared),
									IARG_FAST_ANALYSIS_CALL, IARG_THREAD_ID, IARG_END);
		#endif

							// Log every memory references of the instruction
							if (INS_IsMemoryRead(ins)) {
								INS_InsertCall(ins, IPOINT_BEFORE, AFUNPTR(
										RecordMEMREFInBuffer_Read),
										IARG_FAST_ANALYSIS_CALL, IARG_INST_PTR,
										IARG_MEMORYREAD_EA, IARG_THREAD_ID, IARG_END);
							}
							if (INS_IsMemoryWrite(ins)) {
								INS_InsertCall(ins, IPOINT_BEFORE, AFUNPTR(
										RecordMEMREFInBuffer_Write),
										IARG_FAST_ANALYSIS_CALL, IARG_INST_PTR,
										IARG_MEMORYWRITE_EA, IARG_THREAD_ID, IARG_END);
							}
							if (INS_HasMemoryRead2(ins)) {
								INS_InsertCall(ins, IPOINT_BEFORE, AFUNPTR(
										RecordMEMREFInBuffer_Read),
										IARG_FAST_ANALYSIS_CALL, IARG_INST_PTR,
										IARG_MEMORYREAD2_EA, IARG_THREAD_ID, IARG_END);
							}

		#ifdef LOCK_FOR_SHARED
							// release the global lock
							if (INS_HasFallThrough(ins)) {
								INS_InsertCall(ins, IPOINT_AFTER, AFUNPTR(UnlockForShared),
										IARG_FAST_ANALYSIS_CALL, IARG_THREAD_ID, IARG_END);
							}

							if (INS_IsBranchOrCall(ins)) {
								INS_InsertCall(ins, IPOINT_TAKEN_BRANCH, AFUNPTR(
										UnlockForShared), IARG_FAST_ANALYSIS_CALL,
										IARG_THREAD_ID, IARG_END);
							}
		#endif
						}

		#if CALLS_ENABLED
						if(INS_IsProcedureCall(ins)) {
							INS_InsertCall(ins, IPOINT_BEFORE, AFUNPTR(
									ProcedureCall),
									IARG_FAST_ANALYSIS_CALL, IARG_INST_PTR,
									IARG_BRANCH_TARGET_ADDR, IARG_THREAD_ID, IARG_END);
							INS_InsertCall(ins, IPOINT_BEFORE, AFUNPTR(
									ProcedureReturn),
									IARG_FAST_ANALYSIS_CALL, IARG_INST_PTR,
									IARG_BRANCH_TARGET_ADDR, IARG_THREAD_ID, IARG_END);
						}
		#endif
		#ifdef MAINEXEC
					}
				}
		#endif	
		
			}

		}
	}

}

static bool pthreadLoaded = false;
static UINT32 pthreadIMGID = 0;

VOID ImageLoad(IMG img, VOID *) {

	if(pthreadLoaded) return;

	RTN rtn = RTN_FindByName(img, "pthread_mutex_lock");

	if (RTN_Valid(rtn)) {

		//--------------------------
		pthreadLoaded = true;
		pthreadIMGID = IMG_Id(img);
		//--------------------------

		RTN_Open(rtn);
		RTN_InsertCall(rtn, IPOINT_AFTER, AFUNPTR(LOCKING),
				IARG_FUNCARG_ENTRYPOINT_VALUE, 0, IARG_THREAD_ID, IARG_INST_PTR, IARG_END);
		RTN_Close(rtn);
	}

	rtn = RTN_FindByName(img, "pthread_mutex_unlock");

	if (RTN_Valid(rtn)) {
		RTN_Open(rtn);
		RTN_InsertCall(rtn, IPOINT_BEFORE, AFUNPTR(UNLOCKING),
				IARG_FUNCARG_ENTRYPOINT_VALUE, 0, IARG_THREAD_ID, IARG_INST_PTR, IARG_END);
		RTN_Close(rtn);
	}

	rtn = RTN_FindByName(img, "pthread_rwlock_rdlock");

	if (RTN_Valid(rtn)) {
		RTN_Open(rtn);
		RTN_InsertCall(rtn, IPOINT_BEFORE, AFUNPTR(RLOCKING),
				IARG_FUNCARG_ENTRYPOINT_VALUE, 0, IARG_THREAD_ID, IARG_INST_PTR, IARG_END);
		RTN_Close(rtn);
	}

	rtn = RTN_FindByName(img, "pthread_rwlock_wrlock");

	if (RTN_Valid(rtn)) {
		RTN_Open(rtn);
		RTN_InsertCall(rtn, IPOINT_BEFORE, AFUNPTR(WLOCKING),
				IARG_FUNCARG_ENTRYPOINT_VALUE, 0, IARG_THREAD_ID, IARG_INST_PTR, IARG_END);
		RTN_Close(rtn);
	}

	rtn = RTN_FindByName(img, "pthread_rwlock_unlock");

	if (RTN_Valid(rtn)) {
		RTN_Open(rtn);
		RTN_InsertCall(rtn, IPOINT_BEFORE, AFUNPTR(RWUNLOCKING),
				IARG_FUNCARG_ENTRYPOINT_VALUE, 0, IARG_THREAD_ID, IARG_INST_PTR, IARG_END);
		RTN_Close(rtn);
	}

	rtn = RTN_FindByName(img, "__pthread_create_2_1");

	if (RTN_Valid(rtn)) {

		RTN_Open(rtn);

		RTN_InsertCall(rtn, IPOINT_BEFORE, AFUNPTR(ThreadCreate),
				IARG_FUNCARG_ENTRYPOINT_VALUE, 0, IARG_THREAD_ID, IARG_INST_PTR, IARG_END);

		RTN_Close(rtn);
	}

	rtn = RTN_FindByName(img, "pthread_join");

	if (RTN_Valid(rtn)) {

		RTN_Open(rtn);

		RTN_InsertCall(rtn, IPOINT_AFTER, AFUNPTR(ThreadJoin),
				IARG_FUNCARG_ENTRYPOINT_VALUE, 0, IARG_THREAD_ID, IARG_INST_PTR, IARG_END);

		RTN_Close(rtn);
	}

}

VOID ImageUnload(IMG img, VOID *) {
	if(pthreadLoaded && IMG_Id(img) == pthreadIMGID) {
		pthreadLoaded = false;
		pthreadIMGID = 0;
	}
}

// a wrapper for calling PIN_GetSourceLocation
void PINGetSourceLocationFunc(int addr, int* column, int* line, char** file) {
	string* str = new string();
	PIN_LockClient();
	PIN_GetSourceLocation(addr, column, line, str);
	PIN_UnlockClient();
	*file = (char*) str->c_str();
}
