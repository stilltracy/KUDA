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

#include <iomanip>
#include <limits.h>


#include <string.h>

#include <gelf.h>
#include "pin.H"


#include "eventlist_common.h"
#include "synch.h"

//#undef CALLS_ENABLED

/************************************************/

#include "pin_base.cpp" // code that does instrumentation

/************************************************/
struct timeval startTime;
struct timeval endTime;
struct timeval runningTime;
class VC{
	public:
    		PIN_MUTEX lock;
    		map<unsigned int, int> vc;
		//unsigned long same_epoch;
		//unsigned long read_shared;
		//unsigned long write_shared;
		//unsigned long exclusive;
		//unsigned long slowpath;
		//unsigned long read_racy;
		//unsigned long write_racy;
		bool isRacy;
    		VC(){ 
		  PIN_MutexInit(&lock);
		  isRacy=false;
		  //same_epoch= 0;
		  //read_shared = 0; 
		  //exclusive = 0;
		  //slowpath = 0;
		  //write_shared = 0;
		  //read_racy = 0;
		  //write_racy = 0;
		}
};

typedef map<unsigned int, int> vc;
typedef VC* VCPOINTER;
typedef vc Epoch;

vc races;

class VCs{
	public:
	   	 PIN_MUTEX lock;
   		 map<THREADID, VCPOINTER> vcs;
		 //unsigned long reads;
		 //unsigned long writes;
		// unsigned long threads;
		 //unsigned long locks;
  		  VCs(){ PIN_MutexInit(&lock); /*reads=0; writes=0; threads=0; locks=0;*/}
};

typedef VCs* VCsptr;
typedef map<THREADID, VCPOINTER> vcs;
static VCs glbC;
static VCs glbL;
static VCs glbR;
static VCs glbW;

/************************************************/
VCPOINTER get_vc(VCsptr m, THREADID k);
void vc_cup(VCPOINTER vc1, VCPOINTER vc2);
int clk_get(VCPOINTER vcptr, THREADID t);
void vc_inc(VCPOINTER vcptr, THREADID t);
/************************************************/

inline VCPOINTER get_vc(VCsptr m, THREADID k) 
{
 	map<THREADID, VCPOINTER>::iterator it = (m->vcs).find(k);
 	if(it != (m->vcs).end()) {
 		return it->second;
 	}
 	VCPOINTER tmp = new VC();
	tmp->vc  = map<THREADID, int>();
	if(m == &glbC)
	  tmp->vc[k] = 1;
	else
	  tmp->vc[k] = 0;
 	(m->vcs)[k] = tmp;
 	return tmp;
}

inline int clk_get(VCPOINTER vcptr, THREADID t) {
	vc::iterator it = vcptr->vc.find(t);
	if(it != vcptr->vc.end()) {
		int c = it->second;
		assert(c > 0);
		return c;
	}
	return 0; // rest is 0
}

inline void vc_cup(VCPOINTER vc1, VCPOINTER vc2) {

	for (vc::iterator it=vc2->vc.begin() ; it != vc2->vc.end(); it++) {
 		THREADID t = it->first;
 		int c1 = clk_get(vc1, t);
 		int c2 = it->second;
 		if(c2 > c1) {
 			vc1->vc[t] = c2;
 		}
	}
}

inline void vc_copy(VCPOINTER vc1, VCPOINTER vc2) {
	vc1->vc = vc2->vc;
}

inline Epoch vc_epoch(THREADID t, int clk) {
	vc vc_;
	vc_[t] = clk;
	return vc_;
}

inline void vc_inc(VCPOINTER vcptr, THREADID t) {  
	int c = clk_get(vcptr, t);
	vcptr->vc[t] = c + 1;
}

inline bool vc_is_epoch(VCPOINTER vc_){
	return vc_->vc.size() <= 1;
}

inline bool is_epoch(vc vc_){
        return vc_.size() <= 1;
}


inline int vc_epoch_tid(Epoch e) {
	if(e.size() == 0) {
		return 0;
	}
	assert (e.size() == 1);
	vc::iterator it=e.begin();
	return it->first;
}

inline int vc_epoch_clock(Epoch e) {
	if(e.size() == 0) {
		return 0;
	}
	assert (e.size() == 1);
	vc::iterator it=e.begin();
	return it->second;
}

inline bool vc_epoch_eq(VCPOINTER vp1, Epoch e2) {
	assert(vc_is_epoch(vp1));
	assert(is_epoch(e2));
	return ((vc_epoch_tid(vp1->vc) == vc_epoch_tid(e2)) && (vc_epoch_clock(vp1->vc) == vc_epoch_clock(e2)));
}

inline bool vc_leq_epoch(VCPOINTER e, VCPOINTER vc) {
        assert(vc_is_epoch(e));
        int t = vc_epoch_tid(e->vc);
        int c1 = vc_epoch_clock(e->vc);
        int c2 = clk_get(vc, t);
        return c1 <= c2;
}

inline vc vc_set(vc _vc, int t, int c) {
        if(c > 0) {
                _vc[t] = c;
        } else {
                _vc.erase(t);
        }
        return _vc;
}

inline bool vc_leq_all(VCPOINTER vc1, VCPOINTER vc2) {
	for (vc::iterator it=vc2->vc.begin() ; it != vc2->vc.end(); it++ ) {
		int t = it->first;
		int c1 = clk_get(vc1, t);
		int c2 = it->second;
		if(c1 > c2) {
			return false;
		}
	}
	return true;
}

inline bool isRacy(UINT x)
{
	GLB_LOCK();
	for (vc::iterator it=races.begin() ; it != races.end(); it++ )
		if(it->first == x) return true;
	GLB_UNLOCK();
	return false;
}
inline void setRacy(UINT x)
{
	GLB_LOCK();
	races[x] = x;
	GLB_UNLOCK();
	return;
}
static void PIN_FAST_ANALYSIS_CALL ThreadCreate(VOID* addr, THREADID tid, ADDRINT pc) {
#if EVENT_ENABLED
//	RecordEvent_Fork(tid, totalThreads, pc);
	THREADID t = tid;
	
	PIN_MutexLock(&glbC.lock);
	//glbC.threads++;
	unsigned int u = totalThreads;
	totalThreads++;
	PIN_MutexUnlock(&glbC.lock);
	
	VCPOINTER Ct = get_vc(&glbC, t);
	VCPOINTER Cu = get_vc(&glbC, u);

	vc_cup(Cu, Ct);        //vc_set_vc(glbC, u, vc_cup(Ct, Cu));
	vc_inc(Ct, t);         //vc_set_vc(glbC, t, vc_inc(Ct, t));

#endif
}

static void PIN_FAST_ANALYSIS_CALL ThreadJoin(ADDRINT addr, THREADID tid, ADDRINT pc) {
#if EVENT_ENABLED
//	RecordEvent_Join(tid, pthreadValMap[addr], pc);

	//GLB_LOCK();

	//int t = tid;
	//int u = pthreadValMap[addr];
	//VC Ct = vc_get_vc(glbC, t);
	//VC Cu = vc_get_vc(glbC, u);
	//vc_set_vc(glbC, t, vc_cup(Ct, Cu));
	//vc_set_vc(glbC, u, vc_inc(Cu, u));

	//GLB_UNLOCK();

#endif
}

static void PIN_FAST_ANALYSIS_CALL RecordMEMREFInBuffer_Read(ADDRINT pc, ADDRINT ea, THREADID tid)
{
#if EVENT_ENABLED
//	RecordEvent_SharedRead(tid, ea, pc);
 	THREADID t = tid;
	UINT x = (UINT)ea;

	//PIN_MutexLock(&glbR.lock);
	//glbR.reads++;
	//PIN_MutexUnlock(&glbR.lock);
	VCPOINTER Rx = get_vc(&glbR, x);    //get read vector clocks
	
	PIN_MutexLock(&Rx->lock);      //  lock
	if(!Rx->isRacy) {
		//PIN_MutexLock(&glbW.lock);
		VCPOINTER Wx = get_vc(&glbW, x);    //get write vector clock
		//PIN_MutexUnlock(&glbW.lock);
		VCPOINTER Ct = get_vc(&glbC, t);    //get thread vector clock	
		PIN_MutexLock(&Wx->lock);      //  lock
		Epoch Et = vc_epoch(t, clk_get(Ct, t));   // make epoch of the thread

		if(vc_epoch_eq(Rx, Et)) { // if(x.R == t.epoch) return; // Same epoch 63.4%
			//Rx->same_epoch++;
			PIN_MutexUnlock(&Wx->lock); //unlocking
			PIN_MutexUnlock(&Rx->lock); // unlocking
			return;
		}
		
		if(! vc_leq_epoch(Wx, Ct)){ //if(x.W > t.C[TID(x.W)]) error;     // write-read race
			{Rx->isRacy = true; /* printf("Read-Race!\n");*/ }
		}
		
		if(vc_is_epoch(Rx)) { // if(x.R != READ_SHARED) 

			if(vc_leq_epoch(Rx, Ct)) { //if(x.R <= t.C[TID(x.R)]) //exclusive 15.7%
				//Rx->exclusive++;
				Rx->vc = Et;      //vc_set_vc(glbR, x, Et); // Rx check for this assignment
			}
			else {  //Share 0.1% //4. (SLOW PATH)
				//Rx->slowpath++;
				vc _vc;
				vc_set(_vc, t, clk_get(Ct, t)); // set Et
				vc_set(_vc, vc_epoch_tid(Rx->vc), vc_epoch_clock(Rx->vc)); // set Rx
				Rx->vc = _vc; //vc_set_vc(glbR, x, _vc);
			}
		}
		else { //if(x.R == READ_SHARED) //   //shared 20.8% 2. READ SHARED
			//Rx->read_shared++;
 			vc_set(Rx->vc, t, clk_get(Ct, t));  //vc_set_vc(glbR, x, vc_set(Rx, t, vc_get(Ct,t)));  //vc_set changes the clock
		}
		PIN_MutexUnlock(&Wx->lock); //unlock
	}
//	else {
//		Rx->read_racy++;
//	}
	PIN_MutexUnlock(&Rx->lock); // unlock
	//printf("##  Finished reading\n"); fflush(stdout);

#endif
}

static void PIN_FAST_ANALYSIS_CALL RecordMEMREFInBuffer_Write(ADDRINT pc, ADDRINT ea, THREADID tid)
{
#if EVENT_ENABLED
//	RecordEvent_SharedWrite(tid, ea, pc);

	THREADID t = tid;
	ADDRINT x = (ADDRINT)ea;
	
//	PIN_MutexLock(&glbR.lock);
//	glbR.writes++;
//	PIN_MutexUnlock(&glbR.lock);
	
	VCPOINTER Rx = get_vc(&glbR, x); // get read vector clock
	
	PIN_MutexLock(&Rx->lock); // lock read vc
	if(!Rx->isRacy) {
		//PIN_MutexLock(&glbW.lock);
		VCPOINTER Wx = get_vc(&glbW, x); // get write vector clock 
		//PIN_MutexUnlock(&glbW.lock);
		
		VCPOINTER Ct = get_vc(&glbC, t); // get thread clockvector
		PIN_MutexLock(&Wx->lock); // lock write vc
		
		Epoch Et = vc_epoch(t, clk_get(Ct, t));

		if(vc_epoch_eq(Wx, Et)) {     // same epoch 71.0%
//		      Wx->same_epoch++;
		      PIN_MutexUnlock(&Wx->lock);
		      PIN_MutexUnlock(&Rx->lock);
		      return;	
		}
		
		if(! vc_leq_epoch(Wx, Ct)) {//if(W.x > Ct)  race!!! //
		      Rx->isRacy = true; //printf("Write-Write-Race!\n");
		}
		
		// NOT SAME EPOCH
		if(vc_is_epoch(Rx))    // if(Rx != READ_SHARED ) // 2.  NOT WRITE SHARED it is EPOCH (EXCLUSIVE)
		{// Shared 28.9%
//			Wx->write_shared++;
			if(!vc_leq_epoch(Rx, Ct))
			{ Rx->isRacy = true; /* printf("Write-Race!\n"); */}
		}
		else  {// if(Rx == READ_SHARED) // 3. (SLOW PATH) it is read shared
			//Exclusive 0.1%
//			Wx->slowpath++;
			if(!vc_leq_all(Rx, Ct)) // check all clocks
				{Rx->isRacy = true; /*printf("Write-Race!\n");*/ }
			else
			  	Rx->vc = vc_epoch(t,0);   // vc_set_vc(glbR, x, vc_epoch(0,0));
		}
		Wx->vc = Et;              // vc_set_vc(glbW, x, Et);
		PIN_MutexUnlock(&Wx->lock);
	}
//	else {
//		Rx->write_racy++;
//	}
	PIN_MutexUnlock(&Rx->lock);
	//printf("?? finished writing\n"); fflush(stdout);

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
    	THREADID t = PIN_ThreadId();
	//if(t != PIN_ThreadId()) printf("Different thread ids at locking t:%u tid:%u\n", t, PIN_ThreadId()); fflush(stdout);
        ADDRINT lok;
	PIN_SafeCopy(&lok, (ADDRINT *)value, sizeof(ADDRINT)); //= (UINT)value;
//	PIN_MutexLock(&glbC.lock);
//	glbC.locks++;
//	PIN_MutexUnlock(&glbC.lock);		

	VCPOINTER Ct = get_vc(&glbC, t); //get thread vectorclock pointer
	
	//PIN_MutexLock(&glbL.lock);
	VCPOINTER Lm = get_vc(&glbL, lok); // get lock thread pointer
	//PIN_MutexUnlock(&glbL.lock);

	PIN_MutexLock(&Lm->lock);
	
	vc_copy(Lm, Ct);                   //vc_set_vc(glbL, lok, Ct);  //set 
	vc_inc(Ct, t);       //vc_set_vc(glbC, tid, vc_inc(Ct, tid)); // increment
	
	PIN_MutexUnlock(&Lm->lock);
	//printf("=finished unlocking\n"); fflush(stdout);
#endif

}
static void PIN_FAST_ANALYSIS_CALL LOCKING(ADDRINT value, THREADID tid, ADDRINT pc) {
#if EVENT_ENABLED
	THREADID t = PIN_ThreadId();
	//if(t != PIN_ThreadId()) printf("Different thread ids at locking t:%u tid:%u\n", t, PIN_ThreadId()); fflush(stdout);
    ADDRINT lok;
	PIN_SafeCopy(&lok, (ADDRINT *)value, sizeof(ADDRINT)); //= (UINT)value;
//	PIN_MutexLock(&glbC.lock);
//	glbC.locks++;
//	PIN_MutexUnlock(&glbC.lock);
	
	VCPOINTER Ct = get_vc(&glbC, t);    // get pointer to thread VC
	
	//PIN_MutexLock(&glbL.lock);
	VCPOINTER Lm = get_vc(&glbL, lok);  // get pointer to lock VC
	//PIN_MutexUnlock(&glbL.lock);

	PIN_MutexLock(&Lm->lock);  //lock the lock vector(others may access)
	vc_cup(Ct, Lm); //vc_set_vc(glbC, tid, vc_cup(Ct, Lm)); // set
	
	PIN_MutexUnlock(&Lm->lock); // release locks
		
	//printf("=finished locking\n"); fflush(stdout);
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

	

	if(gettimeofday(&startTime, NULL)) {
		printf("Failed to get the start time!");
		exit(-1);
	}

	PIN_StartProgram();

	return 0;
}
