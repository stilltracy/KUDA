/*******************************************************************************
 * Copyright (c) 2005, 2006 IBM Corporation and others.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     IBM Corporation - initial API and implementation
 *******************************************************************************/
#if false
#include "lockset.h"

#ifdef LOCKSET_ENABLED

/**
 ***********************************************************************************
 ***********************************************************************************
 * Global Variables
 ***********************************************************************************
 ***********************************************************************************
 **/

jls_update_list_t jls_glb_update_list;

#if LOCKSET_CHECK_LOCKSET_SIZE
jls_spinlock_t	jls_glb_max_lockset_spinlock;
unsigned int jls_glb_max_lockset_size;
char * jls_glb_max_lockset_access;
unsigned long jls_glb_max_lockset_start;
unsigned long jls_glb_max_lockset_end;
double jls_glb_max_lockset_count;
double jls_glb_max_lockset_average;
unsigned int jls_glb_max_lockset_size_for_list;
double jls_glb_max_lockset_count_for_list;
double jls_glb_max_lockset_average_for_list;
#endif

/**
 ***********************************************************************************
 ***********************************************************************************
 * Include the implementing functions
 ***********************************************************************************
 ***********************************************************************************
 **/

#include "lockset/lockset-init.c"
#include "lockset/lockset-mutex.c"
#include "lockset/lockset-list.c"
#include "lockset/lockset-map.c"
#include "lockset/lockset-object.c"
#include "lockset/lockset-class.c"
#include "lockset/lockset-array.c"
#include "lockset/lockset-thread.c"
#include "lockset/lockset-check.c"
#include "lockset/lockset-events.c"
#include "lockset/lockset-update.c"
#include "lockset/lockset-rules.c"
#include "lockset/lockset-report.c"
#include "lockset/lockset-readset.c"

#ifdef LOCKSET_ENABLE_STATICINFO
#include "lockset/lockset-static.c"
#endif

#if ALG_RACETRACK || ALG_VCLOCK || ALG_ERASER || ALG_VCLOCK2
	#include "lockset/lockset_vclock.c"
#endif

#include JLS_LOCKSET_IMPL_CSOURCE

#endif // LOCKSET_ENABLED

#include "lockset/lockset-memory.c"

#if LOCKSET_STATISTICS
	#include "lockset/lockset-stat.c"
#endif

#endif




/*BEGIN_LEGAL 
 Intel Open Source License

 Copyright (c) 2002-2010 Intel Corporation. All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are
 met:

 Redistributions of source code must retain the above copyright notice,
 this list of conditions and the following disclaimer.  Redistributions
 in binary form must reproduce the above copyright notice, this list of
 conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.  Neither the name of
 the Intel Corporation nor the names of its contributors may be used to
 endorse or promote products derived from this software without
 specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE INTEL OR
 ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 END_LEGAL */
#include <iostream>
#include <set>
#include <vector>
#include <map>
#include "pin.H"
#include "instlib.H"
#include "sys/time.h"
using namespace INSTLIB;
using namespace std;

KNOB<string> KnobOutputFile(KNOB_MODE_WRITEONCE, "pintool", "o", "output",
		"specify output file name");

typedef struct {
	int owner;
	int alock;
	set<int> LS;
	int pos;
} info;

typedef struct {
	map<THREADID, info> readInfo;
	info writeInfo;
} address;

typedef struct {
	THREADID threadId;
	int value; // for acq and release , it is lock value.  for fork and join, it is child thread id (u)
	int kind; // 0= acq lock 1= release lock 2= fork 3= join
} event;

map<ADDRINT, address> addressMap;
vector<event> eventList;

map<ADDRINT, THREADID> pthreadValMap; //map  from pthread value to threadId map

map<int, int> lockMap;

ofstream myfile;
THREADID totalThreads = 1;

FILTER_LIB filter;
PIN_LOCK lock;

// Note that opening a file in a callback is only supported on Linux systems.
// See buffer-win.cpp for how to work around this issue on Windows.
//
// This routine is executed every time a thread is created.

/*
 VOID BeforeThreadCreate(ADDRINT addr, THREADID threadId)
 {


 PIN_GetLock(&lock, 1);

 // pthreadAddrMap[addr]= totalThreads;





 //myfile<< "thread "<<  totalThreads-1 << " is created by "<< threadId<<endl;


 PIN_ReleaseLock(&lock);



 }


 */

VOID AfterThreadCreate(VOID* addr, THREADID threadId) {
	event newEvent;
	newEvent.threadId = threadId;
	newEvent.value = totalThreads;
	newEvent.kind = 0;
	PIN_GetLock(&lock, 1);

	eventList.push_back(newEvent);
	ADDRINT value;
	PIN_SafeCopy(&value, addr, sizeof(ADDRINT));
	pthreadValMap[value] = totalThreads;

	totalThreads++;

	myfile << "thread " << totalThreads - 1 << " is created by " << threadId
			<< endl;
	PIN_ReleaseLock(&lock);
}

VOID AfterThreadJoin(ADDRINT addr, THREADID threadId) {
	PIN_GetLock(&lock, 1);
	event newEvent;
	newEvent.threadId = threadId;
	newEvent.value = pthreadValMap[addr];
	newEvent.kind = 1;

	eventList.push_back(newEvent);

	myfile << "thread " << newEvent.value << " is joined by " << threadId
			<< "   pthread value: " << addr << endl;
	PIN_ReleaseLock(&lock);
}

VOID AfterLock(VOID* addrP, THREADID threadId) {
	uint64_t addr = reinterpret_cast<uint64_t>( addrP);

	if (addr > 0) {
		PIN_GetLock(&lock, 1);

		lockMap[addr]++;
		if (lockMap[addr] == 2)
			myfile << "*********ALERT:2" << endl;

		event newEvent;
		newEvent.threadId = threadId;
		newEvent.value = addr;
		newEvent.kind = 2;

		eventList.push_back(newEvent);
		myfile << "thread " << threadId << " entered pthread_mutex_lock "
				<< addr << endl;
		PIN_ReleaseLock(&lock);
	}

}

VOID AfterWait(VOID* addrP, THREADID threadId) {
	uint64_t addr = reinterpret_cast<uint64_t>( addrP);

	PIN_GetLock(&lock, 1);

	lockMap[addr]++;
	event newEvent;
	newEvent.threadId = threadId;
	newEvent.value = addr;
	newEvent.kind = 2;

	myfile << "thread " << threadId << " entered pthread_mutex_lock " << addr
			<< endl;

	eventList.push_back(newEvent);
	PIN_ReleaseLock(&lock);

}

VOID BeforeWait(VOID* addrP, THREADID threadId) {
	uint64_t addr = reinterpret_cast<uint64_t>( addrP);

	PIN_GetLock(&lock, 1);

	lockMap[addr]--;
	event newEvent;
	newEvent.threadId = threadId;
	newEvent.value = (int) addr;
	newEvent.kind = 3;

	myfile << "thread " << threadId << " leaved pthread_mutex_lock " << addr
			<< endl;
	eventList.push_back(newEvent);

	PIN_ReleaseLock(&lock);

}

VOID BeforeUnlock(VOID* addrP, THREADID threadId) {
	uint64_t addr = reinterpret_cast<uint64_t>( addrP);
	if (addr > 0) {
		PIN_GetLock(&lock, 1);
		event newEvent;
		newEvent.threadId = threadId;
		newEvent.value = (int) addr;
		newEvent.kind = 3;

		lockMap[addr]--;
		if (lockMap[addr] == -1)
			myfile << "*********ALERT:-1" << endl;

		eventList.push_back(newEvent);
		myfile << "thread " << threadId << " leaved pthread_mutex_lock "
				<< addr << endl;
		PIN_ReleaseLock(&lock);

	}

	//m
	/*
	 map<int,address>::iterator it;


	 for ( it=addressMap.begin() ; it != addressMap.end(); it++ )
	 {
	 set<int> lockSet= (*it).second.lockSet;

	 set<int>::iterator it= lockSet.find(threadId);

	 if(it!= lockSet.end())
	 {
	 lockSet.insert(value);
	 }

	 }

	 */
}

VOID CheckHappensBefore(ADDRINT ip, ADDRINT addr, info *info1, info *info2,
		bool isWrite) {
	if (info1->owner != info2->owner) {
		int i;
		set<int> *LS = &(info1->LS);
		set<int>::iterator it;
		for (i = info1->pos; i < info2->pos; i++) {
			int type = eventList[i].kind;
			THREADID threadId = eventList[i].threadId;
			int value = eventList[i].value;
			switch (type) {
			case 1:

			case 2:

				it = (*LS).find(value);
				if (it != (*LS).end()) {
					(*LS).insert(threadId);
				}

				break;
			case 0:

			case 3:
				it = (*LS).find(threadId);
				if (it != (*LS).end()) {

					(*LS).insert(value);
				}
				break;
			}

		}
		it = (*LS).find(info2->owner);
		if (!((*LS).empty()) && it == (*LS).end()) {
			PIN_LockClient();

			INT32 line = 0;
			string file;

			PIN_GetSourceLocation((ADDRINT) ip, NULL, &line, &file);

			myfile << "DataRace! Threadid: " << info2->owner
					<< " Memory Address: " << addr << "  isWrite:" << isWrite
					<< endl;
			myfile << "DEBUG " << ip << "  " << file << "  " << line << endl;
			myfile << endl << endl;
			PIN_UnlockClient();

		}
	}

}

VOID MemWrite(ADDRINT ip, ADDRINT addr, THREADID threadId) {

	if (!(threadId < totalThreads))
		return;
	map<ADDRINT, address>::iterator it = addressMap.find(addr);

	//myfile<<addr<<"  "<< threadId<<endl;


	info newInfo;

	set<int> LS;
	LS.insert(threadId);
	newInfo.owner = threadId;
	newInfo.pos = eventList.size();
	newInfo.LS = LS;

	PIN_GetLock(&lock, 1);
	if (it == addressMap.end()) {
		map<THREADID, info> readNewInfo;
		addressMap[addr].writeInfo = newInfo;
		addressMap[addr].readInfo = readNewInfo;
	} else {
		map<THREADID, info>::iterator it;
		for (it = addressMap[addr].readInfo.begin(); it
				!= addressMap[addr].readInfo.end(); it++) {
			CheckHappensBefore(ip, addr, &((*it).second), &newInfo, true);
			addressMap[addr].readInfo.erase(it);

		}

		if (addressMap[addr].writeInfo.owner != -1) {
			CheckHappensBefore(ip, addr, &(addressMap[addr].writeInfo),
					&newInfo, true);
		}
		addressMap[addr].writeInfo = newInfo;

	}
	PIN_ReleaseLock(&lock);

}

VOID MemRead(ADDRINT ip, ADDRINT addr, THREADID threadId) {

	if (!(threadId < totalThreads))
		return;
	map<ADDRINT, address>::iterator it = addressMap.find(addr);

	//myfile<<addr<<"  "<< threadId<< " reads"<<endl;
	info newInfo;

	set<int> LS;
	LS.insert(threadId);
	newInfo.owner = threadId;
	newInfo.pos = eventList.size();
	newInfo.LS = LS;

	PIN_GetLock(&lock, 1);
	if (it == addressMap.end()) {
		addressMap[addr].readInfo[threadId] = newInfo;
		info wInfo;
		wInfo.owner = -1;

		addressMap[addr].writeInfo = wInfo;
	} else {
		if (addressMap[addr].writeInfo.owner != -1) {

			CheckHappensBefore(ip, addr, &(addressMap[addr].writeInfo),
					&newInfo, false);

		}

		addressMap[addr].readInfo[threadId] = newInfo;

	}
	PIN_ReleaseLock(&lock);

}

//====================================================================
// Instrumentation Routines
//====================================================================


VOID Trace(TRACE trace, VOID * val) {
	if (!filter.SelectTrace(trace))
		return;

	for (BBL bbl = TRACE_BblHead(trace); BBL_Valid(bbl); bbl = BBL_Next(bbl)) {
		for (INS ins = BBL_InsHead(bbl); INS_Valid(ins); ins = INS_Next(ins)) {
			if (!INS_IsStackRead(ins) && !INS_IsStackWrite(ins)) {

				if (INS_IsMemoryRead(ins)) {
					INS_InsertPredicatedCall(ins, IPOINT_BEFORE,
							(AFUNPTR) MemRead, IARG_INST_PTR,
							IARG_MEMORYREAD_EA, IARG_THREAD_ID, IARG_END);

				}
				if (INS_HasMemoryRead2(ins)) {
					INS_InsertPredicatedCall(ins, IPOINT_BEFORE,
							(AFUNPTR) MemRead, IARG_INST_PTR,
							IARG_MEMORYREAD2_EA, IARG_THREAD_ID, IARG_END);

				}
				if (INS_IsMemoryWrite(ins)) {

					INS_InsertPredicatedCall(ins, IPOINT_BEFORE,
							(AFUNPTR) MemWrite, IARG_INST_PTR,
							IARG_MEMORYWRITE_EA, IARG_THREAD_ID, IARG_END);

				}
			}

		}
	}
}

// This routine is executed for each image.
VOID ImageLoad(IMG img, VOID *) {

	for (SEC sec = IMG_SecHead(img); SEC_Valid(sec); sec = SEC_Next(sec)) {
		for (RTN rtn = SEC_RtnHead(sec); RTN_Valid(rtn); rtn = RTN_Next(rtn)) {
			//myfile<< RTN_Name(rtn)<< endl;
		}
	}
	RTN rtn = RTN_FindByName(img, "pthread_mutex_lock");

	if (RTN_Valid(rtn)) {
		RTN_Open(rtn);

		RTN_InsertCall(rtn, IPOINT_AFTER, AFUNPTR(AfterLock),
				IARG_FUNCARG_ENTRYPOINT_VALUE, 0, IARG_THREAD_ID, IARG_END);

		RTN_Close(rtn);
	}

	rtn = RTN_FindByName(img, "pthread_mutex_unlock");

	if (RTN_Valid(rtn)) {
		RTN_Open(rtn);

		RTN_InsertCall(rtn, IPOINT_BEFORE, AFUNPTR(BeforeUnlock),
				IARG_FUNCARG_ENTRYPOINT_VALUE, 0, IARG_THREAD_ID, IARG_END);

		RTN_Close(rtn);
	}

	// pthread_cond_wait@@GLIBC_2.3.2
	rtn = RTN_FindByName(img, "pthread_cond_wait@@GLIBC_2.3.2");

	if (RTN_Valid(rtn)) {
		RTN_Open(rtn);

		RTN_InsertCall(rtn, IPOINT_BEFORE, AFUNPTR(BeforeWait),
				IARG_FUNCARG_ENTRYPOINT_VALUE, 1, IARG_THREAD_ID, IARG_END);

		RTN_Close(rtn);
	}

	//pthread_cond_wait@@GLIBC_2.3.2
	rtn = RTN_FindByName(img, "pthread_cond_wait@@GLIBC_2.3.2");

	if (RTN_Valid(rtn)) {
		RTN_Open(rtn);

		RTN_InsertCall(rtn, IPOINT_AFTER, AFUNPTR(AfterWait),
				IARG_FUNCARG_ENTRYPOINT_VALUE, 1, IARG_THREAD_ID, IARG_END);

		RTN_Close(rtn);
	}

	/*
	 rtn = RTN_FindByName(img, "__pthread_create_2_1");

	 if ( RTN_Valid( rtn ))
	 {

	 RTN_Open(rtn);

	 RTN_InsertCall(rtn, IPOINT_BEFORE, AFUNPTR(BeforeThreadCreate),  IARG_FUNCARG_ENTRYPOINT_VALUE,0, IARG_THREAD_ID, IARG_END);

	 RTN_Close(rtn);
	 }
	 */
	//__pthread_create_2_1
	rtn = RTN_FindByName(img, "__pthread_create_2_1");

	if (RTN_Valid(rtn)) {

		RTN_Open(rtn);

		RTN_InsertCall(rtn, IPOINT_AFTER, AFUNPTR(AfterThreadCreate),
				IARG_FUNCARG_ENTRYPOINT_VALUE, 0, IARG_THREAD_ID, IARG_END);

		RTN_Close(rtn);
	}

	rtn = RTN_FindByName(img, "pthread_join");

	if (RTN_Valid(rtn)) {

		RTN_Open(rtn);

		RTN_InsertCall(rtn, IPOINT_AFTER, AFUNPTR(AfterThreadJoin),
				IARG_FUNCARG_ENTRYPOINT_VALUE, 0, IARG_THREAD_ID, IARG_END);

		RTN_Close(rtn);
	}

}

// This routine is executed once at the end.
VOID Fini(INT32 code, VOID *v) {
	myfile.close();

}

/* ===================================================================== */
/* Print Help Message                                                    */
/* ===================================================================== */

INT32 Usage() {
	PIN_ERROR("This Pintool prints a trace of pthread calls in the guest application\n"
			+ KNOB_BASE::StringKnobSummary() + "\n");
	return -1;
}

/* ===================================================================== */
/* Main                                                                  */
/* ===================================================================== */

int main(INT32 argc, CHAR **argv) {

	myfile.open(KnobOutputFile.Value().c_str());

	// Initialize the pin lock
	PIN_InitLock(&lock);

	PIN_InitSymbols();

	// Initialize pin
	if (PIN_Init(argc, argv))
		return Usage();

	// Register ImageLoad to be called when each image is loaded.
	TRACE_AddInstrumentFunction(Trace, 0);
	IMG_AddInstrumentFunction(ImageLoad, 0);

	// Register Fini to be called when the application exits
	PIN_AddFiniFunction(Fini, 0);
	filter.Activate();

	// Never returns*/
	PIN_StartProgram();

	return 0;
}

