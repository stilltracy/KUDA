
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
#include <iomanip>
#include <fstream>
#include <set>
#include <list>
#include <vector>
#include <cstdlib>
#include <map>
#include <algorithm>
#include <ctime>
#include "math.h"
#include "pin.H"
#include "instlib.H"
#include "stdlib.h"
#include "sys/time.h"
using namespace INSTLIB;
using namespace std;

KNOB<string> KnobOutputFile(KNOB_MODE_WRITEONCE, "pintool","o", "output", "specify output file name");


typedef struct
{
  int owner;
  int alock;
  set<int> LS;
  int pos;
}info;


typedef struct
{
  map<int, info > readInfo;
  info writeInfo;
}address;

typedef struct
{
  int threadId;
  int value; // for acq and release , it is lock value.  for fork and join, it is child thread id (u)
  int kind; // 0= acq lock 1= release lock 2= fork 3= join
}event;

map<int, address> addressMap;  
vector<event> eventList;
int b;

 //map from os thread id to threadid
map<int, int> osThreadMap;
 //map from pthread* value to threadId map
map<int, int> pthreadAddrMap;
map<UINT32, int> pthreadValMap; //map  from pthread value to threadId map



    struct timeval tv;
   struct timezone tz;


ofstream myfile;
int totalThreads=1;
clock_t start;


FILTER_LIB filter;
PIN_LOCK lock;
int isLock=0;
VOID * WriteAddr;
INT32 WriteSize;





// Note that opening a file in a callback is only supported on Linux systems.
// See buffer-win.cpp for how to work around this issue on Windows.
//
// This routine is executed every time a thread is created.


UINT32 EmitMem(VOID * ea, INT32 size)
{   
  
  
  
  UINT32 val=0;
  
  switch(size)
  {
    
    case 1:
      val= static_cast<UINT32>(*static_cast<UINT8*>(ea));
      break;
      
    case 2:
      val= *static_cast<UINT16*>(ea);
      break;
      
    case 4:
      val= *static_cast<UINT32*>(ea);
      break;
      
    case 8:
      val= *static_cast<UINT64*>(ea);
      break;
  }
  

  return val; 
}


VOID ThreadStart(THREADID threadId, CONTEXT *ctxt, INT32 flags, VOID *v)
{
  event newEvent;
  newEvent.threadId= osThreadMap[PIN_GetParentTid()];
  newEvent.value= threadId;
  newEvent.kind= 0;
  
  
  PIN_GetLock(&lock, 1);  
  osThreadMap[PIN_GetTid()]=threadId;
  eventList.push_back(newEvent);
  PIN_ReleaseLock(&lock);
  
  myfile<< "thread " << threadId <<" starts"<<endl;
  
}


VOID ThreadCreate(VOID* addr, THREADID threadId)
{
  
  
  PIN_GetLock(&lock, 1);
  pthreadAddrMap[(int) addr]= totalThreads;
  b= (int) addr;
  totalThreads++;
  PIN_ReleaseLock(&lock);
  myfile<< "thread "<<  totalThreads-1 << " is created by "<< threadId <<endl;
  
  
}

VOID ThreadJoin(VOID* addr, THREADID threadId)
{
  event newEvent;
  newEvent.threadId= threadId ;
  newEvent.value= pthreadValMap[(UINT32) addr];
  newEvent.kind= 1;
  
  PIN_GetLock(&lock, 1);
  eventList.push_back(newEvent);
  PIN_ReleaseLock(&lock);
  
  myfile<< "thread " << newEvent.value << " is joined by "<< threadId <<endl;
  
}

VOID ThreadFini(THREADID threadId, const CONTEXT *ctxt, INT32 code, VOID *v)
{
  myfile<< "thread " << threadId <<" ends"<< endl;
}

VOID BeforeLock( VOID* addrP, THREADID threadId )
{ 
  int addr=(int) addrP;
  
  
  if(addr>0)
  {
    event newEvent;
    newEvent.threadId= threadId ;
    newEvent.value= addr;
    newEvent.kind= 2;
   
    
    PIN_GetLock(&lock, 1);
    eventList.push_back(newEvent);
    PIN_ReleaseLock(&lock);
  }
  
}







VOID BeforeUnlock( VOID* addrP, THREADID threadId)
{
  int addr=(int) addrP;
  if(addr>0){
    event newEvent;
    newEvent.threadId= threadId ;
    newEvent.value= (int) addr;
    newEvent.kind= 3;
    
    PIN_GetLock(&lock, 1);
    eventList.push_back(newEvent);
    PIN_ReleaseLock(&lock);
    
  }
  //myfile<< "thread " << threadId <<" leaved pthread_mutex_lock "<< addr << endl;
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

VOID CheckHappensBefore(VOID *ip,VOID *addr, info *info1, info *info2)
{
  if(info1->owner != info2->owner)
  {
    int i;
    set<int> *LS= &(info1->LS);
    set<int>::iterator it;
    for(i=info1->pos; i< info2->pos; i++)
    {
      int type= eventList[i].kind;
      int threadId=eventList[i].threadId;
      int value=eventList[i].value;
      switch(type)
      {
	case 1:
	  
	case 2:
	  
	  it= (*LS).find(value);
	  if(it!= (*LS).end())
	  {
	    (*LS).insert(threadId);
	  }
	  
	  
	  break;
	case 0:
	  
	case 3:
	  it= (*LS).find(threadId);
	  if(it!= (*LS).end())
	  {
	    
	    (*LS).insert(value);
	  }
	  break;
      }
      
      
    }
    it= (*LS).find(info2->owner);
    if( !((*LS).empty()) && it== (*LS).end() )
    {
      PIN_LockClient();
      
      
      INT32 line=0;
      string file;
      
      PIN_GetSourceLocation((ADDRINT)ip, NULL, &line, &file);
      
      
      myfile<<"DataRace! Threadid: " << info2->owner << " Memory Address: "<< addr <<endl;
      myfile<< "DEBUG " << ip << "  " <<file << "  " << line << endl;
      myfile<<endl<<endl;
      PIN_UnlockClient();
      
      
      
    }    
  }
  
  
  
  
}



VOID TracePthread(VOID* ip, VOID* add,INT32 size,THREADID threadId, bool isWrite)
{
  PIN_GetLock(&lock, 1);
  map<int,int>::iterator it;

  for ( it=pthreadAddrMap.begin() ; it != pthreadAddrMap.end(); it++ )
  {
    
  if((int)add== it->first)
  {
    pthreadValMap[EmitMem(add,size)]= it->second;
  }
  
  
  }

  
  PIN_ReleaseLock(&lock);
}





VOID MemAccess(VOID* ip, VOID* add,INT32 size,THREADID threadId, bool isWrite,bool isPrefetch)
{
  int addr= (int) add;
  
  
  
  map<int, address >::iterator it;
  
  it= addressMap.find(addr);
  
  
  info newInfo;
  
  set<int> LS;
  LS.insert(threadId);
  newInfo.owner= threadId;
  newInfo.pos= eventList.size();
  newInfo.LS= LS;
  
   PIN_GetLock(&lock, 1);
    
  if(it == addressMap.end() )
  { 
    
    
    map<int,info> readNewInfo;
    if(isWrite)
    {
      addressMap[addr].writeInfo=newInfo;
    addressMap[addr].readInfo=readNewInfo;
    }
    else
    {
      addressMap[addr].readInfo[threadId]=newInfo;
      info writeNewInfo;
      writeNewInfo.owner=-1;
      addressMap[addr].writeInfo=writeNewInfo;
    }
    
  }
  else if(it!= addressMap.end())
  {
    
    if(!isWrite)
    {
      
      if(addressMap[addr].writeInfo.owner != -1)
      {
	
	CheckHappensBefore(ip,add, &(addressMap[addr].writeInfo), &newInfo ); 
	
      }
      
      addressMap[addr].readInfo[threadId]= newInfo;
    }
    else
    {
      map<int,info>::iterator it;
      for ( it=addressMap[addr].readInfo.begin() ; it != addressMap[addr].readInfo.end(); it++ )
      {
	CheckHappensBefore(ip, add,&((*it).second), &newInfo );
	addressMap[addr].readInfo.erase(it);
	
      } 
      
      if(addressMap[addr].writeInfo.owner != -1)
      {
	CheckHappensBefore(ip,add, &(addressMap[addr].writeInfo), &newInfo  );
      }
      
      
      addressMap[addr].writeInfo= newInfo;
    }
    
  }
  
  PIN_ReleaseLock(&lock);   
  
  
  
  
}






VOID WriteAddrSize(VOID * addr, INT32 size)
{
  WriteAddr = addr;
  WriteSize = size;
}



VOID MemWrite(VOID * ip,THREADID threadId)
{
  MemAccess(ip, WriteAddr, WriteSize, threadId,true, false);
}




//====================================================================
// Instrumentation Routines
//====================================================================


VOID Trace(TRACE trace, VOID * val)
{
  if (!filter.SelectTrace(trace))
    return;
  
  for (BBL bbl = TRACE_BblHead(trace); BBL_Valid(bbl); bbl = BBL_Next(bbl))
  {
    for (INS ins = BBL_InsHead(bbl); INS_Valid(ins); ins = INS_Next(ins))
    {
      if(INS_IsStackRead(ins))
      {
	INS_InsertPredicatedCall(
	ins, IPOINT_BEFORE, (AFUNPTR)TracePthread,
				 IARG_INST_PTR,
				 IARG_MEMORYREAD_EA,
				 IARG_MEMORYREAD_SIZE,
				 IARG_THREAD_ID,
				 IARG_BOOL, false,
				 IARG_END);
				 
				 
      }
      
      else if (INS_IsMemoryRead(ins))
      {
	INS_InsertPredicatedCall(
	ins, IPOINT_BEFORE, (AFUNPTR)MemAccess,
				 IARG_INST_PTR,
				 IARG_MEMORYREAD_EA,
				 IARG_MEMORYREAD_SIZE,
				 IARG_THREAD_ID,
				 IARG_BOOL, false,				  
				 IARG_BOOL, INS_IsPrefetch(ins),
				 IARG_END);
				 
				 
      }
      
      if (INS_HasMemoryRead2(ins))
      {
	
	
	
	INS_InsertPredicatedCall(
	ins, IPOINT_BEFORE, (AFUNPTR)MemAccess,
				 IARG_INST_PTR,
				 IARG_MEMORYREAD2_EA,
				 IARG_MEMORYREAD_SIZE,
				 IARG_THREAD_ID,
				 IARG_BOOL, false,				  
				 IARG_BOOL, INS_IsPrefetch(ins),
				 IARG_END);
				 
      }
      
      // instruments stores using a predicated call, i.e.
      // the call happens iff the store will be actually executed
      if (INS_IsMemoryWrite(ins))
      {
	
	if(!INS_IsStackWrite(ins)){
	  INS_InsertPredicatedCall(
	  ins, IPOINT_BEFORE, (AFUNPTR)WriteAddrSize,
				   IARG_MEMORYWRITE_EA,
				   IARG_MEMORYWRITE_SIZE,
				   IARG_END);
				   
				   if (INS_HasFallThrough(ins))
				   {
				     INS_InsertCall(
				     ins, IPOINT_AFTER, (AFUNPTR)MemWrite,
						    IARG_INST_PTR,
						    IARG_THREAD_ID,
						    IARG_END);
				   }
				   if (INS_IsBranchOrCall(ins))
				   {
				     INS_InsertCall(
				     ins, IPOINT_TAKEN_BRANCH, (AFUNPTR)MemWrite,
						    IARG_INST_PTR,
						    IARG_THREAD_ID,
						    IARG_END);
				   }
				   
	}
      }
      
      
      
    }
  }
}

// This routine is executed for each image.
VOID ImageLoad(IMG img, VOID *)
{
  myfile<<IMG_Name(img)<<endl;
  
  RTN rtn = RTN_FindByName(img, "__pthread_mutex_lock");
  
  if ( RTN_Valid( rtn ) )
  {
    RTN_Open(rtn);
    
    RTN_InsertCall(rtn, IPOINT_BEFORE, AFUNPTR(BeforeLock),
		   IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
		   IARG_THREAD_ID, IARG_END);
		   
		   RTN_Close(rtn);
  }
  
  
  rtn = RTN_FindByName(img, "__pthread_mutex_unlock");
  
  if ( RTN_Valid( rtn ))
  {
    RTN_Open(rtn);
    
    RTN_InsertCall(rtn, IPOINT_BEFORE, AFUNPTR(BeforeUnlock),
		   IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
		   IARG_THREAD_ID, IARG_END);
		   
		   RTN_Close(rtn);
  }
  
  
  rtn = RTN_FindByName(img, "pthread_join"); 
  
  if ( RTN_Valid( rtn ))
  {
    
    RTN_Open(rtn);
    
    RTN_InsertCall(rtn, IPOINT_BEFORE, AFUNPTR(ThreadJoin),IARG_FUNCARG_ENTRYPOINT_VALUE,0,IARG_THREAD_ID,IARG_END);
    
    RTN_Close(rtn);
  }
  
  rtn = RTN_FindByName(img, "__pthread_create_2_1"); 
  
  if ( RTN_Valid( rtn ))
  {
    
    RTN_Open(rtn);
    
    RTN_InsertCall(rtn, IPOINT_AFTER, AFUNPTR(ThreadCreate),  IARG_FUNCARG_ENTRYPOINT_VALUE,0,IARG_THREAD_ID, IARG_END);
    
    RTN_Close(rtn);
  }
  
}




// This routine is executed once at the end.
VOID Fini(INT32 code, VOID *v)
{
  
    struct timeval tv2;
   struct timezone tz2;
  gettimeofday(&tv2, &tz2);
  myfile<< tv2.tv_usec -tv.tv_usec<<endl;
  myfile.close();
  
}

/* ===================================================================== */
/* Print Help Message                                                    */
/* ===================================================================== */


INT32 Usage()
{
  PIN_ERROR("This Pintool prints a trace of pthread calls in the guest application\n"
  + KNOB_BASE::StringKnobSummary() + "\n");
  return -1;
}

/* ===================================================================== */
/* Main                                                                  */
/* ===================================================================== */

int main(INT32 argc, CHAR **argv)
{


   gettimeofday(&tv, &tz);
  myfile.open (KnobOutputFile.Value().c_str());
  // Initialize the pin lock
 
  //PIN_InitLock(&lock);
  
  // Initialize pin
  if (PIN_Init(argc, argv)) return Usage();
   
  
  PIN_InitSymbols();
  
  
  
  
  // Register ImageLoad to be called when each image is loaded.
  TRACE_AddInstrumentFunction(Trace, 0);
  IMG_AddInstrumentFunction(ImageLoad, 0);
  
  // Register Analysis routines to be called when a thread begins/ends
  PIN_AddThreadStartFunction(ThreadStart, 0);
  PIN_AddThreadFiniFunction(ThreadFini, 0);
  
  // Register Fini to be called when the application exits
  PIN_AddFiniFunction(Fini, 0);
  filter.Activate();
  
  // Never returns*/
  PIN_StartProgram();
  
 

  return 0;
}

