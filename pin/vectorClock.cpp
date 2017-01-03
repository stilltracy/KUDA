 

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


#include <vector>
#include <ctime>
#include <map>

#include "pin.H"
#include "instlib.H"
using namespace INSTLIB;
using namespace std;

KNOB<string> KnobOutputFile(KNOB_MODE_WRITEONCE, "pintool","o", "output", "specify output file name");



typedef struct
{
  vector<uint64_t> readVector;
  vector<uint64_t> writeVector;
  
}address;

map<uint64_t, address> addressMap;  

map<uint64_t, vector<uint64_t> > lockVectors;

vector<vector<uint64_t> > threadVectors;



ofstream myfile;
unsigned int totalThreads=0;
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


VOID EmitMem(VOID * ea, INT32 size);


VOID ThreadStart(THREADID threadId, CONTEXT *ctxt, INT32 flags, VOID *v)
{

  PIN_GetLock(&lock, 1);  
 
  totalThreads++;
  vector <uint64_t> clockVector;
  
  for(unsigned int i=0; i<totalThreads; i++)
  {
    if(i== threadId)
    {
      clockVector.push_back(1);
    }
    else
    {
      clockVector.push_back(0);
  
    }
  }
  
  threadVectors.push_back(clockVector);
  PIN_ReleaseLock(&lock);
  
  myfile<< "thread " << threadId <<" starts"<<endl;
  
}



VOID ThreadFini(THREADID threadId, const CONTEXT *ctxt, INT32 code, VOID *v)
{
  myfile<< "thread " << threadId <<" ends"<< endl;
}

VOID BeforeLock( VOID* addrP, THREADID threadId )
{ 
 
  uint64_t addr= reinterpret_cast<uint64_t>(addrP );
  if(addr>0)
  {
  
  PIN_GetLock(&lock, 1);
  unsigned  int size1=threadVectors[threadId].size();
  unsigned  int size2=lockVectors[addr].size();
  
  if(!lockVectors[addr].empty())
  {
    unsigned int i;
    for(i=0; i<totalThreads; i++)
    {
      if(i >= size1)
      {
	threadVectors[threadId].push_back(0);
      }
      if(i>= size2)
      {
	lockVectors[addr].push_back(0);
      }
      if(lockVectors[addr].at(i)> threadVectors[threadId].at(i))
      {
	threadVectors[threadId].at(i)= lockVectors[addr].at(i);
      }

    }

  }
  }
  
  PIN_ReleaseLock(&lock);
  
  //myfile<< "thread " << threadId <<" entered pthread_mutex_lock "<< addr << endl;
  
  /*
  map<int,address>::iterator it;
  
  for ( it=addressMap.begin() ; it != addressMap.end(); it++ )
  {
    set<int> lockSet= (*it).second.lockSet;
    
    set<int>::iterator it= lockSet.find(value);
    
    if(it!= lockSet.end())
    {
      lockSet.insert(threadId);
}

}  
// myfile<< "thread " << threadId <<" entered pthread_mutex_lock "<< addr << endl;

*/
  
}







VOID BeforeUnlock( VOID* addrP, THREADID threadId)
{
  
  uint64_t addr=reinterpret_cast<uint64_t>( addrP);
  if(addr>0)
  {
  PIN_GetLock(&lock, 1);
  unsigned int size1=threadVectors[threadId].size();
  unsigned int size2=lockVectors[addr].size();
  

    unsigned int i;
    for(i=0; i<totalThreads; i++)
    {
      if(i >= size1)
      {
	threadVectors[threadId].push_back(0);
      }
      if(i>= size2)
      {
	lockVectors[addr].push_back(0);
      }
      
      lockVectors[addr].at(i)= threadVectors[threadId].at(i);
    }
  threadVectors[threadId].at(threadId)= threadVectors[threadId].at(threadId)+1;
  PIN_ReleaseLock(&lock);
  }
 
} 

bool checkVectors(vector<uint64_t> v1, vector<uint64_t> v2)
{
  unsigned int size1=v1.size();
  unsigned int size2=v2.size();
  

    unsigned int i;
    for(i=0; i<totalThreads; i++)
    {
      if(i >= size1)
      {
	v1.push_back(0);
      }
      if(i>= size2)
      {
	v2.push_back(0);
      }
      
      if(v1[i]>v2[i]){
	return false;
      }
	
    }
  
    return true;
  
}



VOID MemAccess(VOID* ip, VOID* add,INT32 size,THREADID threadId, bool isWrite,bool isPrefetch)
{
  uint64_t addr= reinterpret_cast<uint64_t>( add);
  unsigned int i;
  
  map< uint64_t, address >::iterator it;
  
  it= addressMap.find(addr);
  if(it == addressMap.end())
  {
    
    
      address tmp;
      vector<uint64_t> realVector;
      vector<uint64_t> tmpVector;
      
      for(i=0;i<totalThreads;i++)
      {
	if(threadId==i)
	{
	  realVector.push_back(threadVectors[threadId].at(threadId));
	}
	else
	{
	  realVector.push_back(0);
      
	}
      }
      if(isWrite)
      {
	tmp.writeVector= realVector;
	tmp.readVector= tmpVector;
      }
      else
      {
      	tmp.writeVector= tmpVector;
	tmp.readVector= realVector;
      }
      
      addressMap[addr]= tmp;
  }
  else if( it!= addressMap.end())
  {
    if(isWrite)
    {
      
      unsigned int size=addressMap[addr].writeVector.size();
      for(i=0;i<totalThreads;i++)
      {
	
	if(i>=size)
	{
	  addressMap[addr].writeVector.push_back(0);
	}
	if(threadId==i)
	{
	  addressMap[addr].writeVector[i]= threadVectors[threadId].at(threadId);
	}

      }
      
      if(!checkVectors(addressMap[addr].writeVector,threadVectors[threadId]) || !checkVectors(addressMap[addr].readVector,threadVectors[threadId]) )
      {
	    PIN_LockClient();
	    
	    
	    INT32 line=0;
	    string file;
	    
	    PIN_GetSourceLocation((ADDRINT)ip, NULL, &line, &file);
	    
	    
	    myfile<<"DataRace! Threadid: " << threadId << " Memory Address: "<<  addr <<endl;
	    myfile<< "DEBUG " << ip << "  " <<file << "  " << line <<endl <<endl;
	    //myfile<< "TRAAACE "  <<threadId <<"  last thread: "   << lastThreadid <<"  lock size: " <<size  << "  addr :" << addr;
	    //myfile<< "last size" << it->second.lockSet.size()<< "  " <<  isWrite<< endl;
	    
	    PIN_UnlockClient();	  
      
      
      }
    }
    else
    {
      unsigned int size=addressMap[addr].readVector.size();
      for(i=0;i<totalThreads;i++)
      {
	
	if(i>=size)
	{
	  addressMap[addr].readVector.push_back(0);
	}
	if(threadId==i)
	{
	  addressMap[addr].readVector[i]=threadVectors[threadId].at(threadId);
	}

      }
     
	
        if(  !(checkVectors(addressMap[addr].writeVector,threadVectors[threadId])) )
      {
	    PIN_LockClient();
	    
	    
	    INT32 line=0;
	    string file;
	    
	    PIN_GetSourceLocation((ADDRINT)ip, NULL, &line, &file);
	    
	    
	    myfile<<"DataRace! Threadid: " << threadId << " Memory Address: "<<  addr <<endl;
	    myfile<< "DEBUG " << ip << "  " <<file << "  " << line <<endl <<endl;
	    //myfile<< "TRAAACE "  <<threadId <<"  last thread: "   << lastThreadid <<"  lock size: " <<size  << "  addr :" << addr;
	    //myfile<< "last size" << it->second.lockSet.size()<< "  " <<  isWrite<< endl;
	    
	    PIN_UnlockClient();	  
      
      
      }
    }
  
  
  
  }
  
  

   
  
  
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



VOID TestMalloc(VOID* ip, VOID* addrP, THREADID threadId, ADDRINT stkPtr )
{
  uint64_t addr= reinterpret_cast<uint64_t>( addrP);
   if(addr==134521044)
  {
    myfile<<"malloc"<<endl;
  }

  //myfile<< (ADDRINT) addr << "    "  << threads[threadId].stackBase<< "     " <<  stkPtr<< endl;
  
  //if ( (((ADDRINT) addr < threads[threadId].stackBase )&&  ((ADDRINT) addr> stkPtr )))
  //{
  /*
    PIN_GetLock(&lock,1);
    
  map<int, address >::iterator it;
  
  it= addressMap.find((int)addr);
  if(it == addressMap.end())
  {
    info writeNewInfo;
    writeNewInfo.owner=-1;
    map<int,info> readNewInfo;
  
    addressMap[(int)addr].readInfo=readNewInfo;
    addressMap[(int)addr].writeInfo=writeNewInfo;
    
  }
     PIN_ReleaseLock(&lock);   
    //myfile<< "MALLOC  " << addr << "  " <<threadId << endl;
    */
    //}
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
      
      if (INS_IsMemoryRead(ins))
      {
	if(!INS_IsStackRead(ins)){
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
      }
      
      if (INS_HasMemoryRead2(ins))
      {
	
	if(!INS_IsStackRead(ins)){
	  
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

  RTN rtn;
  
  rtn = RTN_FindByName(img, "__pthread_mutex_lock");
  
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
  
  
  
  
}




// This routine is executed once at the end.
VOID Fini(INT32 code, VOID *v)
{
  
myfile<< ( ( clock() - start ) / (double)CLOCKS_PER_SEC ) <<endl;
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
  // Initialize the pin lock
  PIN_InitLock(&lock);
  
  // Initialize pin
  if (PIN_Init(argc, argv)) return Usage();
  PIN_InitSymbols();
  
  
  myfile.open (KnobOutputFile.Value().c_str());
  
  
  // Register ImageLoad to be called when each image is loaded.
  TRACE_AddInstrumentFunction(Trace, 0);
  IMG_AddInstrumentFunction(ImageLoad, 0);
  
  
  
  
  
  // Register Analysis routines to be called when a thread begins/ends
  PIN_AddThreadStartFunction(ThreadStart, 0);
  PIN_AddThreadFiniFunction(ThreadFini, 0);
  
  // Register Fini to be called when the application exits
  PIN_AddFiniFunction(Fini, 0);
  filter.Activate();
  
  // Never returns
  PIN_StartProgram();
  
  return 0;
}






















/*

    
    
    
    
    VOID RecordMem(VOID * ip, CHAR r, VOID * addr, INT32 size, BOOL isPrefetch)
    {
      myfile << ip << ": " << r << " " << setw(2+2*sizeof(ADDRINT)) << addr << " "
      << dec << setw(2) << size << " "
      << hex << setw(2+2*sizeof(ADDRINT));
      if (!isPrefetch)
      EmitMem(addr, size);
      myfile << endl;
      }
      
      
      */


















/*
UINT32 memOperands = INS_MemoryOperandCount(ins);
//INS_InsertCall(ins, IPOINT_BEFORE, AFUNPTR(TestDebug), IARG_INST_PTR,IARG_THREAD_ID,IARG_END);

// Iterate over each memory operand of the instruction.






for (UINT32 memOp = 0; memOp < memOperands; memOp++)
{
  
  if (INS_MemoryOperandIsRead(ins, memOp))
  {
    if(!INS_IsStackRead(ins))
    INS_InsertPredicatedCall(
    ins, IPOINT_BEFORE, (AFUNPTR)MemAccess,
    IARG_INST_PTR,
    IARG_MEMORYOP_EA, memOp,IARG_MEMORYREAD_SIZE,IARG_THREAD_ID,IARG_BOOL, false,
    IARG_END);
    }
    // Note that in some architectures a single memory operand can be 
    // both read and written (for instance incl (%eax) on IA-32)
    // In that case we instrument it once for read and once for write.
    
    if (INS_MemoryOperandIsWritten(ins, memOp))
    {
      if(!INS_IsStackWrite(ins))
      INS_InsertPredicatedCall(
      ins, IPOINT_BEFORE, (AFUNPTR)MemAccess,
      IARG_INST_PTR,
      IARG_MEMORYOP_EA, memOp,IARG_MEMORYWRITE_SIZE,IARG_THREAD_ID,IARG_BOOL, true,
      IARG_END);
      }
      
      }
      */







/*


  */
