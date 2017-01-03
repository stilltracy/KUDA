
#include <assert.h>
#include <stdio.h>
#include <map>
#include <set>
#include <iostream>
#include <fstream>
#include "pin.H"
#include "portability.H"
#include "instlib.H"

INSTLIB::FILTER_LIB filter;

 
KNOB<string> KnobOutputFile(KNOB_MODE_WRITEONCE, "pintool", "o", "trace.out", "output file");

KNOB<BOOL>   KnobProcessBuffer(KNOB_MODE_WRITEONCE, "pintool", "process_buffs", "1", "process the filled buffers");

KNOB<UINT32> KnobNumBytesInBuffer(KNOB_MODE_WRITEONCE, "pintool", "num_bytes_in_buffer", "1048576", "number of bytes in buffer");



/* Struct of memory reference written to the buffer
 */
struct MEMREF
{
    ADDRINT pc;  // pc (ip) of the instruction doing the memory reference
    ADDRINT ea;  // the address of the memory being referenced
    int type;
};

struct SYNCH
{
	int type;
	ADDRINT value;	
};
  
struct EVENT{
	THREADID tid;
	ADDRINT value; 
	int kind; 
};
// the Pin TLS slot that an application-thread will use to hold the APP_THREAD_BUFFER_HANDLER
// object that it owns
TLS_KEY appThreadRepresentitiveKey;

map<ADDRINT, int> pthreadValMap;
vector<EVENT> eventList;

UINT32 totalBuffersFilled = 0;
UINT64 totalElementsProcessed = 0;
BOOL bufferFull= false;



/* Pin registers per-thread buffer
 */
REG endOfTraceInBufferReg;
REG endOfBufferReg;
REG currentIndex;

THREADID totalThreads = 1;

VOID ThreadCreate(VOID* addr, THREADID tid) {
	EVENT newEvent;
	newEvent.tid = tid;
	newEvent.value = totalThreads;
	newEvent.kind = 3;
	
	eventList.push_back(newEvent);
	
	ADDRINT value;
	PIN_SafeCopy(&value, addr, sizeof(ADDRINT));  // take value from memory
	pthreadValMap[value] = totalThreads;
	totalThreads++;
}

VOID ThreadJoin(ADDRINT addr, THREADID tid) {
	EVENT newEvent;
	newEvent.tid = tid;
	newEvent.value = pthreadValMap[addr];
	newEvent.kind = 4;

	eventList.push_back(newEvent);
}


/*
 * APP_THREAD_REPRESENTITVE
 * Each application thread, creates an object of this class and saves it in it's Pin TLS slot 
 * it manages of the per thread buffer
 */
class APP_THREAD_REPRESENTITVE
{

  public:
    APP_THREAD_REPRESENTITVE(THREADID tid);
    ~APP_THREAD_REPRESENTITVE();
    map<int, SYNCH> synchMap;
    THREADID _myTid;
    
    /*
     * ProcessBuffer 
     */
    void ProcessBuffer(char *end);

    /*
     * Pointer to beginning of the buffer
     */
    char * Begin() { return _buffer; }

    /*
     * Pointer to end of the buffer
     */
    char * End() { return _buffer + KnobNumBytesInBuffer.Value(); }

    UINT32 NumBuffersFilled() {return _numBuffersFilled;}

    UINT32 NumElementsProcessed() {return _numElementsProcessed;}

		
	
	void addMap(int index,int type,ADDRINT value)
	{
		SYNCH newsynch;
		newsynch.type=type;
		newsynch.value=value;
		synchMap[index]= newsynch;
		
	}
	
    /*
     * Analysis routine to record a MEMREF (pc, ea) in the buffer
     */
    static int PIN_FAST_ANALYSIS_CALL RecordMEMREFInBuffer(CHAR * endOfTraceInBuffer, ADDRINT offsetFromEndOfTrace, ADDRINT pc, IARG_TYPE itype, int currentIndex)
    {
    	
        *reinterpret_cast<ADDRINT*>(endOfTraceInBuffer+offsetFromEndOfTrace) = pc;
        *reinterpret_cast<ADDRINT*>(endOfTraceInBuffer+offsetFromEndOfTrace+sizeof(ADDRINT)) =(ADDRINT) itype;
        int type;
        
        if(itype != IARG_MEMORYWRITE_EA)
        {
        	type=5;
        }
        else
        {
        	type=6;
        }
        
        *reinterpret_cast<ADDRINT*>(endOfTraceInBuffer+offsetFromEndOfTrace+sizeof(ADDRINT)+sizeof(int))=type;
        
        if(bufferFull)
        {
        	bufferFull= false;
        	return 0;
        }
        
        return (++currentIndex);
    }
        
    static void UNLOCKING(int currentIndex,ADDRINT value,THREADID tid)
    {
  
    	if((int)value>0)
    	{
    		
    		APP_THREAD_REPRESENTITVE * appThreadRepresentitive 
            	= static_cast<APP_THREAD_REPRESENTITVE*>(PIN_GetThreadData(appThreadRepresentitiveKey, tid));
      		//std::cout<< currentIndex << std::endl;
      		appThreadRepresentitive->addMap(currentIndex,1,value);
      		
      		
      		EVENT newEvent;
			newEvent.tid = tid;
			newEvent.value = value;
			newEvent.kind = 1;

			eventList.push_back(newEvent);
      		
    	}
      // appThreadRepresentitive->a=2;
     //  appThreadRepresentitive->addMap(currentIndex,value);
       
    }
    static void LOCKING(int currentIndex,ADDRINT value,THREADID tid)
    {
  
    	if((int)value>0)
    	{
    		
    		APP_THREAD_REPRESENTITVE * appThreadRepresentitive 
            	= static_cast<APP_THREAD_REPRESENTITVE*>(PIN_GetThreadData(appThreadRepresentitiveKey, tid));
      		//std::cout<< currentIndex << std::endl;
      		appThreadRepresentitive->addMap(currentIndex,0,value);
      		
      		EVENT newEvent;
			newEvent.tid = tid;
			newEvent.value = value;
			newEvent.kind = 0;

			eventList.push_back(newEvent);
    	}
      // appThreadRepresentitive->a=2;
     //  appThreadRepresentitive->addMap(currentIndex,value);
       
    }

    /*
	 * Analysis routine called at beginning of each trace - it is the IF part of the IF THEN analysis routine pair
     *
     * endOfPreviousTraceInBuffer :Pointer to next entry in the buffer
     * bufferEnd     :Pointer to end of the buffer
     * totalSizeOccupiedByTraceInBuffer : Number of bytes required by this TRACE
     */
    static ADDRINT PIN_FAST_ANALYSIS_CALL CheckIfNoSpaceForTraceInBuffer(char * endOfPreviousTraceInBuffer, char * bufferEnd, ADDRINT totalSizeOccupiedByTraceInBuffer)
    {
        return (endOfPreviousTraceInBuffer + totalSizeOccupiedByTraceInBuffer >= bufferEnd);
    }

    static char * PIN_FAST_ANALYSIS_CALL BufferFull(char *endOfTraceInBuffer, ADDRINT tid)
    {
        
        APP_THREAD_REPRESENTITVE * appThreadRepresentitive 
            = static_cast<APP_THREAD_REPRESENTITVE*>(PIN_GetThreadData(appThreadRepresentitiveKey, tid));
        appThreadRepresentitive->ProcessBuffer(endOfTraceInBuffer);
        endOfTraceInBuffer = appThreadRepresentitive->Begin();
		bufferFull=true;
		
        return endOfTraceInBuffer;
    }


    static char * PIN_FAST_ANALYSIS_CALL  AllocateSpaceForTraceInBuffer(char * endOfPreviousTraceInBuffer, 
                                                                        ADDRINT totalSizeOccupiedByTraceInBuffer)
    {
        return (endOfPreviousTraceInBuffer + totalSizeOccupiedByTraceInBuffer);
    }

	
  private:
	ofstream file;
    void ResetCurMEMREFElement(char ** curMEMREFElement);
    char * _buffer;  // actual buffer
    UINT32 _numBuffersFilled;
    UINT32 _numElementsProcessed;	
};





APP_THREAD_REPRESENTITVE::APP_THREAD_REPRESENTITVE(THREADID myTid)
{
    _buffer = new char[KnobNumBytesInBuffer.Value()];
    _numBuffersFilled = 0;
    _numElementsProcessed = 0;
	_myTid = myTid;
	
	string filename = KnobOutputFile.Value() + "." + decstr(getpid_portable()) + "." + decstr(myTid);
    // Open file
    file.open(filename.c_str());
    file << hex;
}

APP_THREAD_REPRESENTITVE::~APP_THREAD_REPRESENTITVE()
{
    delete [] _buffer;
}
    

/*
 * Process the buffer
 */
void APP_THREAD_REPRESENTITVE::ProcessBuffer(char *end)
{
	_numBuffersFilled++;
    if (!KnobProcessBuffer)
    {
        return;
    }
    int i=0;
    struct MEMREF * memref =reinterpret_cast<struct MEMREF*>(Begin());
    struct MEMREF * firstMemref =memref;
    
    map<int, SYNCH>::iterator it;
    it= synchMap.begin();
        
    while(memref < reinterpret_cast<struct MEMREF*>(end))
    {
        if (memref->pc!=0)
        {
          if(it!=synchMap.end()&&  (*it).first==i )
           {
           		//std::cout<<(*it).second <<" " << _myTid << std::endl;
           		SYNCH synch=(*it).second;
           		
           		file << synch.type << " "<< synch.value << endl;
           		it++;
           }
           file << memref->type << " "<< memref->ea<< endl;	
           i++;
           firstMemref->pc += memref->pc + memref->ea;
		   memref->pc = 0;
        }
        memref++;
    }
	
	synchMap.clear();
	
    //printf ("numElements %d (full %d   empty %d)\n", i+j, i, j);
    _numElementsProcessed += i;
}

/*
 * Reset the cursor to the beginning of the APP_THREAD_REPRESENTITVE
 */
void APP_THREAD_REPRESENTITVE::ResetCurMEMREFElement(char ** curMEMREFElement)
{
    *curMEMREFElement = Begin();
}





/*
 * Analysis calls that must be inserted at an INS in the trace are recorded in an
 * ANALYSIS_CALL_INFO object
 *
 */
class ANALYSIS_CALL_INFO
{
  public:
      ANALYSIS_CALL_INFO(INS ins, UINT32 offsetFromTraceStartInBuffer, IARG_TYPE itype) : 
      _ins(ins), 
     _offsetFromTraceStartInBuffer(offsetFromTraceStartInBuffer),
     _type (itype)
    {
    }

    void InsertAnalysisCall(INT32 sizeofTraceInBuffer)
    {
        /* the place in the buffer is
           -sizeofTraceInBuffer +  _offsetFromTraceStartInBuffer(of this _ins)
        */
        INS_InsertCall(_ins, IPOINT_BEFORE, AFUNPTR(APP_THREAD_REPRESENTITVE::RecordMEMREFInBuffer), 
                       IARG_FAST_ANALYSIS_CALL, 
                       IARG_REG_VALUE, endOfTraceInBufferReg, 
                       IARG_ADDRINT, ADDRINT(_offsetFromTraceStartInBuffer - sizeofTraceInBuffer), 
                       IARG_INST_PTR, 
                       _type,
                       IARG_REG_VALUE, currentIndex, 
                       IARG_RETURN_REGS, currentIndex,
                       IARG_END);
    }
    
  private:
    INS _ins;
    INT32 _offsetFromTraceStartInBuffer;
    IARG_TYPE _type;
};



    
/*
 * TRACE_ANALYSIS_CALLS_NEEDED
 *
 * is stores a vector of ANALYSIS_CALL_INFO objects
 *
 */
class TRACE_ANALYSIS_CALLS_NEEDED
{

  public:
    TRACE_ANALYSIS_CALLS_NEEDED() : _currentOffsetFromTraceStartInBuffer(0),  _numAnalysisCallsNeeded(0)
    {}
    
    UINT32 NumAnalysisCallsNeeded() const { return _numAnalysisCallsNeeded; }

    UINT32 TotalSizeOccupiedByTraceInBuffer() const { return _currentOffsetFromTraceStartInBuffer; }


    /*
     * Record a call to store an address in the log
     */
    void RecordAnalysisCallNeeded(INS ins, IARG_TYPE itype);

    /*
     * InsertAnalysisCall all the recorded necessary analysis calls into the trace
     */
    void InsertAnalysisCalls();


    
    private:
    
    INT32 _currentOffsetFromTraceStartInBuffer;
    INT32 _numAnalysisCallsNeeded;
    vector<ANALYSIS_CALL_INFO> _analysisCalls;
};

/*
 * We determined all the required instrumentation, insert the calls
 */
void TRACE_ANALYSIS_CALLS_NEEDED::InsertAnalysisCalls()
{
    for (vector<ANALYSIS_CALL_INFO>::iterator c = _analysisCalls.begin(); 
         c != _analysisCalls.end(); 
         c++)
    {
        c->InsertAnalysisCall(TotalSizeOccupiedByTraceInBuffer());
    }
}

/*
 * Record that we need to insert an analysis call to gather the MEMREF info for this ins
 */
void TRACE_ANALYSIS_CALLS_NEEDED::RecordAnalysisCallNeeded(INS ins,  IARG_TYPE itype)
{
    _analysisCalls.push_back(ANALYSIS_CALL_INFO(ins, _currentOffsetFromTraceStartInBuffer,itype));
    _currentOffsetFromTraceStartInBuffer += sizeof(MEMREF);
    _numAnalysisCallsNeeded++;
}
    

void DetermineBBLAnalysisCalls(BBL bbl, TRACE_ANALYSIS_CALLS_NEEDED * traceAnalysisCallsNeeded)
{
    // Log  memory references of the instruction
    for(INS ins = BBL_InsHead(bbl); INS_Valid(ins); ins=INS_Next(ins))
    {
        
        if (!INS_IsStackRead(ins) && !INS_IsStackWrite(ins)) {

        // Log every memory references of the instruction
        	if (INS_IsMemoryRead(ins))
        	{
        		traceAnalysisCallsNeeded->RecordAnalysisCallNeeded(ins, IARG_MEMORYREAD_EA);
        	   
        	}
        	if (INS_IsMemoryWrite(ins))
        	{
        		traceAnalysisCallsNeeded->RecordAnalysisCallNeeded(ins, IARG_MEMORYWRITE_EA);
        	   
        	}
        	if (INS_HasMemoryRead2(ins))
        	{
        		traceAnalysisCallsNeeded->RecordAnalysisCallNeeded(ins, IARG_MEMORYREAD2_EA);
        	   
        	}
    	}

    }
}

void TraceAnalysisCalls(TRACE trace, void *)
{
	
	if (!filter.SelectTrace(trace))
		return;

    TRACE_ANALYSIS_CALLS_NEEDED traceAnalysisCallsNeeded;
    for (BBL bbl = TRACE_BblHead(trace); BBL_Valid(bbl); bbl = BBL_Next(bbl))
    {
        DetermineBBLAnalysisCalls(bbl, &traceAnalysisCallsNeeded);
    }

    // No addresses in this trace
    if (traceAnalysisCallsNeeded.NumAnalysisCallsNeeded() == 0)
    {
        return;
    }
    
    // Now we know how bytes the analysis calls of this trace will insert into the buffer
    //   Each analysis call inserts a MEMREF into the buffer
 
    // APP_THREAD_REPRESENTITVE::CheckIfNoSpaceForTraceInBuffer will determine if there are NOT enough available bytes in the buffer.
    // If there are NOT then it returns TRUE and the BufferFull function is called
    TRACE_InsertIfCall(trace, IPOINT_BEFORE, AFUNPTR(APP_THREAD_REPRESENTITVE::CheckIfNoSpaceForTraceInBuffer),
                       IARG_FAST_ANALYSIS_CALL,
                       IARG_REG_VALUE, endOfTraceInBufferReg, // previous trace
                       IARG_REG_VALUE, endOfBufferReg,
                       IARG_UINT32, traceAnalysisCallsNeeded.TotalSizeOccupiedByTraceInBuffer(),
                       IARG_END);
    TRACE_InsertThenCall(trace, IPOINT_BEFORE, AFUNPTR(APP_THREAD_REPRESENTITVE::BufferFull),
                         IARG_FAST_ANALYSIS_CALL,
                         IARG_REG_VALUE, endOfTraceInBufferReg,
						 IARG_THREAD_ID,
                         IARG_RETURN_REGS, endOfTraceInBufferReg,
                         IARG_END);
    TRACE_InsertCall(trace, IPOINT_BEFORE,  AFUNPTR(APP_THREAD_REPRESENTITVE::AllocateSpaceForTraceInBuffer),
                     IARG_FAST_ANALYSIS_CALL,
                     IARG_REG_VALUE, endOfTraceInBufferReg,
                     IARG_UINT32, traceAnalysisCallsNeeded.TotalSizeOccupiedByTraceInBuffer(),
                     IARG_RETURN_REGS, endOfTraceInBufferReg,
                     IARG_END);

    // Insert Analysis Calls for each INS on the trace that was recorded as needing one
    //   i.e. each INS that reads and/or writes memory
    traceAnalysisCallsNeeded.InsertAnalysisCalls();
}

VOID ThreadStart(THREADID tid, CONTEXT *ctxt, INT32 flags, VOID *v)
{
    // There is a new APP_THREAD_REPRESENTITVE for every thread
    APP_THREAD_REPRESENTITVE * appThreadRepresentitive = new APP_THREAD_REPRESENTITVE(tid);

    // A thread will need to look up its APP_THREAD_REPRESENTITVE, so save pointer in TLS
    PIN_SetThreadData(appThreadRepresentitiveKey, appThreadRepresentitive, tid);

    // Initialize endOfTraceInBufferReg to point at beginning of buffer
    PIN_SetContextReg(ctxt, endOfTraceInBufferReg, reinterpret_cast<ADDRINT>(appThreadRepresentitive->Begin()));
	
	PIN_SetContextReg(ctxt, currentIndex, (int) 0);
    // Initialize endOfBufferReg to point at end of buffer
    PIN_SetContextReg(ctxt, endOfBufferReg,reinterpret_cast<ADDRINT>(appThreadRepresentitive->End()));
}

VOID ThreadFini(THREADID tid, const CONTEXT *ctxt, INT32 code, VOID *v)
{
    APP_THREAD_REPRESENTITVE * appThreadRepresentitive 
        = static_cast<APP_THREAD_REPRESENTITVE*>(PIN_GetThreadData(appThreadRepresentitiveKey, tid));
    
    appThreadRepresentitive->ProcessBuffer (reinterpret_cast<char *>(PIN_GetContextReg (ctxt, endOfTraceInBufferReg)));
    
    totalBuffersFilled += appThreadRepresentitive->NumBuffersFilled();
    totalElementsProcessed += appThreadRepresentitive->NumElementsProcessed();

    delete appThreadRepresentitive;

    PIN_SetThreadData(appThreadRepresentitiveKey, 0, tid);
}


VOID ImageLoad(IMG img, VOID *) {

	RTN rtn = RTN_FindByName(img, "pthread_mutex_lock");

	if (RTN_Valid(rtn)) {
		RTN_Open(rtn);
		RTN_InsertCall(rtn, IPOINT_AFTER, AFUNPTR(APP_THREAD_REPRESENTITVE::LOCKING),
				IARG_REG_VALUE, currentIndex,
				IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
				IARG_THREAD_ID,
                IARG_END);
		RTN_Close(rtn);
	}
	
	
	rtn = RTN_FindByName(img, "pthread_mutex_unlock");

	if (RTN_Valid(rtn)) {
		RTN_Open(rtn);
		RTN_InsertCall(rtn, IPOINT_BEFORE, AFUNPTR(APP_THREAD_REPRESENTITVE::UNLOCKING),
				IARG_REG_VALUE, currentIndex,
				IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
				IARG_THREAD_ID,
                IARG_END);
		RTN_Close(rtn);
	}
	
	
	rtn = RTN_FindByName(img, "__pthread_create_2_1");

	if (RTN_Valid(rtn)) {

		RTN_Open(rtn);

		RTN_InsertCall(rtn, IPOINT_AFTER, AFUNPTR(ThreadCreate),
				IARG_FUNCARG_ENTRYPOINT_VALUE, 0, IARG_THREAD_ID, IARG_END);

		RTN_Close(rtn);
	}

	rtn = RTN_FindByName(img, "pthread_join");

	if (RTN_Valid(rtn)) {

		RTN_Open(rtn);

		RTN_InsertCall(rtn, IPOINT_AFTER, AFUNPTR(ThreadJoin),
				IARG_FUNCARG_ENTRYPOINT_VALUE, 0, IARG_THREAD_ID, IARG_END);

		RTN_Close(rtn);
	}
	
	
	

}





VOID Fini(INT32 code, VOID *v)
{
	int size=eventList.size();
	int i;
	
	ofstream file;
	file.open("synchList.txt");
   	
	for(i=0; i< size ; i++)
	{
		EVENT event= eventList[i];
    	file << event.tid <<" " << event.kind <<" "<< event.value << endl;
		
	}
	
	file.close();
     return;
     printf ("totalBuffersFilled %u  totalElementsProcessed %14.0f\n", (totalBuffersFilled),  
             static_cast<double>(totalElementsProcessed));
}





INT32 Usage()
{
    printf( "This tool demonstrates simple pin-tool buffer managing\n");
    printf ("The following command line options are available:\n");
    printf ("-num_bytes_in_buffer <num>   :number of bytes allocated for each buffer,                   default 1048576\n");
    printf ("-process_buffs <0 or 1>      :specify 0 to disable processing of the buffers,              default       1\n");
    return -1;
}

int main(int argc, char * argv[])
{
    // Initialize PIN library. Print help message if -h(elp) is specified
    // in the command line or the command line is invalid
    if( PIN_Init(argc,argv) )
    {
        return Usage();
    }
	PIN_InitSymbols();
    appThreadRepresentitiveKey = PIN_CreateThreadDataKey(0);

    // get the registers to be used in each thread for managing the
    // per-thread buffer
    endOfTraceInBufferReg = PIN_ClaimToolRegister();
    endOfBufferReg        = PIN_ClaimToolRegister();
	currentIndex    	  = PIN_ClaimToolRegister();
    if (! (REG_valid(endOfTraceInBufferReg) && REG_valid(endOfBufferReg)&& REG_valid(currentIndex) ) )
    {
        printf ("Cannot allocate a scratch register.\n");
        return 1;
    }
	
    TRACE_AddInstrumentFunction(TraceAnalysisCalls, 0);
	IMG_AddInstrumentFunction(ImageLoad, 0);
    PIN_AddThreadStartFunction(ThreadStart, 0);
    PIN_AddThreadFiniFunction(ThreadFini, 0);

    PIN_AddFiniFunction(Fini, 0);

    //printf ("buffer size in bytes 0x%x\n", KnobNumBytesInBuffer.Value());
    // fflush (stdout);
    filter.Activate();
    PIN_StartProgram();
    
    return 0;
}
