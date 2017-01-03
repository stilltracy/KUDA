#include "eventlist.h"

#ifdef __cplusplus
extern "C" {
#endif

// private optinal parameters for the implementation
#define QUERY_DEVICE (0)

/************************************************/

Configuration glbConfig;
EventList glbEventList;
#ifdef STATISTICS
Statistics glbStats;
#endif // STATISTICS

// global list of data races
DataRaceList glbDataRaces;

#if CHECK_ENABLED == 1
#ifndef CHECK_AT_GPU
BloomFilter h_racyVars;
IndexPairList* h_indexPairs;
#endif // CHECK_AT_GPU
#endif // CHECK_ENABLED

#ifdef USE_PTHREAD
pthread_t worker_pthread;
#endif // USE_PTHREAD

/************************************************/
static volatile bool recording = false;
static volatile bool finalizing = false;
static volatile bool finalized = false;
static volatile bool empty = true;
static volatile unsigned int filled = 0;

#define threadToBlockCacheSize	(32)
static Block* threadToBlockCache[threadToBlockCacheSize];

/************************************************/

static void initConfiguration();

static void checkBlock(Block*, size_t);
static void addEvent(EventKind, int, long, int);
/************************************************/

void EventThreadProc1(void* p) { void* d = EventThreadProc1Ex(p); }

/************************************************/
static int equals(Event*, Event*);
static Event getEvent(unsigned long i);
static void printEvent(FILE*,Event*);

static void try_shift_head(Block*, int );
static void try_shift_tail(Block*);
static Block* try_get_tail();
static inline void freeEventList();

static unsigned long nextEventIndex();
static Block* getBlock(unsigned long);
static void resetBlock(Block*, unsigned long, unsigned long);
static Block* allocBlock(unsigned long, unsigned long);

//static bool transferToEventList(Block*);

void TMP_GetSourceLocationFuncType(int, int*, int*, char**);
/************************************************/

// public interface to record an event
long long timeval_diff(struct timeval *difference, struct timeval *end_time, struct timeval *start_time)
{  
	struct timeval temp_diff;
	if(difference==NULL)  {    difference=&temp_diff;  }
	difference->tv_sec =end_time->tv_sec -start_time->tv_sec ;
	difference->tv_usec=end_time->tv_usec-start_time->tv_usec;
	/* Using while instead of if below makes the code slightly more robust. */
	while(difference->tv_usec<0)  
	{    
		difference->tv_usec+=1000000;
		difference->tv_sec -=1;  
	}
	return 1000000LL*difference->tv_sec + difference->tv_usec;
} /* timeval_diff() */

void RecordEvent_SharedRead(int tid, long mem, int instr) {
	addEvent(EVENT_SHARED_READ, tid, mem, instr);
}

void RecordEvent_SharedWrite(int tid, long mem, int instr) {
	addEvent(EVENT_SHARED_WRITE, tid, mem, instr);
}

void RecordEvent_AtomicRead(int tid, long mem, int instr) {
	addEvent(EVENT_ATOMIC_READ, tid, mem, instr);
}

void RecordEvent_AtomicWrite(int tid, long mem, int instr) {
	addEvent(EVENT_ATOMIC_WRITE, tid, mem, instr);
}

void RecordEvent_Lock(int tid, int lock, int instr) {
	addEvent(EVENT_LOCK, tid, lock, instr);
}

void RecordEvent_Unlock(int tid, int lock, int instr) {
	addEvent(EVENT_UNLOCK, tid, lock, instr);
}

void RecordEvent_RLock(int tid, int lock, int instr) {
	addEvent(EVENT_LOCK, tid, lock, instr);
}

void RecordEvent_WLock(int tid, int lock, int instr) {
	addEvent(EVENT_LOCK, tid, lock, instr);
}

void RecordEvent_RWUnlock(int tid, int lock, int instr) {
	addEvent(EVENT_UNLOCK, tid, lock, instr);
}

void RecordEvent_Fork(int tid, int _tid, int instr) {
	addEvent(EVENT_FORK, tid, _tid, instr);

	IF_COUNT(glbStats.num_threads++);
}

void RecordEvent_Join(int tid, int _tid, int instr) {
	addEvent(EVENT_JOIN, tid, _tid, instr);
}

void RecordEvent_Acquire(int tid, int value, int instr) {
	addEvent(EVENT_ACQUIRE, tid, value, instr);
}

void RecordEvent_Release(int tid, int value, int instr) {
	addEvent(EVENT_RELEASE, tid, value, instr);
}

void RecordEvent_Call(int tid, int proc, int instr) {
	addEvent(EVENT_CALL, tid, proc, instr);
}

void RecordEvent_Return(int tid, int proc, int instr) {
	addEvent(EVENT_RETURN, tid, proc, instr);
}


/************************************************/
// Public functions

void initEventList() {
	// make sure that this is first to call
	initConfiguration();

	// reset the source location function, this may be overwritten later
	setGetSourceLocationFunc(NULL);

	// init thread to block cache
	memset(&threadToBlockCache, 0, sizeof(Block*) * threadToBlockCacheSize);

	// init data race list
	glbDataRaces.size = 0;
	glbDataRaces.head = glbDataRaces.tail = NULL;

	//---------------------------------
	// init global event list
	// create all events
	unsigned long min = 0;
	unsigned long max = BLOCK_SIZE;
	Block* block = NULL;
	Block* prev = NULL;
	prev = glbEventList.tail = glbEventList.head = allocBlock(min, max);
	for(int i = 1; i < INIT_NUM_BLOCKS; ++i) {
		min += BLOCK_SIZE;
		max += BLOCK_SIZE;
		block = allocBlock(min, max); 
		prev = prev->next = block;
	}
	ASSERT(prev != NULL && block != NULL);
	block->next = glbEventList.head; // make cyclic
	ASSERT(block->next == glbEventList.tail);
	glbEventList.size = max;
	glbEventList.num_blocks = 0;
	glbEventList.next_index = 0;
	//---------------------------------

	// init lock
	//ReentrantLock* lock = &glbEventList.lock;
	//lock->owner = -1;
	//lock->count = 0;
	//UNLOCK(&lock->lock);
	MutexLock* lock = &glbEventList.lock;
	initMutex(lock);
	
#if CHECK_ENABLED == 1
#ifndef CHECK_AT_GPU
	// init event index pairs if races are checked at host
	h_indexPairs = (IndexPairList*) malloc(sizeof(IndexPairList));
	h_indexPairs->size = 0;
	bloom_clear(&h_racyVars);
#else //CHECK_AT_GPU
#if QUERY_DEVICE
	IF_DEBUG(printf("Querying GPU device..."));
	deviceQuery();
#endif

	IF_DEBUG(printf("Initializing the race checker..."));
	initRaceChecker();
	IF_DEBUG(printf("Done\n"));
#endif //CHECK_AT_GPU
#endif //CHECK_ENABLED

	// create the worker thread
#ifdef USE_PTHREAD
	if(pthread_create(&worker_pthread, NULL, EventThreadProc1Ex, &worker_pthread) != 0) {
		printf("Failed to create the event thread!");
		exit(-1);
	}
#endif //USE_PTHREAD

	// record the program start time
#ifdef STATISTICS
	if(gettimeofday(&glbStats.startTime, NULL)) {
		printf("Failed to get the start time!");
		exit(-1);
	}
#endif //STATISTICS
}

Event getEvent(unsigned long i) {
	Block* block = getBlock(i);
	return block->events[i % BLOCK_SIZE];
}

void lockForShared(int tid) {
	//lockEventList(tid, 10);
}

void unlockForShared(int tid) {
	//unlockEventList(tid);
}

void initConfiguration() {
	glbConfig.algorithm = Goldilocks;
}

/*******************************************************/
// printing functions
const char* eventKindToString(EventKind kind) {
	switch (kind) {
	case EVENT_SHARED_READ: return "READ";
	case EVENT_SHARED_WRITE: return "WRITE";
	case EVENT_LOCK: return "LOCK";
	case EVENT_UNLOCK: return "UNLOCK";
	case EVENT_ATOMIC_READ: return "VOLREAD";
	case EVENT_ATOMIC_WRITE: "VOLWRITE";
	case EVENT_FORK: return "FORK";
	case EVENT_JOIN: return "JOIN";
	case EVENT_ACQUIRE: return "ACQUIRE";
	case EVENT_RELEASE: return "RELEASE";
	}
	return "UNKNOWN";
}

const char* accessKindToString(AccessKind kind) {
	switch (kind) {
	case ACCESS_READ: return "READ";
	case ACCESS_WRITE: return "WRITE";
	}
	return "UNKNOWN";
}

static GetSourceLocationFuncType GetSourceLocationFunc = NULL;

void TMP_GetSourceLocationFuncType(int i, int* c, int* l, char** f) {
	static const char* unknown = "Unknown.c";
	*c = 1;
	*l = 2;
	*f = (char*) unknown;
}

void setGetSourceLocationFunc(GetSourceLocationFuncType fun) {
	GetSourceLocationFunc = fun;
}

static inline Source getSource(Event* e, EventInfo* info) {
	Source src;

	if(GetSourceLocationFunc == NULL) {
		src.column = src.line = -1;
		src.file = NULL; // marks this source invalid
	} else {
		GetSourceLocationFunc(EVENT_INSTR((*info)), &src.column, &src.line, &src.file);
	}

	return src;
}

static inline Access eventToAccess(Event* e, EventInfo* info) {
	Access a;

	EventKind kind = EVENT_KIND((*e));

	ASSERT(IS_ACCESS(kind));
	a.tid = EVENT_TID((*e));
	a.value = EVENT_VALUE((*e));
	a.kind = IS_READ_ACCESS(kind) ? ACCESS_READ : ACCESS_WRITE;
	a.source = getSource(e, info);

	return a;
}

static inline DataRace* eventsToDataRace(Event* e1, EventInfo* info1, Event* e2, EventInfo* info2) {
	DataRace* dr = (DataRace*) malloc(sizeof(DataRace));

	dr->accesses[0] = eventToAccess(e1, info1);
	dr->accesses[1] = eventToAccess(e2, info2);
	dr->next = NULL;

	return dr;
}

const char* sourceToString(Source* src) {
	if(GetSourceLocationFunc == NULL) {
		return NULL;
	}
	char* msg;
	int n = asprintf(&msg, "%s#%d:%d", src->file, src->line, src->column);
	return msg;
}

const char* eventToString(Event* e, EventInfo* info) {
	char* msg;

	Source s = getSource(e, info);
	const char* src = sourceToString(&s);

	int n = asprintf(&msg, "<%s(%ld) by %d at %s>", eventKindToString(EVENT_KIND((*e))), EVENT_VALUE((*e)), EVENT_TID((*e)), (src == NULL ? "Unknown source!" : src));

	if(src != NULL) {
		free((void*)src);
	}

	return msg;
}

const char* eventToShortString(Event* e) {
	char* msg;

	int n = asprintf(&msg, "<%s(%ld) by %d>", eventKindToString(EVENT_KIND((*e))), EVENT_VALUE((*e)), EVENT_TID((*e)));

	return msg;
}


const char* accessToString(Access* a) {
	char* msg;

	const char* src = sourceToString(&a->source);

	int n = asprintf(&msg, "<%s(%ld) by %d at %s>", accessKindToString(a->kind), a->value, a->tid, (src == NULL ? "Unknown source!" : src));

	if(src != NULL) {
		free((void*)src);
	}

	return msg;
}

const char* dataraceToString(DataRace* race) {
	char* msg;

	const char* str1 = accessToString(&race->accesses[0]);
	const char* str2 = accessToString(&race->accesses[1]);

	int n = asprintf(&msg, "[%s %s]", str1, str2);

	if(str1 != NULL) {
		free((void*)str1);
	}

	if(str2 != NULL) {
		free((void*)str2);
	}

	return msg;
}


static inline Block* getBlock(unsigned long i) {
start:
	Block* block = READ(&glbEventList.head);
	ASSERT (block != NULL);
	while (i >= block->max) {
		block = block->next;
		ASSERT(block != NULL);

		if(block == READ(&glbEventList.tail)) {
			IF_COUNT(glbStats.num_starvation++);
			goto start;
		}

		if(BLOCK_REUSED(block, i)) { // if block is being reused
			IF_COUNT(glbStats.num_restarts++);
			goto start;
		}
	}
	if(BLOCK_REUSED(block, i)) { // if block is being reused
		IF_COUNT(glbStats.num_restarts++);
		goto start;
	}
	ASSERT(block != NULL && IN_BLOCK(block, i));
	return block;
}



//static inline void lockEventList(int tid, int time) {
	//static ReentrantLock* lock = &glbEventList.lock;
	//lockReentrant(lock, tid, time);
//}

//static inline void unlockEventList(int tid) {
	//static ReentrantLock* lock = &glbEventList.lock;
	//unlockReentrant(lock, tid);
//}

static int equals(Event* e1, Event* e2) {
	return EVENT_TID((*e1)) == EVENT_TID((*e2))
			&& EVENT_KIND((*e1)) == EVENT_KIND((*e2))
			&& EVENT_VALUE((*e1)) == EVENT_VALUE((*e2));
			// && EVENT_INSTR((*e1)) == EVENT_INSTR((*e2));
}

static void addEvent(EventKind kind, int tid, long value, int instr) {
	struct timeval earlier;
	struct timeval later;
	struct timeval interval;
	static MutexLock* lock = &glbEventList.lock;
	// check if we have started to record
	if(!recording) {
		if(kind == EVENT_FORK) {
			recording  = true;
		} else {
			return;
		}
	}

	unsigned int index = 0;

	Block* block = NULL;
	lockMutex(lock, 10);
  start:
	block = READ(&glbEventList.head);
	ASSERT (block != NULL);
	//if(READ(&filled) == 62){
		//printf("adding to %p block #%u !\n", &block, block->size);
		//fflush(stdout);
	//}

	index = INC(&block->size); 
	// SLEEP(1); // #!# slowdown application threads
	// int z = 0;
	while(index >= BLOCK_SIZE){
		block = block->next; // we don't have an empty slot on this block -> look next
		ASSERT(block != NULL);
		if((block == READ(&glbEventList.tail))) {
			printf("W%d",tid);
			SLEEP(1000); // #!# 
			printf("W%d",tid);
			fflush(stdout);
			//ASSERT(false);
			IF_COUNT(glbStats.num_starvation++);
			goto start;
		}
		//if(empty && (block == READ(&glbEventList.tail))){
			//if(!(z++ % 100))IF_DEBUG(printf("WTF2?", filled));
			//IF_DEBUG(fflush(stdout));
		//}

		//block = block->next; // we don't have an empty slot on this block -> look next
		//ASSERT(block != NULL);

		index = INC(&block->size); // do we have an empty slot inside this block
		
	} // end while
	
	ASSERT(0 <= index && index < BLOCK_SIZE);
	ASSERT(block != NULL && IN_BLOCK(block, index));
	//-------------------------
	
	block->events[index] = make_event(tid, kind, value);
	block->eventinfos[index].instr = instr;
	
	

	// if we are the last inserter, then try to shift head
	
	if(index == BLOCK_SIZE-1) {
	//lockMutex(lock, 10);
		IF_V_HEAD(printf("Added event to %p. Now try shifting\n", block));
		IF_V_HEAD(fflush(stdout));
		try_shift_head(block, tid);
		IF_V_HEAD(printf("Bravo! Tid %d is last inserter and the block was:%p\n", tid, block));
		IF_V_HEAD(fflush(stdout));
	//unlockMutex(lock);
	}
	unlockMutex(lock);
}

//static inline void try_shift_head(Block* block)
static inline void try_shift_head(Block* block, int tid) {
	IF_V_HEAD(printf("%d++",tid));
	IF_V_HEAD(fflush(stdout));
	Block* head = READ(&glbEventList.head);
	IF_V_HEAD(printf("+++"));
	IF_V_HEAD(fflush(stdout));
	if(head == block) {
		// now try to do it
		IF_V_HEAD(printf("+++"));
		IF_V_HEAD(fflush(stdout));
		while(READ(&head->size) >= BLOCK_SIZE && head->next != READ(&glbEventList.tail)) {
			IF_V_HEAD(printf("traversing..."));
			IF_V_HEAD(fflush(stdout));

			head = head->next;

			IF_COUNT(glbStats.num_blocks += 1);
		}
		while(head->next == READ(&glbEventList.tail)) {
			SLEEP(100); // #!# 
			IF_V_HEAD(printf("%d is waiting tail\n", tid));
			IF_V_HEAD(fflush(stdout));
		}
		if(head != block) {
			ASSERT(block != NULL);
			WRITE(&glbEventList.head, head);
			
			//CAS(&empty,true,false); // #!#
			INC(&filled);
			
			IF_DEBUG(printf("%d number of frames left for detection.", filled));
			IF_DEBUG(fflush(stdout));
		}
		else {
			ASSERT(false);
		}
		IF_V_HEAD(printf("111111111\n"));
		IF_V_HEAD(fflush(stdout));
	}
	else {
		IF_V_HEAD(printf("WTF -> prev head = %p, new head = %p\n", block, head));
		IF_V_HEAD(fflush(stdout));
	}
}

static inline void try_shift_tail(Block* block) {
	IF_V_TAIL(printf("---------resetting...."));
	IF_V_TAIL(fflush(stdout));
	Block* tail = READ(&glbEventList.tail);
	//printf("tail within try_shift_tail: %p \n", tail);
	//printf("arg to try_shift_tail: %p \n", block);
	//fflush(stdout);
	ASSERT(block == tail); 

	Block* head = READ(&glbEventList.head);
	ASSERT(head != tail);

	// reset the block
	unsigned long size = glbEventList.size;
	resetBlock(block, size, size + BLOCK_SIZE);
	// shift tail
	WRITE(&glbEventList.tail, tail->next);
	WRITE(&glbEventList.size, size + BLOCK_SIZE);
	DEC(&filled);
	if(READ(&filled) == 0) CAS(&empty,false,true); // #!#
	
	IF_DEBUG(printf("%d number of frames left for detection.", filled));
	IF_DEBUG(fflush(stdout));

	IF_V_TAIL(printf("000000000\n"));
	IF_V_TAIL(fflush(stdout));
}

static inline Block* try_get_tail() {
	Block* head = READ(&glbEventList.head);
	Block* tail = READ(&glbEventList.tail);

	if(head == tail) {
		return NULL;
	}
	ASSERT(tail != NULL);
	return tail;
}
	
static inline void resetBlock(Block* block, unsigned long min, unsigned long max) {
	ASSERT (block != NULL);
	
	block->min = min;
	block->max = max;
	WRITE(&(block->size), 0);
}
	
static inline Block* allocBlock(unsigned long min, unsigned long max) {
	Block* block = NULL;
	
	block = (Block*) malloc(sizeof(Block));
	
	resetBlock(block, min, max);
	
	return block;
}
// pre: event list is locked
/*
static bool transferToEventList(Block* block) {
	ASSERT (finalizing || block->size >= BLOCK_SIZE);

	resetBlock(block, READ(&glbEventList.size), READ(&glbEventList.size) + BLOCK_SIZE);

	Block* tail = READ(&glbEventList.tail);
	if(tail == NULL) {
		// only happens when finalizing
		ASSERT(finalizing);
		return false;
	}

	tail->next = block; // visible to app threads but they cannot interfere with this

	ADD(&glbEventList.size, BLOCK_SIZE); // commit point that adds the block to the event list

	WRITE(&glbEventList.tail, block);

	INC(&glbEventList.num_blocks);

	IF_COUNT(glbStats.num_blocks += 1);
	IF_COUNT(glbStats.avg_blocks = ((glbStats.avg_blocks * glbStats.avg_blocks_num) + glbEventList.num_blocks) / (glbStats.avg_blocks_num+1));
	IF_COUNT(glbStats.avg_blocks_num++);

	return true;
}
*/
static inline void freeEventList() {
	ASSERT(finalizing && finalized);

	Block* head = glbEventList.head;
	Block* current = head;
	do {
		Block* tmp = current;
		current = current->next;
		free(tmp);
	} while(current != head);
	IF_DEBUG(printf("Freed eventlist!\n"));
	IF_DEBUG(fflush(stdout));
}

void finalizeEventList() {

#ifdef STATISTICS
	if(gettimeofday(&glbStats.endTime_program, NULL)) {
		printf("Failed to get the end time!");
		exit(-1);
	}
	timeval_subtract(&glbStats.runningTime_program, &glbStats.endTime_program, &glbStats.startTime);
#endif

	ASSERT(finalizing == false);
	WRITE(&finalizing, true);
	//shrink_rate = -1; // shrink all events

	IF_DEBUG(printf("Finalizing the event list, wait for the worker thread...\n"));
	IF_DEBUG(printf("%d number of frames left for detection.\n", filled));
	IF_DEBUG(fflush(stdout));
	while(1){
		IF_DEBUG(printf("Deadlock?"));
		IF_DEBUG(fflush(stdout));
		if(READ(&finalized) == true) break;
		else {
			SLEEP(1000);
			IF_DEBUG(printf("Wait4it"));
		}
	}
	IF_DEBUG(printf("Done\n"));
	IF_DEBUG(fflush(stdout));

#if CHECK_ENABLED == 1
#ifdef CHECK_AT_GPU
	IF_DEBUG(printf("Finalizing the race checker..."));
	IF_DEBUG(fflush(stdout));
	finalizeRaceChecker();
	IF_DEBUG(printf("Done\n"));
	IF_DEBUG(fflush(stdout));
#else
// free index pair list if races are checked at host
	free(h_indexPairs);
#endif
#endif

#ifdef USE_PTHREAD
	// wait for the worker thread to end
	ASSERT(false);
	if(pthread_join(worker_pthread, NULL) != 0) {
		printf("Failed to join the event thread 1!");
		exit(-1);
	}
#endif

// CB: analysis time and deallocation 
	IF_COUNT(glbStats.num_events = glbEventList.size);
				
	freeEventList();
	
#ifdef STATISTICS
// record the analysis end time
	if(gettimeofday(&glbStats.endTime_analysis, NULL)) {
		printf("Failed to get the end time!");
		exit(-1);
	}
	timeval_subtract(&glbStats.runningTime_analysis, &glbStats.endTime_analysis, &glbStats.startTime);
#endif
}

/************************************************/

__inline__ uint64_t rdtsc() {
	uint32_t lo, hi;
	__asm__ __volatile__ ( // serialize
			"xorl %%eax,%%eax \n        cpuid"
			::: "%rax", "%rbx", "%rcx", "%rdx");
	/* We cannot use "=A", since this would use %rax on x86_64 */
	__asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
	return (uint64_t) hi << 32 | lo;
}

/************************************************/

#ifdef STATISTICS
void initStatistics() {
	memset(&glbStats, 0, sizeof(Statistics));
}

void printStatistics(FILE* out) {
	Statistics* s = &glbStats;
	if(glbDataRaces.size > 0) {
		fprintf(out, "\n********* DATA RACES (Total: %d) **************\n\n", glbDataRaces.size);
		DataRace* dr = glbDataRaces.head;
		while(dr != NULL) {
			const char* s = dataraceToString(dr);
			fprintf(out, "%s\n", s);
			if(s != NULL) { free((void*)s); }

			// free the data race
			void* tmp = dr;
			dr = dr->next;
			free(tmp);
		}
		fprintf(out, "\n********* END OF DATA RACES *******\n");
	}
	// do not trust glbDataRaces any more, it was reset

	fprintf(out, "\n********* STATISTICS **************\n\n");
	fprintf(out, "Enabled checks:\t\t");
	
#if CHECK_ENABLED
	if(glbConfig.algorithm == Eraser) {
		fprintf(out, "ERASER ");
	}
	else if(glbConfig.algorithm == Goldilocks) {
		fprintf(out, "GOLDILOCKS ");
	}
#endif // CHECK_ENABLED
#if ATOMICITY_ENABLED
	fprintf(out, "ATOMICITY");
#endif // ATOMICITY_ENABLED
	fprintf(out, "\n");
	fprintf(out, "Checked block size:\t\t%d\t(constant)\n", CHECKED_BLOCK_SIZE);
	fprintf(out, "Concurrent kernels:\t\t%d\t(constant)\n", NUM_CONCURRENT_KERNELS);
	fprintf(out, "GPU threads per block:\t%d\t(constant)\n", NUM_THREADS);
	fprintf(out, "\n");
#ifdef COUNTERS_ENABLED
	fprintf(out, "Total # threads:\t\t%d\n", (s->num_threads + 1));	
	fprintf(out, "Total # events:\t\t%lu\n", s->num_events);
	fprintf(out, "Total # blocks:\t\t%lu\n", s->num_blocks);
	fprintf(out, "\n");
	fprintf(out, "Total # restarts:\t\t%lu\n", s->num_restarts);
	fprintf(out, "Total # starvations:\t%lu\n", s->num_starvation);
	fprintf(out, "\n");
	fprintf(out, "Total # race checks:\t%lu\n", s->num_racechecks);
	fprintf(out, "Total # race checks skipped:%lu\n", s->num_racecheckskipped);
	fprintf(out, "Total # races found:\t%u\n", glbDataRaces.size);
	fprintf(out, "\n");
#endif // COUNTERS_ENABLED
	fprintf(out, "TIME for program:\t%lu seconds, %d microseconds\n", glbStats.runningTime_program.tv_sec, glbStats.runningTime_program.tv_usec);
	fprintf(out, "TIME for analysis:\t%lu seconds, %d microseconds\n", glbStats.runningTime_analysis.tv_sec, glbStats.runningTime_analysis.tv_usec);	
	
	fprintf(out, "\n********* END OF STATISTICS *******\n");
}

/************************************************/

// this is from GNU C library manual

/* Subtract the `struct timeval' values X and Y,
 storing the result in RESULT.
 Return 1 if the difference is negative, otherwise 0.  */

int timeval_subtract(struct timeval *result, struct timeval *_x, struct timeval *_y) {
	struct timeval x = *_x;
	struct timeval y = *_y;
	/* Perform the carry for the later subtraction by updating y. */
	if (x.tv_usec < y.tv_usec) {
		int nsec = (y.tv_usec - x.tv_usec) / 1000000 + 1;
		y.tv_usec -= 1000000 * nsec;
		y.tv_sec += nsec;
	}
	if (x.tv_usec - y.tv_usec > 1000000) {
		int nsec = (y.tv_usec - x.tv_usec) / 1000000;
		y.tv_usec += 1000000 * nsec;
		y.tv_sec -= nsec;
	}

	/* Compute the time remaining to wait.
	 tv_usec is certainly positive. */
	result->tv_sec = x.tv_sec - y.tv_sec;
	result->tv_usec = x.tv_usec - y.tv_usec;

	/* Return 1 if result is negative. */
	return x.tv_sec < y.tv_sec;
}

/************************************************/

#endif // STATISTICS

/************************************************/

// map an indexpair to a datarace
static DataRace* mapIndexPairToDataRace(Block* block, IndexPair pair) {
	
	IF_VERBOSE(printf("Index1: %d\n", FST(pair)));
	IF_VERBOSE(printf("Index2: %d\n", SEC(pair)));
	Event e1 = block->events[FST(pair)];
	EventInfo info1 = block->eventinfos[FST(pair)];
	IF_VERBOSE(printf("Event: %s\n", eventToShortString(&e1)));
	ASSERT(IS_ACCESS(EVENT_KIND(e1)));
	Event e2 = block->events[SEC(pair)];
	EventInfo info2 = block->eventinfos[SEC(pair)];
	IF_VERBOSE(printf("Event: %s\n", eventToShortString(&e2)));
	ASSERT(IS_ACCESS(EVENT_KIND(e2)));

	return eventsToDataRace(&e1, &info1, &e2, &info2);
}

static void addToDataRaceList(DataRace* dr) {
	// make sure next is NULL
	if(glbDataRaces.size >= MAX_RACES_TO_STORE) {
		return;
	}

	dr->next = NULL;

	if(glbDataRaces.head == NULL) {
		ASSERT(glbDataRaces.tail == NULL);
		glbDataRaces.head = glbDataRaces.tail = dr;
	} else {
		ASSERT(glbDataRaces.head != NULL);
		glbDataRaces.tail->next = dr;
		glbDataRaces.tail = dr;
	}
	glbDataRaces.size++;
}

void* EventThreadProc1Ex(void* tid) { 
	SLEEP(1000); // #!# 
	
	while (true) {
		
		Block* block = try_get_tail();
		//printf("block=tail in worker thread: %p \n", block);
		//fflush(stdout);
		Block* initBlock = block;
		if(block != NULL) {
			block->max = 0; // mark reused #!# check later for races
		}

		if (block == NULL) {
			// check if the event list has been terminated
			if (READ(&finalizing) == true) {
				WRITE(&finalized, true);
				return NULL;
			}

		} else {

			size_t blksize = READ(&block->size);

			if(blksize > BLOCK_SIZE) {
				blksize = BLOCK_SIZE;
			}

			if(blksize > 0) {
				IF_V_RACE(printf("\nChecking races in the block size of %ld\n", blksize));
				
				checkBlock(block, blksize);
				//Block* cantail = try_get_tail();
				//ASSERT(cantail == initBlock);
			} else {
				IF_COUNT(glbStats.num_racecheckskipped++);
				IF_V_RACE(printf("Skipping empty block\n"));
			}
			ASSERT(block == initBlock); // nobody modifies tail
			try_shift_tail(block);

		}
	}
}


// check the block for races and record the races found. does not change the event list.
// offset is the offset each frame in the block will start
void checkBlock(Block* block, size_t blksize){
#if CHECK_ENABLED == 1
				IndexPairList* result = NULL;
				IndexPairList** results = NULL;
#ifdef CHECK_AT_GPU
				results = raceChecker(block, blksize);

				if(results != NULL) {
					for(int i = 0; i < NUM_CUDA_STREAMS; ++i) {
						unsigned int dr_size = results[i]->size;
						if(dr_size > 0){
							IF_V_RACE(printf("%u races detected\n", dr_size));
							// race found, collect them
							for(unsigned int r = 0; r < dr_size; ++r) {
								DataRace* dr = mapIndexPairToDataRace(block, results[i]->pairs[r]);
								addToDataRaceList(dr); // add to data race
							}
						}
					}
				}
#else // CHECK_AT_GPU
				result = h_raceChecker(block);

				if(result != NULL) {
					unsigned int dr_size = result->size;
					if(dr_size > 0){
						IF_V_RACE(printf("%u races detected\n", dr_size));
						// race found, collect them
						for(unsigned int r = 0; r < dr_size; ++r) {
							DataRace* dr = mapIndexPairToDataRace(block, result->pairs[r]);
							addToDataRaceList(dr); // add to data race
						}

					}
				}
#endif // CHECK_AT_GPU
				IF_V_RACE(printf("Done with checking races in the block size of %ld\n", blksize));
				IF_COUNT(glbStats.num_racechecks++);
#endif // CHECK_ENABLED
}

/************************************************/
#if CHECK_ENABLED == 1
#ifndef CHECK_AT_GPU
IndexPairList* h_raceChecker(Block* block)
{
	Event* events = block->events;
	int size = block->size;

	lockset_t LS;

	int y = 0; // blockIdx.x;
	int num_threads = 1; // blockDim.x;
	int threadIdx = 0;

	h_indexPairs->size = 0;

	#define getEvent(_x_, _y_)	events[((_y_) * CHECKED_BLOCK_SIZE) + (_x_)]

	for(int index = threadIdx; index < size /* CHECKED_BLOCK_SIZE-1*/; index += num_threads) {
		// e is the first access
		Event e = getEvent(index, y);
		EventKind kind = EVENT_KIND(e);
		if(IS_ACCESS(kind)) {
			// check the access to mem
			int mem = EVENT_VALUE(e);
			// check if this variable is already identified to be racy
			if(bloom_lookup(&h_racyVars, mem)) continue;

			int tid = EVENT_TID(e);

			bool initLS = true;

			for(int i = index + 1, j = index + 1; i < size; ++i) {
				Event e2 = getEvent(i, y);
				int tid2 = EVENT_TID(e2);
				EventKind kind2 = EVENT_KIND(e2);

				if(IS_ACCESS(kind2)
					&& EVENT_VALUE(e2) == mem
					&& tid != tid2
					&& (IS_WRITE_ACCESS(kind) || IS_WRITE_ACCESS(kind2)))
				{
					bool racy = true;
					// initialize lockset
					if(initLS){
						lockset_init(&LS, tid);
						initLS = false;
					}

					// update the lockset
					for(; j < i; ++j) {
						// apply the lockset rule to j. event
						Event e3 = getEvent(j, y);
						int tid3 = EVENT_TID(e3);
						EventKind kind3 = EVENT_KIND(e3);

						if(IS_ACQUIRE(kind3)) {
							if(lockset_lookup(&LS, EVENT_VALUE(e3))) {
								// check if we are adding the tid of the second access
								if(tid3 == tid2 && !IS_READ_ACCESS(kind2)) {
									racy = false;
									// break to the the end of the loop
									break;
								}
								lockset_add(&LS, tid3);
							}
						} else if(IS_RELEASE(kind3)) {
							if(lockset_lookup(&LS, tid3)) {
								lockset_add(&LS, EVENT_VALUE(e3));
							}
						}
					}
					// check if the current tid is in the lockset
					if(racy && !lockset_lookup(&LS, tid2)) {
						// mark the variable as racy, to prevent checking the race again
						bloom_add(&h_racyVars, mem);
						// report the data race
						if(h_indexPairs->size < MAX_RACES_TO_REPORT) {
							unsigned int ipair = h_indexPairs->size;
							h_indexPairs->size = ipair + 1;
							if(ipair < MAX_RACES_TO_REPORT) {
								int idx1 = ((y * CHECKED_BLOCK_SIZE) + index);
								IF_VERBOSE(printf("Index1: %d\n", idx1));
								printf("Event: %s\n", eventToShortString(&e));
								ASSERT(equals(&events[idx1], &e));
								ASSERT(IS_ACCESS(EVENT_KIND(events[idx1])));
								int idx2 = ((y * CHECKED_BLOCK_SIZE) + i);
								IF_VERBOSE(printf("Index2: %d\n", idx2));
								IF_VERBOSE(printf("Event: %s\n", eventToShortString(&e2)));
								ASSERT(equals(&events[idx2], &e2));
								ASSERT(IS_ACCESS(EVENT_KIND(events[idx2])));
								h_indexPairs->pairs[ipair] = make_indexpair(idx1, idx2);

								IF_DEBUG(printf("** %u races detected\n", h_indexPairs->size));
							} else {
								h_indexPairs->size = MAX_RACES_TO_REPORT;
							}
						}
						break; // restart for another access
					} else {
						// decide whether to continue or not
						if(!IS_READ_ACCESS(kind2)) {
							break;
						}
					}
				} // end of checking access
			}
		}
	}
	return h_indexPairs;
}
#endif // CHECK_AT_GPU
#endif // CHECK_ENABLED
#ifdef __cplusplus
}
#endif
