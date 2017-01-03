/*
 * goldilocks.h
 *
 *  Created on: Sep 25, 2010
 *      Author: elmas
 */

#ifndef GOLDILOCKS_H_
#define GOLDILOCKS_H_

#define __INLINE__		static inline

/**
 ***********************************************************************************
 ***********************************************************************************
 * Atomic operations
 ***********************************************************************************
 ***********************************************************************************
 **/

#define CAS(m,c,s)  cas((intptr_t)(s),(intptr_t)(c),(intptr_t*)(m))

__INLINE__ intptr_t
cas (intptr_t newVal, intptr_t oldVal, volatile intptr_t* ptr)
{
    intptr_t prevVal;

    __asm__ __volatile__ (
        "lock \n"
#ifdef __LP64__
        "cmpxchgq %1,%2 \n"
#else
        "cmpxchgl %k1,%2 \n"
#endif
        : "=a" (prevVal)
        : "q"(newVal), "m"(*ptr), "0" (oldVal)
        : "memory"
    );

    return prevVal;
}

#define INC(m)  atomic_inc((intptr_t*)(m))
#define DEC(m)  atomic_dec((intptr_t*)(m))


__INLINE__ intptr_t
atomic_inc (volatile intptr_t* addr)
{
    intptr_t v;
    for (v = *addr; CAS(addr, v, v+1) != v; v = *addr) {}
    return (v+1);
}

__INLINE__ intptr_t
atomic_dec (volatile intptr_t* addr)
{
    intptr_t v;
    for (v = *addr; CAS(addr, v, v-1) != v; v = *addr) {}
    return (v-1);
}

typedef unsigned long long TIMER_T;

#define TIMER_READ() ({ \
    unsigned int lo; \
    unsigned int hi; \
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi)); \
    ((TIMER_T)hi) << 32 | lo; \
})




/**
 ***********************************************************************************
 ***********************************************************************************
 * Compile-time Parameters
 ***********************************************************************************
 ***********************************************************************************
 **/

#define LOCKSET_GC_MAX_REMOVE		256
#define LOCKSET_GC_MAX_FREE_UPDATES	1000
#define LOCKSET_EVENT_ALLOC_BLOCK_SIZE	256

#define LOCKSET_EAGER_LOCKSET_COMPUTATION	        1
#define LOCKSET_LIMIT_FOR_EAGER_PROPAGATE	        100000
#define LOCKSET_MAX_LIST_SIZE_FOR_EAGER_PROPAGATE	1000000

#define LOCKSET_GROW_INCREASE		32


/**
 ***********************************************************************************
 ***********************************************************************************
 * Primitive Types
 ***********************************************************************************
 ***********************************************************************************
 **/

typedef unsigned long jls_address_t;

typedef jls_address_t jls_syncvar_t;
typedef jls_address_t jls_lockvar_t;
typedef jls_address_t jls_volvar_t;
typedef jls_address_t jls_shrvar_t;
typedef jls_address_t jls_thrvar_t;

/**
 ***********************************************************************************
 ***********************************************************************************
 * Includes
 ***********************************************************************************
 ***********************************************************************************
 **/

#include "core/lockset-impl.h"

#include "core/lockset-stat.h"

/**
 ***********************************************************************************
 ***********************************************************************************
 * Struct Types
 ***********************************************************************************
 ***********************************************************************************
 **/

// spinlocks
typedef volatile int jls_spinlock_t;
#define jls_spin_acquire(l)		while(!jls_check_and_set((&(l)), 0, 1)) { KTHREAD(yield)(); } assert((l) == 1); while(false){} { (++(l)); } assert((l) == 2);
#define jls_spin_release(l)		{ assert((l) == 2); while(false) {} { (l) = 0; } } //if(!jls_check_and_set((&(l)), 1, 0)) { fprintf(stderr, "spinlock error"); exit(EXIT_FAILURE); }
#define jls_create_spinlock()	(0)
#define jls_destroy_spinlock(l)

/****************************************/

typedef struct _jls_list_item_t {
	void* data;
 	struct _jls_list_item_t* next;
 	struct _jls_list_item_t* prev;
 } jls_list_item_t;
/****************************************/
typedef struct _jls_list_t {
 	jls_list_item_t* head;
  	jls_list_item_t* tail;
 	jls_spinlock_t glb_lock;
 	size_t size;
 } jls_list_t;
/****************************************/
typedef enum {
	RdWrAccess  = 0,
	ReadAccess  = 1,
	WriteAccess = 2,
} jls_access_type_t;
 /****************************************/
 typedef struct {
 	jls_access_type_t		rdwr;
 	jthread_t				thread;
 	int						pc;
 } _jls_var_access_t;
 typedef _jls_var_access_t * jls_var_access_t;

/****************************************/

JLS_LOCKSET_DEFINITION

/****************************************/

typedef enum {
	Virgin1,
	Exclusive1,
	SharedRead,
	SharedModify1,
	Exclusive2,
	SharedModify2
} jls_vclock_state_t;

typedef enum {
	Virgin,
	Exclusive,
	Shared,
	SharedModified
} jls_eraser_state_t;

typedef struct __jls_vclock_t {
	jls_thread_t	owner;
	unsigned int	ticks;
	struct __jls_vclock_t * next;
} _jls_vclock_t;
typedef _jls_vclock_t * jls_vclock_t;
/****************************************/
typedef struct {
	jls_vclock_t	head;
	jls_vclock_t	tail;
	jls_spinlock_t	lock;
} _jls_vector_clock_t;
typedef _jls_vector_clock_t * jls_vector_clock_t;

/****************************************/

typedef enum {
	Dummy,			// 0
	Acquire,		// 1
	Release,		// 2
	Fork,			// 3
	Join,			// 4
	VolatileRead,	// 5
	VolatileWrite,	// 6
	ObjAllocate, 	// 7
	ObjFinalize 	// 8
} jls_update_type_t;

struct _jls_thread_data_t;

typedef struct _jls_update_t {
 	jls_update_type_t	type;
 	jls_syncvar_t		elt_chk;
 	jls_syncvar_t		elt_add;
 	struct _jls_thread_data_t * owner;
 	unsigned long id; // increases for each new update, for comparing last update of accessor thread and the update of the variable accessed
	volatile int ref_count;
	// jls_spinlock_t		lock;

	struct _jls_update_t * next;
} jls_update_t;
/****************************************/
typedef struct _jls_update_list_t {
 	jls_update_t	head;
  	jls_update_t	tail;
  	jls_update_t	xtail;
  	jls_spinlock_t	lock;
  	_jls_update_t	dummy_update;
} jls_update_list_t;
/****************************************/
typedef struct _jls_readset_item_t {
	jls_thread_t		owner_thread;
	jls_lockvar_t		random_lock;
	jls_update_t		last_update;
	jls_lockset_t		lockset;
	struct __jls_readset_item_t * next;
} jls_readset_item_t;
/****************************************/
typedef struct _jls_readset_t {
	jls_readset_item_t	head;
	unsigned int		size;
} jls_readset_t;
/****************************************/
// TODO: write a clone method _jls_lockinfo_t jls_clone_lockinfo(_jls_lockinfo_t)
typedef struct _jls_lockinfo_t {
	bool				visited;
	jls_thread_t		owner_thread;
	jls_lockvar_t		random_lock;
	jls_update_t		last_update;
	jls_lockset_t		lockset;
	_jls_readset_t		readset;
	jls_spinlock_t		lock;
	_jls_var_access_t	last_access;
} jls_lockinfo_t;
/****************************************/

typedef struct _jls_lockinfo_list_t {
	bool 			isLIv;
	union {
		jls_lockinfo_t		LIv;
		jls_readset_item_t	item;
	}	info;
	struct _jls_lockinfo_list_t* next;
 } jls_lockinfo_list_t;
/****************************************/

typedef struct _jls_thread_data_t {
 	jls_lockvar_t		random_lock;
 	jls_lockset_t		lockset; // rename it , scratch lockset
 	// jls_update_t		last_update;
 	unsigned long		last_update_id;
 	bool				xaction;
 	jls_update_t		reads;
 	jls_update_t		writes;
	jls_thread_t		tid;
 	_jls_var_access_t	current_access;
} jls_thread_data_t;
/****************************************/
typedef struct _jls_object_data_t {
 	bool				check_enabled;
 	jls_spinlock_t		lock;
 	jls_lockinfo_t *	LImap; // for non-volatile fields, static or instance
 	jls_spinlock_t *	vollockmap; // for volatile fields, static or instance, for making access and adding corresponding update into the global update list atomic, for a volatile field
	jls_thread_t		local_owner;
} jls_object_data_t;

/**
 ***********************************************************************************
 ***********************************************************************************
 * Global Variables
 ***********************************************************************************
 ***********************************************************************************
 **/

extern jls_update_list_t jls_glb_update_list;
extern jls_update_t dummy_update;

extern volatile bool initialize_completed;


extern jls_lockinfo_list_t jls_glb_lockinfo_list_head;
extern jls_lockinfo_list_t jls_glb_lockinfo_list_tail;
extern bool jls_glb_eager_lockset_computation_enabled;

/**
 ***********************************************************************************
 ***********************************************************************************
 * Macros
 ***********************************************************************************
 ***********************************************************************************
 **/


#define jls_print(s)				fprintf(stderr, (s))

#define jls_thread_none 			((jls_thread_t)0)
#define jls_thread_any  			(~(jls_thread_none))

#define jls_join_threads(t,u) 		((t) & (u))

#define jls_current_thread()		jls_jthread_to_thrvar(KTHREAD(current)())

#define jls_get_thread_state(t) 		KTHREAD(get_data)((jthread_t)(t))->jlstate
#define jls_get_current_thread_state()	jls_get_thread_state(jls_current_thread())

/**
 * Conversion Macros
 */

#define jls_object_to_lockvar(o)  	((jls_lockvar_t) o)
#define jls_thread_to_syncvar(t)  	((jls_syncvar_t) t)
#define jls_lockvar_to_syncvar(l) 	((jls_syncvar_t) l)
#define jls_syncvar_to_lockvar(l) 	((jls_lockvar_t) l)
#define jls_jthread_to_thrvar(t)	((jls_thrvar_t) t)
#define jls_thrvar_to_jthread(t) 	((jthread_t) t)
#define jls_volvar_to_syncvar(l) 	((jls_syncvar_t) l)
#define jls_address_to_shrvar(a) 	((jls_shrvar_t) a)
#define jls_address_to_volvar(a) 	((jls_volvar_t) a)
#define jls_shrvar_to_syncvar(v) 	((jls_syncvar_t) v)
#define jls_mutex_to_syncvar(m) 	((jls_syncvar_t) m)
#define jls_mutex_to_lockvar(m) 	((jls_lockvar_t) m)
#define jls_lockvar_to_jobject(l)	((Hjava_lang_Object*)(l))
#define jls_lockvar_to_iLock(l)		((iLock**)(jls_lockvar_to_jobject(l)->lock))

#define compare_listdata(x,y)		((x) == (y))
#define is_list_empty(l)			((l)->head == NULL)
#define is_set_empty(l)				(is_list_empty((l)->elements))
#define jls_list_size(l) 			((l)->size)
#define ASSERT_EMPTY_LIST(l) 		assert((l)->head == NULL);assert((l)->tail == NULL);assert((l)->size == 0)

#define JLS_MAP_NUM_BITS			0x08
#define JLS_MAP_SIZE 				(0x00000001 << JLS_MAP_NUM_BITS)
#define JLS_MAP_MASK 				(JLS_MAP_SIZE - 1)
#define JLS_MAP_INITHASH  			(0xdeadbeef)

#define compare_mapkeys(x,y)		((x) == (y))
#define compare_mapdata(x,y)		((x) == (y))

#define JLS_COMPARE_SYNCVAR(x,y) 	((x) == (y))

#define JLS_SIZEOF(T) 				sizeof(_##T)
#define JLS_ALLOC(T) 				((T)jls_malloc(JLS_SIZEOF(T)))
#define JLS_ALLOC_BULK(T,N) 		(T)jls_malloc(N * JLS_SIZEOF(T))
#define JLS_ALLOC_BULKP(T,N) 		(T*)jls_malloc((N) * sizeof(T))
#define JLS_FREE(T) 				jls_free(T); (T) = NULL
#define JLS_MEMZERO(O,T) 			memset((O), 0, JLS_SIZEOF(T));

#define JLS_SET_INITIAL_CAPACITY 8

#define IS_ARRAY_ACCESS(a)			((a)->type == ArrayIndex)
#define IS_INSTANCE_ACCESS(a)		((a)->type == InstanceField)
#define IS_STATIC_ACCESS(a)			((a)->type == StaticField)


#define FUNC_LIST_APPLY(f) 			void (*f) (void*, void*)
#define FUNC_SET_APPLY(f) 			void (*f) (void*, void*)

#define FUNC_LIST_APPLY_T 			void (*) (void*, void*)
#define FUNC_SET_APPLY_T 			void (*) (void*, void*)

/**
 * Set abstraction
 */

#define jls_set_apply(s, f, a)		jls_list_apply((s), (f), (a))
#define jls_set_add(s, d) 			jls_list_enqueue((s), (d))
#define jls_set_remove(s, d) 		jls_list_remove((s), (d))
#define jls_set_lookup(s, d) 		jls_list_lookup((s), (d))
#define jls_set_clear(s) 			jls_list_clear((s))
#define jls_set_size(s) 			jls_list_size((s))
#define jls_create_set() 			jls_create_list()
#define jls_destroy_set(s) 			jls_destroy_list(s)


extern bool jls_is_lock_owner(jls_lockvar_t, jls_thrvar_t);

#define jls_check_random_lock(l)		(jls_is_lock_owner((l),(jls_current_thread())))

#define jls_get_thread_random_lock(t)	(jls_get_thread_state((t))->random_lock)
#define jls_get_current_thread_random_lock()	(jls_get_current_thread_state()->random_lock)
#define jls_get_thread_lockset(t)		(jls_get_thread_state((t))->lockset)

#define jls_acquire_thread_random_lock(o)		\
	if(jls_get_current_thread_random_lock() == NULL) { jls_get_current_thread_random_lock() = (o); }
	// jls_check_and_set((&jls_get_current_thread_random_lock()), NULL, (o))

#define jls_release_thread_random_lock(o)		\
	if(jls_get_current_thread_random_lock() == (o)) { jls_get_current_thread_random_lock() = NULL; }
	// jls_check_and_set((&jls_get_current_thread_random_lock()), (o), NULL)


#define jls_is_checking_lockinfo(LIv)	((LIv)->lock != 0)
#define jls_is_updating_event_list()	((jls_glb_update_list)->lock != 0)

#define JLS_MAX_UPDATE_ID				(0xFFFFFFFFUL)


// macros about checking accesses
#define checkOwnerThread(owner, last)	((jls_join_threads((owner), (last))) == (last))
#define checkOwnerLock(lock, thread)	(((lock) != NULL) && (jls_is_lock_owner((lock), (thread))))

/**
 ***********************************************************************************
 ***********************************************************************************
 * External Functions
 ***********************************************************************************
 ***********************************************************************************
 **/

/**
 * Initialization Functions
 */

extern void jls_initialize();

/*
 * Event Notification Entries
 */

extern void jls_notify_event_acquire(jls_thread_t, jls_lockvar_t);
extern void jls_notify_event_release(jls_thread_t, jls_lockvar_t);

extern void jls_notify_event_fork(jls_thread_t, jls_thread_t);
extern void jls_notify_event_join(jls_thread_t, jls_thread_t);
extern void jls_notify_event_shrread(jls_var_access_t);
extern void jls_notify_event_shrwrite(jls_var_access_t);
extern void jls_notify_event_volread(jls_thread_t, jls_volvar_t);
extern void jls_notify_event_volwrite(jls_thread_t, jls_volvar_t);

extern void jls_notify_event_objfinalize(jls_thread_t, jls_shrvar_t);

/**
 * Threads
 */

//extern jls_thread_t jls_current_thread();

extern void jls_do_check_variable_access(jls_var_access_t);

extern jls_thread_data_t jls_init_thread_data(threadData*, jls_thrvar_t);
extern void jls_destroy_thread_data(threadData*);

extern jls_lockinfo_t jls_create_lockinfo();

extern jls_lockinfo_t jls_get_slockinfo(Field*, Hjava_lang_Class*);
extern jls_lockinfo_t jls_get_ilockinfo(Field*, Hjava_lang_Object*);
extern jls_lockinfo_t jls_get_alockinfo(jsize, Hjava_lang_Object*);

/**
 * Readset functions
 */

extern void jls_readset_init(jls_readset_t);
extern void jls_readset_clear(jls_readset_t);
extern void jls_readset_enqueue(jls_readset_t, jls_thrvar_t, jls_lockvar_t, jls_update_t);
extern jls_readset_item_t jls_readset_dequeue(jls_readset_t);
extern bool jls_readset_isempty(jls_readset_t);

/**
 * List Functions
 */

extern jls_list_t jls_create_list();
extern void jls_list_reset(jls_list_t);
extern jls_list_item_t jls_list_enqueue(jls_list_t, jls_listdata_t);
extern jls_listdata_t jls_list_dequeue(jls_list_t);
extern bool jls_list_lookup(jls_list_t, jls_listdata_t);
extern jls_listdata_t jls_list_item_remove(jls_list_t, jls_list_item_t);
extern void jls_list_remove(jls_list_t, jls_listdata_t);
extern void jls_list_apply(jls_list_t, FUNC_LIST_APPLY_T, void*);
extern jls_list_item_t jls_list_add_sorted(jls_list_t, jls_listdata_t);
extern void jls_list_clear(jls_list_t);
extern void jls_destroy_list(jls_list_t);

/**
 * Mutex Functions
 */

extern jls_mutex_t jls_create_mutex();
extern void jls_acquire(jls_mutex_t);
extern void jls_release(jls_mutex_t);
extern bool jls_holds_mutex(jls_mutex_t, jthread_t);
extern void jls_destroy_mutex(jls_mutex_t);


/**
 * Map Functions
 */

extern jls_map_t jls_create_map();
extern void jls_map_set(jls_map_t, jls_mapkey_t, jls_mapdata_t);
extern jls_mapdata_t jls_map_get(jls_map_t, jls_mapkey_t);
extern jls_mapdata_t jls_map_remove(jls_map_t, jls_mapkey_t);

/*
 * Memory functions
 */

extern void* jls_malloc(size_t sz);
extern void* jls_realloc(void* mem, size_t sz);
extern void  jls_free(void* mem);

/**
 * Rule and Update Functions
 */

extern void jls_init_rules();
extern jls_update_t jls_create_update(jls_update_type_t, jls_syncvar_t, jls_syncvar_t);
extern jls_update_list_t jls_create_update_list();
extern jls_update_t jls_get_update_list_tail();
// extern void jls_enqueue_update(jls_update_t);
extern void jls_enqueue_update(jls_update_type_t type, jls_syncvar_t chk, jls_syncvar_t add);
extern void jls_print_ref_counts(jls_update_list_t);
// extern jls_update_t jls_get_dummy_update();
// extern bool jls_is_dummy_update(jls_update_t);
extern void jls_apply_updates_for_thread(jls_thread_t);
// extern void jls_set_current_thread_last_update(jls_update_t);
// extern void jls_set_thread_last_update(jls_thread_t, jls_update_t);
// extern jls_update_t jls_get_thread_last_update(jls_thread_t);
extern unsigned long jls_get_thread_last_update_id(jls_thread_t);
extern void jls_set_thread_last_update_id(jls_thread_t, unsigned long);
// extern void jls_set_current_thread_last_update_id(unsigned long);
// extern bool checkOwnerThread(jls_thrvar_t, jls_thrvar_t);
// extern bool checkOwnerLock(jls_lockvar_t, jls_thrvar_t);

#if LOCKSET_GC_UPDATE_LIST
	extern int jls_garbage_collect_update_list(int);
#endif

/**
 * Rule creation functions
 */

extern jls_update_t jls_create_rule_acquire(jls_thread_t, jls_lockvar_t);
extern jls_update_t jls_create_rule_release(jls_thread_t, jls_lockvar_t);

extern jls_update_t jls_create_rule_fork(jls_thread_t, jls_thread_t);
extern jls_update_t jls_create_rule_join(jls_thread_t, jls_thread_t);

extern jls_update_t jls_create_rule_volread(jls_thread_t, jls_volvar_t);
extern jls_update_t jls_create_rule_volwrite(jls_thread_t, jls_volvar_t);

extern jls_update_t jls_create_rule_objfinalize(jls_thread_t, jls_shrvar_t);

/**
 * Object data functions
 */

extern void jls_init_object_data(bool, jls_object_data_t, int, int);
extern void jls_destroy_object_data(Hjava_lang_Object*, Hjava_lang_Class*);

extern int jls_compute_cfield_indexes(Hjava_lang_Class*);

extern int jls_compute_ifield_index(Hjava_lang_Class*, int);
extern int jls_compute_ivfield_index(Hjava_lang_Class*, int);
extern int jls_compute_sfield_index(Hjava_lang_Class*, int);
extern int jls_compute_svfield_index(Hjava_lang_Class*, int);

extern int jls_compute_num_ifields(Hjava_lang_Class*);
extern int jls_compute_num_ivfields(Hjava_lang_Class*);
extern int jls_compute_num_sfields(Hjava_lang_Class*);
extern int jls_compute_num_svfields(Hjava_lang_Class*);

extern int jls_get_ifield_index(Field*);
extern int jls_get_sfield_index(Field*);
extern int jls_get_ivfield_index(Field*);
extern int jls_get_svfield_index(Field*);


/**
 * Error reporting functions
 */
extern jls_var_access_t jls_create_ivar_access(Field*,Hjava_lang_Object*,jthread_t,Method*,int);
extern jls_var_access_t jls_create_svar_access(Field*,Hjava_lang_Class*,jthread_t,Method*,int);
extern jls_var_access_t jls_create_avar_access(jsize,Hjava_lang_Object*,jthread_t,Method*,int);
extern void jls_print_var_access(jls_var_access_t);
extern void jls_report_race(jls_lockinfo_t, jls_var_access_t);
extern int32 jls_get_method_line_number(Method*, uintp);
//extern void jls_throw_race_exception(jls_lockinfo_t, jls_var_access_t);
extern void jls_throw_race_exception(jls_var_access_t, jls_var_access_t);
extern char* jls_var_access_to_string(jls_var_access_t);
extern char* jls_field_access_to_string(jls_var_access_t);
extern char* jls_array_access_to_string(jls_var_access_t);

/**
 * Static information functions
 */
extern void jls_init_check_field(Hjava_lang_Class*);
extern void jls_enable_check_for_class(Hjava_lang_Class*, int);
extern bool jls_is_class_check_enabled(Hjava_lang_Class*);
extern void jls_disable_check_for_access(jls_var_access_t);

/**
 * Eager lockset computation functions
 */
extern void jls_start_eager_gc();
extern void jls_finish_eager_gc();
extern void jls_walk_object(Hjava_lang_Object*);
extern void jls_walk_class(Hjava_lang_Class*);



#endif /* GOLDILOCKS_H_ */
