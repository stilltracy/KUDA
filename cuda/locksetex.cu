
#ifndef LOCKSETEX_CU
#define LOCKSETEX_CU


#include <stdio.h>
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
#include <math.h>
#include <stdint.h>

#include "bloom_kernel.cu"

#ifdef CHECK_AT_GPU

/************************************************/

#define LOCKSETEX_MAXSIZE  8

typedef struct {
	lockset_t bloom;
	int size;
	int set[LOCKSETEX_MAXSIZE];
} locksetex_t;

/************************************************/

__device__  void locksetex_add(locksetex_t* ls, int element) {
	lockset_add(&ls->bloom, element);
	if(ls->size < LOCKSETEX_MAXSIZE) {
		ls->set[ls->size] = element;
		ls->size++;
	}
}

__device__  bool locksetex_lookup(locksetex_t* ls, int element) {
	return lockset_lookup(&ls->bloom, element);
}

__device__ void locksetex_clear(locksetex_t* ls) {
	lockset_clear(&ls->bloom);
	ls->size = 0;
}

__device__  void locksetex_init(locksetex_t* ls, int element) {
	lockset_init(&ls->bloom, element);
	ls->set[0] = element;
	ls->size = 1;
}

__device__  bool locksetex_is_intersect_empty(locksetex_t* ls1, locksetex_t* ls2) {
	for(int i = 0; i < ls1->size; ++i) {
		if(locksetex_lookup(ls2, ls1->set[i])) {
			return true;
		}
	}
	return false;
}

/************************************************/

#endif // CHECK_AT_GPU


#endif // LOCKSETEX_CU