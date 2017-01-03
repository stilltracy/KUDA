/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Template project which demonstrates the basics on how to setup a project 
 * example application.
 * Device code.
 */

#ifndef _ERASER_KERNEL_H_
#define _ERASER_KERNEL_H_

#include <stdio.h>
#include "eventlist.h"

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

#ifdef CHECK_AT_GPU
#include "cuPrintf.cu"
#include "bloom_kernel.cu"
#include "kernel_common.cu"
#include "locksetex.cu"

#if (MEMORY_MODEL == TEXTURE_MEMORY_MODEL) 

texture<int4, cudaTextureType2D, cudaReadModeElementType> tex;
__global__ void raceCheckerKernelEraser(int size, int offset, IndexPairList* d_indexPairs)

#elif (MEMORY_MODEL == CONSTANT_MEMORY_MODEL) 

__constant__ Event events[BLOCK_SIZE];
__global__ void raceCheckerKernelEraser(int size, int offset, IndexPairList* d_indexPairs)

#elif (MEMORY_MODEL == SHARED_MEMORY_MODEL) 

__global__ void raceCheckerKernelEraser(Event* events, int size, int offset, IndexPairList* d_indexPairs)

#endif

{
	locksetex_t LSA1, LSR1, LSA2, LSR2;

	const int y = cuda_frame_id();
	const int num_threads = cuda_num_threads();
	const int tid = cuda_thread_id();
		
	// if last thread and offset is defined, then quit
	if((offset > 0) && cuda_is_last_thread()) {
		return;
	}

	for(int index = tid; index < size-1; index += num_threads) {
		// e is the first access
		Event e = getEvent(index, y);
		EventKind kind = EVENT_KIND(e);
		if(IS_ACCESS(kind)) {
			// check the access to mem
			int mem = EVENT_VALUE(e);
			// check if this variable is already identified to be racy
			if(bloom_kernel_lookup(&d_racyVars, mem)) continue;

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
						locksetex_clear(&LSA1);
						locksetex_clear(&LSR1);
						initLS = false;
					}

					// update the locksets
					// phase 1: Compute LSR1
					for(; j < i; ++j) {
						// apply the lockset rule to j. event
						Event e3 = getEvent(j, y);
						int tid3 = EVENT_TID(e3);
						EventKind kind3 = EVENT_KIND(e3);
						
						if(tid3 == tid) {
							int val3 = EVENT_VALUE(e3);	
							if(IS_ACQUIRE(kind3)) {
								locksetex_add(&LSA1, val3);
							} else if(IS_RELEASE(kind3) && !locksetex_lookup(&LSA1, val3)) {
								locksetex_add(&LSR1, val3);
							}
						}
					}
					
					// update the locksets
					// phase 2: Compute LSA2
					locksetex_clear(&LSA2);
					locksetex_clear(&LSR2);
					for(j = i-1; j > index; --j) {
						// apply the lockset rule to j. event
						Event e3 = getEvent(j, y);
						int tid3 = EVENT_TID(e3);
						EventKind kind3 = EVENT_KIND(e3);
						
						if(tid3 == tid2) {
							int val3 = EVENT_VALUE(e3);	
							if(IS_RELEASE(kind3)) {
								locksetex_add(&LSR2, val3);
							} else if(IS_ACQUIRE(kind3) && !locksetex_lookup(&LSR2, val3)) {
								locksetex_add(&LSA2, val3);
							}
						}
					}
					
					// check if the current tid is in the lockset
					if(racy && locksetex_is_intersect_empty(&LSR1, &LSA2)) {
						
						d_reportRace(d_indexPairs, mem, index, i, y, offset);
						
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
}
	

#endif // #ifndef _ERASER_KERNEL_H_

#endif // CHECK_AT_GPU
