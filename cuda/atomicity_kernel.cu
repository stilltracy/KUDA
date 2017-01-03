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

#ifndef _ATOMICITY_KERNEL_H_
#define _ATOMICITY_KERNEL_H_

#include <stdio.h>
#include "eventlist.h"

#include "cuPrintf.cu"
#include "bloom_kernel.cu"


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

typedef struct {
	bool left;
	bool right;
} Mover;

#define MakeBothMover(m)	(m).left = (m).right = true
#define MakeLeftMover(m)	(m).left = true; (m).right = false
#define MakeRighttMover(m)	(m).left = false; (m).right = true
#define MakeNonMover(m)		(m).left = (m).right = false

#define IsLeftMover(m)		((m).left == true)
#define IsRightMover(m)		((m).right == true)
#define IsBothMover(m)		(IsLeftMover(m) && IsRightMover(m))
#define IsNonMover(m)		(!IsLeftMover(m) && !IsRightMover(m))



#if (MEMORY_MODEL == TEXTURE_MEMORY_MODEL) 

// texture<int2, cudaTextureType2D, cudaReadModeElementType> tex;
__global__ void atomicityCheckerKernel(int size, IndexPairList* d_indexPairs)

#elif (MEMORY_MODEL == CONSTANT_MEMORY_MODEL) 

// __constant__ Event events[BLOCK_SIZE];
__global__ void atomicityCheckerKernel(int size, int offset, IndexPairList* d_indexPairs)

#elif (MEMORY_MODEL == SHARED_MEMORY_MODEL) 

__global__ void atomicityCheckerKernel(Event* events, int size, int offset, IndexPairList* d_indexPairs)

#endif

{
	Mover mover;

	const int y = blockIdx.x;
	const int num_threads = blockDim.x;
	
	for(int index = threadIdx.x; index < size-1; index += num_threads) {
		// e is the first access
		Event e = getEvent(index, y);
		EventKind kind = EVENT_KIND(e);
		if(IS_CALL(kind)) {
			// init the mover type
			MakeBothMover(mover);
			
			int proc = EVENT_VALUE(e);
			int tid = EVENT_TID(e);

			int depth = 1;
			
			for(int i = index + 1; i < size; ++i) {
				Event e2 = getEvent(i, y);
				int tid2 = EVENT_TID(e2);
				
				// check the owner of the event
				if(tid != tid2) continue;
				
				EventKind kind2 = EVENT_KIND(e2);
				int proc2 = EVENT_VALUE(e2);
				
				if(IS_CALL(kind2)) {
					depth++;
				} 
				else 
				if(IS_RETURN(kind2)) {
					depth--;
					
					// check if we reached the end of the call
					if(depth == 0 && proc == proc2) {
						// assume we have checked it
						// TODO: what to do here? report if there is a violation?
						break;
					}
				} 
				else 
				if(IS_LOCK(kind2)) {
					if(!IsRightMover(mover)) {
						// error
						break;
					}	
					mover.left = false; // only right mover
				} 
				else 
				if(IS_UNLOCK(kind2)) {
					MakeLeftMover(mover); // become left mover from now on
				} 
				else 
				if(IS_ACCESS(kind2)) {
					// check if racy
					int mem = EVENT_VALUE(e2);
					if(bloom_kernel_lookup(&d_racyVars, mem)) {
						// racy
						if(!IsRightMover(mover)) {
							// error
							break;
						}
						MakeNonMover(mover);
					}
					// race-free variables are both movers
				}
			}
		}
	}
}

#endif // #ifndef _ATOMICITY_KERNEL_H_

#endif // CHECK_AT_GPU
