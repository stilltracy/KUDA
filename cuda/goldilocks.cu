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
* Host code.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <math.h>

// utilities and system includes
#include <shrUtils.h>

// CUDA-C includes
#include <cuda_runtime_api.h>

// includes, project
#include <cutil_inline.h>

// includes, kernels
#include <goldilocks_kernel.cu>
#include <eraser_kernel.cu>
#include <atomicity_kernel.cu>


/************************************************/
#ifdef __cplusplus
extern "C" {
#endif
/************************************************/

#ifdef CHECK_AT_GPU

/********************************************/
// globals:

static inline void waitForKernel(cudaEvent_t stop) {
	cudaError_t err;
	while((err = cudaEventQuery(stop)) != cudaSuccess) {
		ASSERT (err == cudaErrorNotReady);
		SLEEP(10);
	}
}

// timer
unsigned int timer = 0;

#if (MEMORY_MODEL == TEXTURE_MEMORY_MODEL)

	// host memory
	Event* h_block;
	
	// channel descriptor
	cudaChannelFormatDesc channelDesc;
	
	// cuda array
	cudaArray* cu_array;

#elif (MEMORY_MODEL == SHARED_MEMORY_MODEL) 
	// host memory
	Event* h_block;
	// device memory
	Event* d_block;
	
	cudaStream_t streams[NUM_CUDA_STREAMS];
#endif

	
// memory for output
IndexPairList* h_indexPairs[NUM_CUDA_STREAMS];

IndexPairList* d_indexPairs[NUM_CUDA_STREAMS];

/********************************************/
/********************************************/

// setup the host and device memory
void initRaceChecker()
{
	cutilSafeCall( cudaSetDeviceFlags(cudaDeviceScheduleYield) );
	
	BloomFilter bloom_tmp;
	size_t sizeof_block = sizeof(Event) * BLOCK_SIZE;
	
	// init the timer
	//cutilCheckError( cutCreateTimer( &timer ) );
	
#if (MEMORY_MODEL == TEXTURE_MEMORY_MODEL) 
	
	size_t width = CHECKED_BLOCK_SIZE;
	size_t height = NUM_CONCURRENT_KERNELS;


	// init host memory
	h_block = NULL;
	cutilSafeCall( cudaHostAlloc( (void**)&h_block, sizeof_block, cudaHostAllocWriteCombined) );
	
    // allocate array and copy image data
	channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindSigned);
	cutilSafeCall( cudaMallocArray( &cu_array, &channelDesc, width, height )); 
	
	// set texture parameters
	tex.addressMode[0] = cudaAddressModeClamp;
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.filterMode = cudaFilterModePoint;
	tex.normalized = false;    // access with normalized texture coordinates

	// Bind the array to the texture
	cutilSafeCall( cudaBindTextureToArray(tex, cu_array, channelDesc));
	
#elif (MEMORY_MODEL == SHARED_MEMORY_MODEL)
	
	// init host memory
	h_block = NULL;
	cutilSafeCall( cudaHostAlloc( (void**)&h_block, sizeof_block, cudaHostAllocWriteCombined) );
	
	// init device memory
	cutilSafeCall(cudaMalloc((void**) &d_block,  sizeof_block));
	
	if(NUM_CUDA_STREAMS > 1) {
		for (int i = 0; i < NUM_CUDA_STREAMS; ++i) {
			cudaStreamCreate(&streams[i]);
		}
	} else {
		streams[0] = 0;
	}
	
#endif
		
	
	for(int i = 0; i < NUM_CUDA_STREAMS; ++i) {
		// setup the output memory for host
		cutilSafeCall(cudaHostAlloc((void**) &h_indexPairs[i],  sizeof(IndexPairList), cudaHostAllocWriteCombined));
	
		// setup the putput memory for device
		cutilSafeCall(cudaMalloc((void**) &d_indexPairs[i],  sizeof(IndexPairList)));
	}
	
	// initialize the bloom filter 
	ASSERT(sizeof(BloomFilter) == sizeof(BloomKernelFilter));
	bloom_clear(&bloom_tmp);
	cudaMemcpyToSymbol("d_racyVars", &bloom_tmp, sizeof(BloomKernelFilter), 0, cudaMemcpyHostToDevice);
//	cutilSafeCall(cudaMemcpyToSymbol("d_racyVars", &bloom_tmp, sizeof(BloomKernelFilter), 0, cudaMemcpyHostToDevice));
}

/********************************************/
/********************************************/

// dealloc the host and device memory
void finalizeRaceChecker()
{
	cudaEvent_t stop;
	cutilSafeCall( cudaEventCreate( &stop ) );
	waitForKernel(stop);
	// remove the timer
	//cutilCheckError( cutDeleteTimer( timer));
	cutilSafeCall( cudaEventDestroy(stop) );
	
#if (MEMORY_MODEL == TEXTURE_MEMORY_MODEL) 
	
	// Unbind the array from the texture
	cutilSafeCall( cudaUnbindTexture(tex) );

	// free host memory
	cutilSafeCall( cudaFreeHost(h_block) );
	
	// free cuda memory
	cutilSafeCall( cudaFreeArray(cu_array) );
	
#elif (MEMORY_MODEL == SHARED_MEMORY_MODEL)

	// free host memory
	cutilSafeCall( cudaFreeHost(h_block));
	// free device memory
	cutilSafeCall( cudaFree(d_block) );
	if(NUM_CUDA_STREAMS > 1) {
		for (int i = 0; i < NUM_CUDA_STREAMS; ++i) {
			cudaStreamDestroy(streams[i]);
		}
	}
#endif
	
	for(int i = 0; i < NUM_CUDA_STREAMS; ++i) {
		// free output memory for host
		cutilSafeCall( cudaFreeHost(h_indexPairs[i]) );
		
		// free output memory for device
		cutilSafeCall( cudaFree(d_indexPairs[i]) );
	}
}

/********************************************/
/********************************************/

// Event == int4
IndexPairList** raceChecker(Block* block, size_t num_events)
{
	
//	size_t num_events = block->size;
//	if(num_events > BLOCK_SIZE) {
//		num_events = BLOCK_SIZE;
//	}
	size_t width = CHECKED_BLOCK_SIZE;
	size_t height = num_events / width;
	if(height <= 0) return NULL; //#!#
	//return NULL;
	
	unsigned int timer = 0;
    float elapsedTimeInMs = 0.0f;
    cudaEvent_t start, stop;
    cutilSafeCall( cudaEventCreate( &start ) );
    cutilSafeCall( cudaEventCreate( &stop ) );
    
    //cutilCheckError( cutStartTimer(timer));
    cutilSafeCall( cudaEventRecord(start, 0 ) );
    
    // reset d_indexPairs->size (important that size is the first field of IndexPairList
    for(int i = 0; i < NUM_CUDA_STREAMS; ++i) {
		unsigned int zero = 0; 
		cutilSafeCall(cudaMemcpyAsync(&d_indexPairs[i]->size, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice, NULL));
    }
    
//-----------------------------------------------------------------
// prepare the memory
//-----------------------------------------------------------------
#if (MEMORY_MODEL == SHARED_MEMORY_MODEL)	
    size_t sizeof_block = num_events * sizeof(Event);
    ASSERT(sizeof_block > 0);
    
	//initialize the host memory
	memcpy((void*)h_block, (void*)block->events, sizeof_block);
	 
	// cutilSafeCall(cudaMemcpyAsync((void*)d_block, (void*)h_block, sizeof_block, cudaMemcpyHostToDevice, NULL));

	size_t sizeof_half_block = sizeof_block >> 1;
	ASSERT (sizeof_half_block > 0);
	
	for (int i = 0; i < NUM_CUDA_STREAMS; ++i) { 
		// copy to device
		Event* d_block_v = &d_block[(i * (num_events / NUM_CUDA_STREAMS))];  // ((void*)d_block) + (i * sizeof_half_block);
		Event* h_block_v = &h_block[(i * (num_events / NUM_CUDA_STREAMS))];  // ((void*)h_block) + (i * sizeof_half_block);
		
		cutilSafeCall(cudaMemcpyAsync((void*)d_block_v, (void*)h_block_v, sizeof_half_block, cudaMemcpyHostToDevice, streams[i]));
		
		//-----------------------------------------------------------------
		// call the kernel #!# after synchronization 
		//-----------------------------------------------------------------
		waitForKernel(start);
		cutilSafeCall( cudaEventRecord(stop, 0 ) );
		
		if(glbConfig.algorithm == Goldilocks) {
			raceCheckerKernelGoldilocks <<< (height / NUM_CUDA_STREAMS), NUM_THREADS, 0, streams[i] >>> (d_block_v, CHECKED_BLOCK_SIZE, 0, d_indexPairs[i]);
			raceCheckerKernelGoldilocks <<< (height / NUM_CUDA_STREAMS), NUM_THREADS, 0, streams[i] >>> (d_block_v, CHECKED_BLOCK_SIZE, (CHECKED_BLOCK_SIZE >> 1), d_indexPairs[i]);
		} 
		else 
		if(glbConfig.algorithm == Eraser){
			raceCheckerKernelEraser <<< (height >> 1), NUM_THREADS, 0, streams[i] >>> (d_block_v, CHECKED_BLOCK_SIZE, 0, d_indexPairs[i]);
			raceCheckerKernelEraser <<< (height >> 1), NUM_THREADS, 0, streams[i] >>> (d_block_v, CHECKED_BLOCK_SIZE, (CHECKED_BLOCK_SIZE >> 1), d_indexPairs[i]);
		}
		
		// read the number of races
		cutilSafeCall(cudaMemcpyAsync(&h_indexPairs[i]->size, &d_indexPairs[i]->size,  sizeof(unsigned int), cudaMemcpyDeviceToHost, streams[i]));
		
		if(height < NUM_CUDA_STREAMS) break; // if there is only one checked block, then there is only one iteration
	}
    	
#else 
	#if (MEMORY_MODEL == TEXTURE_MEMORY_MODEL) 
	
		size_t sizeof_block = height * width * sizeof(Event);
		
		//initialize the memory
		memcpy((void*)h_block, (void*)block->events, sizeof_block);
			
		// copy image data
		cutilSafeCall( cudaMemcpyToArrayAsync( cu_array, 0, 0, (void*)h_block, sizeof_block, cudaMemcpyHostToDevice, NULL));

	#elif (MEMORY_MODEL == CONSTANT_MEMORY_MODEL)
	
		size_t sizeof_block = num_events * sizeof(Event);
		
		cudaMemcpyToSymbolAsync("events", block->events, num_events * sizeof(Event), 0, cudaMemcpyHostToDevice, NULL);
//		cutilSafeCall(cudaMemcpyToSymbolAsync("events", block->events, num_events * sizeof(Event), 0, cudaMemcpyHostToDevice, NULL));
     
	#endif
		
//-----------------------------------------------------------------
// call the kernel
//-----------------------------------------------------------------
#if NUM_CUDA_STREAMS != 1
#error "NUM_CUDA_STREAMS must be 1 for texture and constant memory"
#endif
		
	if(glbConfig.algorithm == Goldilocks) {
		raceCheckerKernelGoldilocks <<< height, NUM_THREADS >>> (CHECKED_BLOCK_SIZE, 0, d_indexPairs[0]);
		raceCheckerKernelGoldilocks <<< height, NUM_THREADS >>> (CHECKED_BLOCK_SIZE, (CHECKED_BLOCK_SIZE >> 1) , d_indexPairs[0]);
	} 
	else 
	if(glbConfig.algorithm == Eraser){
		raceCheckerKernelEraser <<< height, NUM_THREADS >>> (CHECKED_BLOCK_SIZE, 0, d_indexPairs[0]);
		raceCheckerKernelEraser <<< height, NUM_THREADS >>> (CHECKED_BLOCK_SIZE, (CHECKED_BLOCK_SIZE >> 1) , d_indexPairs[0]);
	}
	
	// read the number of races
	cutilSafeCall(cudaMemcpyAsync(&h_indexPairs[0]->size, &d_indexPairs[0]->size,  sizeof(unsigned int), cudaMemcpyDeviceToHost, NULL));
	
#endif
   	///*
//-----------------------------------------------------------------
// get the results of the check
//-----------------------------------------------------------------
   	cutilSafeCall( cudaEventRecord( stop, 0 ) );

	waitForKernel(stop); // wait for all streams
	
//-----------------------------------------------------------------
// check atomicity
//-----------------------------------------------------------------
#if ATOMICITY_ENABLED
	cudaEvent_t stop2;
	cutilSafeCall( cudaEventCreate( &stop2 ) );
#if (MEMORY_MODEL != ASYNC_SHARED_MEMORY_MODEL)	
	atomicityCheckerKernel <<< height, NUM_THREADS >>> (
		#if (MEMORY_MODEL == SHARED_MEMORY_MODEL)   
				d_block, 
		#endif 
				CHECKED_BLOCK_SIZE, d_indexPairs);
	
	cutilSafeCall( cudaEventRecord( stop2, 0 ) );
	waitForKernel(stop2);
#else
	// code for async memory
#endif
	cutilSafeCall( cudaEventDestroy(stop2) );
#endif // ATOMICITY_ENABLED
	
    //total elapsed time in ms
    //cutilCheckError( cutStopTimer( timer));
    //cutilSafeCall( cudaEventElapsedTime( &elapsedTimeInMs, start, stop ) );
    //cutilCheckError( cutResetTimer( timer));

    int num_races = 0;
#if (MEMORY_MODEL == SHARED_MEMORY_MODEL) && (NUM_CUDA_STREAMS > 1)
    cudaEvent_t stop3;
	cutilSafeCall( cudaEventCreate( &stop3 ) );
    for (int i = 0; i < NUM_CUDA_STREAMS; ++i) {
		if(h_indexPairs[i]->size > 0) {
			num_races += h_indexPairs[i]->size; 
			IF_DEBUG(printf("%u races detected\n", h_indexPairs[i]->size));
			IF_DEBUG(fflush(stdout));
			cutilSafeCall(cudaMemcpyAsync(&h_indexPairs[i]->pairs, &d_indexPairs[i]->pairs,  (h_indexPairs[i]->size * sizeof(IndexPair)), cudaMemcpyDeviceToHost, streams[i]));
		
		}
    }
    cutilSafeCall( cudaEventRecord( stop3, 0 ) );
	waitForKernel(stop3);
	cutilSafeCall( cudaEventDestroy(stop3) );
#else
	if(h_indexPairs[0]->size > 0) {
		num_races += h_indexPairs[0]->size; 
		IF_DEBUG(printf("%u races detected\n", h_indexPairs[0]->size));
		IF_DEBUG(fflush(stdout));
		cutilSafeCall(cudaMemcpy(&h_indexPairs[0]->pairs, &d_indexPairs[0]->pairs,  (h_indexPairs[0]->size * sizeof(IndexPair)), cudaMemcpyDeviceToHost));
	
	}
#endif
    
    //clean up memory
	cutilSafeCall( cudaEventDestroy(stop) );
	cutilSafeCall( cudaEventDestroy(start) );
    
    return (num_races > 0 ? h_indexPairs : NULL);
//*/return NULL;
}

/*****************************************************/


int deviceQuery()
{
    shrSetLogFileName ("deviceQuery.txt");
    shrLog("Starting...\n\n");
    shrLog(" CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

    int deviceCount = 0;
	if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
		shrLog("cudaGetDeviceCount FAILED CUDA Driver and Runtime version may be mismatched.\n");
		shrLog("\nFAILED\n");
		return EXIT_FAILURE;// shrEXIT(argc, argv);
	}

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
        shrLog("There is no device supporting CUDA\n");

    int dev;
	int driverVersion = 0, runtimeVersion = 0;
    for (dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        if (dev == 0) {
			// This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
            if (deviceProp.major == 9999 && deviceProp.minor == 9999)
                shrLog("There is no device supporting CUDA.\n");
            else if (deviceCount == 1)
                shrLog("There is 1 device supporting CUDA\n");
            else
                shrLog("There are %d devices supporting CUDA\n", deviceCount);
        }
        shrLog("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

    #if CUDART_VERSION >= 2020
        // Console log
		cudaDriverGetVersion(&driverVersion);
		shrLog("  CUDA Driver Version:                           %d.%d\n", driverVersion/1000, driverVersion%100);
		cudaRuntimeGetVersion(&runtimeVersion);
		shrLog("  CUDA Runtime Version:                          %d.%d\n", runtimeVersion/1000, runtimeVersion%100);
    #endif
        shrLog("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

		char msg[256];
		sprintf(msg, "  Total amount of global memory:                 %llu bytes\n", (unsigned long long) deviceProp.totalGlobalMem);
		shrLog(msg);
    #if CUDART_VERSION >= 2000
        shrLog("  (%2d) Multiprocessors x (%2d) CUDA Cores/MP:     %d CUDA Cores\n",
			deviceProp.multiProcessorCount,
			ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
			ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
    #endif
        shrLog("  Total amount of constant memory:               %u bytes\n", deviceProp.totalConstMem);
        shrLog("  Total amount of shared memory per block:       %u bytes\n", deviceProp.sharedMemPerBlock);
        shrLog("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
        shrLog("  Warp size:                                     %d\n", deviceProp.warpSize);
        shrLog("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
        shrLog("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        shrLog("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        shrLog("  Maximum memory pitch:                          %u bytes\n", deviceProp.memPitch);
    #if CUDART_VERSION >= 4000
//		shrLog("  Memory Bus Width:                              %d-bit\n", deviceProp.memBusWidth);
//		shrLog("  Memory Clock rate:                             %.2f Mhz\n", deviceProp.memoryClock * 1e-3f);
    #endif

		shrLog("  Texture alignment:                             %u bytes\n", deviceProp.textureAlignment);
        shrLog("  Clock rate:                                    %.2f GHz\n", deviceProp.clockRate * 1e-6f);
    #if CUDART_VERSION >= 2000
        shrLog("  Concurrent copy and execution:                 %s\n", deviceProp.deviceOverlap ? "Yes" : "No");
    #endif
    #if CUDART_VERSION >= 4000
		shrLog("  # of Asynchronous Copy Engines:                %d\n", deviceProp.asyncEngineCount);
    #endif
    #if CUDART_VERSION >= 2020
        shrLog("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
        shrLog("  Integrated:                                    %s\n", deviceProp.integrated ? "Yes" : "No");
        shrLog("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
        shrLog("  Compute mode:                                  %s\n", deviceProp.computeMode == cudaComputeModeDefault ?
			                                                            "Default (multiple host threads can use this device simultaneously)" :
		                                                                deviceProp.computeMode == cudaComputeModeExclusive ?
																		"Exclusive (only one host thread at a time can use this device)" :
		                                                                deviceProp.computeMode == cudaComputeModeProhibited ?
																		"Prohibited (no host thread can use this device)" :
																		"Unknown");
    #endif
    #if CUDART_VERSION >= 3000
        shrLog("  Concurrent kernel execution:                   %s\n", deviceProp.concurrentKernels ? "Yes" : "No");
    #endif
    #if CUDART_VERSION >= 3010
        shrLog("  Device has ECC support enabled:                %s\n", deviceProp.ECCEnabled ? "Yes" : "No");
    #endif
    #if CUDART_VERSION >= 3020
		shrLog("  Device is using TCC driver mode:               %s\n", deviceProp.tccDriver ? "Yes" : "No");
    #endif

	}

    // csv masterlog info
    // *****************************
    // exe and CUDA driver name
    shrLog("\n");
	std::string sProfileString = "deviceQuery, CUDA Driver = CUDART";
    char cTemp[10];

    // driver version
    sProfileString += ", CUDA Driver Version = ";
    #ifdef WIN32
	    sprintf_s(cTemp, 10, "%d.%d", driverVersion/1000, driverVersion%100);
    #else
	    sprintf(cTemp, "%d.%d", driverVersion/1000, driverVersion%100);
    #endif
    sProfileString +=  cTemp;

    // Runtime version
    sProfileString += ", CUDA Runtime Version = ";
    #ifdef WIN32
	    sprintf_s(cTemp, 10, "%d.%d", runtimeVersion/1000, runtimeVersion%100);
    #else
	    sprintf(cTemp, "%d.%d", runtimeVersion/1000, runtimeVersion%100);
    #endif
    sProfileString +=  cTemp;

    // Device count
    sProfileString += ", NumDevs = ";
    #ifdef WIN32
        sprintf_s(cTemp, 10, "%d", deviceCount);
    #else
        sprintf(cTemp, "%d", deviceCount);
    #endif
    sProfileString += cTemp;

    // First 2 device names, if any
    for (dev = 0; dev < ((deviceCount > 2) ? 2 : deviceCount); ++dev)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        sProfileString += ", Device = ";
        sProfileString += deviceProp.name;
    }
    sProfileString += "\n";
    shrLogEx(LOGBOTH | MASTER, 0, sProfileString.c_str());

    // finish
    shrLog("\n\nPASSED\n");
    return EXIT_SUCCESS; // shrEXIT(argc, argv);
}

#endif // CHECK_AT_GPU

/************************************************/
#ifdef __cplusplus
}
#endif
/************************************************/
