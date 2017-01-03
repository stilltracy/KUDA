#ifndef CUDAHELPER_H_
#define CUDAHELPER_H_

#define cuda_frame_id()			(blockIdx.x)
#define cuda_num_threads()		(blockDim.x)
#define cuda_thread_id()		(threadIdx.x)

#define cuda_is_last_thread()	(cuda_thread_id() == (cuda_num_threads() - 1))


#if (MEMORY_MODEL == TEXTURE_MEMORY_MODEL)
	#define getEvent(_x_, _y_)	tex2D(tex, (float)(_x_), (float)(_y_))
#else
	#define getEvent(_x_, _y_)	events[((_y_) * CHECKED_BLOCK_SIZE) + (_x_) + offset]	
#endif


// global racy fields, initialized by initRaceChecker()
__device__ BloomKernelFilter d_racyVars;


__device__ inline void d_reportRace(IndexPairList* d_indexPairs, int mem, int x1, int x2, int y, size_t offset) {
	// mark the variable as racy, to prevent checking the race again
	bloom_kernel_add(&d_racyVars, mem);
	// report the data race
	if(d_indexPairs->size < MAX_RACES_TO_REPORT) {
		unsigned int ipair = d_indexPairs->size;
		d_indexPairs->size = ipair + 1;
		if(ipair < MAX_RACES_TO_REPORT) {
			int idx1 = ((y * CHECKED_BLOCK_SIZE) + x1) + offset;
			int idx2 = ((y * CHECKED_BLOCK_SIZE) + x2) + offset;
			d_indexPairs->pairs[ipair] = make_indexpair(idx1, idx2);
		} else {
			d_indexPairs->size = MAX_RACES_TO_REPORT;
		}
	}
	
}
#endif // CUDAHELPER_H_
