
#ifndef LOCKSET_H_
#define LOCKSET_H_

#include <iostream>
#include <iomanip>
#include <fstream>

#include <vector>
#include <set>
#include <ctime>
#include <map>

/************************************************/

#include "eventlist_common.h"
#include "bloom.h"

/************************************************/

struct timeval startTime;
struct timeval endTime;
struct timeval runningTime;

/************************************************/

typedef set<int>	lockset_t;

/************************************************/

static map<int, lockset_t> glbLS;
static map<int, lockset_t> glbLH;
static map<int, lockset_t> glbWLH;

static BloomFilter glbRacyVars;

/************************************************/

inline bool lockset_is_var_initialized(int var) {
	map<int, lockset_t>::iterator it = glbLS.find(var);
	return (it != glbLS.end());
}

inline lockset_t lockset_get_for_tid(int tid) {
	map<int, lockset_t>::iterator it = glbLH.find(tid);
	if(it != glbLH.end()) {
		return it->second;
	}
	lockset_t lh;
	return lh;
}

inline lockset_t lockset_getw_for_tid(int tid) {
	map<int, lockset_t>::iterator it = glbWLH.find(tid);
	if(it != glbWLH.end()) {
		return it->second;
	}
	lockset_t lh;
	return lh;
}

inline lockset_t lockset_get_for_var(int var) {
	map<int, lockset_t>::iterator it = glbLS.find(var);
	if(it != glbLS.end()) {
		return it->second;
	}
	lockset_t ls;
	return ls;
}

inline void lockset_set_for_tid(int tid, lockset_t lh) {
	glbLH[tid] = lh;
}

inline void lockset_setw_for_tid(int tid, lockset_t lh) {
	glbWLH[tid] = lh;
}

inline void lockset_set_for_var(int var, lockset_t ls) {
	glbLS[var] = ls;
}

inline lockset_t lockset_intersect(lockset_t LS, lockset_t LH) {
	lockset_t LS2;
	lockset_t::iterator it;
	for ( it=LS.begin() ; it != LS.end(); it++ ) {
		lockset_t::iterator it2 = LH.find(*it);
		if(it2 != LH.end()) {
			LS2.insert(*it);
		}
	}
	return LS2;
}

inline bool isRacy(int var) {
	return bloom_lookup(&glbRacyVars, var);
}

inline void setRacy(int var) {
	return bloom_add(&glbRacyVars, var);
}

inline void freeAll() {

}

/************************************************/

#endif // LOCKSET_H_
