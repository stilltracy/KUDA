
#ifndef VC_H_
#define VC_H_

#include <iostream>
#include <iomanip>
#include <fstream>

#include <vector>
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

typedef map<int,int> VC;
typedef VC Epoch;

/************************************************/

static map<int, VC> glbC;
static map<int, VC> glbL;
static map<int, VC> glbW;
static map<int, VC> glbR;
static map<int, Epoch> glbE;

/************************************************/

static BloomFilter glbRacyVars;

/************************************************/

Epoch vc_epoch(int t, int c);
int vc_epoch_tid(Epoch e);
int vc_epoch_clock(Epoch e);
VC vc_get_vc(map<int, VC>& m, int k);
void vc_set_vc(map<int, VC>& m, int k, VC vc);
VC vc_new();
int vc_get(VC vc, int t);
VC vc_set(VC vc, int t, int c);
VC vc_inc(VC vc, int t);
VC vc_cup(VC vc1, VC vc2);
bool vc_leq(VC vc1, VC vc2, int t);
bool vc_leq_all(VC vc1, VC vc2);
bool vc_leq_epoch(Epoch e, VC vc);
bool vc_is_epoch(VC vc);
bool vc_epoch_eq(Epoch e1, Epoch e2);

/************************************************/

inline Epoch vc_epoch(int t, int c) {
	assert(c >= 0);
	Epoch e = vc_new();
	if(c > 0) {
		vc_set(e, t, c);
	}
	return e;
}

inline bool vc_is_epoch(VC vc) {
	return vc.size() <= 1;
}

inline bool vc_epoch_eq(Epoch e1, Epoch e2) {
	assert(vc_is_epoch(e1));
	assert(vc_is_epoch(e2));
	return ((vc_epoch_tid(e1) == vc_epoch_tid(e2)) && (vc_epoch_clock(e1) == vc_epoch_clock(e2)));
}


inline int vc_epoch_tid(Epoch e) {
	if(e.size() == 0) {
		return 0;
	}
	assert (e.size() == 1);
	VC::iterator it=e.begin();
	return it->first;
}

inline int vc_epoch_clock(Epoch e) {
	if(e.size() == 0) {
		return 0;
	}
	assert (e.size() == 1);
	VC::iterator it=e.begin();
	return it->second;
}

inline VC vc_get_vc(map<int, VC>& m, int k) {
	map<int, VC>::iterator it = m.find(k);
	if(it != m.end()) {
		return it->second;
	}
	return VC();
}

inline void vc_set_vc(map<int, VC>& m, int k, VC vc) {
	m[k] = vc;
}

inline VC vc_new() {
	VC _vc_;
	return _vc_;
}

inline int vc_get(VC vc, int t) {
	VC::iterator it = vc.find(t);
	if(it != vc.end()) {
		int c = it->second;
		assert(c > 0);
		return c;
	}
	return 0; // rest is 0
}

inline VC vc_set(VC vc, int t, int c) {
	if(c > 0) {
		vc[t] = c;
	} else {
		vc.erase(t);
	}
	return vc;
}

inline VC vc_inc(VC vc, int t) {
	VC _vc_ = vc;
	int c = vc_get(_vc_, t);
	_vc_[t] = c + 1;
	return _vc_;
}

inline VC vc_cup(VC vc1, VC vc2) {
	VC _vc_ = vc1;
	for (VC::iterator it=vc2.begin() ; it != vc2.end(); it++ ) {
		int t = it->first;
		int c1 = vc_get(_vc_, t);
		int c2 = it->second;
		if(c2 > c1) {
			_vc_[t] = c2;
		}
	}
	return _vc_;
}

inline bool vc_leq(VC vc1, VC vc2, int t) {
	int c1 = vc_get(vc1, t);
	int c2 = vc_get(vc2, t);
	return c1 <= c2;
}

inline bool vc_leq_all(VC vc1, VC vc2) {
	for (VC::iterator it=vc2.begin() ; it != vc2.end(); it++ ) {
		int t = it->first;
		int c1 = vc_get(vc1, t);
		int c2 = it->second;
		if(c1 > c2) {
			return false;
		}
	}
	return true;
}

inline bool vc_leq_epoch(Epoch e, VC vc) {
	assert(vc_is_epoch(e));
	int t = vc_epoch_tid(e);
	int c1 = vc_epoch_clock(e);
	int c2 = vc_get(vc, t);
	return c1 <= c2;
}

/************************************************/

inline bool isRacy(int var) {
	return bloom_lookup(&glbRacyVars, var);
}

inline void setRacy(int var) {
	return bloom_add(&glbRacyVars, var);
}


/************************************************/

#endif // VC_H_
