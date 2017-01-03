/*******************************************************************************
 * Copyright (c) 2005, 2006 IBM Corporation and others.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     IBM Corporation - initial API and implementation
 *******************************************************************************/

#include "lockset.h"

#ifdef LOCKSET_ENABLED

/**
 ***********************************************************************************
 ***********************************************************************************
 * Global Variables
 ***********************************************************************************
 ***********************************************************************************
 **/
 
jls_update_list_t jls_glb_update_list;

#if LOCKSET_CHECK_LOCKSET_SIZE
jls_spinlock_t	jls_glb_max_lockset_spinlock;
unsigned int jls_glb_max_lockset_size;
char * jls_glb_max_lockset_access;
unsigned long jls_glb_max_lockset_start;
unsigned long jls_glb_max_lockset_end;
double jls_glb_max_lockset_count;
double jls_glb_max_lockset_average;
unsigned int jls_glb_max_lockset_size_for_list;
double jls_glb_max_lockset_count_for_list;
double jls_glb_max_lockset_average_for_list;
#endif

/**
 ***********************************************************************************
 ***********************************************************************************
 * Include the implementing functions
 ***********************************************************************************
 ***********************************************************************************
 **/

#include "lockset/lockset-init.c"
#include "lockset/lockset-mutex.c"
#include "lockset/lockset-list.c"
#include "lockset/lockset-map.c"
#include "lockset/lockset-object.c"
#include "lockset/lockset-class.c"
#include "lockset/lockset-array.c"
#include "lockset/lockset-thread.c"
#include "lockset/lockset-check.c"
#include "lockset/lockset-events.c"
#include "lockset/lockset-update.c"
#include "lockset/lockset-rules.c"
#include "lockset/lockset-report.c"
#include "lockset/lockset-readset.c"

#ifdef LOCKSET_ENABLE_STATICINFO
#include "lockset/lockset-static.c"
#endif

#if ALG_RACETRACK || ALG_VCLOCK || ALG_ERASER || ALG_VCLOCK2
	#include "lockset/lockset_vclock.c"
#endif

#include JLS_LOCKSET_IMPL_CSOURCE

#endif // LOCKSET_ENABLED

#include "lockset/lockset-memory.c"

#if LOCKSET_STATISTICS
	#include "lockset/lockset-stat.c"
#endif

