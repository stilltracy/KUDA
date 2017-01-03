
#include "kuda_KudaInterface.h"
#include "eventlist_common.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline int identityHashCode(JNIEnv *env, jobject obj)
{
	// compute the memory address
	jclass systemClass = env->FindClass("java/lang/System");

	jmethodID mid = env->GetStaticMethodID(systemClass, "identityHashCode", "(Ljava/lang/Object;)I");

	if (mid == NULL) {
		printf("ERROR! Could not find the method identityHashCode");
		return NULL;  /* method not found */
	}

	jint hashCode = env->CallStaticIntMethod(systemClass, mid, obj);

	return hashCode;
}

static inline int threadId(JNIEnv *env, jobject thr)
{
	// compute the memory address
	jclass threadClass = env->FindClass("java/lang/Thread");

	jmethodID mid = env->GetMethodID(threadClass, "getId", "()J");

	if (mid == NULL) {
		printf("ERROR! Could not find the method getId");
		return NULL;  /* method not found */
	}

	jlong tid = env->CallLongMethod(threadClass, mid);

#define HALF_LONG_BITS ((sizeof(jlong)*8)>>1)
	return (int) ((tid << HALF_LONG_BITS) >> HALF_LONG_BITS);
}

/*
 * Class:     kuda_KudaInterface
 * Method:    Native_InitEventList
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_kuda_KudaInterface_Native_1InitEventList
  (JNIEnv * env, jclass cls)
{
	initEventList();
}

/*
 * Class:     kuda_KudaInterface
 * Method:    Native_FinalizeEventList
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_kuda_KudaInterface_Native_1FinalizeEventList
  (JNIEnv * env, jclass cls)
{
	finalizeEventList();
}

/*
 * Class:     kuda_KudaInterface
 * Method:    Native_InitStatistics
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_kuda_KudaInterface_Native_1InitStatistics
  (JNIEnv * env, jclass cls)
{
	initStatistics();
}

/*
 * Class:     kuda_KudaInterface
 * Method:    Native_PrintStatistics
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_kuda_KudaInterface_Native_1PrintStatistics
  (JNIEnv * env, jclass cls, jstring str)
{
	if(str == NULL) {
		printStatistics(stdout);
	} else {
		jboolean isCopy = JNI_FALSE;
		const char* file_name = env->GetStringUTFChars(str, &isCopy);

		FILE * file = fopen (file_name, "w");
		printStatistics(file);
		fclose(file);

		env->ReleaseStringUTFChars(str, file_name);
	}
}

/*
 * Class:     kuda_KudaInterface
 * Method:    Native_RecordEvent_SharedRead
 * Signature: (III)V
 */
JNIEXPORT void JNICALL Java_kuda_KudaInterface_Native_1RecordEvent_1SharedRead
(JNIEnv *env, jclass cls, jint tid, jlong mem, jint instr)
{
	RecordEvent_SharedRead(tid, mem, instr);
}

/*
 * Class:     kuda_KudaInterface
 * Method:    Native_RecordEvent_SharedWrite
 * Signature: (III)V
 */
JNIEXPORT void JNICALL Java_kuda_KudaInterface_Native_1RecordEvent_1SharedWrite
(JNIEnv *env, jclass cls, jint tid, jlong mem, jint instr)
{
	RecordEvent_SharedWrite(tid, mem, instr);
}


/*
 * Class:     kuda_KudaInterface
 * Method:    Native_RecordEvent_VolatileRead
 * Signature: (III)V
 */
JNIEXPORT void JNICALL Java_kuda_KudaInterface_Native_1RecordEvent_1VolatileRead
(JNIEnv *env, jclass cls, jint tid, jlong mem, jint instr)
{
	RecordEvent_AtomicRead(tid, mem, instr);
}


/*
 * Class:     kuda_KudaInterface
 * Method:    Native_RecordEvent_VolatileWrite
 * Signature: (III)V
 */
JNIEXPORT void JNICALL Java_kuda_KudaInterface_Native_1RecordEvent_1VolatileWrite
(JNIEnv *env, jclass cls, jint tid, jlong mem, jint instr)
{
	RecordEvent_AtomicWrite(tid, mem, instr);
}


/*
 * Class:     kuda_KudaInterface
 * Method:    Native_RecordEvent_Lock
 * Signature: (III)V
 */
JNIEXPORT void JNICALL Java_kuda_KudaInterface_Native_1RecordEvent_1Lock
(JNIEnv *env, jclass cls, jint tid, jint lock, jint instr)
{
	RecordEvent_Lock(tid, lock, instr);
}


/*
 * Class:     kuda_KudaInterface
 * Method:    Native_RecordEvent_Unlock
 * Signature: (III)V
 */
JNIEXPORT void JNICALL Java_kuda_KudaInterface_Native_1RecordEvent_1Unlock
(JNIEnv *env, jclass cls, jint tid, jint lock, jint instr)
{
	RecordEvent_Unlock(tid, lock, instr);
}


/*
 * Class:     kuda_KudaInterface
 * Method:    Native_RecordEvent_RLock
 * Signature: (III)V
 */
JNIEXPORT void JNICALL Java_kuda_KudaInterface_Native_1RecordEvent_1RLock
(JNIEnv *env, jclass cls, jint tid, jint lock, jint instr)
{
	RecordEvent_RLock(tid, lock, instr);
}


/*
 * Class:     kuda_KudaInterface
 * Method:    Native_RecordEvent_WLock
 * Signature: (III)V
 */
JNIEXPORT void JNICALL Java_kuda_KudaInterface_Native_1RecordEvent_1WLock
(JNIEnv *env, jclass cls, jint tid, jint lock, jint instr)
{
	RecordEvent_WLock(tid, lock, instr);
}


/*
 * Class:     kuda_KudaInterface
 * Method:    Native_RecordEvent_RWUnlock
 * Signature: (III)V
 */
JNIEXPORT void JNICALL Java_kuda_KudaInterface_Native_1RecordEvent_1RWUnlock
(JNIEnv *env, jclass cls, jint tid, jint lock, jint instr)
{
	RecordEvent_RWUnlock(tid, lock, instr);
}


/*
 * Class:     kuda_KudaInterface
 * Method:    Native_RecordEvent_Fork
 * Signature: (III)V
 */
JNIEXPORT void JNICALL Java_kuda_KudaInterface_Native_1RecordEvent_1Fork
(JNIEnv *env, jclass cls, jint tid, jint _tid, jint instr)
{
	RecordEvent_Fork(tid, _tid, instr);
}


/*
 * Class:     kuda_KudaInterface
 * Method:    Native_RecordEvent_Join
 * Signature: (III)V
 */
JNIEXPORT void JNICALL Java_kuda_KudaInterface_Native_1RecordEvent_1Join
(JNIEnv *env, jclass cls, jint tid, jint _tid, jint instr)
{
	RecordEvent_Join(tid, _tid, instr);
}


/*
 * Class:     kuda_KudaInterface
 * Method:    Native_LockForShared
 * Signature: (II)V
 */
JNIEXPORT void JNICALL Java_kuda_KudaInterface_Native_1LockForShared
(JNIEnv *env, jclass cls, jint tid, jlong mem)
{
	lockForShared(tid);
}


/*
 * Class:     kuda_KudaInterface
 * Method:    Native_UnlockForShared
 * Signature: (II)V
 */
JNIEXPORT void JNICALL Java_kuda_KudaInterface_Native_1UnlockForShared
(JNIEnv *env, jclass cls, jint tid, jlong mem)
{
	unlockForShared(tid);
}

JNIEXPORT void JNICALL Java_kuda_KudaInterface_Native_1Wild
(JNIEnv *env, jclass cls, jintArray arr)
{
	 jint* buf;
     jint i,opcd,arg1,arg2,arg3;
     i = 0;
     jsize limit;
     buf = env->GetIntArrayElements(arr, NULL);
     jint* buf_rel = buf;
     limit = env->GetArrayLength(arr);
     
     if (buf == NULL) {
         printf("ERROR! Could not get array! \n");
     }
     while(i < limit-3)
     {
      	opcd = buf[i++];
       	arg1 = buf[i++];
       	arg2 = buf[i++];
       	arg3 = buf[i++];
       	switch(opcd)
       	{
	       	case 1: RecordEvent_SharedRead(arg1, arg2, arg3); break;
	       	case 2: RecordEvent_SharedWrite(arg1, arg2, arg3); break;
			case 3: RecordEvent_AtomicRead(arg1, arg2, arg3); break;
			case 4: RecordEvent_AtomicWrite(arg1, arg2, arg3); break;
			case 5: RecordEvent_Lock(arg1, arg2, arg3); break;
			case 6: RecordEvent_Unlock(arg1, arg2, arg3); break;
			case 7: RecordEvent_Fork(arg1, arg2, arg3); break;
			case 8: RecordEvent_Join(arg1, arg2, arg3); break;
			default: ;//printf("end of jni event list");
       	}
     }

     env->ReleaseIntArrayElements(arr, buf_rel, 0);
}


#ifdef __cplusplus
}
#endif
