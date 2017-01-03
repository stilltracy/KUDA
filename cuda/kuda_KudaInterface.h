/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class kuda_KudaInterface */

#ifndef _Included_kuda_KudaInterface
#define _Included_kuda_KudaInterface
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     kuda_KudaInterface
 * Method:    Native_InitEventList
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_kuda_KudaInterface_Native_1InitEventList
  (JNIEnv *, jclass);

/*
 * Class:     kuda_KudaInterface
 * Method:    Native_FinalizeEventList
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_kuda_KudaInterface_Native_1FinalizeEventList
  (JNIEnv *, jclass);

/*
 * Class:     kuda_KudaInterface
 * Method:    Native_InitStatistics
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_kuda_KudaInterface_Native_1InitStatistics
  (JNIEnv *, jclass);

/*
 * Class:     kuda_KudaInterface
 * Method:    Native_PrintStatistics
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_kuda_KudaInterface_Native_1PrintStatistics
  (JNIEnv *, jclass, jstring);

/*
 * Class:     kuda_KudaInterface
 * Method:    Native_RecordEvent_SharedRead
 * Signature: (III)V
 */
JNIEXPORT void JNICALL Java_kuda_KudaInterface_Native_1RecordEvent_1SharedRead
  (JNIEnv *, jclass, jint, jlong, jint);

/*
 * Class:     kuda_KudaInterface
 * Method:    Native_RecordEvent_SharedWrite
 * Signature: (III)V
 */
JNIEXPORT void JNICALL Java_kuda_KudaInterface_Native_1RecordEvent_1SharedWrite
  (JNIEnv *, jclass, jint, jlong, jint);

/*
 * Class:     kuda_KudaInterface
 * Method:    Native_RecordEvent_VolatileRead
 * Signature: (III)V
 */
JNIEXPORT void JNICALL Java_kuda_KudaInterface_Native_1RecordEvent_1VolatileRead
  (JNIEnv *, jclass, jint, jlong, jint);

/*
 * Class:     kuda_KudaInterface
 * Method:    Native_RecordEvent_VolatileWrite
 * Signature: (III)V
 */
JNIEXPORT void JNICALL Java_kuda_KudaInterface_Native_1RecordEvent_1VolatileWrite
  (JNIEnv *, jclass, jint, jlong, jint);

/*
 * Class:     kuda_KudaInterface
 * Method:    Native_RecordEvent_Lock
 * Signature: (III)V
 */
JNIEXPORT void JNICALL Java_kuda_KudaInterface_Native_1RecordEvent_1Lock
  (JNIEnv *, jclass, jint, jint, jint);

/*
 * Class:     kuda_KudaInterface
 * Method:    Native_RecordEvent_Unlock
 * Signature: (III)V
 */
JNIEXPORT void JNICALL Java_kuda_KudaInterface_Native_1RecordEvent_1Unlock
  (JNIEnv *, jclass, jint, jint, jint);

/*
 * Class:     kuda_KudaInterface
 * Method:    Native_RecordEvent_RLock
 * Signature: (III)V
 */
JNIEXPORT void JNICALL Java_kuda_KudaInterface_Native_1RecordEvent_1RLock
  (JNIEnv *, jclass, jint, jint, jint);

/*
 * Class:     kuda_KudaInterface
 * Method:    Native_RecordEvent_WLock
 * Signature: (III)V
 */
JNIEXPORT void JNICALL Java_kuda_KudaInterface_Native_1RecordEvent_1WLock
  (JNIEnv *, jclass, jint, jint, jint);

/*
 * Class:     kuda_KudaInterface
 * Method:    Native_RecordEvent_RWUnlock
 * Signature: (III)V
 */
JNIEXPORT void JNICALL Java_kuda_KudaInterface_Native_1RecordEvent_1RWUnlock
  (JNIEnv *, jclass, jint, jint, jint);

/*
 * Class:     kuda_KudaInterface
 * Method:    Native_RecordEvent_Fork
 * Signature: (III)V
 */
JNIEXPORT void JNICALL Java_kuda_KudaInterface_Native_1RecordEvent_1Fork
  (JNIEnv *, jclass, jint, jint, jint);

/*
 * Class:     kuda_KudaInterface
 * Method:    Native_RecordEvent_Join
 * Signature: (III)V
 */
JNIEXPORT void JNICALL Java_kuda_KudaInterface_Native_1RecordEvent_1Join
  (JNIEnv *, jclass, jint, jint, jint);

/*
 * Class:     kuda_KudaInterface
 * Method:    Native_LockForShared
 * Signature: (II)V
 */
JNIEXPORT void JNICALL Java_kuda_KudaInterface_Native_1LockForShared
  (JNIEnv *, jclass, jint, jlong);

/*
 * Class:     kuda_KudaInterface
 * Method:    Native_UnlockForShared
 * Signature: (II)V
 */
JNIEXPORT void JNICALL Java_kuda_KudaInterface_Native_1UnlockForShared
  (JNIEnv *, jclass, jint, jlong);
  
JNIEXPORT void JNICALL Java_kuda_KudaInterface_Native_1Wild
  (JNIEnv *env, jclass cls, jintArray arr);

#ifdef __cplusplus
}
#endif
#endif
