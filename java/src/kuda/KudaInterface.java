package kuda;

import acme.util.Util;

import java.io.File;

/**
 * Copyright (c) 2010-2011,
 * Tayfun Elmas    <elmas@cs.berkeley.edu>
 * All rights reserved.
 * <p/>
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 * <p/>
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * <p/>
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 * <p/>
 * 3. The names of the contributors may not be used to endorse or promote
 * products derived from this software without specific prior written
 * permission.
 * <p/>
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

public class KudaInterface {

    public static final int
        OP_SHARED_READ    = 1,
        OP_SHARED_WRITE   = 2,
        OP_LOCK           = 3,
        OP_UNLOCK         = 4,
        OP_FORK           = 5,
        OP_JOIN           = 6,
        OP_VOLATILE_READ  = 7,
        OP_VOLATILE_WRITE = 8;

    public static final boolean VERBOSE = false;
    public static void log(String s, int...params) {
        if(VERBOSE) {
            StringBuffer strb = new StringBuffer();
            strb.append(s);
            if(params != null && params.length > 0) {
                strb.append("[");
                String comma = "";
                for(int i : params) {
                    strb.append(comma).append(i);
                    comma = ",";
                }
                strb.append("]");
            }
            System.out.println(strb.toString());
        }
    }
	
	//---------------------------------------
	public static final String KUDA_HOME_JAVA = System.getProperty("kuda.home.java");
	static {
        try {
            String os = System.getProperty("os.name");
            String libkuda = (os.equalsIgnoreCase("MAC OS X") ? "libkuda.jnilib" : "libkuda.so");
            System.load(KUDA_HOME_JAVA + File.separator + "lib" + File.separator + libkuda);
//            System.loadLibrary("kuda");
        } catch(UnsatisfiedLinkError e) {
            System.out.println("Java library path: " + System.getProperty("java.library.path"));
            throw e;
        }
    }
	
	//---------------------------------------
	
	static public void Initialize() {
		// initialize everything
		Native_InitStatistics();
		Native_InitEventList();

		// schedule the finalizer when the program ends
		//final String filename = "./statistics.txt"; // Can
		Runtime.getRuntime().addShutdownHook(new Thread(new Runnable() {
			public void run() {
				Native_FinalizeEventList();
				Native_PrintStatistics(null); // null means: write to stdout/console
				//Native_PrintStatistics(null); // null means: write to stdout/console
			}
		}));
        log("Kuda interface initialized.");
	}
	
	//---------------------------------------
	// Methods to compute identifiers for threads and objects
	//---------------------------------------
	
	static public int currentThreadId() {
		return threadId(Thread.currentThread());
	}
	
	static public int threadId(Thread thr) {
		return new Long(thr.getId()).hashCode();
	}
	
	static public int objectId(Object obj) {
		return System.identityHashCode(obj);
	}
	
	static public int classId(Class cls) {
		return System.identityHashCode(cls);
	}
	
	static public long variableId(Object obj, int fld) {
		return variableId(objectId(obj), fld);
	}
	
	static public long variableId(Class cls, int fld) {
		return variableId(classId(cls), fld);
	}

    static public long variableId(int obj, int fld) {
		return idLong(obj, fld);
	}

    static public long idLong(int f, int s) {
        long l = f;
        l = l << 32;
        l += s;
        return l;
    }

    //---------------------------------------
	// Non-native instrumentation methods
	//---------------------------------------

    static public void RecordEvent_SharedRead(int tid, long mem, int iid) {

        Native_RecordEvent_SharedRead(tid, (int)mem, iid);

        log("SharedRead", tid, (int)mem, iid);
	}

	static public void RecordEvent_SharedWrite(int tid, long mem, int iid) {

        Native_RecordEvent_SharedWrite(tid, (int)mem, iid);

        log("SharedWrite", tid, (int)mem, iid);
	}

    static public void RecordEvent_SharedRead(Thread thr, long mem, int iid) {
        final int tid = threadId(thr);

		RecordEvent_SharedRead(tid, mem, iid);
	}

	static public void RecordEvent_SharedWrite(Thread thr, long mem, int iid) {
		final int tid = threadId(thr);

        RecordEvent_SharedWrite(tid, mem, iid);
	}

    static public void RecordEvent_SharedRead(Thread thr, Object obj, int fld, int iid) {
		final long mem = variableId(obj, fld);

        RecordEvent_SharedRead(thr, mem, iid);
	}
	
	static public void RecordEvent_SharedWrite(Thread thr, Object obj, int fld, int iid) {
		final long mem = variableId(obj, fld);

        RecordEvent_SharedWrite(thr, mem, iid);
	}

    static public void RecordEvent_SharedRead(Thread thr, Class cls, int fld, int iid) {
		final long mem = variableId(cls, fld);

		RecordEvent_SharedRead(thr, mem, iid);
	}

	static public void RecordEvent_SharedWrite(Thread thr, Class cls, int fld, int iid) {
		final long mem = variableId(cls, fld);

		RecordEvent_SharedWrite(thr, mem, iid);
	}
	
	static public void RecordEvent_SharedRead(Thread thr, int cls, int fld, int iid) {
		final long mem = variableId(cls, fld);

		RecordEvent_SharedRead(thr, mem, iid);
	}
	
	static public void RecordEvent_SharedWrite(Thread thr, int cls, int fld, int iid) {
		final long mem = variableId(cls, fld);

		RecordEvent_SharedWrite(thr, mem, iid);
	}

	//---------------------------------------

    static public void RecordEvent_VolatileRead(int tid, long mem, int iid) {

        Native_RecordEvent_VolatileRead(tid, (int)mem, iid);

        log("AtomicRead", tid, (int)mem, iid);
	}

	static public void RecordEvent_VolatileWrite(int tid, long mem, int iid) {

        Native_RecordEvent_VolatileWrite(tid, (int)mem, iid);

        log("AtomicWrite", tid, (int)mem, iid);
	}

    static public void RecordEvent_VolatileRead(Thread thr, long mem, int iid) {
		final int tid = threadId(thr);

		RecordEvent_VolatileRead(tid, mem, iid);
	}

	static public void RecordEvent_VolatileWrite(Thread thr, long mem, int iid) {
		final int tid = threadId(thr);

        RecordEvent_VolatileWrite(tid, mem, iid);
	}
	
	static public void RecordEvent_VolatileRead(Thread thr, Object obj, int fld, int iid) {
		final long mem = variableId(obj, fld);

		RecordEvent_VolatileRead(thr, mem, iid);
	}

	static public void RecordEvent_VolatileWrite(Thread thr, Object obj, int fld, int iid) {
		final long mem = variableId(obj, fld);

		RecordEvent_VolatileWrite(thr, mem, iid);
	}

    static public void RecordEvent_VolatileRead(Thread thr, Class cls, int fld, int iid) {
		final long mem = variableId(cls, fld);

		RecordEvent_VolatileRead(thr, mem, iid);
	}

	static public void RecordEvent_VolatileWrite(Thread thr, Class cls, int fld, int iid) {
		final long mem = variableId(cls, fld);

		RecordEvent_VolatileWrite(thr, mem, iid);
	}

	static public void RecordEvent_VolatileRead(Thread thr, int cls, int fld, int iid) {
		final int mem = cls;

		RecordEvent_VolatileRead(thr, mem, iid);
	}
	
	static public void RecordEvent_VolatileWrite(Thread thr, int cls, int fld, int iid) {
		final int mem = cls;

		RecordEvent_VolatileWrite(thr, mem, iid);
	}
	
	//---------------------------------------

    static public void RecordEvent_Lock(int tid, Object obj, int iid) {
		final int lock = objectId(obj);

		Native_RecordEvent_Lock(tid, lock, iid);

        log("Lock", tid, lock, iid);
	}

	static public void RecordEvent_Unlock(int tid, Object obj, int iid) {
		final int lock = objectId(obj);

		Native_RecordEvent_Unlock(tid, lock, iid);

        log("Unlock", tid, lock, iid);
	}

	static public void RecordEvent_Lock(Thread thr, Object obj, int iid) {
		final int tid = threadId(thr);
		final int lock = objectId(obj);

		RecordEvent_Lock(tid, lock, iid);
	}
	
	static public void RecordEvent_Unlock(Thread thr, Object obj, int iid) {
		final int tid = threadId(thr);
		final int lock = objectId(obj);

		RecordEvent_Unlock(tid, lock, iid);
	}
	
	//---------------------------------------
	
	static public void RecordEvent_RLock(Thread thr, Object obj, int iid) {
		final int tid = threadId(thr);
		final int lock = objectId(obj);

		Native_RecordEvent_RLock(tid, lock, iid);

        log("RLock", tid, lock, iid);
	}
	
	static public void RecordEvent_WLock(Thread thr, Object obj, int iid) {
		final int tid = threadId(thr);
		final int lock = objectId(obj);

		Native_RecordEvent_WLock(tid, lock, iid);

        log("WLock", tid, lock, iid);
	}
	
	static public void RecordEvent_RWUnlock(Thread thr, Object obj, int iid) {
		final int tid = threadId(thr);
		final int lock = objectId(obj);

		Native_RecordEvent_RWUnlock(tid, lock, iid);

        log("RWUnlock", tid, lock, iid);
	}
	
	//---------------------------------------

    static public void RecordEvent_Fork(int tid, int _tid, int iid) {

		Native_RecordEvent_Fork(tid, _tid, iid);

        log("Fork", tid, _tid, iid);
	}

	static public void RecordEvent_Join(int tid, int _tid, int iid) {

        Native_RecordEvent_Join(tid, _tid, iid);

        log("Join", tid, _tid, iid);
	}

	static public void RecordEvent_Fork(Thread thr, Thread _thr, int iid) {
		final int tid = threadId(thr);
		final int _tid = threadId(_thr);

		RecordEvent_Fork(tid, _tid, iid);
	}
	
	static public void RecordEvent_Join(Thread thr, Thread _thr, int iid) {
		final int tid = threadId(thr);
		final int _tid = threadId(_thr);

		RecordEvent_Join(tid, _tid, iid);
	}
	
	//---------------------------------------
	
	static public void LockForShared(Thread thr, Object obj, int fld) {
		final int tid = threadId(thr);
		final long mem = variableId(obj, fld);

		Native_LockForShared(tid, mem);
	}
	
	static public void UnlockForShared(Thread thr, Object obj, int fld) {
		final int tid = threadId(thr);
		final long mem = variableId(obj, fld);

		Native_UnlockForShared(tid, mem);
	}

    static public void LockForShared(Thread thr, Class cls, int fld) {
		final int tid = threadId(thr);
		final long mem = variableId(cls, fld);

		Native_LockForShared(tid, mem);
	}

	static public void UnlockForShared(Thread thr, Class cls, int fld) {
		final int tid = threadId(thr);
		final long mem = variableId(cls, fld);

		Native_UnlockForShared(tid, mem);
	}
	
	static public void LockForShared(Thread thr, int cls, int fld) {
		final int tid = threadId(thr);
		final int mem = cls;

		Native_LockForShared(tid, mem);
	}
	
	static public void UnlockForShared(Thread thr, int cls, int fld) {
		final int tid = threadId(thr);
		final int mem = cls;

		Native_UnlockForShared(tid, mem);
	}
	
	//---------------------------------------	
	
	
	
	//---------------------------------------
	// Native methods
	//---------------------------------------

    static private native void Native_InitEventList();
	static private native void Native_FinalizeEventList();

    //---------------------------------------

	static private native void Native_InitStatistics();
	static private native void Native_PrintStatistics(String filename);

    //---------------------------------------

	static private native void Native_RecordEvent_SharedRead(int thr, long mem, int iid);
	static private native void Native_RecordEvent_SharedWrite(int thr, long mem, int iid);
	
	//---------------------------------------
	
	static private native void Native_RecordEvent_VolatileRead(int thr, long mem, int iid);
	static private native void Native_RecordEvent_VolatileWrite(int thr, long mem, int iid);

	//---------------------------------------
	
	static private native void Native_RecordEvent_Lock(int thr, int lock, int iid);
	static private native void Native_RecordEvent_Unlock(int thr, int lock, int iid);
	
	//---------------------------------------
	
	static private native void Native_RecordEvent_RLock(int thr, int lock, int iid);
	static private native void Native_RecordEvent_WLock(int thr, int lock, int iid);
	static private native void Native_RecordEvent_RWUnlock(int thr, int lock, int iid);
	
	//---------------------------------------
	
	static private native void Native_RecordEvent_Fork(int thr, int _thr, int iid);
	static private native void Native_RecordEvent_Join(int thr, int _thr, int iid);
	
	//---------------------------------------
	
	static private native void Native_LockForShared(int thr, long obj);
	static private native void Native_UnlockForShared(int thr, long obj);
	
	//---------------------------------------
}
