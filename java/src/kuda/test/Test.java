package kuda.test;

import gnu.trove.iterator.TIntIterator;
import gnu.trove.set.hash.TIntHashSet;
import kuda.KudaInterface;

import java.util.*;

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

public class Test implements Runnable {

    static int numThreads = 100;
    static int numOps = 1000;
    static int numAccesses = 1000;
    static int numVars = 200;
    static int numLocks = 100;

    static Random rnd = new Random(System.currentTimeMillis());
    static int nextInt() { return Math.abs(rnd.nextInt()); }
    static int nextInt(int i) { return Math.abs(rnd.nextInt(i)); }

    static int iid() { return nextInt(); }
    static Object shrvar() { return new Object(); }
    static Object lock() { return new Object(); }
    static Object volvar() { return new Object(); }

    static int[] varToLock = new int[numVars];
    static Object[] vars = new Object[numVars];
    static Object[] locks = new Object[numLocks];

    static {
        for(int i = 0; i < numLocks; ++i) {
            locks[i] = lock();
        }

        for(int i = 0; i < numVars; ++i) {
            vars[i] = shrvar();
            varToLock[i] = nextInt(numLocks); // choose protecting lock
        }
    }

    public static void main(String[] args) {

        System.out.println("Java library path: " + System.getProperty("java.library.path"));

        KudaInterface.Initialize();

        List<Thread> threadList = new ArrayList<Thread>(numThreads);

        for(int i = 0; i < numThreads; ++i) {
            Thread t = new Thread(new Test());
            threadList.add(t);
        }

        for(Thread t : threadList) {
            KudaInterface.RecordEvent_Fork(Thread.currentThread(), t, iid());
            t.start();
        }

        for(Thread t : threadList) {
            try {
                t.join();
                KudaInterface.RecordEvent_Join(Thread.currentThread(), t, iid());
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    public void run() {
        for(int i = 0; i < numOps; ++i) {

            // determine the accesses at each operation
            TIntHashSet vs = new TIntHashSet();
            TIntHashSet ls = new TIntHashSet();
            for(int j = 0; j < rnd.nextInt(numAccesses); ++j) {
                int var = rnd.nextInt(numVars);
                int lock = varToLock[var];
                vs.add(var);
                ls.add(lock);
            }

            // lock protecting locks
            for(TIntIterator itr = ls.iterator(); itr.hasNext();) {
                KudaInterface.RecordEvent_Lock(Thread.currentThread(), locks[itr.next()], iid());
            }

            // access variables
            for(TIntIterator itr = vs.iterator(); itr.hasNext();) {
                if(rnd.nextBoolean()) {
                    KudaInterface.RecordEvent_SharedRead(Thread.currentThread(), vars[itr.next()], 0, iid());
                } else {
                    KudaInterface.RecordEvent_SharedWrite(Thread.currentThread(), vars[itr.next()], 0, iid());
                }

            }

            // unlock protecting locks
            for(TIntIterator itr = ls.iterator(); itr.hasNext();) {
                KudaInterface.RecordEvent_Unlock(Thread.currentThread(), locks[itr.next()], iid());
            }

        }
    }
}
