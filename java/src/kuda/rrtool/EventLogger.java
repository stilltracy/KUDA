package kuda.rrtool;

import kuda.util.MutexSpinLock;

import java.io.DataOutputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.concurrent.locks.ReentrantLock;

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
public class EventLogger {

    public static final String DEFAULT_LOG_FILE_NAME = "kudalog";

    private DataOutputStream out;
    private MutexSpinLock lock;
    public EventLogger(String file_name) {
        try {
            this.out = new DataOutputStream(new FileOutputStream(file_name));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            System.exit(-1);
        }
        this.lock = new MutexSpinLock();
    }

    public void close () {
        try {
            out.writeInt(-1); // marks the end of the file
            out.flush();
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(-1);
        }
    }

    public void write (int op, int tid, long memId, int iid) {
        lock.lock();
        try {
            out.writeInt(op);
            out.writeInt(tid);
            out.writeLong(memId);
            out.writeInt(iid);
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(-1);
        } finally {
            lock.unlock();
        }
    }

    public void write (int op, int tid, int objId, int iid) {
        lock.lock();
        try {
            out.writeInt(op);
            out.writeInt(tid);
            out.writeInt(objId);
            out.writeInt(iid);
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(-1);
        } finally {
            lock.unlock();
        }
    }
}
