package kuda.util;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

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
public class TaskPool {

    static class FutureNode {
        public Future future;
        public FutureNode next;

        public FutureNode(Future future) {
            this.future = future;
            this.next = null;
        }
    }

    private ExecutorService executor;
    private static final FutureNode dummy = new FutureNode(null);
    private FutureNode head, tail;

    public TaskPool(int numThreads) {
        if (numThreads == 0) {
            this.executor = Executors.newCachedThreadPool();
        } else if (numThreads == 1) {
            this.executor = Executors.newSingleThreadExecutor();
        } else {
            this.executor = Executors.newFixedThreadPool(numThreads);
        }
    }

    public void beginJob() {
        head = tail = dummy;
        dummy.next = null;
    }

    public void addTask(Runnable task) {
        FutureNode node = new FutureNode(executor.submit(task));
        tail.next = node;
        tail = node;
    }

    public boolean tryEndJob() {
        while (head.next != null) {
            try {
                FutureNode node = head.next;
                if (node.future.get() == null) {
                    head = node;
                } else {
                    return false;
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        return true;
    }

    public void waitEndJob() {
        while (head.next != null) {
            try {
                FutureNode node = head.next;
                while (node.future.get() != null) ;
                head = node;
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        assert head == tail;
    }

    public void shutdown() {
        this.executor.shutdown();
    }
}
