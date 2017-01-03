package kuda.util;

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

public class Timer {
    private long startTime, endTime;

    public Timer() {
        startTime = endTime = 0;
    }

    public void start() {
        startTime = endTime = System.nanoTime();
    }

    public void end() {
        endTime = System.nanoTime();
    }

    public long nanos() {
        return endTime - startTime;
    }

    public double seconds() {
        return (endTime - startTime) * 1e-9;
    }

    public static String toString(long nanos) {
        Timer t = new Timer();
        t.endTime = nanos;
        return t.toString();
    }

    public static String toString(double seconds) {
        Timer t = new Timer();
        t.endTime = (long) (seconds * 1e9);
        return t.toString();
    }

    @Override
    public String toString() {
        long millis = nanos() / (1000 * 1000);
        if (millis < 10000)
            return String.format("%dms", millis);
        long hours = millis / (60 * 60 * 1000);
        millis -= hours * 60 * 60 * 1000;
        long minutes = millis / (60 * 1000);
        millis -= minutes * 60 * 1000;
        long seconds = millis / 1000;
        millis -= seconds * 1000;
        return String.format("%d:%02d:%02d.%1d", hours, minutes, seconds, millis / 100);
    }
}