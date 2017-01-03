package kuda.rrtool;

import acme.util.Assert;
import acme.util.option.CommandLine;
import kuda.KudaInterface;
import rr.annotations.Abbrev;
import rr.error.ErrorMessage;
import rr.error.ErrorMessages;
import rr.event.*;
import rr.meta.AccessInfo;
import rr.meta.ArrayAccessInfo;
import rr.meta.FieldInfo;
import rr.meta.MetaDataInfoMaps;
import rr.simple.LastTool;
import rr.state.ShadowLock;
import rr.state.ShadowThread;
import rr.state.ShadowVar;
import rr.tool.Tool;

import java.io.FileWriter;
import java.io.IOException;

@Abbrev("KUDALOG")
final public class KudaLoggingTool extends KudaTool {

	public KudaLoggingTool (String name, Tool next, CommandLine commandLine) {
		super(name, next, commandLine);
	}

	EventLogger eventLogger;

    static ThreadLocalState ts_get_tls(ShadowThread ts) { Assert.panic("Bad");	return null; }
	static void ts_set_tls(ShadowThread ts, ThreadLocalState tls) { Assert.panic("Bad");  }

    @Override
	public void create(NewThreadEvent e) {
		ShadowThread thread = e.getThread();
		ThreadLocalState tls = ts_get_tls(thread);

		if (tls == null) {
			tls = new ThreadLocalState();
			ts_set_tls(thread, tls);
		} else {
            tls.reset();
        }

	}

    @Override
	public void init() {
        eventLogger = new EventLogger(EventLogger.DEFAULT_LOG_FILE_NAME);

		super.init();
	}

	public void fini() {
		eventLogger.close();

		super.fini();
	}

	@Override
	public void access(final AccessEvent fae) {
		final ShadowVar g = fae.getOriginalShadow();

		if (g instanceof KudaShadowVar) {
			assert g instanceof KudaShadowDataVar;
			final KudaShadowDataVar var = (KudaShadowDataVar) g;
			final ShadowThread thread = fae.getThread();
			final int tid = thread.getTid();
			assert tid >= 0;
			if (var.lastTid != tid) {
				var.lastTid = tid;

				// iid must be non-zero
				// positive iid indicates a field access
				// negative iid indicates an array access
				int iid = getIid(fae.getAccessInfo(), fae.getKind());

				if (fae.isWrite()) {
					eventLogger.write(KudaInterface.OP_SHARED_WRITE, tid, var.memId, iid);
				} else {
					eventLogger.write(KudaInterface.OP_SHARED_READ, tid, var.memId, iid);
				}
			}
		}
	}

	@Override
	public void volatileAccess(VolatileAccessEvent fae) {
		final ShadowVar g = fae.getOriginalShadow();

		if (g instanceof KudaShadowVar) {
			assert g instanceof KudaShadowVolatileVar;
			final KudaShadowVolatileVar var = (KudaShadowVolatileVar) g;
			final ShadowThread thread = fae.getThread();

			if (fae.isWrite()) {
				eventLogger.write(KudaInterface.OP_VOLATILE_WRITE, thread.getTid(), var.memId, -1);
			} else {
				eventLogger.write(KudaInterface.OP_VOLATILE_READ, thread.getTid(), var.memId, -1);
			}
		}
	}

	@Override
	public void acquire(AcquireEvent ae) {
		final ShadowLock lock = ae.getLock();
		final ShadowThread thread = ae.getThread();

        eventLogger.write(KudaInterface.OP_LOCK, thread.getTid(), KudaInterface.objectId(lock.getLock()), -1);
	}

	@Override
	public void release(ReleaseEvent ae) {
		final ShadowLock lock = ae.getLock();
		final ShadowThread thread = ae.getThread();

        eventLogger.write(KudaInterface.OP_UNLOCK, thread.getTid(), KudaInterface.objectId(lock.getLock()), -1);
	}

	@Override
	public void preStart(StartEvent se) {
		final ShadowThread thread = se.getThread();
		final ShadowThread forked = se.getNewThread();

        eventLogger.write(KudaInterface.OP_FORK, thread.getTid(), forked.getTid(), -1);
	}

	@Override
	public void postJoin(JoinEvent je) {
		final ShadowThread thread = je.getThread();
		final ShadowThread joined = je.getJoiningThread();
		
		eventLogger.write(KudaInterface.OP_JOIN, thread.getTid(), joined.getTid(), -1);
	}
}