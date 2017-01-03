package kuda.rrtool;

import acme.util.Assert;
import acme.util.Util;
import kuda.KudaInterface;
import acme.util.option.CommandLine;
import kuda.util.BackgroundWorker;
import rr.annotations.Abbrev;
import rr.error.ErrorMessage;
import rr.error.ErrorMessages;
import rr.event.*;
import rr.meta.*;
import rr.simple.LastTool;
import rr.state.ShadowLock;
import rr.state.ShadowThread;
import rr.state.ShadowVar;
import rr.tool.Tool;
import tools.fasttrack.Epoch;
import tools.fasttrack.FastTrackVolatileData;
import tools.util.CV;

@Abbrev("KUDA")
public class KudaTool extends Tool {

    public KudaTool(String name, Tool next, CommandLine commandLine) {
		super(name, next, commandLine);

        if (!(next instanceof LastTool)) {
			fieldErrors.setMax(1);
			arrayErrors.setMax(1);
		}
	}

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

		super.create(e);
	}
	@Override
	public void init() {
		KudaInterface.Initialize();
        super.init();
	}

    public void fini() {
//        BackgroundWorker.close();
        super.fini();
    }

	@Override
	public void access(final AccessEvent fae) {
		final ShadowVar g = fae.getOriginalShadow();

		if (g instanceof KudaShadowVar) {
            assert g instanceof KudaShadowDataVar;
            final KudaShadowDataVar var = (KudaShadowDataVar)g;
			final ShadowThread thread = fae.getThread();
            final int tid = thread.getTid();
            assert tid >= 0;
            if(var.lastTid != tid) {
                var.lastTid = tid;

                // iid must be non-zero
                // positive iid indicates a field access
                // negative iid indicates an array access
                int iid = getIid(fae.getAccessInfo(), fae.getKind());

                if (fae.isWrite()) {
                    KudaInterface.RecordEvent_SharedWrite(tid, var.memId, iid);
                } else {
                    KudaInterface.RecordEvent_SharedRead(tid, var.memId, iid);
                }
            }
		}

		super.access(fae);
	}

    @Override
	public void volatileAccess(VolatileAccessEvent fae) {
        final ShadowVar g = fae.getOriginalShadow();

		if (g instanceof KudaShadowVar) {
            assert g instanceof KudaShadowVolatileVar;
            final KudaShadowVolatileVar var = (KudaShadowVolatileVar)g;
			final ShadowThread thread = fae.getThread();

            if (fae.isWrite()) {
                KudaInterface.RecordEvent_VolatileWrite(thread.getTid(), var.memId, -1);
			} else {
				KudaInterface.RecordEvent_VolatileRead(thread.getTid(), var.memId, -1);
			}
		}

		super.volatileAccess(fae);
	}

    @Override
	public void acquire(AcquireEvent ae) {
		final ShadowLock lock = ae.getLock();
        final ShadowThread thread = ae.getThread();

		KudaInterface.RecordEvent_Lock(thread.getTid(), lock.getLock(), -1);

        super.acquire(ae);
	}

	@Override
	public void release(ReleaseEvent ae) {
		final ShadowLock lock = ae.getLock();
        final ShadowThread thread = ae.getThread();

		KudaInterface.RecordEvent_Unlock(thread.getTid(), lock.getLock(), -1);

        super.release(ae);
	}

    @Override
	public void preStart(StartEvent se) {
        final ShadowThread thread = se.getThread();
		final ShadowThread forked = se.getNewThread();

		KudaInterface.RecordEvent_Fork(thread.getTid(), forked.getTid(), -1);

		super.preStart(se);
	}

	@Override
	public void postJoin(JoinEvent je) {
		final ShadowThread thread = je.getThread();
		final ShadowThread joined = je.getJoiningThread();

		KudaInterface.RecordEvent_Join(thread.getTid(), joined.getTid(), -1);

		super.postJoin(je);
	}

    public final ErrorMessage<FieldInfo> fieldErrors =
		ErrorMessages.makeFieldErrorMessage("KudaTool");

	public final ErrorMessage<ArrayAccessInfo> arrayErrors =
		ErrorMessages.makeArrayErrorMessage("KudaTool");

    @Override
	public ShadowVar makeShadowVar(AccessEvent ae) {
        if (ae.getKind() == AccessEvent.Kind.VOLATILE) {
			return new KudaShadowVolatileVar(ae);
		} else {
			return new KudaShadowDataVar(ae);
		}
	}

    protected int getIid (AccessInfo info, AccessEvent.Kind kind) {
        int iid = info.getId();
        assert iid > 0;
        if(kind == AccessEvent.Kind.FIELD) {
            return iid;
        } else if(kind == AccessEvent.Kind.ARRAY){
            return 0-iid;
        } else {
            throw new IllegalArgumentException("Kind must be either Field or Array!");
        }
    }

    protected AccessInfo getAccessInfo(int iid) {
        assert iid != 0 : "iid must be non-zero!";
        if(iid > 0) {
            return MetaDataInfoMaps.getFieldAccesses().get(iid);
        } else {
            return MetaDataInfoMaps.getArrayAccesses().get(Math.abs(iid));
        }
    }


}