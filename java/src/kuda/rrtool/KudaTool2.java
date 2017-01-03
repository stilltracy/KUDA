//package kuda.rrtool;
//
//import acme.util.option.CommandLine;
//import kuda.KudaInterface;
//import rr.annotations.Abbrev;
//import rr.event.*;
//import rr.meta.AccessInfo;
//import rr.meta.ClassInfo;
//import rr.meta.FieldInfo;
//import rr.state.ShadowLock;
//import rr.tool.Tool;
//
//@Abbrev("KUDA")
//final public class KudaTool2 extends Tool {
//
//	public KudaTool2 (String name, Tool next, CommandLine commandLine) {
//		super(name, next, commandLine);
//	}
//
//	@Override
//	public void init() {
//		KudaInterface.Initialize();
//	}
//
//	@Override
//	public void access(final AccessEvent fae) {
//		final AccessInfo ai = fae.getAccessInfo();
//		final ClassInfo  ci = ai.getEnclosing().getOwner();
//		final FieldInfo  fi = ((FieldAccessEvent)fae).getInfo().getField();
//		if(fae.isWrite()) {
//			if(fi.isStatic()) {
//				if(ci.isClass()) {
//					System.out.println("[ACC] Static Field Write Access with Class: " + fae.toString());
//					KudaInterface.RecordEvent_SharedWrite(ci.getClass(),fae.getTarget().hashCode(),fae.toString().hashCode());
//				} else {
//					System.out.println("[ACC] Static Field Write Access without Class: " + fae.toString());
//					KudaInterface.RecordEvent_SharedWrite(ci.hashCode(),fae.getTarget().hashCode(),fae.toString().hashCode());
//				}
//			} else {
//				System.out.println("[ACC] Instance Field Write Access: " + fae.toString());
//				KudaInterface.RecordEvent_SharedWrite(fae.getTarget(),fae.getTarget().hashCode(),fae.toString().hashCode());
//			}
//		} else {
//			if(fi.isStatic()) {
//				if(ci.isClass()) {
//					System.out.println("[ACC] Static Field Read Access with Class: " + fae.toString());
//					KudaInterface.RecordEvent_SharedRead(ci.getClass(),fae.getTarget().hashCode(),fae.toString().hashCode());
//				} else {
//					System.out.println("[ACC] Static Field Read Access without Class: " + fae.toString());
//					KudaInterface.RecordEvent_SharedRead(ci.hashCode(),fae.getTarget().hashCode(),fae.toString().hashCode());
//				}
//			} else {
//				System.out.println("[ACC] Instance Field Read Access: " + fae.toString());
//				KudaInterface.RecordEvent_SharedRead(fae.getTarget(),fae.getTarget().hashCode(),fae.toString().hashCode());
//			}
//		}
//	}
//
//	@Override
//	public void volatileAccess(VolatileAccessEvent fae) {
//		System.out.println("volatile access: " + fae.toString());
//		final AccessInfo ai = fae.getAccessInfo();
//		final ClassInfo  ci = ai.getEnclosing().getOwner();
//		final FieldInfo  fi = ((FieldAccessEvent)fae).getInfo().getField();
//		if(fae.isWrite()) {
//			if(fi.isStatic()) {
//				if(ci.isClass()) {
//					KudaInterface.RecordEvent_VolatileWrite(ci.getClass(),fae.getTarget().hashCode(),fae.toString().hashCode());
//				} else {
//					KudaInterface.RecordEvent_VolatileWrite(ci.hashCode(),fae.getTarget().hashCode(),fae.toString().hashCode());
//				}
//			} else {
//				KudaInterface.RecordEvent_VolatileWrite(fae.getTarget(),fae.getTarget().hashCode(),fae.toString().hashCode());
//			}
//		} else {
//			if(fi.isStatic()) {
//				if(ci.isClass()) {
//					KudaInterface.RecordEvent_VolatileRead(ci.getClass(),fae.getTarget().hashCode(),fae.toString().hashCode());
//				} else {
//					KudaInterface.RecordEvent_VolatileRead(ci.hashCode(),fae.getTarget().hashCode(),fae.toString().hashCode());
//				}
//			} else {
//				KudaInterface.RecordEvent_VolatileRead(fae.getTarget(),fae.getTarget().hashCode(),fae.toString().hashCode());
//			}
//		}
//	}
//
//	@Override
//	public void acquire(AcquireEvent ae) {
//		System.out.println("acquire: " + ae.toString());
//		final ShadowLock lk = ae.getLock();
//		KudaInterface.RecordEvent_Lock(lk.getLock(), ae.toString().hashCode());
//	}
//
//	@Override
//	public void release(ReleaseEvent ae) {
//		System.out.println("release: " + ae.toString());
//		final ShadowLock lk = ae.getLock();
//		KudaInterface.RecordEvent_Unlock(lk.getLock(), ae.toString().hashCode());
//	}
//
//	@Override
//	public void postJoin(JoinEvent je) {
//		System.out.println("join: " + je.toString());
//		KudaInterface.RecordEvent_Join(je.getJoiningThread().getThread(), je.toString().hashCode());
//	}
//
//	@Override
//	public void postStart(StartEvent se) {
//		System.out.println("start: " + se.toString());
//		KudaInterface.RecordEvent_Fork(se.getNewThread().getThread(), se.toString().hashCode());
//	}
//}