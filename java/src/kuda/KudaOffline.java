package kuda;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import kuda.KudaInterface;
import kuda.rrtool.EventLogger;

public class KudaOffline {

    public static void main(String[] args){
		int opcode = -1;
		DataInputStream dis = null;
		try {
			dis = new DataInputStream(new FileInputStream(EventLogger.DEFAULT_LOG_FILE_NAME));
		} catch (FileNotFoundException e1) {
			e1.printStackTrace();
            System.exit(-1);
		}

		KudaInterface.Initialize();

		try {

            while ((opcode = dis.readInt()) != -1)   {

				switch (opcode) {
				
                    case KudaInterface.OP_SHARED_WRITE:
                        KudaInterface.RecordEvent_SharedWrite(dis.readInt(), dis.readLong(), dis.readInt());
                        break;

                    case KudaInterface.OP_SHARED_READ:
                        KudaInterface.RecordEvent_SharedRead(dis.readInt(), dis.readLong(), dis.readInt());
                        break;

                    case KudaInterface.OP_VOLATILE_WRITE:
                        KudaInterface.RecordEvent_VolatileWrite(dis.readInt(), dis.readLong(), dis.readInt());
                        break;

                     case KudaInterface.OP_VOLATILE_READ:
                        KudaInterface.RecordEvent_VolatileRead(dis.readInt(), dis.readLong(), dis.readInt());
                         break;

                     case KudaInterface.OP_LOCK:
                        KudaInterface.RecordEvent_Lock(dis.readInt(), dis.readInt(), dis.readInt());
                         break;

                     case KudaInterface.OP_UNLOCK:
                        KudaInterface.RecordEvent_Unlock(dis.readInt(), dis.readInt(), dis.readInt());
                         break;

                     case KudaInterface.OP_FORK:
                        KudaInterface.RecordEvent_Fork(dis.readInt(), dis.readInt(), dis.readInt());
                         break;

                     case KudaInterface.OP_JOIN:
                        KudaInterface.RecordEvent_Join(dis.readInt(), dis.readInt(), dis.readInt());
                         break;

                     default: throw new RuntimeException("Unknown opcode: " + opcode);
				}
						
			}
		} catch (IOException e) {
			e.printStackTrace();
            System.exit(-1);
		}
	}
}