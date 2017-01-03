package kuda.tools;

import org.apache.bcel.classfile.Field;
import org.apache.bcel.classfile.JavaClass;
import org.apache.bcel.classfile.Method;
import org.apache.bcel.util.ClassPath;
import org.apache.bcel.util.SyntheticRepository;

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

public class DumpFlags {

	public static void main(String[] args) {
		String classPath = args[0];
		String className = args[1];

		System.out.printf("Starting dumping.\nClassPath: %s\nClass: %s\n",
				classPath, className);

		dump(classPath, className);

		System.out.printf("Finished dumping.\n");
		System.exit(0);
	}

	static JavaClass currentClass = null;

	static SyntheticRepository repository = null;

	private static void dump(String classPath, String className) {
		repository = SyntheticRepository.getInstance(new ClassPath(classPath));

		try {
			currentClass = repository.loadClass(className);
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}

		int flags = currentClass.getAccessFlags();
		if ((flags & Instrumentor.MASK_CLASS) > 0) {
			System.out.println("Checking class is enabled.\n");
		} else {
			System.out.println("Checking class is disabled.\n");
			return;
		}

		Field[] fields = currentClass.getFields();
		for (Field field : fields) {
			flags = field.getAccessFlags();
			if ((flags & Instrumentor.MASK_FIELD) > 0) {
				System.out.printf("Checking field %s is enabled.\n", field
						.getName());
			}
		}

		Method[] methods = currentClass.getMethods();
		for (Method method : methods) {
			flags = method.getAccessFlags();
			if ((flags & Instrumentor.MASK_METHOD) > 0) {
				System.out.printf("Checking method %s is enabled.\n", method
						.getName());
			}
		}
	}

}
