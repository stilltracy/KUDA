package kuda.tools;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

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

public class Instrumentor {

	public static final String HEADING_CLASS = "c";

	public static final String HEADING_FIELD = "f";

	public static final String HEADING_METHOD = "m";

	public static final int MASK_CLASS = 0x8000;

	public static final int MASK_FIELD = 0x8000;

	public static final int MASK_METHOD = 0x8000;

	static int numClassesInstrumented = 0;

	static int numFieldsInstrumented = 0;

	static int numMethodsInstrumented = 0;

	public static void main(String args[]) {
		classPath = args[0];
		File confFile = Helper.chooseFile();

		System.out
				.printf(
						"Starting instrumentation.\nClassPath: %s\nConfigurationFile: %s\n",
						classPath, confFile.getName());

		instrument(classPath, confFile);

		System.out
				.printf(
						"Finished instrumentation.\nNumberOfClassesInstrumented: %d\nNumberOfFieldsInstrumented: %d\nNumberOfMethodsInstrumented: %d\n",
						numClassesInstrumented, numFieldsInstrumented,
						numMethodsInstrumented);
		System.exit(0);
	}

	static JavaClass currentClass = null;

	static SyntheticRepository repository = null;

	static String classPath = null;

	static boolean skip = false;

	private static void instrument(String classPath, File confFile) {
		BufferedReader reader = null;

		try {

			repository = SyntheticRepository.getInstance(new ClassPath(
					classPath));

			reader = new BufferedReader(new FileReader(confFile));

			while (reader.ready()) {
				String line = reader.readLine().trim();
				if (line.equals(""))
					return;

				String[] parts = line.split("[ ]");
				assert (parts.length == 2);
				String heading = parts[0];
				String name = parts[1];

				if (heading.equalsIgnoreCase(HEADING_CLASS)) {
					skip = false;
					System.out.printf("Instrumenting class %s\n", name);
					instrumentClass(name);
					numClassesInstrumented++;
				} else if ((!skip) && heading.equalsIgnoreCase(HEADING_FIELD)) {
					System.out.printf("Instrumenting field %s\n", name);
					instrumentField(name);
					numFieldsInstrumented++;
				} else if ((!skip) && heading.equalsIgnoreCase(HEADING_METHOD)) {
					System.out.printf("Instrumenting method %s\n", name);
					instrumentMethod(name);
					numMethodsInstrumented++;
				}
			}

			if (currentClass != null) {
				saveClass(currentClass);
			}

			reader.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static void instrumentField(String name) {
		Field[] fields = currentClass.getFields();
		for (Field field : fields) {
			if (field.getName().equals(name)) {
				int flags = field.getAccessFlags();

				flags |= MASK_FIELD;

				field.setAccessFlags(flags);

				break;
			}
		}

	}

	private static void instrumentMethod(String name) {
		Method[] methods = currentClass.getMethods();
		for (Method method : methods) {
			if (method.getName().equals(name)) {
				int flags = method.getAccessFlags();

				flags |= MASK_FIELD;

				method.setAccessFlags(flags);

				break;
			}
		}

	}

	private static void instrumentClass(String name) {
		// write old class
		if (currentClass != null) {
			saveClass(currentClass);
		}

		// start new class
		try {
			currentClass = repository.loadClass(name);

			int flags = currentClass.getAccessFlags();

			flags |= MASK_CLASS;

			currentClass.setAccessFlags(flags);

		} catch (ClassNotFoundException e) {
			skip = true;
		}
	}

	private static void saveClass(JavaClass cls) {
		String className = cls.getClassName();
		String path = null;
		try {
			path = repository.getClassPath().getPath(
					className.replace('.', '/'), ".class");
			cls.dump(path);
		} catch (IOException e) {

		}
	}

}
