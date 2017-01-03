package kuda.tools;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.HashSet;

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

public class MergeConfFiles {

	public static final String HEADING_CLASS = "c";

	public static final String HEADING_FIELD = "f";

	public static final String HEADING_METHOD = "m";

	static HashMap<String, HashSet<String>> classFields = new HashMap<String, HashSet<String>>();

	static HashMap<String, HashSet<String>> classMethods = new HashMap<String, HashSet<String>>();

	public static void main(String args[]) {
		File[] inputFiles = Helper.chooseFiles();
		File confFile = Helper.chooseFile();

		merge(inputFiles);

		saveOutput(confFile);

		System.out.println("Finished instrumentation.");

		System.exit(0);
	}

	static String currentClassName = null;

	private static void merge(File[] inputFiles) {
		BufferedReader reader = null;

		try {

			for (int i = 0; i < inputFiles.length; ++i) {

				reader = new BufferedReader(new FileReader(inputFiles[i]));

				while (reader.ready()) {
					String line = reader.readLine().trim();
					if (line.equals(""))
						return;

					String[] parts = line.split("[ ]");
					assert (parts.length == 2);
					String heading = parts[0];
					String name = parts[1];

					if (heading.equalsIgnoreCase(HEADING_CLASS)) {
						System.out.printf("Instrumenting class %s\n", name);
						currentClassName = name;
					} else if (heading.equalsIgnoreCase(HEADING_FIELD)) {
						System.out.printf("Instrumenting field %s\n", name);
						assert (currentClassName != null);
						if (!classFields.containsKey(currentClassName)) {
							classFields.put(currentClassName,
									new HashSet<String>());
						}
						HashSet<String> fields = classFields
								.get(currentClassName);
						fields.add(name);
					} else if (heading.equalsIgnoreCase(HEADING_METHOD)) {
						System.out.printf("Instrumenting method %s\n", name);
						assert (currentClassName != null);
						if (!classMethods.containsKey(currentClassName)) {
							classMethods.put(currentClassName,
									new HashSet<String>());
						}
						HashSet<String> methods = classMethods
								.get(currentClassName);
						methods.add(name);
					}
				}

				reader.close();
			}

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static void saveOutput(File outputFile) {
		PrintWriter writer = null;

		try {
			writer = new PrintWriter(outputFile);

			for (String className : classFields.keySet()) {
				HashSet<String> fields = classFields.get(className);
				writer.printf("c %s\n", className);
				for (String fieldName : fields) {
					writer.printf("f %s\n", fieldName);
				}
			}

			for (String className : classMethods.keySet()) {
				HashSet<String> methods = classMethods.get(className);
				writer.printf("c %s\n", className);
				for (String methodName : methods) {
					writer.printf("m %s\n", methodName);
				}
			}

			writer.close();

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
