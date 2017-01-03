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

public class ReadRccOutput {

	static HashMap<String, HashSet> classInfos = new HashMap<String, HashSet>();

	public static void main(String[] args) {
		File inputFile = Helper.chooseFile();
		File outputFile = Helper.chooseFile();

		System.out.printf(
				"Starting parsing.\nInput file: %s, Output file %s\n",
				inputFile.getName(), outputFile.getName());

		parse(inputFile, outputFile);

		System.out.printf("Finished parsing instrumentation.");
		System.exit(0);

	}

	private static void parse(File inputFile, File outputFile) {
		BufferedReader reader = null;

		try {
			PrintWriter writer = null;

			reader = new BufferedReader(new FileReader(inputFile));

			while (reader.ready()) {
				String line = reader.readLine().trim();

				if (line.indexOf("/*# guarded_by") != -1) {
					System.out.printf("\nParsing %s\n\n", line);
					String[] parts = line.split("[: ]");
					String className = parts[0];
					int i = 4;
					while (!parts[i].startsWith("/*#")) {
						++i;
					}
					String fieldName = parts[i - 1];
					while (!parts[i].equals("guarded_by")) {
						++i;
					}
					String guard = parts[i + 1];

					if (guard.equals("no_guard") || guard.startsWith("$p$")) {
						System.out.printf("Adding field: %s.%s\n", className,
								fieldName);

						if (!classInfos.containsKey(className)) {
							classInfos.put(className, new HashSet<String>());
						}
						HashSet fields = classInfos.get(className);
						fields.add(fieldName);
					}
				}

			}

			reader.close();

			writer = new PrintWriter(outputFile);

			for (String className : classInfos.keySet()) {
				HashSet<String> fields = classInfos.get(className);
				writer.printf("c %s\n", className);
				for (String fieldName : fields) {
					writer.printf("f %s\n", fieldName);
				}
			}

			writer.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
