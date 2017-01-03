package kuda.tools;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
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

public class ReadChordOutput {

	static HashMap<String, HashSet> classInfos = new HashMap<String, HashSet>();

	public static void main(String[] args) {
		File inputDir = new File(args[0]);
		File outputFile = Helper.chooseFile();

		System.out.printf(
				"Starting parsing.\nInput file: %s, Output file %s\n", inputDir
						.getName(), outputFile.getName());

		parse(inputDir, outputFile);

		System.out.printf("Finished parsing instrumentation.");
		System.exit(0);

	}

	static int numUnlockedRacePairs = 0;

	static File fileUnlockedRacePairs = null;

	static int[][] unlockedRacePairs;

	private static void parseCounters(File inputDir) {
		BufferedReader reader = null;

		try {
			reader = new BufferedReader(new FileReader(new File(inputDir,
					"doms.txt")));

			while (reader.ready()) {
				String line = reader.readLine().trim();
				String[] parts = line.split("[ ]");
				if (parts[0].equals("E")) {
					numUnlockedRacePairs = Integer.parseInt(parts[1]);
					fileUnlockedRacePairs = new File(inputDir, parts[2]);
					System.out.printf("There are %d racy pairs.\n",
							numUnlockedRacePairs);
				}
			}

			reader.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static void parseUnlockedRacePairs(File inputDir) {
		BufferedReader reader = null;

		try {
			unlockedRacePairs = new int[2][numUnlockedRacePairs];

			reader = new BufferedReader(new FileReader(new File(inputDir,
					"UnlockedracePair.tuples")));
			reader.readLine(); // first line
			int i = 0;
			while (reader.ready()) {
				String line = reader.readLine().trim();
				String[] parts = line.split("[ ]");
				unlockedRacePairs[0][i] = Integer.parseInt(parts[0]);
				unlockedRacePairs[1][i] = Integer.parseInt(parts[1]);
				++i;
			}
			assert (i == numUnlockedRacePairs);
			reader.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static void parseRacyFields(File inputDir) {
		HashSet<Integer> set = new HashSet<Integer>();
		for (int i = 0; i < numUnlockedRacePairs; ++i) {
			set.add(new Integer(unlockedRacePairs[0][i]));
			set.add(new Integer(unlockedRacePairs[1][i]));
		}

		BufferedReader reader = null;

		try {
			reader = new BufferedReader(new FileReader(new File(inputDir,
					"E.map")));
			assert (reader.markSupported());
			reader.mark((int) new File(inputDir, "E.map").length());

			for (Integer access : set) {
				String fieldName = null, className = null;
				String line = readNthLine(reader, access);
				String[] parts = line.split("[ ]");
				if (parts[0].trim().equals("field")) {
					className = parts[1].trim();
					className = className.substring(0, className.length() - 1);
					fieldName = parts[3];

					System.out.printf("Adding field: %s.%s\n", className,
							fieldName);

					if (!classInfos.containsKey(className)) {
						classInfos.put(className, new HashSet<String>());
					}
					HashSet fields = classInfos.get(className);
					fields.add(fieldName);
				}
			}

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static String readNthLine(BufferedReader reader, Integer num)
			throws IOException {
		reader.reset();
		String line = null;
		int i = 0;
		for (; i < num; ++i) {
			line = reader.readLine();
			assert (reader.ready());
		}
		return line;
	}

	private static void saveOutput(File outputFile) {
		PrintWriter writer = null;

		try {
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

	private static void parse(File inputDir, File outputFile) {
		try {
			parseCounters(inputDir);
			parseUnlockedRacePairs(inputDir);
			parseRacyFields(inputDir);
			saveOutput(outputFile);

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
