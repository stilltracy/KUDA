/*
    Lonestar BarnesHut: Simulation of the gravitational forces in a
    galactic cluster using the Barnes-Hut n-body algorithm

    Author: Martin Burtscher
    Center for Grid and Distributed Computing
    The University of Texas at Austin

    Copyright (C) 2007, 2008 The University of Texas at Austin

    Licensed under the Eclipse Public License, Version 1.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

      http://www.eclipse.org/legal/epl-v10.html

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    File: BarnesHut.java
    Modified: Dec. 13, 2007 by Martin Burtscher (initial Java version)
    Modified: Jan 26, 2008 by Nicholas Chen (refactoring to decompose methods into smaller logical chunks)
 */

import java.util.*;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.io.*;
import java.lang.Math;
import java.text.DecimalFormat;

abstract class OctTreeNode {
	double mass;
	double posx;
	double posy;
	double posz;
}

class OctTreeLeafNode extends OctTreeNode { // the tree leaves are the bodies
	double velx;
	double vely;
	double velz;
	private double accx;
	private double accy;
	private double accz;

	void setVelocity(double x, double y, double z) {
		velx = x;
		vely = y;
		velz = z;
	}

	// advances a body's velocity and position by one time step
	void advance() {
		double dvelx, dvely, dvelz;
		double velhx, velhy, velhz;

		dvelx = accx * BarnesHut.dthf;
		dvely = accy * BarnesHut.dthf;
		dvelz = accz * BarnesHut.dthf;

		velhx = velx + dvelx;
		velhy = vely + dvely;
		velhz = velz + dvelz;

		posx += velhx * BarnesHut.dtime;
		posy += velhy * BarnesHut.dtime;
		posz += velhz * BarnesHut.dtime;

		velx = velhx + dvelx;
		vely = velhy + dvely;
		velz = velhz + dvelz;
	}

	// computes the acceleration and velocity of a body
	void computeForce(OctTreeInternalNode root, double size) {
		double ax, ay, az;

		ax = accx;
		ay = accy;
		az = accz;

		accx = 0.0;
		accy = 0.0;
		accz = 0.0;

		recurseForce(root, size * size * BarnesHut.itolsq);

		if (BarnesHut.step > 0) {
			velx += (accx - ax) * BarnesHut.dthf;
			vely += (accy - ay) * BarnesHut.dthf;
			velz += (accz - az) * BarnesHut.dthf;
		}
	}

	// recursively walks the tree to compute the force
	// on a body
	private void recurseForce(OctTreeNode n, double dsq) {
		double drx, dry, drz, drsq, nphi, scale, idr;

		drx = n.posx - posx;
		dry = n.posy - posy;
		drz = n.posz - posz;
		drsq = drx * drx + dry * dry + drz * drz;
		if (drsq < dsq) {
			if (n instanceof OctTreeInternalNode) { // n is a cell
				OctTreeInternalNode in = (OctTreeInternalNode) n;
				dsq *= 0.25;
				if (in.child[0] != null) {
					recurseForce(in.child[0], dsq);
					if (in.child[1] != null) {
						recurseForce(in.child[1], dsq);
						if (in.child[2] != null) {
							recurseForce(in.child[2], dsq);
							if (in.child[3] != null) {
								recurseForce(in.child[3], dsq);
								if (in.child[4] != null) {
									recurseForce(in.child[4], dsq);
									if (in.child[5] != null) {
										recurseForce(in.child[5], dsq);
										if (in.child[6] != null) {
											recurseForce(in.child[6], dsq);
											if (in.child[7] != null) {
												recurseForce(in.child[7], dsq);
											}
										}
									}
								}
							}
						}
					}
				}
			} else { // n is a body
				if (n != this) {
					drsq += BarnesHut.epssq;
					idr = 1 / Math.sqrt(drsq);
					nphi = n.mass * idr;
					scale = nphi * idr * idr;
					accx += drx * scale;
					accy += dry * scale;
					accz += drz * scale;
				}
			}
		} else { // node is far enough away, don't recurse any deeper
			drsq += BarnesHut.epssq;
			idr = 1 / Math.sqrt(drsq);
			nphi = n.mass * idr;
			scale = nphi * idr * idr;
			accx += drx * scale;
			accy += dry * scale;
			accz += drz * scale;
		}
	}
}

// the internal nodes are cells that summarize their children's properties
class OctTreeInternalNode extends OctTreeNode {
	OctTreeNode child[] = new OctTreeNode[8];

	static OctTreeInternalNode newNode(double px, double py, double pz) {
		OctTreeInternalNode in;
		in = new OctTreeInternalNode();

		in.mass = 0.0;
		in.posx = px;
		in.posy = py;
		in.posz = pz;
		for (int i = 0; i < 8; i++)
			in.child[i] = null;

		return in;
	}

	// builds the tree
	void insert(OctTreeLeafNode b, double r) {
		int i = 0;
		double x = 0.0, y = 0.0, z = 0.0;

		if (posx < b.posx) {
			i = 1;
			x = r;
		}
		if (posy < b.posy) {
			i += 2;
			y = r;
		}
		if (posz < b.posz) {
			i += 4;
			z = r;
		}

		if (child[i] == null) {
			child[i] = b;
		} else if (child[i] instanceof OctTreeInternalNode) {
			((OctTreeInternalNode) (child[i])).insert(b, 0.5 * r);
		} else {
			double rh = 0.5 * r;
			OctTreeInternalNode cell = newNode(posx - rh + x, posy - rh + y, posz - rh + z);
			cell.insert(b, rh);
			cell.insert((OctTreeLeafNode) (child[i]), rh);
			child[i] = cell;
		}
	}

	// recursively summarizes info about subtrees
	void computeCenterOfMass() {
		double m, px = 0.0, py = 0.0, pz = 0.0;
		OctTreeNode ch;

		int j = 0;
		mass = 0.0;
		for (int i = 0; i < 8; i++) {
			ch = child[i];
			if (ch != null) {
				child[i] = null; // move non-null children to the front (needed
				// later to make other code faster)
				child[j++] = ch;

				if (ch instanceof OctTreeLeafNode) {
					BarnesHut.body[BarnesHut.curr++] = (OctTreeLeafNode) ch;
					// sort bodies in tree order ( approximation of putting
					// nearby nodes together for locality )
				} else {
					((OctTreeInternalNode) ch).computeCenterOfMass();
				}
				m = ch.mass;
				mass += m;
				px += ch.posx * m;
				py += ch.posy * m;
				pz += ch.posz * m;
			}
		}

		m = 1.0 / mass;
		posx = px * m;
		posy = py * m;
		posz = pz * m;
	}
}

final class BarnesHut {
	static int nbodies; // number of bodies in system
	static int ntimesteps; // number of time steps to run
	static double dtime; // length of one time step
	static double eps; // potential softening parameter
	static double tol; // tolerance for stopping recursion, should be less than 0.57 for 3D case to bound error

	static double dthf, epssq, itolsq;

	static int step = 0;
	static int curr = 0;
	static OctTreeLeafNode body[]; // the n bodies

	static double diameter, centerx, centery, centerz;

	// For parallel version
	static Lock partitionLock = new ReentrantLock();
	static Condition partitionsCondition = partitionLock.newCondition();
	volatile static boolean partitionsDone;

	static int numberOfProcessors = Runtime.getRuntime().availableProcessors();
	static CyclicBarrier barrier;
	static CyclicBarrier specialLatch;
	static ComputePartitionTask[] workers;
	static Thread[] threads;

	static long starttime, endtime, runtime, lasttime, mintime, run;

	private static void readInput(String filename) {
		double vx, vy, vz;

		Scanner in = null;
		try {
			in = new Scanner(new BufferedReader(new FileReader(filename)));
			in.useLocale(Locale.US);
		} catch (FileNotFoundException e) {
			System.err.println(e);
			System.exit(-1);
		}

		nbodies = in.nextInt();
		ntimesteps = in.nextInt();
		dtime = in.nextDouble();
		eps = in.nextDouble();
		tol = in.nextDouble();

		dthf = 0.5 * dtime;
		epssq = eps * eps;
		itolsq = 1.0 / (tol * tol);

		if (body == null) {
			System.err.println("configuration: " + nbodies + " bodies, " + ntimesteps + " time steps");

			body = new OctTreeLeafNode[nbodies];
			for (int i = 0; i < nbodies; i++)
				body[i] = new OctTreeLeafNode();
		}

		for (int i = 0; i < nbodies; i++) {
			body[i].mass = in.nextDouble();
			body[i].posx = in.nextDouble();
			body[i].posy = in.nextDouble();
			body[i].posz = in.nextDouble();
			vx = in.nextDouble();
			vy = in.nextDouble();
			vz = in.nextDouble();
			body[i].setVelocity(vx, vy, vz);
		}
	}

	private static void computeCenterAndDiameter() {
		double minx, miny, minz;
		double maxx, maxy, maxz;
		double posx, posy, posz;

		minx = 1.0E90;
		miny = 1.0E90;
		minz = 1.0E90;
		maxx = -1.0E90;
		maxy = -1.0E90;
		maxz = -1.0E90;

		for (int i = 0; i < nbodies; i++) {
			posx = body[i].posx;
			posy = body[i].posy;
			posz = body[i].posz;

			if (minx > posx)
				minx = posx;
			if (miny > posy)
				miny = posy;
			if (minz > posz)
				minz = posz;

			if (maxx < posx)
				maxx = posx;
			if (maxy < posy)
				maxy = posy;
			if (maxz < posz)
				maxz = posz;
		}

		diameter = maxx - minx;
		if (diameter < (maxy - miny))
			diameter = (maxy - miny);
		if (diameter < (maxz - minz))
			diameter = (maxz - minz);

		centerx = (maxx + minx) * 0.5;
		centery = (maxy + miny) * 0.5;
		centerz = (maxz + minz) * 0.5;
	}

	public static void main(String args[]) {

		printHeader();

		checkValidInputs(args);

		runtime = 0;
		lasttime = Long.MAX_VALUE;
		mintime = Long.MAX_VALUE;
		run = 0;

		BarnesHut barnesHutRunner = new BarnesHut();
		workers = new ComputePartitionTask[numberOfProcessors];
		threads = new Thread[numberOfProcessors];
		barrier = new CyclicBarrier(numberOfProcessors);
		specialLatch = new CyclicBarrier(numberOfProcessors + 1);

		while (((run < 3) || (Math.abs(lasttime - runtime) * 64 > Math.min(lasttime, runtime))) && (run < 7)) {

			readInput(args[0]);

			System.gc();
			System.gc();
			System.gc();
			System.gc();
			System.gc();

			barnesHutRunner.runSimulation();

		}

		System.err.println("runtime: " + (mintime / 1000000) + " ms");
		System.err.println("");

		DecimalFormat df = new DecimalFormat("0.0000E00");
		for (int i = 0; i < nbodies; i++) {
			// print result
			System.out.println(df.format(body[i].posx) + " " + df.format(body[i].posy) + " " + df.format(body[i].posz));
		}

	}

	static class ComputePartitionTask implements Runnable {
		OctTreeInternalNode root;
		int lowerBound;
		int upperBound;
		BarnesHut hut;

		protected ComputePartitionTask(int _lowerBound, int _upperBound, BarnesHut _hut) {
			lowerBound = _lowerBound;
			upperBound = _upperBound;
			hut = _hut;
		}

		public void run() {
			int step = 0;
			while (step < ntimesteps) {
				try {

					specialLatch.await();

					// Compute Force first
					for (int i = lowerBound; i < upperBound; i++) {
						body[i].computeForce(root, diameter);
					}

					barrier.await();

					// Then advance once all forces have been computed
					for (int i = lowerBound; i < upperBound; i++) {
						body[i].advance();
					}

					if (barrier.await() == 0) {
						// Notify main thread that we are ready
						try {
							partitionLock.lock();
							partitionsDone = true;
							partitionsCondition.signal();
						} finally {
							partitionLock.unlock();
						}
					}

					step++;
				} catch (InterruptedException e) {
					return;
				} catch (BrokenBarrierException e) {
					return;
				}
			}
		}
	}

	private synchronized void runSimulation() {

		for (int i = 0; i < numberOfProcessors; i++) {
			int[] bounds = getLowerAndUpperBoundsFor(i, numberOfProcessors);
			workers[i] = new ComputePartitionTask(bounds[0], bounds[1], this);
			threads[i] = new Thread(workers[i]);
			threads[i].start();
		}

		lasttime = runtime;
		endtime = 0;
		starttime = System.nanoTime();

		for (step = 0; step < ntimesteps; step++) {
			computeCenterAndDiameter();

			// create the tree's root
			OctTreeInternalNode root = OctTreeInternalNode.newNode(centerx, centery, centerz);

			double radius = diameter * 0.5;
			for (int i = 0; i < nbodies; i++) {
				root.insert(body[i], radius);
			}

			for (int i = 0; i < numberOfProcessors; i++)
				workers[i].root = root;

			curr = 0;
			root.computeCenterOfMass();

			try {
				specialLatch.await();
			} catch (InterruptedException e) {
				return;
			} catch (BrokenBarrierException e) {
				return;
			}

			// Each time step needs to wait until the previous one is done first
			// Still need to keep this in a while loop because the await() could be
			// spuriouosly awaken.
			while (!partitionsDone) {
				try {
					partitionLock.lock();
					try {
						partitionsCondition.await();
					} catch (InterruptedException e) {
						return;
					}
				} finally {
					partitionLock.unlock();
				}
			}

			partitionsDone = false;
		}

		endtime = System.nanoTime();
		runtime = endtime - starttime;

		if ((run == 0) || (runtime < mintime))
			mintime = runtime;
		run++;
	}

	private static int[] getLowerAndUpperBoundsFor(int count, int numberOfProcessors) {
		int[] results = new int[2];
		int increment = nbodies / numberOfProcessors;
		results[0] = count * increment;
		if (count + 1 == numberOfProcessors) {
			results[1] = nbodies;
		} else {
			results[1] = (count + 1) * increment;
		}

		return results;
	}

	private static void checkValidInputs(String[] args) {
		if (args.length != 1) {
			System.err.println("");
			System.err.println("arguments: input_file_name");
			System.exit(-1);
		}
	}

	private static void printHeader() {
		System.err.println("");
		System.err.println("Lonestar benchmark suite");
		System.err.println("Copyright (C) 2007, 2008 The University of Texas at Austin");
		System.err.println("http://iss.ices.utexas.edu/lonestar/");
		System.err.println("");
		System.err.println("application: BarnesHut v1.0");
		System.err.println("Running with " + numberOfProcessors + " processors");
	}
}
