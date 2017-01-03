#! /bin/sh

if [ $# -ne 1 ]; then
  echo 1>&2 Usage: $0 "(runA | runB | runC)"
  exit -1
fi

if [ ! -f ../$1/input/BarnesHut.in ]; then
  echo 1>&2 Error: cannot find input file ../$1/input/BarnesHut.in
  echo 1>&2 Usage: $0 "(runA | runB | runC)"
  exit -1
fi

JAVA=java

rm -f barneshut.out
$JAVA -Xms32M -Xmx64M -jar BarnesHut.jar ../$1/input/BarnesHut.in > barneshut.out && \
if diff -q -w barneshut.out ../$1/output/BarnesHut.out > /dev/null; then
  echo "completed successfully"
  rm -f barneshut.out
else
  echo "Error in output: barneshut.out and ../$1/output/BarnesHut.out differ"
fi
