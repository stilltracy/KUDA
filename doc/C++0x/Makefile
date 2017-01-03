
default : test

n2427.bash : n2427.html
	sed -f n2427.sed n2427.html > n2427.bash

stdatomic.h cstdatomic impatomic.h impatomic.c n2427.c : n2427.bash
	bash n2427.bash

impatomic.o : impatomic.h impatomic.c
	gcc -std=c99 -c impatomic.c

n2427.c.exe : n2427.c stdatomic.h impatomic.o
	gcc -std=c99 -o n2427.c.exe n2427.c impatomic.o

n2427.c++.exe : n2427.c stdatomic.h impatomic.o
	g++ -o n2427.c++.exe n2427.c impatomic.o

test : n2427.c.exe n2427.c++.exe

clean :
	rm -f n2427.bash stdatomic.h cstdatomic impatomic.h impatomic.c
	rm -f impatomic.o n2427.c.exe n2427.c++.exe

