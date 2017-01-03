#include <stdio.h>
#include <pthread.h>

typedef struct
{
  int a;
  int b;
}test;

//int *global;
long double b=0;
test tst;
test *tstP;
pthread_mutex_t lck;

int k=250000;

void* mythread(void* se)
{
	int i,j,temp;
	for(i=0;i<k;i++) {
		//if(i%(k/10)) printf("%d",b++);
		//pthread_mutex_lock(&lck);
		temp = b;
		j=10000;
		while(j--){}
		temp++;
		b = temp;
		//pthread_mutex_unlock(&lck);
	}
}



int main(int argc, char *argv[])
{
	int i,j,temp;
	pthread_mutex_init (&lck, NULL);
  
	pthread_t thread1;
	// pthread_t thread2;

	pthread_create (&thread1,NULL,&mythread,NULL);
	for(i=0;i<k;i++) {
		//pthread_mutex_lock(&lck);
		////if(i%(k/10))printf("%d",b++);
		temp = b;
		j=10000;
		while(j--){}
		temp++;
		b = temp;
		//pthread_mutex_unlock(&lck);
	}
	pthread_join(thread1,NULL);
	printf("Result of the parallel sum is: %Lf\n",b);
	pthread_exit(NULL);
}
