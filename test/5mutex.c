
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <pthread.h>


int common=0;
int common2=0;
//first way to create mutex
//pthread_mutex_t lock= PTHREAD_MUTEX_INITIALIZER;

//second way to create mutex
pthread_mutex_t lck;
pthread_mutex_t lck2;
//
                     
void* print(void* unused)
 {
        int x=1,i;
        int retval;
 
 //**** to see the effect of using mutex let's remove the "pthread_mutex_lock(&lock)" and "pthread_mutex_unlock(&lock)" 
 
	pthread_mutex_lock(&lck); //if mutex is used then only one thread can run the code between "lock" and "unlock" at the same time
	common++;
	sleep(1); 
	printf("Common: %d\tx: %d    ( Thread ID: %d )\n",common,x,(int)pthread_self());
	fflush(stdout);
	x++;
        pthread_mutex_unlock(&lck);
	
	pthread_mutex_lock(&lck2);
	
	printf("a");
	fflush(stdout);
	pthread_mutex_unlock(&lck2);
	
  }

  
  int main(int argc, char *argv[])
{
     pthread_mutex_init (&lck, NULL);
     pthread_mutex_init (&lck2, NULL);
     int i;
    
     pthread_t threads[3];
 

      for(i=0;i<3;i++)
		pthread_create (&(threads[i]),NULL,&print,NULL); //create 5 threads
    
      for(i=0;i<3;i++)
		pthread_join(threads[i],NULL); //wait for termination of threads

      pthread_exit(NULL);
}
