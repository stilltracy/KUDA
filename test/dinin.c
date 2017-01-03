#include <sys/time.h>
#include <stdio.h>
#include <pthread.h> 
#include <errno.h>

#define NUMP 5

pthread_mutex_t fork_mutex[NUMP];
pthread_mutex_t eat_mutex;
pthread_cond_t space_avail = PTHREAD_COND_INITIALIZER;;
int num_diners=0;
 
main()  
{
  int i;
  pthread_t diner_thread[NUMP]; 
  int dn[NUMP];
  void *diner();
  pthread_mutex_init(&eat_mutex, NULL);
  for (i=0;i<NUMP;i++)
    pthread_mutex_init(&fork_mutex[i], NULL);

   for (i=0;i<NUMP;i++){
      dn[i] = i;
      pthread_create(&diner_thread[i],NULL,diner,&dn[i]);
   }
   for (i=0;i<NUMP;i++)
      pthread_join(diner_thread[i],NULL);
  pthread_exit(0);

}

void *diner(int *i)
{
int v;
int eating = 0;
printf("I'm diner %d\n",*i);
v = *i;
while (eating < 10) {
   printf("%d is thinking\n", v);
   sleep( v/2);
   printf("%d is hungry\n", v);

   pthread_mutex_lock(&eat_mutex);
   if (num_diners == (NUMP-1)) 
       pthread_cond_wait(&space_avail,&eat_mutex);   
   num_diners++;
   pthread_mutex_unlock(&eat_mutex);

   pthread_mutex_lock(&fork_mutex[v]);
   pthread_mutex_lock(&fork_mutex[(v+1)%NUMP]);
   printf("%d is eating\n", v);
   eating++;
   sleep(1);
   printf("%d is done eating\n", v);
   pthread_mutex_unlock(&fork_mutex[v]);
   pthread_mutex_unlock(&fork_mutex[(v+1)%NUMP]);

   pthread_mutex_lock(&eat_mutex);
   if (num_diners == (NUMP-1)) pthread_cond_signal(&space_avail);
   num_diners--;
   pthread_mutex_unlock(&eat_mutex);

}
pthread_exit(NULL);
}


 
 
