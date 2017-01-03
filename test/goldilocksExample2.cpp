
#include <iostream>
#include <fstream>
#include "stdlib.h"
#include "pthread.h"
using namespace std;

class IntBox {
    
  public:
    int data;
    
};


IntBox *a,*b,*tmp1,*tmp2,*tmp3;
pthread_mutex_t lck1;
pthread_mutex_t lck2;



int v;
void* method1(void* s)
{
  tmp1= new IntBox;
  tmp1->data=0;
  pthread_mutex_lock(&lck1);
  
  a= tmp1;
  pthread_mutex_unlock(&lck1);
}

void* method2(void* s)
{
  
  pthread_mutex_lock(&lck1);
  tmp2=a;
  pthread_mutex_unlock(&lck1);
  pthread_mutex_lock(&lck2);
  b=tmp2;
  pthread_mutex_unlock(&lck2);
  
  
}


void* method3(void* s)
{
  pthread_mutex_lock(&lck2);
  b->data=2;
  tmp3=b;
  pthread_mutex_unlock(&lck2);
  tmp3->data=3;

}



int main()
{
  
  
  pthread_mutex_init (&lck1, NULL);
  pthread_mutex_init (&lck2, NULL);
  pthread_t thread1;
  pthread_t thread2;
  pthread_t thread3;
  pthread_create (&thread1,NULL,&method1,NULL);
  pthread_create (&thread2,NULL,&method2,NULL);
  pthread_create (&thread3,NULL,&method3,NULL);
  
 
  pthread_join(thread1,NULL);
  pthread_join(thread2,NULL);
  pthread_join(thread3,NULL);
  


  
  return 0;
}

