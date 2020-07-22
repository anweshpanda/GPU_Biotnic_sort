%%cu
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <cuda.h>

int no_blocks=64;
int no_threads=1024;
int total_size=no_blocks*no_threads;

void sequential_sort(float* a ,int n)
{
   for(int i=2;i<=n;i=i*2)
    {
      for(int j=i/2;j>0;j=j/2)
      {
        for(int index =0 ;index<n;index++)
         { 
          int k=index^j;
            if(index<k)
            {
                if((i&index) )
                {
                    if(a[k]>a[index])
                    {
                        float tmp=a[k];
                         a[k]=a[index];
                         a[index]=tmp;
                    }
                }
                else
                {
                    if(a[k]<a[index])
                    {
                        float tmp=a[k];
                        a[k]=a[index];
                        a[index]=tmp;
                    }
                }
                     
             }
           }
        }
    }
}


__global__ void sort(float *glob_arr,int n) 
{
    int index=threadIdx.x ;
    int glob_index=threadIdx.x + blockDim.x * blockIdx.x;
 


    extern __shared__ int a[1024];
    a[index]=glob_arr[glob_index];
    
    for(int i=2;i<=1024;i=i*2)
    {
      for(int j=i/2;j>0;j=j/2)
      {
          int k=index^j;
          if(index<k)
          {
            if((i&index) )
              {
                if(a[k]>a[index])
                {
                  float tmp=a[k];
                  a[k]=a[index];
                  a[index]=tmp;
                }
              }
          else
              {
                if(a[k]<a[index])
                {
                  float tmp=a[k];
                  a[k]=a[index];
                  a[index]=tmp;
                }
              }
            }
        __syncthreads();
    }
  } 
 glob_arr[glob_index]=a[index]; 
 
 }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 __global__ void merge(float *glob_arr,float* final_arr,int n) 
{
   
    int glob_index=threadIdx.x + blockDim.x * blockIdx.x;
 
 int i = glob_index;
 float key =glob_arr[i];
 int pos =0;
 int offset =1024;  
 int x = (i/offset)*offset;

 for (int j = 0; j < n; j+=offset)
 {
  
  int s=j;
  int e=j+offset-1;
  
  
   if(j<x)
   {
    while(s<=e)
    {
      int mid =(s+e)/2;
      if(glob_arr[mid] == key)
      {
       
        s=mid+1;
      }
      else if(glob_arr[mid]>key)
      {
        e = mid -1;
      }
      else
      {
        s = mid+1;
      }
    }
    pos += (s-j);
   }
   if(j>x)
   {
    while(s<=e)
    {
      int mid =(s+e)/2;
      if(glob_arr[mid] ==key)
      {
       
        e = mid -1;
      }
      else if(glob_arr[mid]>key)
      {
        e=mid-1;
      }
      else
      {
        s = mid+1;
      }
    }
    pos += (s-j);

   }
}
 pos += (i-x);
 final_arr[pos] = key;
 }


 //////////////////////////////////////////////////////////////////////////////////////////
int main() 
{
  double average_time1 = 0;
  double average_time2 = 0;
  double average_time3 = 0;

  for (int j = 0; j <10 ; ++j)
  {
       float* a = (float*)malloc(total_size*sizeof(float));
       float* arr = (float*)malloc(total_size*sizeof(float));
       struct timespec start1,start2,start3,end1,end2,end3;
       

       for(int i=0;i<total_size;i++)
       {
           a[i]=rand();
           arr[i] =a[i];
       }
       clock_gettime( CLOCK_REALTIME,&start1);
       sequential_sort(arr,total_size);
       clock_gettime( CLOCK_REALTIME,&end1);

       for (int i = 1; i < total_size; ++i)
       {
         if(arr[i]<arr[i-1])
         {
          printf("wrong answer\n");
         }
       }


       int size = sizeof(float);
       float *d_a,*d_b;
       cudaMalloc((void **)&d_a, size*total_size);
       cudaMalloc((void **)&d_b, size*total_size);

       clock_gettime( CLOCK_REALTIME,&start2);

       cudaMemcpy(d_a, a, size*total_size, cudaMemcpyHostToDevice);

       clock_gettime( CLOCK_REALTIME,&start3);

       sort<<<no_blocks,no_threads>>>(d_a,total_size);
       merge<<<no_blocks,no_threads>>>(d_a,d_b,total_size);
       clock_gettime( CLOCK_REALTIME,&end3);

      cudaMemcpy(a, d_b, size*total_size, cudaMemcpyDeviceToHost);

       clock_gettime( CLOCK_REALTIME,&end2);
       

      for(int i=0;i<total_size;i++)
      {
        if(arr[i]!=a[i])
        {
            printf("wrong answer\n");
        }
      }
       
      double elapsed1 = (end1.tv_sec-start1.tv_sec)*1000000000 + end1.tv_nsec-start1.tv_nsec;
      double elapsed2 = (end2.tv_sec-start2.tv_sec)*1000000000 + end2.tv_nsec-start2.tv_nsec;
      double elapsed3 = (end3.tv_sec-start3.tv_sec)*1000000000 + end3.tv_nsec-start3.tv_nsec;
      average_time1 += elapsed1;
      average_time3 += elapsed3;
      average_time2 += elapsed2;


       
       


       cudaFree(d_a);
       cudaFree(d_b);
  }
    printf("time taken for sequential sorting\n");
    printf("%lf\n",average_time1/10);

    printf("time taken for gpu sorting without memcopy\n");
    printf("%lf\n",average_time3/10);

    printf("time taken for gpu sorting with memcopy\n");
    printf("%lf\n",average_time2/10);

    printf("Speed up achieved is\n");
    printf("%lf\n",(average_time1/average_time2));

    printf("Speed up achieved without memcopy is\n");
    printf("%lf\n",(average_time1/average_time3));



 return 0;
}
