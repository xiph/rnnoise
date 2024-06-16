#include <cassert>
#include <cstring>
#include "xcorr_offload_kernel.hpp"

#include <stdio.h>

inline float MAC16_16(float c, float a, float b){ 
    return c+a*b; 
} 
 
#define MAX_PROCESSING_BLOCK_SIZE 1024
void xcorr_kernel(const float * x, const float * y, float* sum, int len) 
{ 

#pragma HLS INTERFACE mode=s_axilite port=return bundle=CONTROL_BUS 
#pragma HLS INTERFACE mode=s_axilite port=len bundle=CONTROL_BUS 
 
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=INPUT 
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=INPUT 
#pragma HLS INTERFACE m_axi port=sum offset=slave bundle=OUTPUT 
 
   int j; 
   float y_0, y_1, y_2, y_3; 
   assert(len>=3); 

   float x_copy[MAX_PROCESSING_BLOCK_SIZE]; 
   float y_copy[MAX_PROCESSING_BLOCK_SIZE];

   float sum_copy[4]; 
 
   float* x_arr_ptr = x_copy; 
   float* y_arr_ptr = y_copy; 
 
   memcpy(x_copy, x, len); 
   memcpy(y_copy, y, len); 
   memcpy(sum_copy, sum, 4); 
 
   y_3=0; /* gcc doesn't realize that y_3 can't be used uninitialized */ 
   y_0=*y_arr_ptr++; 
   y_1=*y_arr_ptr++; 
   y_2=*y_arr_ptr++; 
 
xcoor_1st_loop: 
   for (j=0;j<len-3;j+=4) 
   { 
      float tmp; 
      tmp = *x_arr_ptr++; 
      y_3=*y_arr_ptr++; 
      sum_copy[0] = MAC16_16(sum_copy[0],tmp,y_0); 
      sum_copy[1] = MAC16_16(sum_copy[1],tmp,y_1); 
      sum_copy[2] = MAC16_16(sum_copy[2],tmp,y_2); 
      sum_copy[3] = MAC16_16(sum_copy[3],tmp,y_3); 
      tmp=*x_arr_ptr++; 
      y_0=*y_arr_ptr++; 
      sum_copy[0] = MAC16_16(sum_copy[0],tmp,y_1); 
      sum_copy[1] = MAC16_16(sum_copy[1],tmp,y_2); 
      sum_copy[2] = MAC16_16(sum_copy[2],tmp,y_3); 
      sum_copy[3] = MAC16_16(sum_copy[3],tmp,y_0); 
      tmp=*x_arr_ptr++; 
      y_1=*y_arr_ptr++; 
      sum_copy[0] = MAC16_16(sum_copy[0],tmp,y_2); 
      sum_copy[1] = MAC16_16(sum_copy[1],tmp,y_3); 
      sum_copy[2] = MAC16_16(sum_copy[2],tmp,y_0); 
      sum_copy[3] = MAC16_16(sum_copy[3],tmp,y_1); 
      tmp=*x_arr_ptr++; 
      y_2=*y_arr_ptr++; 
      sum_copy[0] = MAC16_16(sum_copy[0],tmp,y_3); 
      sum_copy[1] = MAC16_16(sum_copy[1],tmp,y_0); 
      sum_copy[2] = MAC16_16(sum_copy[2],tmp,y_1); 
      sum_copy[3] = MAC16_16(sum_copy[3],tmp,y_2); 
   } 
xcoor_2nd_loop: 
   if (j++<len) 
   { 
      float tmp = *x_arr_ptr++; 
      y_3=*y_arr_ptr++; 
      sum_copy[0] = MAC16_16(sum_copy[0],tmp,y_0); 
      sum_copy[1] = MAC16_16(sum_copy[1],tmp,y_1); 
      sum_copy[2] = MAC16_16(sum_copy[2],tmp,y_2); 
      sum_copy[3] = MAC16_16(sum_copy[3],tmp,y_3); 
   } 
xcoor_3rd_loop: 
   if (j++<len) 
   { 
      float tmp=*x_arr_ptr++; 
      y_0=*y_arr_ptr++; 
      sum_copy[0] = MAC16_16(sum_copy[0],tmp,y_1); 
      sum_copy[1] = MAC16_16(sum_copy[1],tmp,y_2); 
      sum_copy[2] = MAC16_16(sum_copy[2],tmp,y_3); 
      sum_copy[3] = MAC16_16(sum_copy[3],tmp,y_0); 
   } 
xcoor_4th_loop: 
   if (j<len) 
   { 
      float tmp=*x_arr_ptr++; 
      y_1=*y_arr_ptr++; 
      sum_copy[0] = MAC16_16(sum_copy[0],tmp,y_2); 
      sum_copy[1] = MAC16_16(sum_copy[1],tmp,y_3); 
      sum_copy[2] = MAC16_16(sum_copy[2],tmp,y_0); 
      sum_copy[3] = MAC16_16(sum_copy[3],tmp,y_1); 
   } 
   memcpy(sum,sum_copy,4); 
}