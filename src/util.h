#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>
#include <time.h>
#include <sys/time.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )

#define BLOCK_SIZE 128

typedef struct timeval Time;
void getTimeNow(Time *t);
double elapsedTime(Time t1, Time t2);

#endif