#ifndef UTIL_H
#define UTIL_H

#include <iostream>
using namespace std;

#include <string>

#include <time.h>
#include <sys/time.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#define CHECK_CUDA_ERROR(ans) { checkCudaError((ans), __FILE__, __LINE__); }
inline void checkCudaError(cudaError_t code, const char *file, int line, bool abort=false) {
    cudaDeviceSynchronize();
    if (code != cudaSuccess) {
        cout << "CUDA Error: " << cudaGetErrorString(code) << " " << file <<  " " << line << endl;
        if (abort) exit(code);
    }
}

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )
#define SUB_ABS(lhs, rhs) ((lhs > rhs) ? (lhs - rhs) : (rhs - lhs))

#define BLOCK_SIZE 128

#define BLACK       "\e[0;30m"
#define RED         "\e[0;31m"
#define GREEN       "\e[0;32m"
#define YELLOW      "\e[0;33m"
#define BLUE        "\e[0;34m"
#define PURPLE      "\e[0;35m"
#define CYAN        "\e[0;36m"
#define WHITE       "\e[0;37m"
#define DFT_COLOR   "\e[0m"

uint32_t log2(uint32_t n);

typedef struct timeval Time;
void getTimeNow(Time *t);
double elapsedTime(Time t1, Time t2);

int find_int_arg(int argc, char **argv, char *arg, int def);
char *find_char_arg(int argc, char **argv, char *arg, char *def);

bool save_output(double *CallResult, double *PutResult, int N);
bool compare_output_with_golden(double *CallResult, double *PutResult, int N, char *goldOutputFilename);

#endif