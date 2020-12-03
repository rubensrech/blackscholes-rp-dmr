#include "dmrFunctions.h"
#include "util.h"

////////////////////////////////////////////////////////////////////////////////
// Check error functions
////////////////////////////////////////////////////////////////////////////////

__device__ unsigned long long errors = 0;

__forceinline__  __device__ void relativeError(double rhs, float lhs, float REL_ERR_THRESHOLD) {
    float relErr = abs(1 - lhs / float(rhs));
    if (relErr > REL_ERR_THRESHOLD) {
        atomicAdd(&errors, 1);
    }
}

__forceinline__  __device__ void uintError(double rhs, float lhs, float UINT_ERR_THRESHOLD) {
	float rhs_as_float = float(rhs);
	uint32_t lhs_data = *((uint32_t*) &lhs);
	uint32_t rhs_data = *((uint32_t*) &rhs_as_float);

	uint32_t uintErr = SUB_ABS(lhs_data, rhs_data);

	if (uintErr > UINT_ERR_THRESHOLD) {
		atomicAdd(&errors, 1);
	}
}

__global__ void checkErrorsKernel(double *array, float *array_rp, int n, float THRESHOLD) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
#if ERROR_METRIC == UINT_ERROR
        uintError(array[tid], array_rp[tid], THRESHOLD);
#else
        relativeError(array[tid], array_rp[tid], THRESHOLD);
#endif
}

void checkErrorsGPU(double *array, float *array_rp, int n, float THRESHOLD) {
    checkErrorsKernel<<<DIV_UP(n, BLOCK_SIZE), BLOCK_SIZE>>>(array, array_rp, n, THRESHOLD);
    CHECK_CUDA_ERROR(cudaPeekAtLastError());
}

// > Getters

unsigned long long getDMRErrors() {
    unsigned long long ret = 0;
    cudaMemcpyFromSymbol(&ret, errors, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
    return ret;
}