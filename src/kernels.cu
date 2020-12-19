#include "util.h"
#include "kernels.h"
#include "dmr-constants.h"

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

__device__ void checkErrors(double rhs, float lhs, float THRESHOLD) {
#if ERROR_METRIC == UINT_ERROR
    uintError(rhs, lhs, THRESHOLD);
#else
    relativeError(rhs, lhs, THRESHOLD);
#endif
}

// > Getters

unsigned long long getDMRErrors() {
    unsigned long long ret = 0;
    cudaMemcpyFromSymbol(&ret, errors, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
    return ret;
}

////////////////////////////////////////////////////////////////////////////////
// BLACK SCHOLES
////////////////////////////////////////////////////////////////////////////////

const double RISKFREE    = 0.02f;
const double VOLATILITY  = 0.30f;

__device__ inline double cndGPU(double d) {
    const double       A1 = 0.31938153f;
    const double       A2 = -0.356563782f;
    const double       A3 = 1.781477937f;
    const double       A4 = -1.821255978f;
    const double       A5 = 1.330274429f;
    const double RSQRT2PI = 0.39894228040143267793994605993438f;

    double K = 1.0f / (1.0f + 0.2316419f * fabs(d));

    double cnd = RSQRT2PI * exp(- 0.5f * d * d) *
            (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0f - cnd;

    return cnd;
}

__device__ inline float cndGPU(float d) {
    const float       A1 = 0.31938153f;
    const float       A2 = -0.356563782f;
    const float       A3 = 1.781477937f;
    const float       A4 = -1.821255978f;
    const float       A5 = 1.330274429f;
    const float RSQRT2PI = 0.39894228040143267793994605993438f;

    float K = 1.0f / (1.0f + 0.2316419f * fabsf(d));

    float cnd = RSQRT2PI * __expf(- 0.5f * d * d) *
            (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0f - cnd;

    return cnd;
}

template<typename dmr_t>
__device__ inline void BlackScholes_DMR_body(
    double &CallResult, double &PutResult,
    double S, double X, double T, double R, double V
) {

    // > Full-precision
    double sqrtT, expRT;
    double d1, d2, CNDD1, CNDD2;
    // > Reduced-precision
    dmr_t S_rp = dmr_t(S), X_rp = dmr_t(X), T_rp = dmr_t(T), R_rp = dmr_t(R), V_rp = dmr_t(V);
    dmr_t CallResult_rp, PutResult_rp;
    dmr_t sqrtT_rp, expRT_rp;
    dmr_t d1_rp, d2_rp, CNDD1_rp, CNDD2_rp;
    
    // > Full-precision
    sqrtT = sqrt(T);
    d1 = (log(S / X) + (R + 0.5f * V * V) * T) / (V * sqrtT);
    d2 = d1 - V * sqrtT;
    // > Reduced-precision
    sqrtT_rp = sqrtf(T_rp);
    d1_rp = (__logf(S_rp / X_rp) + (R_rp + 0.5f * V_rp * V_rp) * T_rp) / (V_rp * sqrtT_rp);
    d2_rp = d1_rp - V_rp * sqrtT_rp;

    // > Full-precision
    CNDD1 = cndGPU(d1);
    CNDD2 = cndGPU(d2);
    // > Reduced-precision
    CNDD1_rp = cndGPU(d1_rp);
    CNDD2_rp = cndGPU(d2_rp);

    // > Full-precision
    expRT = exp(- R * T);
    CallResult = S * CNDD1 - X * expRT * CNDD2;
    // > Reduced-precision
    expRT_rp = __expf(- R_rp * T_rp);
    CallResult_rp = S_rp * CNDD1_rp - X_rp * expRT_rp * CNDD2_rp;

    // > Full-precision
    PutResult  = X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1);
    // > Reduced-precision
    PutResult_rp  = X_rp * expRT_rp * (1.0f - CNDD2_rp) - S_rp * (1.0f - CNDD1_rp);

    checkErrors(CallResult, CallResult_rp, CALL_RESULT_REL_ERR_THRESHOLD);
    checkErrors(PutResult, PutResult_rp, PUT_RESULT_REL_ERR_THRESHOLD);
}

__device__ inline void BlackScholes_noDMR_body(
    double &CallResult, double &PutResult,
    double S, double X, double T, double R, double V
) {

    double sqrtT, expRT;
    double d1, d2, CNDD1, CNDD2;
    
    sqrtT = sqrt(T);
    d1 = (log(S / X) + (R + 0.5f * V * V) * T) / (V * sqrtT);
    d2 = d1 - V * sqrtT;

    CNDD1 = cndGPU(d1);
    CNDD2 = cndGPU(d2);

    expRT = exp(- R * T);
    CallResult = S * CNDD1 - X * expRT * CNDD2;
    PutResult  = X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1);
}

// >


template<typename dmr_t>
__global__ void BlackScholes_DMR_kernel(
    double *CallResult, double *PutResult,
    double *StockPrice, double *OptionStrike, double *OptionYears, double Riskfree, double Volatility, int optN
) {
    const int opt = blockDim.x * blockIdx.x + threadIdx.x;
    if (opt < optN) {
        BlackScholes_DMR_body<dmr_t>(
            CallResult[opt], PutResult[opt],
            StockPrice[opt], OptionStrike[opt], OptionYears[opt], Riskfree, Volatility
        );
    }
}

__global__ void BlackScholes_noDMR_kernel(
    double *CallResult, double *PutResult,
    double *StockPrice, double *OptionStrike, double *OptionYears, double Riskfree, double Volatility, int optN
) {
    const int opt = blockDim.x * blockIdx.x + threadIdx.x;
    if (opt < optN) {
        BlackScholes_noDMR_body(
            CallResult[opt], PutResult[opt],
            StockPrice[opt], OptionStrike[opt], OptionYears[opt], Riskfree, Volatility
        );
    }
}

// >

void BlackScholesGPU_DMR_double(
    double *CallResult, double *PutResult,
    double *StockPrice, double *OptionStrike, double *OptionYears, int optN
) {
    BlackScholes_DMR_kernel<double><<<DIV_UP(optN, BLOCK_SIZE), BLOCK_SIZE>>>(
        CallResult, PutResult,
        StockPrice, OptionStrike, OptionYears, RISKFREE, VOLATILITY, optN
    );
}

void BlackScholesGPU_DMR_float(
    double *CallResult, double *PutResult,
    double *StockPrice, double *OptionStrike, double *OptionYears, int optN
) {
    BlackScholes_DMR_kernel<float><<<DIV_UP(optN, BLOCK_SIZE), BLOCK_SIZE>>>(
        CallResult, PutResult,
        StockPrice, OptionStrike, OptionYears, RISKFREE, VOLATILITY, optN
    );
}

void BlackScholesGPU_noDMR(
    double *CallResult, double *PutResult,
    double *StockPrice, double *OptionStrike, double *OptionYears, int optN
) {
    BlackScholes_noDMR_kernel<<<DIV_UP(optN, BLOCK_SIZE), BLOCK_SIZE>>>(
        CallResult, PutResult,
        StockPrice, OptionStrike, OptionYears, RISKFREE, VOLATILITY, optN
    );
}




