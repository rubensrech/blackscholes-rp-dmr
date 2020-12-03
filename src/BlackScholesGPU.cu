#include "util.h"

const double RISKFREE    = 0.02f;
const double VOLATILITY  = 0.30f;

///////////////////////////////////////////////////////////////////////////////
// Polynomial approximation of cumulative normal distribution function
///////////////////////////////////////////////////////////////////////////////
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

///////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
///////////////////////////////////////////////////////////////////////////////
__device__ inline void BlackScholesBodyGPU(double &CallResult, double &PutResult, float &CallResult_rp,
        float &PutResult_rp, double S, double X, double T, double R, double V) {
    // > Full-precision
    double sqrtT, expRT;
    double d1, d2, CNDD1, CNDD2;
    // > Reduced-precision
    float S_rp = float(S), X_rp = float(X), T_rp = float(T), R_rp = float(R), V_rp = float(V);
    float sqrtT_rp, expRT_rp;
    float d1_rp, d2_rp, CNDD1_rp, CNDD2_rp;
    
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
}


////////////////////////////////////////////////////////////////////////////////
//Process an array of optN options on GPU
////////////////////////////////////////////////////////////////////////////////
__global__ void BlackScholesKernel(double *CallResult, double *PutResult, float *CallResult_rp, float *PutResult_rp,
        double *StockPrice, double *OptionStrike, double *OptionYears, double Riskfree, double Volatility, int optN) {

    const int opt = blockDim.x * blockIdx.x + threadIdx.x;
    if (opt < optN) {
        BlackScholesBodyGPU(CallResult[opt], PutResult[opt], CallResult_rp[opt], PutResult_rp[opt],
                StockPrice[opt], OptionStrike[opt], OptionYears[opt], Riskfree, Volatility);
    }
}

void BlackScholesGPU(double *CallResult, double *PutResult, float *CallResult_rp, float *PutResult_rp,
        double *StockPrice, double *OptionStrike, double *OptionYears, int optN) {
    BlackScholesKernel<<<DIV_UP(optN, BLOCK_SIZE), BLOCK_SIZE>>>(CallResult, PutResult, CallResult_rp,
            PutResult_rp, StockPrice, OptionStrike, OptionYears, RISKFREE, VOLATILITY, optN);
}



