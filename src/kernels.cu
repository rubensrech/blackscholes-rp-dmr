#include "util.h"
#include "kernels.h"
#include "dmr-constants.h"

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

__device__ inline void BlackScholesBodyGPU(double &CallResult, double &PutResult,
        double S, double X, double T, double R, double V) {
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

__global__ void BlackScholesKernel(double *CallResult, double *PutResult,
        double *StockPrice, double *OptionStrike, double *OptionYears, double Riskfree, double Volatility, int optN) {

    const int opt = blockDim.x * blockIdx.x + threadIdx.x;
    if (opt < optN) {
        BlackScholesBodyGPU(CallResult[opt], PutResult[opt],
                StockPrice[opt], OptionStrike[opt], OptionYears[opt], Riskfree, Volatility);
    }
}

void BlackScholesGPU(double *CallResult, double *PutResult,
        double *StockPrice, double *OptionStrike, double *OptionYears, int optN) {
    BlackScholesKernel<<<DIV_UP(optN, BLOCK_SIZE), BLOCK_SIZE>>>(CallResult, PutResult,
            StockPrice, OptionStrike, OptionYears, RISKFREE, VOLATILITY, optN);
}




