#include <fstream>
using namespace std;

#include <string.h>

////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options on CPU
////////////////////////////////////////////////////////////////////////////////
extern "C" void BlackScholesCPU(
    double *h_CallResult,
    double *h_PutResult,
    double *h_StockPrice,
    double *h_OptionStrike,
    double *h_OptionYears,
    double Riskfree,
    double Volatility,
    int optN
);

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
__device__ inline void BlackScholesBodyGPU(
    double &CallResult,
    double &PutResult,
    float &CallResult_rp,
    float &PutResult_rp,
    double S, //Stock price
    double X, //Option strike
    double T, //Option years
    double R, //Riskless rate
    double V  //Volatility rate
) {
    // > Full-precision
    double sqrtT, expRT;
    double d1, d2, CNDD1, CNDD2;
    double parrotOutput[1];
    // > Reduced-precision
    float S_rp = float(S), X_rp = float(X), T_rp = float(T), R_rp = float(R), V_rp = float(V);
    float sqrtT_rp, expRT_rp;
    float d1_rp, d2_rp, CNDD1_rp, CNDD2_rp;
    float parrotOutput_rp[1];
    
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

    // Calculate Call and Put simultaneously
    // > Full-precision
    expRT = exp(- R * T);
    CallResult = S * CNDD1 - X * expRT * CNDD2;
    parrotOutput[0] = CallResult / 10.0;
    // > Reduced-precision
    expRT_rp = __expf(- R_rp * T_rp);
    CallResult_rp = S_rp * CNDD1_rp - X_rp * expRT_rp * CNDD2_rp;
    parrotOutput_rp[0] = CallResult_rp / 10.0;

    // > Full-precision
    CallResult = parrotOutput[0] * 10.0;
    PutResult  = X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1);
    // > Reduced-precision
    CallResult_rp = parrotOutput_rp[0] * 10.0;
    PutResult_rp  = X_rp * expRT_rp * (1.0f - CNDD2_rp) - S_rp * (1.0f - CNDD1_rp);
}


////////////////////////////////////////////////////////////////////////////////
//Process an array of optN options on GPU
////////////////////////////////////////////////////////////////////////////////
__global__ void BlackScholesGPU(
    double *d_CallResult,
    double *d_PutResult,
    float *d_CallResult_rp,
    float *d_PutResult_rp,
    double *d_StockPrice,
    double *d_OptionStrike,
    double *d_OptionYears,
    double Riskfree,
    double Volatility,
    int optN
) {
    const int opt = blockDim.x * blockIdx.x + threadIdx.x;
    if (opt < optN)
        BlackScholesBodyGPU(
            d_CallResult[opt],
            d_PutResult[opt],
            d_CallResult_rp[opt],
            d_PutResult_rp[opt],
            d_StockPrice[opt],
            d_OptionStrike[opt],
            d_OptionYears[opt],
            Riskfree,
            Volatility
        );
}

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const int OPT_N = 4000000;
const int  NUM_ITERATIONS = 1;

const int   OPT_SZ      = OPT_N * sizeof(double);
const int   OPT_SZ_RP   = OPT_N * sizeof(float);
const double RISKFREE    = 0.02f;
const double VOLATILITY  = 0.30f;

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {

    if (argc < 3) {
        printf("Usage: %s <input-file> <output-file>\n", argv[0]);
        exit(1);
    }

    char *inputFilename = argv[1];
    char *outputFilename = argv[2];

    char output32Filename[100];
    strcpy(output32Filename, outputFilename);
    strcat(output32Filename, "32");

    // ======================================
    // == Declaring variables
    // ======================================
    // > Host data
    // >> Full-precision
    double *h_CallResultGPU, *h_PutResultGPU; // CPU copy of GPU results
    double *h_StockPrice, *h_OptionStrike, *h_OptionYears; // CPU instance of input data
    // >> Reduced-precision
    float *h_CallResultGPU_rp, *h_PutResultGPU_rp;
    
    // > Device data
    // >> Full-precision
    double *d_CallResult, *d_PutResult; // Results calculated by GPU
    double *d_StockPrice, *d_OptionStrike, *d_OptionYears; // GPU instance of input data
    // >> Reduced-precision
    float *d_CallResult_rp, *d_PutResult_rp;

    int i;

    // ======================================
    // == Allocating memory
    // ======================================
    // > Host data
    // >> Full-precision
    h_CallResultGPU = (double *)malloc(OPT_SZ);
    h_PutResultGPU  = (double *)malloc(OPT_SZ);
    h_StockPrice    = (double *)malloc(OPT_SZ);
    h_OptionStrike  = (double *)malloc(OPT_SZ);
    h_OptionYears   = (double *)malloc(OPT_SZ);
    // >> Reduced-precision
    h_CallResultGPU_rp = (float *)malloc(OPT_SZ_RP);
    h_PutResultGPU_rp  = (float *)malloc(OPT_SZ_RP);

    // > Device data
    // >> Full-precision
    cudaMalloc((void **)&d_CallResult,   OPT_SZ);
    cudaMalloc((void **)&d_PutResult,    OPT_SZ);
    cudaMalloc((void **)&d_StockPrice,   OPT_SZ);
    cudaMalloc((void **)&d_OptionStrike, OPT_SZ);
    cudaMalloc((void **)&d_OptionYears,  OPT_SZ);
    // >> Reduced-precision
    cudaMalloc((void **)&d_CallResult_rp,   OPT_SZ_RP);
    cudaMalloc((void **)&d_PutResult_rp,    OPT_SZ_RP);


    // ======================================
    // == Reading input data
    // ======================================
    std::ifstream dataFile(inputFilename);

    int numberOptions;
    dataFile >> numberOptions;

    double stockPrice, optionStrike, optionYear;
    for (i = 0; i < numberOptions; i++) {
        dataFile >> stockPrice >> optionStrike >> optionYear;
        h_StockPrice[i] = stockPrice;
        h_OptionStrike[i] = optionStrike;
        h_OptionYears[i] =  optionYear;      
    }

    // ======================================
    // == Copying data to device
    // ======================================
    cudaMemcpy(d_StockPrice,    h_StockPrice,   numberOptions * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_OptionStrike,  h_OptionStrike, numberOptions * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_OptionYears,   h_OptionYears,  numberOptions * sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // ======================================
    // == Executing on device
    // ======================================
    for (i = 0; i < NUM_ITERATIONS; i++) {
        BlackScholesGPU<<<DIV_UP(numberOptions, 128), 128>>>(
            d_CallResult,
            d_PutResult,
            d_CallResult_rp,
            d_PutResult_rp,
            d_StockPrice,
            d_OptionStrike,
            d_OptionYears,
            RISKFREE,
            VOLATILITY,
            numberOptions
        ); 
    }

    cudaDeviceSynchronize();

    // ======================================
    // == Reading back results from device
    // ======================================
    // > Full-precision
    cudaMemcpy(h_CallResultGPU, d_CallResult, numberOptions * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_PutResultGPU,  d_PutResult,  numberOptions * sizeof(double), cudaMemcpyDeviceToHost);
    // > Reduced-precision
    cudaMemcpy(h_CallResultGPU_rp, d_CallResult_rp, numberOptions * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_PutResultGPU_rp,  d_PutResult_rp,  numberOptions * sizeof(float), cudaMemcpyDeviceToHost);

    // ======================================
    // == Writing results to output file
    // ======================================
    ofstream callResultFile;
    ofstream callResultFile32;
    callResultFile.open(outputFilename);
    callResultFile32.open(output32Filename);
    for (i = 0 ; i < numberOptions; i++) {
        callResultFile << h_CallResultGPU[i] << std::endl;
        callResultFile32 << h_CallResultGPU_rp[i] << std::endl;
    }
    callResultFile.close();
    callResultFile32.close();

    // ======================================
    // == Deallocating memory
    // ======================================
    // > Device data
    // >> Full-precision
    cudaFree(d_OptionYears);
    cudaFree(d_OptionStrike);
    cudaFree(d_StockPrice);
    cudaFree(d_PutResult);
    cudaFree(d_CallResult);
    // >> Reduced-precision
    cudaFree(d_CallResult_rp);
    cudaFree(d_PutResult_rp);

    // > Host data
    // >> Full-precision
    free(h_OptionYears);
    free(h_OptionStrike);
    free(h_StockPrice);
    free(h_PutResultGPU);
    free(h_CallResultGPU);
    // >> Reduced-precision
    free(h_CallResultGPU_rp);
    free(h_PutResultGPU_rp);

    cudaDeviceReset();
    exit(EXIT_SUCCESS);
}
