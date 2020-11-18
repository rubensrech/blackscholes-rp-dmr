#include <fstream>
using namespace std;

////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options on CPU
////////////////////////////////////////////////////////////////////////////////
extern "C" void BlackScholesCPU(
    float *h_CallResult,
    float *h_PutResult,
    float *h_StockPrice,
    float *h_OptionStrike,
    float *h_OptionYears,
    float Riskfree,
    float Volatility,
    int optN
);

///////////////////////////////////////////////////////////////////////////////
// Polynomial approximation of cumulative normal distribution function
///////////////////////////////////////////////////////////////////////////////
__device__ inline float cndGPU(float d) {
    const float       A1 = 0.31938153f;
    const float       A2 = -0.356563782f;
    const float       A3 = 1.781477937f;
    const float       A4 = -1.821255978f;
    const float       A5 = 1.330274429f;
    const float RSQRT2PI = 0.39894228040143267793994605993438f;

    float
    K = 1.0f / (1.0f + 0.2316419f * fabsf(d));

    float
    cnd = RSQRT2PI * __expf(- 0.5f * d * d) *
          (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0f - cnd;

    return cnd;
}


///////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
///////////////////////////////////////////////////////////////////////////////
__device__ inline void BlackScholesBodyGPU(
    float &CallResult,
    float &PutResult,
    float S, //Stock price
    float X, //Option strike
    float T, //Option years
    float R, //Riskless rate
    float V  //Volatility rate
) {
    float sqrtT, expRT;
    float d1, d2, CNDD1, CNDD2;

    float parrotOutput[1];
    
    sqrtT = sqrtf(T);
    d1 = (__logf(S / X) + (R + 0.5f * V * V) * T) / (V * sqrtT);
    d2 = d1 - V * sqrtT;

    CNDD1 = cndGPU(d1);
    CNDD2 = cndGPU(d2);

    // Calculate Call and Put simultaneously
    expRT = __expf(- R * T);
    CallResult = S * CNDD1 - X * expRT * CNDD2;
    parrotOutput[0] = CallResult / 10.0;

    CallResult = parrotOutput[0] * 10.0;
    PutResult  = X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1);
}


////////////////////////////////////////////////////////////////////////////////
//Process an array of optN options on GPU
////////////////////////////////////////////////////////////////////////////////
__global__ void BlackScholesGPU(
    float *d_CallResult,
    float *d_PutResult,
    float *d_StockPrice,
    float *d_OptionStrike,
    float *d_OptionYears,
    float Riskfree,
    float Volatility,
    int optN
) {
    const int opt = blockDim.x * blockIdx.x + threadIdx.x;
    if (opt < optN)
        BlackScholesBodyGPU(
            d_CallResult[opt],
            d_PutResult[opt],
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

const int   OPT_SZ      = OPT_N * sizeof(float);
const float RISKFREE    = 0.02f;
const float VOLATILITY  = 0.30f;

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

    // ======================================
    // == Declaring variables
    // ======================================
    // > Host data
    float *h_CallResultGPU, *h_PutResultGPU; // CPU copy of GPU results
    float *h_StockPrice, *h_OptionStrike, *h_OptionYears; // CPU instance of input data
    
    // > Device data
    float *d_CallResult, *d_PutResult; // Results calculated by GPU
    float *d_StockPrice, *d_OptionStrike, *d_OptionYears; // GPU instance of input data
    
    int i;

    // ======================================
    // == Allocating memory
    // ======================================
    // > Host data
    h_CallResultGPU = (float *)malloc(OPT_SZ);
    h_PutResultGPU  = (float *)malloc(OPT_SZ);
    h_StockPrice    = (float *)malloc(OPT_SZ);
    h_OptionStrike  = (float *)malloc(OPT_SZ);
    h_OptionYears   = (float *)malloc(OPT_SZ);

    // > Device data
    cudaMalloc((void **)&d_CallResult,   OPT_SZ);
    cudaMalloc((void **)&d_PutResult,    OPT_SZ);
    cudaMalloc((void **)&d_StockPrice,   OPT_SZ);
    cudaMalloc((void **)&d_OptionStrike, OPT_SZ);
    cudaMalloc((void **)&d_OptionYears,  OPT_SZ);

    // ======================================
    // == Reading input data
    // ======================================
    std::ifstream dataFile(inputFilename);

    int numberOptions;
    dataFile >> numberOptions;

    float stockPrice, optionStrike, optionYear;
    for (i = 0; i < numberOptions; i++) {
        dataFile >> stockPrice >> optionStrike >> optionYear;
        h_StockPrice[i] = stockPrice;
        h_OptionStrike[i] = optionStrike;
        h_OptionYears[i] =  optionYear;      
    }

    // ======================================
    // == Copying data to device
    // ======================================
    cudaMemcpy(d_StockPrice,    h_StockPrice,   numberOptions * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_OptionStrike,  h_OptionStrike, numberOptions * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_OptionYears,   h_OptionYears,  numberOptions * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // ======================================
    // == Executing on device
    // ======================================
    for (i = 0; i < NUM_ITERATIONS; i++) {
        BlackScholesGPU<<<DIV_UP(numberOptions, 128), 128>>>(
            d_CallResult,
            d_PutResult,
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
    cudaMemcpy(h_CallResultGPU, d_CallResult, numberOptions * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_PutResultGPU,  d_PutResult,  numberOptions * sizeof(float), cudaMemcpyDeviceToHost);

    // ======================================
    // == Writing results to output file
    // ======================================
    ofstream callResultFile;
    callResultFile.open(outputFilename);
    for (i = 0 ; i < numberOptions; i++) {
        callResultFile << h_CallResultGPU[i] << std::endl;
    }
    callResultFile.close();

    // ======================================
    // == Deallocating memory
    // ======================================
    cudaFree(d_OptionYears);
    cudaFree(d_OptionStrike);
    cudaFree(d_StockPrice);
    cudaFree(d_PutResult);
    cudaFree(d_CallResult);
    free(h_OptionYears);
    free(h_OptionStrike);
    free(h_StockPrice);
    free(h_PutResultGPU);
    free(h_CallResultGPU);

    cudaDeviceReset();
    exit(EXIT_SUCCESS);
}
