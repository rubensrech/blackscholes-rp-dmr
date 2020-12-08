#include <fstream>
using namespace std;

#define BLACK       "\e[0;30m"
#define RED         "\e[0;31m"
#define GREEN       "\e[0;32m"
#define YELLOW      "\e[0;33m"
#define BLUE        "\e[0;34m"
#define PURPLE      "\e[0;35m"
#define CYAN        "\e[0;36m"
#define WHITE       "\e[0;37m"
#define DFT_COLOR   "\e[0m"

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

    double
    K = 1.0f / (1.0f + 0.2316419f * fabsf(d));

    double
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
    double &CallResult,
    double &PutResult,
    double S, //Stock price
    double X, //Option strike
    double T, //Option years
    double R, //Riskless rate
    double V  //Volatility rate
) {
    double sqrtT, expRT;
    double d1, d2, CNDD1, CNDD2;

    double parrotOutput[1];
    
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
    double *d_CallResult,
    double *d_PutResult,
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

const int   OPT_SZ      = OPT_N * sizeof(double);
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

    // ======================================
    // == Declaring variables
    // ======================================
    // > Host data
    double *h_CallResultGPU, *h_PutResultGPU; // CPU copy of GPU results
    double *h_StockPrice, *h_OptionStrike, *h_OptionYears; // CPU instance of input data
    
    // > Device data
    double *d_CallResult, *d_PutResult; // Results calculated by GPU
    double *d_StockPrice, *d_OptionStrike, *d_OptionYears; // GPU instance of input data
    cudaEvent_t start, stop;

    int i;

    // ======================================
    // == Allocating memory
    // ======================================
    // > Host data
    h_CallResultGPU = (double *)malloc(OPT_SZ);
    h_PutResultGPU  = (double *)malloc(OPT_SZ);
    h_StockPrice    = (double *)malloc(OPT_SZ);
    h_OptionStrike  = (double *)malloc(OPT_SZ);
    h_OptionYears   = (double *)malloc(OPT_SZ);

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
    float memCpyToDeviceTimeMs;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaMemcpy(d_StockPrice,    h_StockPrice,   numberOptions * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_OptionStrike,  h_OptionStrike, numberOptions * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_OptionYears,   h_OptionYears,  numberOptions * sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&memCpyToDeviceTimeMs, start, stop);
    printf("%s* MemCpy to device: %f ms\n%s", GREEN, memCpyToDeviceTimeMs, DFT_COLOR);

    // ======================================
    // == Executing on device
    // ======================================
    float kernelTimeMs;
    cudaEventRecord(start, 0);

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

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernelTimeMs, start, stop);
    printf("%s* Kernel: %f ms\n%s", GREEN, kernelTimeMs, DFT_COLOR);

    cudaDeviceSynchronize();

    // ======================================
    // == Reading back results from device
    // ======================================
    float memCpyToHostTimeMs;
    cudaEventRecord(start, 0);

    cudaMemcpy(h_CallResultGPU, d_CallResult, numberOptions * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_PutResultGPU,  d_PutResult,  numberOptions * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&memCpyToHostTimeMs, start, stop);
    printf("%s* MemCpy to host: %f ms\n%s", GREEN, memCpyToHostTimeMs, DFT_COLOR);

    float noDmrTotalTimeMs = memCpyToDeviceTimeMs + kernelTimeMs + memCpyToHostTimeMs;
    printf("%s* Total NO-DMR time: %f ms%s\n", GREEN, noDmrTotalTimeMs, DFT_COLOR);


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
