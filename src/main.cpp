#include <fstream>
#include <iostream>
using namespace std;

#include "util.h"
#include "kernels.h"
#include "dmr-constants.h"

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

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    // ======================================
    // == Managing arguments
    // ======================================

    // * Input filename
    char *inputFilename = find_char_arg(argc, argv, (char*)"-input", (char*)"test.data/input/blackscholes_4000K.data");
    // * Measure time
    bool measureTime = find_int_arg(argc, argv, (char*)"-measureTime", 0);
    // * Iterations
    int iterations = find_int_arg(argc, argv, (char*)"-it", 20);

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
    // >> Extra
    cudaEvent_t start, stop;

    Time t0, t1;
    int i;

    if (measureTime) getTimeNow(&t0);

    // ======================================
    // == Allocating memory
    // ======================================

    // > Reading size of input
    std::ifstream dataFile(inputFilename);
    if (!dataFile.is_open()) {
        cerr << "ERROR: could not open input file" << endl;
        exit(-1);
    }

    int numberOptions;
    dataFile >> numberOptions;

    const int optionsSize = numberOptions * sizeof(double);
    const int optionsSize_rp = numberOptions * sizeof(double);

    // > Host data
    // >> Full-precision
    h_CallResultGPU = (double *)malloc(optionsSize);
    h_PutResultGPU  = (double *)malloc(optionsSize);
    h_StockPrice    = (double *)malloc(optionsSize);
    h_OptionStrike  = (double *)malloc(optionsSize);
    h_OptionYears   = (double *)malloc(optionsSize);
    // >> Reduced-precision
    h_CallResultGPU_rp = (float *)malloc(optionsSize_rp);
    h_PutResultGPU_rp  = (float *)malloc(optionsSize_rp);

    // > Device data
    // >> Full-precision
    cudaMalloc((void **)&d_CallResult,   optionsSize);
    cudaMalloc((void **)&d_PutResult,    optionsSize);
    cudaMalloc((void **)&d_StockPrice,   optionsSize);
    cudaMalloc((void **)&d_OptionStrike, optionsSize);
    cudaMalloc((void **)&d_OptionYears,  optionsSize);
    // >> Reduced-precision
    cudaMalloc((void **)&d_CallResult_rp,   optionsSize_rp);
    cudaMalloc((void **)&d_PutResult_rp,    optionsSize_rp);


    // ======================================
    // == Reading input data
    // ======================================
    double stockPrice, optionStrike, optionYear;
    for (i = 0; i < numberOptions; i++) {
        dataFile >> stockPrice >> optionStrike >> optionYear;
        h_StockPrice[i] = stockPrice;
        h_OptionStrike[i] = optionStrike;
        h_OptionYears[i] =  optionYear;      
    }


for (i = 0; i < iterations; i++) {
    // ======================================
    // == Copying data to device
    // ======================================

    float memCpyToDeviceTimeMs;
    if (measureTime) {
        // cudaEventCreate(&start);
        // cudaEventCreate(&stop);
        // cudaEventRecord(start, 0);
    }

    cudaMemcpy(d_StockPrice,    h_StockPrice,   optionsSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_OptionStrike,  h_OptionStrike, optionsSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_OptionYears,   h_OptionYears,  optionsSize, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // ======================================
    // == Executing on device
    // ======================================

    BlackScholesGPU(d_CallResult, d_PutResult, d_CallResult_rp, d_PutResult_rp,
                    d_StockPrice, d_OptionStrike, d_OptionYears, numberOptions);

    // ======================================
    // == Reading back results from device
    // ======================================

    cudaMemcpy(h_CallResultGPU, d_CallResult, optionsSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_PutResultGPU,  d_PutResult,  optionsSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    if (measureTime) {
        // float totalTimeMs;
        // cudaEventRecord(stop, 0);
        // cudaEventSynchronize(stop);
        // cudaEventElapsedTime(&totalTimeMs, start, stop);
        // printf("%s* Total CUDA event time: %f ms (it: %d)%s\n", GREEN, totalTimeMs, i, DFT_COLOR);
    }

    // ======================================
    // == Checking for faults
    // ======================================

    unsigned long long dmrErrors = getDMRErrors();
    bool faultDetected = dmrErrors > 0;
    // cout << "> Faults detected?  " << (faultDetected ? "YES" : "NO") << " (DMR errors: " << dmrErrors << ")" << endl;
}

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

    if (measureTime) {
        getTimeNow(&t1);
        cout << GREEN << "* Total execution time: " << elapsedTime(t0, t1) << " ms (" << iterations << " iterations)" << DFT_COLOR << endl;
    }

    exit(EXIT_SUCCESS);
}
