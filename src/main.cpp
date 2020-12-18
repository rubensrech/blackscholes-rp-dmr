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
    // * DMR Type => NO-DMR: 0, DMR-FLOAT: 32, DMR: 64
    int dmrType = find_int_arg(argc, argv, (char*)"-dmr", 0);
    // * Iterations
    int iterations = find_int_arg(argc, argv, (char*)"-it", 1000);

    cout << "> Input file: " << inputFilename <<  endl << endl;
    
    if (dmrType == NO_DMR)
        cout << "> DMR: NO-DMR" << endl;
    if (dmrType == DMR_FLOAT)
        cout << "> DMR: Float" << endl;
    if (dmrType == DMR_DOUBLE)
        cout << "> DMR: Double" << endl;

    // ======================================
    // == Declaring variables
    // ======================================
    // > Host data
    double *h_CallResultGPU, *h_PutResultGPU; // CPU copy of GPU results
    double *h_StockPrice, *h_OptionStrike, *h_OptionYears; // CPU instance of input data
    
    // > Device data
    double *d_CallResult, *d_PutResult; // Results calculated by GPU
    double *d_StockPrice, *d_OptionStrike, *d_OptionYears; // GPU instance of input data
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
    h_CallResultGPU = (double *)malloc(optionsSize);
    h_PutResultGPU  = (double *)malloc(optionsSize);
    h_StockPrice    = (double *)malloc(optionsSize);
    h_OptionStrike  = (double *)malloc(optionsSize);
    h_OptionYears   = (double *)malloc(optionsSize);

    // > Device data
    cudaMalloc((void **)&d_CallResult,   optionsSize);
    cudaMalloc((void **)&d_PutResult,    optionsSize);
    cudaMalloc((void **)&d_StockPrice,   optionsSize);
    cudaMalloc((void **)&d_OptionStrike, optionsSize);
    cudaMalloc((void **)&d_OptionYears,  optionsSize);


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

    // ======================================
    // == Copying data to device
    // ======================================
    float memCpyToDeviceTimeMs;
    if (measureTime) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
    }

    cudaMemcpy(d_StockPrice,    h_StockPrice,   optionsSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_OptionStrike,  h_OptionStrike, optionsSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_OptionYears,   h_OptionYears,  optionsSize, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    if (measureTime) {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&memCpyToDeviceTimeMs, start, stop);
        printf("%s* MemCpy to device: %f ms\n%s", GREEN, memCpyToDeviceTimeMs, DFT_COLOR);
    }

    // ======================================
    // == Executing on device
    // ======================================
    float kernelTimeMs;
    if (measureTime) {
        cudaEventRecord(start, 0);
    }


    if (dmrType == NO_DMR)
        for (i = 0; i < iterations; i++)
            BlackScholesGPU_noDMR(d_CallResult, d_PutResult, d_StockPrice, d_OptionStrike, d_OptionYears, numberOptions);
    if (dmrType == DMR_FLOAT)
        for (i = 0; i < iterations; i++)
            BlackScholesGPU_DMR_float(d_CallResult, d_PutResult, d_StockPrice, d_OptionStrike, d_OptionYears, numberOptions);
    if (dmrType == DMR_DOUBLE)
        for (i = 0; i < iterations; i++)
            BlackScholesGPU_DMR_double(d_CallResult, d_PutResult, d_StockPrice, d_OptionStrike, d_OptionYears, numberOptions);

    if (measureTime) {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&kernelTimeMs, start, stop);
        if (dmrType == NO_DMR)
            printf("%s* Kernel: %f ms (%d iterations)\n%s", GREEN, kernelTimeMs, iterations, DFT_COLOR);
        else
            printf("%s* Kernel + check faults: %f ms (%d iterations)\n%s", GREEN, kernelTimeMs, iterations, DFT_COLOR);
    }

    // ======================================
    // == Reading back results from device
    // ======================================
    float memCpyToHostTimeMs;
    if (measureTime) {
        cudaEventRecord(start, 0);
    }

    // > Full-precision
    cudaMemcpy(h_CallResultGPU, d_CallResult, optionsSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_PutResultGPU,  d_PutResult,  optionsSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    if (measureTime) {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&memCpyToHostTimeMs, start, stop);
        printf("%s* MemCpy to host: %f ms\n%s", GREEN, memCpyToHostTimeMs, DFT_COLOR);

        float dmrTotalTimeMs = memCpyToDeviceTimeMs + kernelTimeMs + memCpyToHostTimeMs;
        printf("%s* Total DMR time: %f ms%s\n", GREEN, dmrTotalTimeMs, DFT_COLOR);
    }

    // ======================================
    // == Checking for faults
    // ======================================

    if (dmrType != NO_DMR) {
        unsigned long long dmrErrors = getDMRErrors();
        bool faultDetected = dmrErrors > 0;
        cout << "> Faults detected?  " << (faultDetected ? "YES" : "NO") << " (DMR errors: " << dmrErrors << ")" << endl;
    }

    // ======================================
    // == Deallocating memory
    // ======================================
    // > Device data
    cudaFree(d_OptionYears);
    cudaFree(d_OptionStrike);
    cudaFree(d_StockPrice);
    cudaFree(d_PutResult);
    cudaFree(d_CallResult);

    // > Host data
    // >> Full-precision
    free(h_OptionYears);
    free(h_OptionStrike);
    free(h_StockPrice);
    free(h_PutResultGPU);
    free(h_CallResultGPU);

    if (measureTime) {
        getTimeNow(&t1);
        cout << endl;
        cout << "> Total execution time: " << elapsedTime(t0, t1) << " ms" << endl;
    }

    exit(EXIT_SUCCESS);
}
