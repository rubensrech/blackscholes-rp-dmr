#include <fstream>
#include <iostream>
using namespace std;

#include "util.h"

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

extern void BlackScholesGPU(
    double *d_CallResult,
    double *d_PutResult,
    float *d_CallResult_rp,
    float *d_PutResult_rp,
    double *d_StockPrice,
    double *d_OptionStrike,
    double *d_OptionYears,
    int optN
);

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

    // > Reading size of input
    std::ifstream dataFile(inputFilename);

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

    // ======================================
    // == Copying data to device
    // ======================================
    cudaMemcpy(d_StockPrice,    h_StockPrice,   optionsSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_OptionStrike,  h_OptionStrike, optionsSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_OptionYears,   h_OptionYears,  optionsSize, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // ======================================
    // == Executing on device
    // ======================================
    BlackScholesGPU(d_CallResult, d_PutResult, d_CallResult_rp, d_PutResult_rp,
                    d_StockPrice, d_OptionStrike, d_OptionYears, numberOptions);
    cudaDeviceSynchronize();

    // ======================================
    // == Reading back results from device
    // ======================================
    // > Full-precision
    cudaMemcpy(h_CallResultGPU, d_CallResult, optionsSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_PutResultGPU,  d_PutResult,  optionsSize, cudaMemcpyDeviceToHost);
    // > Reduced-precision
    cudaMemcpy(h_CallResultGPU_rp, d_CallResult_rp, optionsSize_rp, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_PutResultGPU_rp,  d_PutResult_rp,  optionsSize_rp, cudaMemcpyDeviceToHost);

    // ======================================
    // == Comparing: FP64 vs FP32
    // ======================================
    float maxRelErr = -999;
    for (i = 0 ; i < numberOptions; i++) {
        float relErr = abs(1 - h_CallResultGPU_rp[i] / float(h_CallResultGPU[i]));
        if (relErr > maxRelErr) maxRelErr = relErr;
    }
    cout << "Max relative error: " << maxRelErr << endl;

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
