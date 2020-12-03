#include <fstream>
#include <iostream>
using namespace std;

#include "util.h"
#include "dmrFunctions.h"

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
    // ======================================
    // == Managing arguments
    // ======================================

    // * Input filename
    char *inputFilename = find_char_arg(argc, argv, (char*)"-input", (char*)"test.data/input/blackscholes_4000K.data");
    // * Save output
    bool saveOutput = find_int_arg(argc, argv, (char*)"-saveOutput", 0);
    // * Measure time
    bool measureTime = find_int_arg(argc, argv, (char*)"-measureTime", 0);

    cout << "> Input file: " << inputFilename <<  endl << endl;

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

    Time t0, t1;
    int i;

    if (measureTime) getTimeNow(&t0);

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
    // == Saving (gold) output
    // ======================================
    if (saveOutput) {
        if (save_output(h_CallResultGPU, h_PutResultGPU, numberOptions)) {
            cout << "OUTPUT SAVED SUCCESSFULY" << endl;
        } else {
            cerr << "ERROR: could not save output" << endl;
        }
    }

#ifdef FIND_THRESHOLD
    // ======================================
    // == Finding DMR-RP thresholds
    // ======================================
    cout << "FINDING THRESHOLDS:" << endl;

    int zerosCount_call = 0;
    float maxRelErr_call = -999, maxAbsErr_call = -999;
    uint32_t maxUintErr_call = -999;

    // > Call result
    for (i = 0 ; i < numberOptions; i++) {
        float lhs = h_CallResultGPU_rp[i];
        float rhs = float(h_CallResultGPU[i]);
        uint32_t lhs_data = *((uint32_t*) &lhs);
        uint32_t rhs_data = *((uint32_t*) &rhs);
        
        float relErr = abs(1 - lhs / rhs);
        float absErr = SUB_ABS(lhs, rhs);
        uint32_t uintErr = SUB_ABS(lhs_data, rhs_data);

        if (relErr > maxRelErr_call) maxRelErr_call = relErr;
        if (absErr > maxAbsErr_call) maxAbsErr_call = absErr;
        if (uintErr > maxUintErr_call) maxUintErr_call = uintErr;
        if (lhs == 0 || rhs == 0) zerosCount_call++;
    }

    cout << " > Call results:" << endl;
    cout << "   * Number of zeros: " << zerosCount_call << endl;
    cout << "   * Max relative error: " << maxRelErr_call << endl;
    cout << "   * Max absolute error: " << maxAbsErr_call << endl;
    cout << "   * Max UINT error: " << maxUintErr_call << " (Bit: " << log2(maxUintErr_call) << ")" << endl;

    int zerosCount_put = 0;
    float maxRelErr_put = -999, maxAbsErr_put = -999;
    uint32_t maxUintErr_put = -999;

    // > Put result
    for (i = 0 ; i < numberOptions; i++) {
        float lhs = h_PutResultGPU_rp[i];
        float rhs = float(h_PutResultGPU[i]);
        uint32_t lhs_data = *((uint32_t*) &lhs);
        uint32_t rhs_data = *((uint32_t*) &rhs);
        
        float relErr = abs(1 - lhs / rhs);
        float absErr = SUB_ABS(lhs, rhs);
        uint32_t uintErr = SUB_ABS(lhs_data, rhs_data);

        if (relErr > maxRelErr_put) maxRelErr_put = relErr;
        if (absErr > maxAbsErr_put) maxAbsErr_put = absErr;
        if (uintErr > maxUintErr_put) maxUintErr_put = uintErr;
        if (lhs == 0 || rhs == 0) zerosCount_put++;
    }

    cout << " > Put results:" << endl;
    cout << "   * Number of zeros: " << zerosCount_put << endl;
    cout << "   * Max relative error: " << maxRelErr_put << endl;
    cout << "   * Max absolute error: " << maxAbsErr_put << endl;
    cout << "   * Max UINT error: " << maxUintErr_put << " (Bit: " << log2(maxUintErr_put) << ")" << endl;
    
#else
    string errMetric = ERROR_METRIC == HYBRID ? "Hybrid (Rel + Abs)" : (ERROR_METRIC == UINT_ERROR ? "UINT Error" : "Relative Error");
    cout << "> Error metric: " << errMetric << endl;

    // ======================================
    // == Checking for faults
    // ======================================

    checkErrorsGPU(d_CallResult, d_CallResult_rp, numberOptions, CALL_RESULT_REL_ERR_THRESHOLD);
    checkErrorsGPU(d_PutResult, d_PutResult_rp, numberOptions, PUT_RESULT_REL_ERR_THRESHOLD);

    unsigned long long dmrErrors = getDMRErrors();
    bool faultDetected = dmrErrors > 0;
    cout << "> Faults detected?  " << (faultDetected ? "YES" : "NO") << " (DMR errors: " << dmrErrors << ")" << endl;
    
    // ======================================
    // == Comparing output with Golden output
    // ======================================
    bool outputIsCorrect = compare_output_with_golden(h_CallResultGPU, h_PutResultGPU, numberOptions);
    cout << "> Output corrupted? " << (!outputIsCorrect ? "YES" : "NO") << endl;

    // ======================================
    // == Classifing
    // ======================================
    cout << "> DMR classification: ";
    if (faultDetected && outputIsCorrect) cout << "FALSE POSITIVE" << endl;
    if (faultDetected && !outputIsCorrect) cout << "TRUE POSITIVE" << endl;
    if (!faultDetected && outputIsCorrect) cout << "TRUE NEGATIVE" << endl;
    if (!faultDetected && !outputIsCorrect) cout << "FALSE NEGATIVE" << endl;

#endif

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

    if (faultDetected) {
        exit(2);
    }

    if (measureTime) {
        getTimeNow(&t1);
        cout << endl;
        cout << "> Total execution time: " << elapsedTime(t0, t1) << " ms" << endl;
    }

    exit(EXIT_SUCCESS);
}
