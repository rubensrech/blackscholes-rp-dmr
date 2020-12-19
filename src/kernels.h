#ifndef KERNELS_H
#define KERNELS_H

unsigned long long getDMRErrors();

void BlackScholesGPU_noDMR(
    double *CallResult, double *PutResult,
    double *StockPrice, double *OptionStrike, double *OptionYears, int optN
);

void BlackScholesGPU_DMR_float(
    double *CallResult, double *PutResult,
    double *StockPrice, double *OptionStrike, double *OptionYears, int optN
);

void BlackScholesGPU_DMR_double(
    double *CallResult, double *PutResult,
    double *StockPrice, double *OptionStrike, double *OptionYears, int optN
);

#endif