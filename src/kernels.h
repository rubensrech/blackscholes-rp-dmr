#ifndef KERNELS_H
#define KERNELS_H

unsigned long long getDMRErrors();

void BlackScholesGPU(double *CallResult, double *PutResult, float *CallResult_rp, float *PutResult_rp,
        double *StockPrice, double *OptionStrike, double *OptionYears, int optN);
#endif