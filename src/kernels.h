#ifndef KERNELS_H
#define KERNELS_H

void BlackScholesGPU(double *CallResult, double *PutResult,
        double *StockPrice, double *OptionStrike, double *OptionYears, int optN);
#endif