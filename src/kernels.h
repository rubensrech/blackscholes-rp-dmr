#ifndef DMR_FUNCTIONS
#define DMR_FUNCTIONS

#define REL_ERROR    0
#define UINT_ERROR   1
#define HYBRID       2

#define CALL_RESULT_REL_ERR_THRESHOLD   1.25200e-05 // 1.25170e-05
#define PUT_RESULT_REL_ERR_THRESHOLD    4.76900e-07 // 4.76837e-07

unsigned long long getDMRErrors();

void BlackScholesGPU(double *CallResult, double *PutResult, float *CallResult_rp, float *PutResult_rp,
        double *StockPrice, double *OptionStrike, double *OptionYears, int optN);
#endif