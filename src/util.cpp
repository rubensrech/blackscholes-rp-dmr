#include <fstream>
using namespace std;

#include "util.h"

uint32_t log2(uint32_t n) {
    return (n > 1) ? 1 + log2(n >> 1) : 0;
}

// > Timing functions

void getTimeNow(Time *t) {
    gettimeofday(t, 0);
}

double elapsedTime(Time t1, Time t2) {
    return (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
}

// > Arguments functions

void del_arg(int argc, char **argv, int index) {
    int i;
    for(i = index; i < argc-1; ++i) argv[i] = argv[i+1];
    argv[i] = 0;
}

int find_int_arg(int argc, char **argv, char *arg, int def) {
    int i;
    for (i = 0; i < argc-1; ++i) {
        if(!argv[i]) continue;
        if (0==strcmp(argv[i], arg)) {
            def = atoi(argv[i+1]);
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}

char *find_char_arg(int argc, char **argv, char *arg, char *def) {
    int i;
    for(i = 0; i < argc-1; ++i){
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg)){
            def = argv[i+1];
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}

// > Input/Output management functions

bool save_output(double *CallResult, double *PutResult, int N) {
    ofstream f("gold_output.data", ios::out | ios::binary);

    if (!f.is_open()) {
        cerr << "ERROR: could not save output to file" << endl;
        exit(-1);
    }

    f.write((char*)&N, sizeof(int));
    f.write((char*)CallResult, sizeof(double) * N);
    f.write((char*)PutResult, sizeof(double) * N);

    f.close();
    return true;
}

bool compare_output_with_golden(double *CallResult, double *PutResult, int N, char *goldOutputFilename) {
    ifstream f(goldOutputFilename, ios::in | ios::binary);
    double *gold_CallResult, *gold_PutResult;
    int n, i = 0;

    if (!f.is_open()) {
        cerr << "ERROR: could not read output from file" << endl;
        exit(-1);
    }

    f.read((char*)&n, sizeof(int));
    if (n != N) {
        cerr << "ERROR: Output data doesn't match the expected size" << endl;
        exit(-1);
    }

    gold_CallResult = (double*)malloc(N * sizeof(double));
    f.read((char*)gold_CallResult, sizeof(double) * N);

    gold_PutResult = (double*)malloc(N * sizeof(double));
    f.read((char*)gold_PutResult, sizeof(double) * N);

    bool outputsMatch = true;
    int countDiffs = 0;
    double absErr, relErr;
    double minAbsErr = __DBL_MAX__, maxAbsErr = __DBL_MIN__, sumAbsErr = 0, avgAbsErr = 0;
    double minRelErr = __DBL_MAX__, maxRelErr = __DBL_MIN__, sumRelErr = 0, avgRelErr = 0;
    for (i = 0; i < N; i++) {
        if (CallResult[i] != gold_CallResult[i]) {
            outputsMatch = false;
            countDiffs++;
            // Abs diff
            absErr = SUB_ABS(CallResult[i], gold_CallResult[i]);
            if (absErr > maxAbsErr) maxAbsErr = absErr;
            if (absErr < minAbsErr) minAbsErr = absErr;
            sumAbsErr += absErr;
            // Rel err
            relErr = abs(1 - CallResult[i]/gold_CallResult[i]);
            if (relErr > maxRelErr) maxRelErr = relErr;
            if (relErr < minRelErr) minRelErr = relErr;
            sumRelErr += relErr;
        }
        if (PutResult[i] != gold_PutResult[i]) {
            outputsMatch = false;
            countDiffs++;
            // Abs diff
            absErr = SUB_ABS(PutResult[i], gold_PutResult[i]);
            if (absErr > maxAbsErr) maxAbsErr = absErr;
            if (absErr < minAbsErr) minAbsErr = absErr;
            sumAbsErr += absErr;
            // Rel err
            relErr = abs(1 - PutResult[i]/gold_PutResult[i]);
            if (relErr > maxRelErr) maxRelErr = relErr;
            if (relErr < minRelErr) minRelErr = relErr;
            sumRelErr += relErr;
        }
    }

    if (countDiffs > 0) {
        avgAbsErr = sumAbsErr / countDiffs;
        avgRelErr = sumRelErr / countDiffs;
    } else {
        avgAbsErr = sumRelErr = minAbsErr = maxAbsErr = 0;
        avgRelErr = sumRelErr = minRelErr = maxRelErr = 0;
    }
    
    // Save stats to file
    FILE *fstats = fopen("out-vs-gold-stats.txt", "w");
    fprintf(fstats, "===================================================\n");
    fprintf(fstats, "========== Output VS Golden output stats ==========\n");
    fprintf(fstats, "===================================================\n");
    fprintf(fstats, "  > Outputs match: %s\n", outputsMatch ? "YES" : "NO");
    fprintf(fstats, "  > Number of diff values: %u\n", countDiffs);
    fprintf(fstats, "\n");
    fprintf(fstats, "  > Max absolute err: %.20e\n", maxAbsErr);
    fprintf(fstats, "  > Min absolute err: %.20e\n", minAbsErr);
    fprintf(fstats, "  > Avg absolute err: %.20e\n", avgAbsErr);
    fprintf(fstats, "  > Sum absolute err: %.20e\n", sumAbsErr);
    fprintf(fstats, "\n");
    fprintf(fstats, "  > Max relative err: %.20e\n", maxRelErr);
    fprintf(fstats, "  > Min relative err: %.20e\n", minRelErr);
    fprintf(fstats, "  > Avg relative err: %.20e\n", avgRelErr);
    fprintf(fstats, "  > Sum relative err: %.20e\n", sumRelErr);
    fprintf(fstats, "\n");
    fclose(fstats);

    f.close();
    return outputsMatch;
}