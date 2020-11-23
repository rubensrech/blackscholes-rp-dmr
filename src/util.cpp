#include "util.h"

// > Timing functions

void getTimeNow(Time *t) {
    gettimeofday(t, 0);
}

double elapsedTime(Time t1, Time t2) {
    return (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
}