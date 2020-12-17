import sys
import os
import pandas as pd

if len(sys.argv) < 2:
    print("Usage: python3 %s <filename>" % sys.argv[0])
    exit(-1)

filename = sys.argv[1]

df = pd.read_csv(filename)

df['benchmark'] = 'blackScholes'

cols = ['benchmark', 'error_model', 'detection_outcome', 'min_err', 'max_err', 'avg_err', 'relative_sum']
df = df[cols]

outFilename = os.path.dirname(filename) + '/sdcs-relative-sum.csv'

df.to_csv(outFilename, index=False)
