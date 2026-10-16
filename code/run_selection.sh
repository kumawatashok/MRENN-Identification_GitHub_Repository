#!/bin/sh
cd "$(dirname "$0")"
for w in MRENN ENN BPTT; do python3 selection.py $w > ../results/sel_$w.log 2>&1; done
