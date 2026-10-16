#!/bin/sh
# usage: loop.sh group seeds
cd "$(dirname "$0")" && timeout 165 python3 drive_exp.py $1 $2 105 2>&1 | tail -2
