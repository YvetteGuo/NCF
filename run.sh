#!/bin/bash
for bs in {1,256,2048,5000,10000,20000,30000,50000,1048576,2000000,5000000}
do
    python inference.py --batch-size $bs
    
    > bash run.sh >&1 | tee log
done