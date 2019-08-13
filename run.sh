#!/bin/bash
for bs in {1,256,2048,5000,10000,20000,30000,50000,100000,1048576,2000000,5000000}
do
    python inference.py --batch-size $bs 
done