#!/bin/bash
for bs in {2000000,5000000}
do
    python inference.py --batch-size $bs
done