# Neural Collaborative Filtering
This is mxnet implementation for the paper:

Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu and Tat-Seng Chua (2017). Neural Collaborative Filtering. In Proceedings of WWW '17, Perth, Australia, April 03-07, 2017.

Three collaborative filtering models: Generalized Matrix Factorization (GMF), Multi-Layer Perceptron (MLP), and Neural Matrix Factorization (NeuMF). To target the models for implicit feedback and ranking task, we optimize them using log loss with negative sampling.

Author: Dr. Xiangnan He (http://www.comp.nus.edu.sg/~xiangnan/)

Code Reference: https://github.com/hexiangnan/neural_collaborative_filtering

## Environment Settings
We use MXnet with MKL-DNN as the backend.

- MXNet version for CPU: '1.5.0'

        pip install mxnet-mkl --pre
- MXNet version for GPU: '1.5.0' 
        
        pip install mxnet-cu90mkl --pre

## DataSet
MovieLens 1 Million (ml-1m) 
MovieLens 20 Million (ml-20m)

train.rating:
 - Train file.
 - Each Line is a training instance: userID\t itemID\t rating\t timestamp (if have)

test.rating:
- Test file (positive instances).
- Each Line is a testing instance: userID\t itemID\t rating\t timestamp (if have)

test.negative:
- test file (negative instances).
- Each line corresponds to the line of test.rating, containing 99 negative samples.
- Each line is in the format: (userID,itemID)\t negativeItemID1\t negativeItemID2 ...

## Training on CPU
    # KMP/OMP Settings
    export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
    export OMP_NUM_THREADS=28
    python train.py --factor-size 64 --layer=[256,256,128,64]

## Inference on CPU
    python inference.py --load-epoch 2 --batch-size=256 

## Training on GPU
    python train.py --factor-size 64 --layer=[256,256,128,64] --gpus 0

## Inference on GPU
    python inference.py --load-epoch 2 --batch-size=256 --gpus 0 

## Evalute accuray
    python inference.py --load-epoch 2 --batch-size=256 --benchmark

## bash file (multiple batch sizes)
    ./run.sh 2>&1 |tee log
