# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# 
import argparse
import logging
import mxnet as mx
import numpy as np
import multiprocessing as mp
from data import get_movielens_iter, get_movielens_data
import os
from time import time
from data import get_train_iters
from data import get_eval_iters
from Dataset import Dataset
from evaluate import evaluate_model
from mxnet import profiler
logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Run neural collaborative filtering inference",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')                                 
parser.add_argument('--batch-size', type=int, default=256,
                        help='number of examples per batch')
parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance')
parser.add_argument('--gpus', type=str,
                        help="list of gpus to run, e.g. 0 or 0,2. empty means using cpu().")
parser.add_argument('--benchmark', action='store_true', default=False,
                    help='run the script for benchmark mode, not set for accuracy test.')
parser.add_argument('--model-prefix', type=str, default='checkpoint',
                    help='the model prefix') 
parser.add_argument('--load-epoch', type=int, default=0,
                    help='loading the params of the corresponding training epoch.')                                       

if __name__ == '__main__':
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.INFO, format=head)

    # arg parser
    args = parser.parse_args()
    logging.info(args)

    batch_size = args.batch_size
    num_negatives = args.num_neg
    benchmark = args.benchmark
    model_prefix = args.model_prefix
    load_epoch = args.load_epoch

    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')] if args.gpus else [mx.cpu()]
    topK = 10
    evaluation_threads = 1 #mp.cpu_count()

    # prepare dataset and iterators
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    max_user, max_movies = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(time()-t1,  max_user, max_movies, train.nnz, len(testRatings)))
    train_iter = get_train_iters(train, num_negatives, batch_size) 
    val_iter = get_movielens_iter(args.path + args.dataset + '.test.rating', batch_size)

   # load parameters and symbol
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, load_epoch) 

     # initialize the module
    mod = mx.module.Module(symbol=sym, context=ctx, data_names=['user', 'item'], label_names=['score'])
    mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)  
    
    # get the sparse weight parameter
    mod.set_params(arg_params=arg_params, aux_params=aux_params)
   
    # profile
    profiler.set_config(profile_all=True,aggregate_stats=True, filename='profile_neumf.json')
    profiler.set_state('run')
    
    if benchmark:
        logging.info('Evaluating...')
        (hits, ndcgs) = evaluate_model(mod, testRatings, testNegatives, topK, evaluation_threads)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        print('HR = %.4f, NDCG = %.4f'  % (hr, ndcg))  
        logging.info('Evaluating completed')
        profiler.set_state('stop')    
    else:
        logging.info('Inference started ...')
        nbatch=0
        tic = time()
        num_samples = 0
        for batch in train_iter:     
            mod.forward(batch, is_train=False)
            mx.nd.waitall()
            nbatch += 1
        toc = time()
        fps = (nbatch * batch_size)/(toc - tic)
        logging.info('Inference completed')
        logging.info('batch size %d, process %.4f samples/s' % (batch_size, fps))
        

