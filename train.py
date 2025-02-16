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
from data import get_movielens_iter
from MF import mf_model
from MLP import mlp_model
from GMF import gmf_model
from NeuMF import neumf_model
import os
from time import time
from data import get_train_iters
from Dataset import Dataset
from evaluate import evaluate_model
# from mxnet import profiler
logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Run neural collaborative filtering inference",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')                                 
parser.add_argument('--num-epoch', type=int, default=3,
                        help='number of epochs to train')
# parser.add_argument('-l','--learning-rate', type=float, default=0.001,
#                         help='learning rate for optimizer')
parser.add_argument('-b','--batch-size', type=int, default=256,
                        help='number of examples per batch')
parser.add_argument('-f','--factor-size', type=int, default=8,
                        help="the factor size of the embedding operation")
parser.add_argument('-l','--layers', nargs='?', default='[64,32,16,8]',
                        help="Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
parser.add_argument('--log-interval', type=int, default=100,
                        help='logging interval')
parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance')
parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
parser.add_argument('--gpus', type=str,
                        help="list of gpus to run, e.g. 0 or 0,2. empty means using cpu().")
parser.add_argument('--dense', action='store_true', help="whether to use dense embedding")

def batch_row_ids(data_batch):
    """ Generate row ids based on the current mini-batch """
    item = data_batch.data[0]
    user = data_batch.data[1]
    return {'user_weight': user.astype(np.int64),
            'item_weight': item.astype(np.int64)}

def cross_entropy(label, pred):
    ce = 0
    for l, p in zip(label, pred):
        ce += -( l*np.log(p) + (1-l)*np.log(1-p) )
    return ce

if __name__ == '__main__':
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.INFO, format=head)

    # arg parser
    args = parser.parse_args()
    logging.info(args)
    
    num_epoch = args.num_epoch
    num_negatives = args.num_neg
    # learning_rate=args.learning_rate
    batch_size = args.batch_size
    factor_size = args.factor_size
    model_layers= eval(args.layers)
    learner = args.learner
    log_interval = args.log_interval

    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')] if args.gpus else [mx.cpu()]
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    topK = 10
    evaluation_threads = 1 #mp.cpu_count()#mp.cpu_count()

    # prepare dataset and iterators
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    max_user, max_movies = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(time()-t1,  max_user, max_movies, train.nnz, len(testRatings)))
    train_iter = get_train_iters(train, num_negatives, batch_size) 

    # construct the model
    # net = mf_model(factor_size, factor_size, max_user, max_movies, dense=args.dense)
    # net = mlp_model(model_layers, max_user, max_movies, dense=args.dense) # MLP
    # net = gmf_model(factor_size, max_user, max_movies, dense=args.dense) # GMF
    net = neumf_model(factor_size, model_layers, max_user, max_movies, dense=args.dense) #NeuMF

    # initialize the module
    mod = mx.module.Module(net, context=ctx, data_names=['user', 'item'], label_names=['score'])
    mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)  
    mod.init_params(initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))
    mod.init_optimizer(optimizer=learner, kvstore='device')
    # use cross entropy as the metric
    metric = mx.metric.create(cross_entropy)
    speedometer = mx.callback.Speedometer(batch_size, log_interval)
    # profile
    # profiler.set_config(profile_all=True,aggregate_stats=True,filename='profile_neumf.json')
    # profiler.set_state('run')

    best_hr, best_ndcg, best_iter = -1, -1, -1
    train = True
    if train:
        logging.info('Training started ...')
        for epoch in range(num_epoch): 
            profiler.pause()
            t1 = time()
            nbatch = 0
            metric.reset()
            # profiler.resume()
            for batch in train_iter:
                nbatch += 1
                mod.prepare(batch, sparse_row_id_fn=batch_row_ids)
                mod.forward(batch)
                pred = mod.get_outputs()[0]
                # update all parameters
                mod.backward()
                mod.update()
                # update training metric
                label = batch.label[0]
                metric.update(label, pred)
                speedometer_param = mx.model.BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                        eval_metric=metric, locals=locals())
                speedometer(speedometer_param) # print epoch nbatch metric
            # profiler.pause()
            # reset iterator
            train_iter.reset()
            # save model
            mod.save_checkpoint("checkpoint", epoch, save_optimizer_states=True)
            
            # profiler.resume()
            t2 = time()
            # compute hit ratio
            (hits, ndcgs) = evaluate_model(mod, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
            
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f [%.1f s]'  % (epoch,  t2-t1, hr, ndcg, time()-t2))
            # best hit ratio
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
        # profiler.set_state('stop')      
        print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
        logging.info('Training completed.')
            
    else:
        logging.info('Evaluating...')
        sym, arg_params, aux_params = mx.model.load_checkpoint('checkpoint', 0)
        mod.set_params(arg_params=arg_params, aux_params=aux_params)

        (hits, ndcgs) = evaluate_model(mod, testRatings, testNegatives, topK, evaluation_threads)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        logging.info('Evaluating completed')
       

