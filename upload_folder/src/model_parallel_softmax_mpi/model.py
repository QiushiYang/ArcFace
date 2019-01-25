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

# pylint: disable=fixme, invalid-name, too-many-arguments, too-many-locals, too-many-lines
# pylint: disable=too-many-branches, too-many-statements
"""MXNet model module"""
from __future__ import absolute_import, print_function

import os
import time
import logging
import warnings
import math
from collections import namedtuple
import numpy as np

import gc
import mxnet.io as io
import mxnet.ndarray as nd
import mxnet.symbol as sym
import mxnet.optimizer as opt
import mxnet.metric as metric
import mxnet.kvstore as kvs
import mxnet.autograd as autograd
from mxnet.context import Context, cpu
from mxnet.initializer import Uniform
from mxnet.optimizer import get_updater
from mxnet.executor_manager import DataParallelExecutorManager, _check_arguments, _load_data
from mxnet.io import DataDesc
from mxnet.base import mx_real_t

from . import machine_parallel_loss

import mpi4py.MPI as MPI

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

BASE_ESTIMATOR = object

try:
    from sklearn.base import BaseEstimator
    BASE_ESTIMATOR = BaseEstimator
except ImportError:
    SKLEARN_INSTALLED = False

# Parameter to pass to batch_end_callback
BatchEndParam = namedtuple('BatchEndParams',
                           ['epoch',
                            'nbatch',
                            'eval_metric',
                            'locals'])

def _create_kvstore(kvstore, num_device, arg_params, hybrid_num_batches_per_block=0):
    """Create kvstore
    This function select and create a proper kvstore if given the kvstore type.

    Parameters
    ----------
    kvstore : KVStore or str
        The kvstore.
    num_device : int
        The number of devices
    arg_params : dict of str to `NDArray`.
        Model parameter, dict of name to `NDArray` of net's weights.
    """
    update_on_kvstore = True
    if kvstore is None:
        kv = None
    elif isinstance(kvstore, kvs.KVStore):
        kv = kvstore
    elif isinstance(kvstore, str):
        # create kvstore using the string type
        if num_device is 1 and 'dist' not in kvstore:
            # no need to use kv for single device and single machine
            kv = None
        else:
            kv = kvs.create(kvstore, hybrid_num_batches_per_block)
            if kvstore is 'local':
            # automatically select a proper local
                max_size = max(np.prod(param.shape) for param in
                               arg_params.values())
                if max_size > 1024 * 1024 * 16 and not isinstance(kv, kvs.HybridKVStore):
                    update_on_kvstore = False
    else:
        raise TypeError('kvstore must be KVStore, str or None')

    if kv is None:
        update_on_kvstore = False

    return (kv, update_on_kvstore)

def _initialize_kvstore(kvstore, param_arrays, arg_params, param_names, update_on_kvstore):
    """Initialize kvstore"""
    for idx, param_on_devs in enumerate(param_arrays):
        name = param_names[idx]
        kvstore.init(name, arg_params[name])

        if update_on_kvstore:
            kvstore.pull(name, param_on_devs, priority=-idx)

def _initialize_softmax_kvstore(kvstore, split_param, params, data_slices, param_names, update_on_kvstore):
    """Initialize softmax kvstore"""
    name = 'weight'
    kvstore.init(name, params[name][data_slices[comm_rank],:])
    if update_on_kvstore:
        kvstore.pull(name, split_param[name], priority=0)
    name = 'bias'
    kvstore.init(name, params[name][data_slices[comm_rank]])
    if update_on_kvstore:
        kvstore.pull(name, split_param[name], priority=-1)

def _update_params_on_kvstore_nccl(param_arrays, grad_arrays, kvstore, param_names):
    """Perform update of param_arrays from grad_arrays on NCCL kvstore."""
    valid_indices = [index for index, grad_list in
                     enumerate(grad_arrays) if grad_list[0] is not None]
    valid_grad_arrays = [grad_arrays[i] for i in valid_indices]
    valid_param_arrays = [param_arrays[i] for i in valid_indices]
    valid_param_names = [param_names[i] for i in valid_indices]
    size = len(valid_grad_arrays)
    start = 0
    # Use aggregation by default only with NCCL
    default_batch = 16
    batch = int(os.getenv('MXNET_UPDATE_AGGREGATION_SIZE', default_batch))
    while start < size:
        end = start + batch if start + batch < size else size
        # push gradient, priority is negative index
        kvstore.push(valid_param_names[start:end], valid_grad_arrays[start:end], priority=-start)
        # pull back the weights
        kvstore.pull(valid_param_names[start:end], valid_param_arrays[start:end], priority=-start)
        start = end

def _update_params_on_kvstore(param_arrays, grad_arrays, kvstore, param_names):
    """Perform update of param_arrays from grad_arrays on kvstore."""
    for index, pair in enumerate(zip(param_arrays, grad_arrays)):
        arg_list, grad_list = pair
        if grad_list[0] is None:
            continue
        name = param_names[index]
        # push gradient, priority is negative index
        kvstore.push(name, grad_list, priority=-index)
        # pull back the weights
        kvstore.pull(name, arg_list, priority=-index)

def _update_params(param_arrays, grad_arrays, updater, num_device,
                   kvstore=None, param_names=None):
    """Perform update of param_arrays from grad_arrays not on kvstore."""
    for i, pair in enumerate(zip(param_arrays, grad_arrays)):
        arg_list, grad_list = pair
        if grad_list[0] is None:
            continue
        index = i
        if kvstore:
            name = param_names[index]
            # push gradient, priority is negative index
            kvstore.push(name, grad_list, priority=-index)
            # pull back the sum gradients, to the same locations.
            kvstore.pull(name, grad_list, priority=-index)
        for k, p in enumerate(zip(arg_list, grad_list)):
            # faked an index here, to make optimizer create diff
            # state for the same index but on diff devs, TODO(mli)
            # use a better solution later
            w, g = p
            updater(index*num_device+k, g, w)

def _clear_param_grads(grad_arrays):
    for param_grad in grad_arrays:
        for grad in param_grad:
            grad[:] = 0.0

def _multiple_callbacks(callbacks, *args, **kwargs):
    """Sends args and kwargs to any configured callbacks.
    This handles the cases where the 'callbacks' variable
    is ``None``, a single function, or a list.
    """
    if isinstance(callbacks, list):
        for cb in callbacks:
            cb(*args, **kwargs)
        return
    if callbacks:
        callbacks(*args, **kwargs)

def mpi_allgather(input_nd, output_nd):
    assert input_nd.size*comm_size == output_nd.size
    sendbuf = input_nd.asnumpy()
    recvbuf = np.empty([comm_size]+list(sendbuf.shape), dtype=sendbuf.dtype)
    comm.Allgather(sendbuf, recvbuf)
    output_nd[:] = recvbuf.reshape(output_nd.shape)

def mpi_reduce_allgather(input_nd, output_nd):
    assert input_nd.size == output_nd.size
    sendbuf = input_nd.asnumpy()
    tempbuf = np.empty(sendbuf.shape, dtype=sendbuf.dtype) if comm_rank==0 else None
    comm.Reduce(sendbuf, tempbuf, root=0, op=MPI.SUM)
    recvbuf = comm.bcast(tempbuf if comm_rank==0 else None, root=0)
    output_nd[:] = recvbuf[:]

def mpi_gather_bcast(input_max_nd, input_sum_nd, output_max_nd, output_sum_nd):
    assert input_max_nd.size == output_max_nd.size
    assert input_sum_nd.size == output_sum_nd.size
    sendmaxbuf = input_max_nd.asnumpy()
    sendsumbuf = input_sum_nd.asnumpy()
    recvmaxbuf = np.empty([comm_size]+list(sendmaxbuf.shape), dtype=sendmaxbuf.dtype)
    recvsumbuf = np.empty([comm_size]+list(sendsumbuf.shape), dtype=sendsumbuf.dtype)
    comm.Allgather(sendmaxbuf, recvmaxbuf)
    comm.Allgather(sendsumbuf, recvsumbuf)
    totalmax = np.max(recvmaxbuf, axis=0)
    totalsum = np.sum(recvsumbuf*(np.exp(recvmaxbuf-totalmax)), axis=0)

    totalmax = comm.bcast(totalmax if comm_rank == 0 else None, root=0)
    totalsum = comm.bcast(totalsum if comm_rank == 0 else None, root=0)

    output_max_nd[:] = totalmax
    output_sum_nd[:] = totalsum

def _eval_data(eval_datas, eval_metric, part_size, executor_manager, eval_batch_end_callback, epoch, logger):
    for k, _eval_data in enumerate(eval_datas):
        eval_metric.reset()
        _eval_data.reset()
        for i, eval_batch in enumerate(_eval_data):
            assert _eval_data.batch_size == part_size
            out_shape = tuple([_eval_data.batch_size] + list(executor_manager.output_shapes[0][1:]))
            cpu_out_feature = nd.zeros(out_shape)
            executor_manager.load_data_batch(eval_batch)
            executor_manager.forward(is_train=False)
            executor_manager.output_copyto(cpu_out_feature)

            if eval_batch.pad > 0:
                real_batch_size = _eval_data.batch_size - eval_batch.pad
                eval_metric.update([a[:real_batch_size] for a in eval_batch.label],
                                   [a[:real_batch_size] for a in [cpu_out_feature]])
            else:
                eval_metric.update(eval_batch.label, [cpu_out_feature])

            if eval_batch_end_callback is not None:
                batch_end_params = BatchEndParam(epoch=epoch,
                                                 nbatch=i,
                                                 eval_metric=eval_metric,
                                                 locals=locals())
                if isinstance(eval_batch_end_callback, list):
                    for call in eval_batch_end_callback:
                        call(batch_end_params)
                else:
                    eval_batch_end_callback(batch_end_params)
        name_value = eval_metric.get_name_value()
        for name, value in name_value:
            logger.info('Epoch[%d] eval data-%d Validation-%s=%f', epoch, k, name, value)

def _train_multi_device(symbol, ctx, arg_names, param_names, aux_names,
                        arg_params, aux_params,
                        begin_epoch, end_epoch, epoch_size,
                        optimizer, fc_optimizer,
                        embedding_size, num_classes, no_bias,
                        margin, easy_margin, margin_s, margin_m,
                        fullyconnect_params,
                        kvstore, update_on_kvstore,
                        train_data, eval_data=None,
                        train_metric=None, eval_metric=None,
                        split_num=None,
                        epoch_end_callback=None, batch_end_callback=None,
                        logger=None, work_load_list=None, monitor=None,
                        eval_end_callback=None,
                        eval_batch_end_callback=None, sym_gen=None):
    """Internal training function on multiple devices.
    This function will also work for single device as well.

    Parameters
    ----------
    symbol : Symbol
        The network configuration.
    ctx : list of Context
        The training devices.
    arg_names: list of str
        Name of all arguments of the network.
    param_names: list of str
        Name of all trainable parameters of the network.
    aux_names: list of str
        Name of all auxiliary states of the network.
    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    aux_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.
    begin_epoch : int
        The begining training epoch.
    end_epoch : int
        The end training epoch.
    epoch_size : int, optional
        Number of batches in a epoch. In default, it is set to
        ``ceil(num_train_examples / batch_size)``.
    optimizer : Optimizer
        The optimization algorithm
    train_data : DataIter
        Training data iterator.
    eval_data : DataIter
        Validation data iterator.
    eval_metric : EvalMetric
        An evaluation function or a list of evaluation functions.
    epoch_end_callback : callable(epoch, symbol, arg_params, aux_states)
        A callback that is invoked at end of each epoch.
        This can be used to checkpoint model each epoch.
    batch_end_callback : callable(BatchEndParams)
        A callback that is invoked at end of each batch.
        This can be used to measure speed, get result from evaluation metric. etc.
    kvstore : KVStore
        The KVStore.
    update_on_kvstore : bool
        Whether or not perform weight updating on kvstore.
    logger : logging logger
        When not specified, default logger will be used.
    work_load_list : list of float or int, optional
        The list of work load for different devices,
        in the same order as ``ctx``.
    monitor : Monitor, optional
        Monitor installed to executor,
        for monitoring outputs, weights, and gradients for debugging.
    Notes
    -----
    - This function will inplace update the NDArrays in `arg_params` and `aux_states`.
    """
    if logger is None:
        logger = logging

    # split data_batch to small batches
    split_parts = 1 if split_num is None else split_num
    split_size = int(train_data.batch_size / split_parts)
    assert split_size*split_parts == train_data.batch_size
    split_slices = [slice(i*split_size, (i+1)*split_size) for i in range(split_parts)]

    executor_manager = DataParallelExecutorManager(symbol=symbol,
                                                   sym_gen=sym_gen,
                                                   ctx=ctx,
                                                   train_data=train_data,
                                                   split_num_small_batches=split_num,
                                                   param_names=param_names,
                                                   arg_names=arg_names,
                                                   aux_names=aux_names,
                                                   work_load_list=work_load_list,
                                                   logger=logger)
    feature_shape = executor_manager.output_shapes[0][1:]
    # num_workers = kvstore.num_workers if update_on_kvstore else 1
    num_workers = comm_size
    device_part_size = int(num_classes / num_workers)
    device_slices = [slice(i*device_part_size, (i+1)*device_part_size) for i in range(num_workers)]
    assert device_part_size*num_workers is num_classes # for allgather fc params
    device_part_shape = []
    for i in range(num_workers):
        device_part_shape.append(device_slices[i].stop-device_slices[i].start)

    # if not update_on_kvstore:
    updater = get_updater(optimizer)
    fc_updater = get_updater(fc_optimizer)

    softmax_executor = machine_parallel_loss.SoftmaxModelParallelOperator(train_data.batch_size*num_workers/split_parts,
                                                                          feature_shape,
                                                                          ctx, comm, device_part_size,
                                                                          fc_updater, no_bias, margin, easy_margin,
                                                                          margin_s, margin_m)

    if monitor:
        executor_manager.install_monitor(monitor)

    executor_manager.set_params(arg_params, aux_params)

    if kvstore:
        _initialize_kvstore(kvstore=kvstore,
                            param_arrays=executor_manager.param_arrays,
                            arg_params=arg_params,
                            param_names=executor_manager.param_names,
                            update_on_kvstore=update_on_kvstore)

        split_param={}
        split_param['weight'] = nd.zeros((device_part_shape[comm_rank], embedding_size))
        split_param['bias'] = nd.zeros((device_part_shape[comm_rank],))
        _initialize_softmax_kvstore(kvstore=kvstore,
                                    split_param=split_param,
                                    params=fullyconnect_params,
                                    data_slices=device_slices,
                                    param_names=split_param.keys(),
                                    update_on_kvstore=update_on_kvstore)

        softmax_executor.set_params(split_param)
    else:
        softmax_executor.set_params(fullyconnect_params)

    if update_on_kvstore:
        kvstore.set_optimizer(optimizer)

    # Now start training
    train_data.reset()
    batch_size = train_data.batch_size
    slice_batch_size = batch_size/split_parts
    for epoch in range(begin_epoch, end_epoch):
        # Training phase
        tic = time.time()
        train_metric.reset()
        nbatch = 0
        # Iterate over training data.
        while True:
            do_reset = True
            for data_batch in train_data:
                if monitor is not None:
                    monitor.tic()

                # for predict
                predict_index = nd.empty((batch_size*num_workers,))
                predict_value = nd.empty((batch_size*num_workers,))
                batch_loss = nd.empty((batch_size*num_workers,))
                eval_total_label = nd.empty((batch_size*num_workers,))

                if len(split_slices)>1:
                    _clear_param_grads(executor_manager.grad_arrays)

                for islice in split_slices:
                    # forward to get feature
                    executor_manager.load_data_batch(data_batch, islice)
                    cpu_out_feature = nd.empty((slice_batch_size,)+feature_shape)
                    total_feature = nd.empty((slice_batch_size*num_workers,)+feature_shape)
                    total_label = nd.empty((slice_batch_size*num_workers,))
                    executor_manager.forward(is_train=True)
                    executor_manager.output_copyto(cpu_out_feature)

                    if update_on_kvstore:
                        mpi_allgather(cpu_out_feature, total_feature)
                        mpi_allgather(data_batch.label[0][islice], total_label)
                        softmax_executor.load_data(total_feature, total_label)
                    else:
                        softmax_executor.load_data(cpu_out_feature, data_batch.label[0][islice])

                    # fully connected and softmax
                    max_exp = nd.empty((slice_batch_size*num_workers,))
                    sum_exp = nd.empty((slice_batch_size*num_workers,))
                    softmax_executor.forward(max_exp, sum_exp)
                    if update_on_kvstore:
                        mpi_gather_bcast(max_exp, sum_exp, max_exp, sum_exp)

                    softmax_executor.backward(max_exp, sum_exp)

                    out_grad = nd.empty(cpu_out_feature.shape)
                    if update_on_kvstore:
                        out_grad_kv = nd.empty((slice_batch_size*num_workers,)+feature_shape)
                        softmax_executor.output_copyto(out_grad_kv)
                        mpi_reduce_allgather(out_grad_kv, out_grad_kv)
                        out_grad[:] = out_grad_kv[slice_batch_size*comm_rank: slice_batch_size*(comm_rank+1)]
                    else:
                        softmax_executor.output_copyto(out_grad)

                    # backward
                    executor_manager.backward(out_grad)

                    # predict
                    worker_slice = slice(islice.start*num_workers, islice.stop*num_workers)
                    softmax_executor.output_predict(predict_value, predict_index, worker_slice)
                    softmax_executor.output_loss(batch_loss, worker_slice)
                    if update_on_kvstore:
                        eval_total_label[worker_slice] = total_label[:]

                # batch feature callback
                #train_data.provide_loss_queue.put_nowait(batch_loss)
                #train_data.provide_index_queue.put_nowait(data_batch.index[0])

                # update
                softmax_executor.update_param()

                if update_on_kvstore:
                    if 'nccl' in kvstore.type:
                        _update_params_on_kvstore_nccl(executor_manager.param_arrays,
                                                       executor_manager.grad_arrays,
                                                       kvstore, executor_manager.param_names)
                    else:
                        _update_params_on_kvstore(executor_manager.param_arrays,
                                                  executor_manager.grad_arrays,
                                                  kvstore, executor_manager.param_names)
                else:
                    _update_params(executor_manager.param_arrays,
                                   executor_manager.grad_arrays,
                                   updater=updater,
                                   num_device=len(ctx),
                                   kvstore=kvstore,
                                   param_names=executor_manager.param_names)

                if monitor is not None:
                    monitor.toc_print()

                # evaluate at end, so we can lazy copy
                if update_on_kvstore:
                    predict_value_kv = nd.empty((num_workers, batch_size*num_workers))
                    predict_index_kv = nd.empty((num_workers, batch_size*num_workers))
                    mpi_allgather(predict_value, predict_value_kv)
                    mpi_allgather(predict_index, predict_index_kv)
                    
                    predict_max_index = nd.argmax(predict_value_kv, axis=0)
                    predict_max_value = nd.max(predict_value_kv, axis=0)

                    device_begin = nd.array([device_slices[i].start for i in range(len(device_slices))])
                    total_predict = device_begin[predict_max_index] + predict_index_kv[predict_max_index, nd.arange(batch_size*num_workers)]
                    train_metric.update(total_predict, eval_total_label)
                else:
                    train_metric.update(predict_index, data_batch.label[0])

                nbatch += 1
                loss = nd.sum(batch_loss).asnumpy()[0] / batch_size / num_workers
                if nbatch % 50 == 0:
                    logger.info('Epoch[%d] Batch[%d] Train-clsloss:%.6f', epoch, nbatch, loss)

                # batch callback (for print purpose)
                if batch_end_callback is not None:
                    batch_end_params = BatchEndParam(epoch=epoch,
                                                     nbatch=nbatch,
                                                     eval_metric=train_metric,
                                                     locals=locals())
                    _multiple_callbacks(batch_end_callback, batch_end_params)

                # this epoch is done possibly earlier
                if epoch_size is not None and nbatch >= epoch_size:
                    do_reset = False
                    break

            if do_reset:
                logger.info('Epoch[%d] Resetting Data Iterator', epoch)
                train_data.reset()

            # this epoch is done
            if epoch_size is None or nbatch >= epoch_size:
                break

        toc = time.time()
        logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc - tic))

        if epoch_end_callback or epoch + 1 == end_epoch:
            executor_manager.copy_to(arg_params, aux_params)

            weight_tmp = nd.empty(fullyconnect_params['weight'][device_slices[comm_rank]].shape)
            bias_tmp = nd.empty(fullyconnect_params['bias'][device_slices[comm_rank]].shape)
            softmax_executor.dumps(weight_tmp, bias_tmp)
            if update_on_kvstore:
                fullyconnect_weight_tmp = nd.empty(fullyconnect_params['weight'].shape)
                fullyconnect_bias_tmp = nd.empty(fullyconnect_params['bias'].shape)
                # mpi rank -> kvstore rank
                cur_rank = nd.array([comm_rank, kvstore.rank])
                all_rank = nd.empty((comm_size, 2))

                mpi_allgather(weight_tmp, fullyconnect_weight_tmp)
                mpi_allgather(bias_tmp, fullyconnect_bias_tmp)
                mpi_allgather(cur_rank, all_rank)

                for comm_sz in range(comm_size):
                    rank_index = all_rank[comm_sz]
                    kvstore_slices = device_slices[rank_index[1]]
                    mpi_slices = device_slices[rank_index[0]]

                    fullyconnect_params['weight'][kvstore_slices] = fullyconnect_weight_tmp[mpi_slices]
                    fullyconnect_params['bias'][kvstore_slices] = fullyconnect_bias_tmp[mpi_slices]
            else:
                fullyconnect_params['weight'][:] = weight_tmp[:]
                fullyconnect_params['bias'][:] = bias_tmp[:]

        _multiple_callbacks(epoch_end_callback, epoch, symbol, arg_params, aux_params, fullyconnect_params)

        # evaluation
        if eval_data:
            _eval_data(eval_data, eval_metric, eval_data[0].batch_size, executor_manager, eval_batch_end_callback, epoch, logger)

    # end of all epochs
    return


def save_checkpoint(prefix, epoch, symbol, arg_params, aux_params, fullyconnect_params):
    """Checkpoint the model data into file.

    Parameters
    ----------
    prefix : str
        Prefix of model name.
    epoch : int
        The epoch number of the model.
    symbol : Symbol
        The input Symbol.
    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    aux_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.
    Notes
    -----
    - ``prefix-symbol.json`` will be saved for symbol.
    - ``prefix-epoch.params`` will be saved for parameters.
    """
    if symbol is not None:
        symbol.save('%s-symbol.json' % prefix)

    save_dict = {('arg:%s' % k) : v.as_in_context(cpu()) for k, v in arg_params.items()}
    save_dict.update({('aux:%s' % k) : v.as_in_context(cpu()) for k, v in aux_params.items()})
    param_name = '%s-%04d.params' % (prefix, epoch)
    nd.save(param_name, save_dict)

    fullyconnect_param_name = '%s-%04d.fullyconnect_params' % (prefix, epoch)
    nd.save(fullyconnect_param_name, fullyconnect_params)
    logging.info('Saved checkpoint to \"%s\"', param_name)
    # for name, array in arg_params.items():
    #     print name, array.shape


def load_checkpoint(prefix, epoch):
    """Load model checkpoint from file.

    Parameters
    ----------
    prefix : str
        Prefix of model name.
    epoch : int
        Epoch number of model we would like to load.

    Returns
    -------
    symbol : Symbol
        The symbol configuration of computation network.
    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    aux_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.

    Notes
    -----
    - Symbol will be loaded from ``prefix-symbol.json``.
    - Parameters will be loaded from ``prefix-epoch.params``.
    """
    symbol = sym.load('%s-symbol.json' % prefix)
    save_dict = nd.load('%s-%04d.params' % (prefix, epoch))
    fullyconnect_dict = nd.load('%s-%04d.fullyconnect_params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return (symbol, arg_params, aux_params, fullyconnect_dict)

from mxnet.callback import LogValidationMetricsCallback # pylint: disable=wrong-import-position

class FeedForward(BASE_ESTIMATOR):
    """Model class of MXNet for training and predicting feedforward nets.
    This class is designed for a single-data single output supervised network.

    Parameters
    ----------
    symbol : Symbol
        The symbol configuration of computation network.
    ctx : Context or list of Context, optional
        The device context of training and prediction.
        To use multi GPU training, pass in a list of gpu contexts.
    num_epoch : int, optional
        Training parameter, number of training epochs(epochs).
    epoch_size : int, optional
        Number of batches in a epoch. In default, it is set to
        ``ceil(num_train_examples / batch_size)``.
    optimizer : str or Optimizer, optional
        Training parameter, name or optimizer object for training.
    initializer : initializer function, optional
        Training parameter, the initialization scheme used.
    numpy_batch_size : int, optional
        The batch size of training data.
        Only needed when input array is numpy.
    arg_params : dict of str to NDArray, optional
        Model parameter, dict of name to NDArray of net's weights.
    aux_params : dict of str to NDArray, optional
        Model parameter, dict of name to NDArray of net's auxiliary states.
    allow_extra_params : boolean, optional
        Whether allow extra parameters that are not needed by symbol
        to be passed by aux_params and ``arg_params``.
        If this is True, no error will be thrown when ``aux_params`` and ``arg_params``
        contain more parameters than needed.
    begin_epoch : int, optional
        The begining training epoch.
    kwargs : dict
        The additional keyword arguments passed to optimizer.
    """
    def __init__(self, symbol, ctx=None,
                 num_epoch=None, epoch_size=None, optimizer='sgd',
                 initializer=Uniform(0.01),
                 fc_initializer=None,
                 fc_optimizer=None,
                 fc_opt_param={},
                 numpy_batch_size=128,
                 arg_params=None, aux_params=None,
                 fullyconnect_params=None,
                 allow_extra_params=False,
                 begin_epoch=0,
                 **kwargs):
        warnings.warn(
            '\033[91mmxnet.model.FeedForward has been deprecated. ' + \
            'Please use mxnet.mod.Module instead.\033[0m',
            DeprecationWarning, stacklevel=2)

        if isinstance(symbol, sym.Symbol):
            self.symbol = symbol
            self.sym_gen = None
        else:
            assert(callable(symbol))
            self.symbol = None
            self.sym_gen = symbol

        # model parameters
        self.arg_params = arg_params
        self.aux_params = aux_params
        self.fullyconnect_params = fullyconnect_params
        self.allow_extra_params = allow_extra_params

        self.argument_checked = False
        if self.sym_gen is None:
            self._check_arguments()

        # basic configuration
        if ctx is None:
            ctx = [cpu()]
        elif isinstance(ctx, Context):
            ctx = [ctx]
        self.ctx = ctx
        # training parameters
        self.num_epoch = num_epoch
        self.epoch_size = epoch_size
        self.kwargs = kwargs.copy()
        self.optimizer = optimizer
        self.initializer = initializer
        self.fc_init = fc_initializer
        self.fc_optimizer = fc_optimizer
        self.fc_opt_param = fc_opt_param
        self.numpy_batch_size = numpy_batch_size
        # internal helper state
        self._pred_exec = None
        self.begin_epoch = begin_epoch

    def _check_arguments(self):
        """verify the argument of the default symbol and user provided parameters"""
        if self.argument_checked:
            return

        assert(self.symbol is not None)
        self.argument_checked = True

        # check if symbol contain duplicated names.
        _check_arguments(self.symbol)
        # rematch parameters to delete useless ones
        if self.allow_extra_params:
            if self.arg_params:
                arg_names = set(self.symbol.list_arguments())
                self.arg_params = {k : v for k, v in self.arg_params.items()
                                   if k in arg_names}
            if self.aux_params:
                aux_names = set(self.symbol.list_auxiliary_states())
                self.aux_params = {k : v for k, v in self.aux_params.items()
                                   if k in aux_names}
            if self.fullyconnect_params:
                assert 'weight' in self.fullyconnect_params
                assert 'bias' in self.fullyconnect_params

    @staticmethod
    def _is_data_arg(name):
        """Check if name is a data argument."""
        return name.endswith('data') or name.endswith('label')

    def _init_params(self, inputs, num_classes, embedding_size, overwrite=False):
        """Initialize weight parameters and auxiliary states."""
        inputs = [x if isinstance(x, DataDesc) else DataDesc(*x) for x in inputs]
        input_shapes = {item.name: item.shape for item in inputs}
        arg_shapes, _, aux_shapes = self.symbol.infer_shape(**input_shapes)
        assert arg_shapes is not None
        input_dtypes = {item.name: item.dtype for item in inputs}
        arg_dtypes, _, aux_dtypes = self.symbol.infer_type(**input_dtypes)
        assert arg_dtypes is not None

        arg_names = self.symbol.list_arguments()
        input_names = input_shapes.keys()
        param_names = [key for key in arg_names if key not in input_names]
        aux_names = self.symbol.list_auxiliary_states()

        param_name_attrs = [x for x in zip(arg_names, arg_shapes, arg_dtypes)
                            if x[0] in param_names]
        arg_params = {k : nd.zeros(shape=s, dtype=t)
                      for k, s, t in param_name_attrs}
        aux_name_attrs = [x for x in zip(aux_names, aux_shapes, aux_dtypes)
                          if x[0] in aux_names]
        aux_params = {k : nd.zeros(shape=s, dtype=t)
                      for k, s, t in aux_name_attrs}

        for k, v in arg_params.items():
            if self.arg_params and k in self.arg_params and (not overwrite):
                arg_params[k][:] = self.arg_params[k][:]
            else:
                self.initializer(k, v)

        for k, v in aux_params.items():
            if self.aux_params and k in self.aux_params and (not overwrite):
                aux_params[k][:] = self.aux_params[k][:]
            else:
                self.initializer(k, v)

        # fully connect param
        fullyconnect_params = {}
        fullyconnect_params['weight'] = nd.zeros((num_classes, embedding_size))
        fullyconnect_params['bias'] = nd.zeros((num_classes))

        for k, v in fullyconnect_params.items():
            if self.fullyconnect_params:
                fullyconnect_params['weight'] = self.fullyconnect_params['weight'][:]
                fullyconnect_params['bias'] = self.fullyconnect_params['bias'][:]
            else:
                if self.fc_init:
                    self.fc_init('weight', fullyconnect_params['weight'])
                    self.fc_init('bias', fullyconnect_params['bias'])
                else:
                    self.initializer('weight', fullyconnect_params['weight'])
                    self.initializer('bias', fullyconnect_params['bias'])

        self.arg_params = arg_params
        self.aux_params = aux_params
        self.fullyconnect_params = fullyconnect_params
        return (arg_names, list(param_names), aux_names)

    def __getstate__(self):
        this = self.__dict__.copy()
        this['_pred_exec'] = None
        return this

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _init_predictor(self, input_shapes, type_dict=None):
        """Initialize the predictor module for running prediction."""
        if self._pred_exec is not None:
            arg_shapes, _, _ = self.symbol.infer_shape(**dict(input_shapes))
            assert arg_shapes is not None, "Incomplete input shapes"
            pred_shapes = [x.shape for x in self._pred_exec.arg_arrays]
            if arg_shapes == pred_shapes:
                return
        # for now only use the first device
        pred_exec = self.symbol.simple_bind(
            self.ctx[0], grad_req='null', type_dict=type_dict, **dict(input_shapes))
        pred_exec.copy_params_from(self.arg_params, self.aux_params)

        _check_arguments(self.symbol)
        self._pred_exec = pred_exec

    def _init_iter(self, X, y, is_train):
        """Initialize the iterator given input."""
        if isinstance(X, (np.ndarray, nd.NDArray)):
            if y is None:
                if is_train:
                    raise ValueError('y must be specified when X is numpy.ndarray')
                else:
                    y = np.zeros(X.shape[0])
            if not isinstance(y, (np.ndarray, nd.NDArray)):
                raise TypeError('y must be ndarray when X is numpy.ndarray')
            if X.shape[0] != y.shape[0]:
                raise ValueError("The numbers of data points and labels not equal")
            if y.ndim == 2 and y.shape[1] == 1:
                y = y.flatten()
            if y.ndim != 1:
                raise ValueError("Label must be 1D or 2D (with 2nd dimension being 1)")
            if is_train:
                return io.NDArrayIter(X, y, min(X.shape[0], self.numpy_batch_size),
                                      shuffle=is_train, last_batch_handle='roll_over')
            else:
                return io.NDArrayIter(X, y, min(X.shape[0], self.numpy_batch_size), shuffle=False)
        if not isinstance(X, io.DataIter):
            raise TypeError('X must be DataIter, NDArray or numpy.ndarray')
        return X

    def _init_eval_iter(self, eval_data):
        """Initialize the iterator given eval_data."""
        if eval_data is None:
            return eval_data
        if isinstance(eval_data, (tuple, list)) and len(eval_data) == 2:
            if eval_data[0] is not None:
                if eval_data[1] is None and isinstance(eval_data[0], io.DataIter):
                    return eval_data[0]
                input_data = (np.array(eval_data[0]) if isinstance(eval_data[0], list)
                              else eval_data[0])
                input_label = (np.array(eval_data[1]) if isinstance(eval_data[1], list)
                               else eval_data[1])
                return self._init_iter(input_data, input_label, is_train=True)
            else:
                raise ValueError("Eval data is NONE")
        if not isinstance(eval_data, io.DataIter):
            raise TypeError('Eval data must be DataIter, or ' \
                            'NDArray/numpy.ndarray/list pair (i.e. tuple/list of length 2)')
        return eval_data

    def predict(self, X, num_batch=None, return_data=False, reset=True):
        """Run the prediction, always only use one device.

        Parameters
        ----------
        X : mxnet.DataIter
        num_batch : int or None
            The number of batch to run. Go though all batches if ``None``.
        Returns
        -------
        y : numpy.ndarray or a list of numpy.ndarray if the network has multiple outputs.
            The predicted value of the output.
        """
        X = self._init_iter(X, None, is_train=False)

        if reset:
            X.reset()
        data_shapes = X.provide_data
        data_names = [x[0] for x in data_shapes]
        type_dict = dict((key, value.dtype) for (key, value) in self.arg_params.items())
        for x in X.provide_data:
            if isinstance(x, DataDesc):
                type_dict[x.name] = x.dtype
            else:
                type_dict[x[0]] = mx_real_t

        self._init_predictor(data_shapes, type_dict)
        batch_size = X.batch_size
        data_arrays = [self._pred_exec.arg_dict[name] for name in data_names]
        output_list = [[] for _ in range(len(self._pred_exec.outputs))]
        if return_data:
            data_list = [[] for _ in X.provide_data]
            label_list = [[] for _ in X.provide_label]

        i = 0
        for batch in X:

            _load_data(batch, data_arrays)
            self._pred_exec.forward(is_train=False)
            padded = batch.pad
            real_size = batch_size - padded

            for o_list, o_nd in zip(output_list, self._pred_exec.outputs):
                o_list.append(o_nd[0:real_size].asnumpy())

            if return_data:
                for j, x in enumerate(batch.data):
                    data_list[j].append(x[0:real_size].asnumpy())
                for j, x in enumerate(batch.label):
                    label_list[j].append(x[0:real_size].asnumpy())
            i += 1
            if num_batch is not None and i == num_batch:
                break

        outputs = [np.concatenate(x) for x in output_list]
        if len(outputs) == 1:
            outputs = outputs[0]

        if return_data:
            data = [np.concatenate(x) for x in data_list]
            label = [np.concatenate(x) for x in label_list]
            if len(data) == 1:
                data = data[0]
            if len(label) == 1:
                label = label[0]
            return outputs, data, label
        else:
            return outputs

    def score(self, X, eval_metric='acc', num_batch=None, batch_end_callback=None, reset=True, display=False):
        """Run the model given an input and calculate the score
        as assessed by an evaluation metric.

        Parameters
        ----------
        X : mxnet.DataIter
        eval_metric : metric.metric
            The metric for calculating score.
        num_batch : int or None
            The number of batches to run. Go though all batches if ``None``.
        Returns
        -------
        s : float
            The final score.
        """
        # setup metric
        if not isinstance(eval_metric, metric.EvalMetric):
            eval_metric = metric.create(eval_metric)

        X = self._init_iter(X, None, is_train=False)
        if reset:
            X.reset()

        data_shapes = X.provide_data
        data_names = [x[0] for x in data_shapes]
        type_dict = dict((key, value.dtype) for (key, value) in self.arg_params.items())
        for x in X.provide_data:
            if isinstance(x, DataDesc):
                type_dict[x.name] = x.dtype
            else:
                type_dict[x[0]] = mx_real_t

        self._init_predictor(data_shapes, type_dict)
        data_arrays = [self._pred_exec.arg_dict[name] for name in data_names]

        for i, batch in enumerate(X):
            if num_batch is not None and i == num_batch:
                break
            _load_data(batch, data_arrays)
            self._pred_exec.forward(is_train=False)
            eval_metric.update(batch.label, self._pred_exec.outputs)

            if batch_end_callback is not None:
                batch_end_params = BatchEndParam(epoch=0,
                                                 nbatch=i,
                                                 eval_metric=eval_metric,
                                                 locals=locals())
                _multiple_callbacks(batch_end_callback, batch_end_params)
        if display:
            name_value = eval_metric.get_name_value()
            for name, value in name_value:
                logging.info('Validation-%s=%f', name, value)
        return eval_metric.get()[1]

    def fit(self, X, y=None, eval_data=None, train_metric='acc', eval_metric=None,
            split_num=None, embedding_size=500, num_classes=10000, no_bias=False,
            margin=False, easy_margin=True, margin_s=1, margin_m=0,
            epoch_end_callback=None, batch_end_callback=None, kvstore='local', logger=None,
            work_load_list=None, monitor=None, eval_end_callback=LogValidationMetricsCallback(),
            eval_batch_end_callback=None):
        """Fit the model.

        Parameters
        ----------
        X : DataIter, or numpy.ndarray/NDArray
            Training data. If `X` is a `DataIter`, the name or (if name not available)
            the position of its outputs should match the corresponding variable
            names defined in the symbolic graph.
        y : numpy.ndarray/NDArray, optional
            Training set label.
            If X is ``numpy.ndarray`` or `NDArray`, `y` is required to be set.
            While y can be 1D or 2D (with 2nd dimension as 1), its first dimension must be
            the same as `X`, i.e. the number of data points and labels should be equal.
        eval_data : DataIter or numpy.ndarray/list/NDArray pair
            If eval_data is numpy.ndarray/list/NDArray pair,
            it should be ``(valid_data, valid_label)``.
        eval_metric : metric.EvalMetric or str or callable
            The evaluation metric. This could be the name of evaluation metric
            or a custom evaluation function that returns statistics
            based on a minibatch.
        epoch_end_callback : callable(epoch, symbol, arg_params, aux_states)
            A callback that is invoked at end of each epoch.
            This can be used to checkpoint model each epoch.
        batch_end_callback: callable(epoch)
            A callback that is invoked at end of each batch for purposes of printing.
        kvstore: KVStore or str, optional
           The KVStore or a string kvstore type: 'local', 'dist_sync', 'dist_async'
           In default uses 'local', often no need to change for single machiine.
        logger : logging logger, optional
            When not specified, default logger will be used.
        work_load_list : float or int, optional
            The list of work load for different devices,
            in the same order as `ctx`.

        Note
        ----
        KVStore behavior
        - 'local', multi-devices on a single machine, will automatically choose best type.
        - 'dist_sync', multiple machines communicating via BSP.
        - 'dist_async', multiple machines with asynchronous communication.
        """

        data = self._init_iter(X, y, is_train=True)
        #eval_data = self._init_eval_iter(eval_data)

        if self.sym_gen:
            self.symbol = self.sym_gen(data.default_bucket_key) # pylint: disable=no-member
            self._check_arguments()
        self.kwargs["sym"] = self.symbol

        arg_names, param_names, aux_names = \
                self._init_params(data.provide_data+data.provide_label, num_classes, embedding_size)

        # setup metric
        if not isinstance(eval_metric, metric.EvalMetric):
            eval_metric = metric.create(eval_metric)
        if not isinstance(train_metric, metric.EvalMetric):
            train_metric = metric.create(train_metric)

        # create kvstore
        (kvstore, update_on_kvstore) = _create_kvstore(
            kvstore, len(self.ctx), self.arg_params)

        param_idx2name = {}
        if update_on_kvstore:
            param_idx2name.update(enumerate(param_names))
        else:
            for i, n in enumerate(param_names):
                for k in range(len(self.ctx)):
                    param_idx2name[i*len(self.ctx)+k] = n
        self.kwargs["param_idx2name"] = param_idx2name

        # init optmizer
        if isinstance(self.optimizer, str):
            batch_size = data.batch_size
            if kvstore and 'dist' in kvstore.type and not '_async' in kvstore.type:
                batch_size *= kvstore.num_workers
            optimizer = opt.create(self.optimizer,
                                   rescale_grad=(1.0/batch_size),
                                   **(self.kwargs))
        elif isinstance(self.optimizer, opt.Optimizer) or isinstance(self.optimizer, list):
            optimizer = self.optimizer

        if self.fc_optimizer:
            if isinstance(self.fc_optimizer, str):
                batch_size = data.batch_size
                if kvstore and 'dist' in kvstore.type and not '_async' in kvstore.type:
                    batch_size *= kvstore.num_workers
                fc_opt = opt.create(self.fc_optimizer,
                                    rescale_grad=(1.0/batch_size),
                                    **(self.fc_opt_param))
            elif isinstance(self.fc_optimizer, opt.Optimizer) or isinstance(self.fc_optimizer, list):
                fc_opt = self.fc_optimizer

        else:
            fc_opt = optimizer

        # do training
        _train_multi_device(self.symbol, self.ctx, arg_names, param_names, aux_names,
                            self.arg_params, self.aux_params,
                            begin_epoch=self.begin_epoch, end_epoch=self.num_epoch,
                            epoch_size=self.epoch_size,
                            optimizer=optimizer,
                            fc_optimizer=fc_opt,
                            embedding_size=embedding_size,
                            num_classes=num_classes,
                            no_bias=no_bias,
                            margin=margin, easy_margin=easy_margin,
                            margin_s=margin_s, margin_m=margin_m,
                            fullyconnect_params=self.fullyconnect_params,
                            train_data=data, eval_data=eval_data,
                            train_metric=train_metric,
                            eval_metric=eval_metric,
                            split_num=split_num,
                            epoch_end_callback=epoch_end_callback,
                            batch_end_callback=batch_end_callback,
                            kvstore=kvstore, update_on_kvstore=update_on_kvstore,
                            logger=logger, work_load_list=work_load_list, monitor=monitor,
                            eval_end_callback=eval_end_callback,
                            eval_batch_end_callback=eval_batch_end_callback,
                            sym_gen=self.sym_gen)


    def save(self, prefix, epoch=None):
        """Checkpoint the model checkpoint into file.
        You can also use `pickle` to do the job if you only work on Python.
        The advantage of `load` and `save` (as compared to `pickle`) is that
        the resulting file can be loaded from other MXNet language bindings.
        One can also directly `load`/`save` from/to cloud storage(S3, HDFS)

        Parameters
        ----------
        prefix : str
            Prefix of model name.

        Notes
        -----
        - ``prefix-symbol.json`` will be saved for symbol.
        - ``prefix-epoch.params`` will be saved for parameters.
        """
        if epoch is None:
            epoch = self.num_epoch
        assert epoch is not None
        save_checkpoint(prefix, epoch, self.symbol, self.arg_params, self.aux_params, self.fullyconnect_params)

    @staticmethod
    def load(prefix, epoch, ctx=None, **kwargs):
        """Load model checkpoint from file.

        Parameters
        ----------
        prefix : str
            Prefix of model name.
        epoch : int
            epoch number of model we would like to load.
        ctx : Context or list of Context, optional
            The device context of training and prediction.
        kwargs : dict
            Other parameters for model, including `num_epoch`, optimizer and `numpy_batch_size`.

        Returns
        -------
        model : FeedForward
            The loaded model that can be used for prediction.

        Notes
        -----
        - ``prefix-symbol.json`` will be saved for symbol.
        - ``prefix-epoch.params`` will be saved for parameters.
        """
        symbol, arg_params, aux_params, fullyconnect_params = load_checkpoint(prefix, epoch)
        return FeedForward(symbol, ctx=ctx,
                           arg_params=arg_params, aux_params=aux_params,
                           fullyconnect_params=fullyconnect_params,
                           begin_epoch=epoch,
                           **kwargs)

    @staticmethod
    def create(symbol, X, y=None, ctx=None,
               num_epoch=None, epoch_size=None, optimizer='sgd', initializer=Uniform(0.01),
               eval_data=None, eval_metric='acc',
               epoch_end_callback=None, batch_end_callback=None,
               kvstore='local', logger=None, work_load_list=None,
               eval_end_callback=LogValidationMetricsCallback(),
               eval_batch_end_callback=None, **kwargs):
        """Functional style to create a model.
        This function is more consistent with functional
        languages such as R, where mutation is not allowed.

        Parameters
        ----------
        symbol : Symbol
            The symbol configuration of a computation network.
        X : DataIter
            Training data.
        y : numpy.ndarray, optional
            If `X` is a ``numpy.ndarray``, `y` must be set.
        ctx : Context or list of Context, optional
            The device context of training and prediction.
            To use multi-GPU training, pass in a list of GPU contexts.
        num_epoch : int, optional
            The number of training epochs(epochs).
        epoch_size : int, optional
            Number of batches in a epoch. In default, it is set to
            ``ceil(num_train_examples / batch_size)``.
        optimizer : str or Optimizer, optional
            The name of the chosen optimizer, or an optimizer object, used for training.
        initializer : initializer function, optional
            The initialization scheme used.
        eval_data : DataIter or numpy.ndarray pair
            If `eval_set` is ``numpy.ndarray`` pair, it should
            be (`valid_data`, `valid_label`).
        eval_metric : metric.EvalMetric or str or callable
            The evaluation metric. Can be the name of an evaluation metric
            or a custom evaluation function that returns statistics
            based on a minibatch.
        epoch_end_callback : callable(epoch, symbol, arg_params, aux_states)
            A callback that is invoked at end of each epoch.
            This can be used to checkpoint model each epoch.
        batch_end_callback: callable(epoch)
            A callback that is invoked at end of each batch for print purposes.
        kvstore: KVStore or str, optional
           The KVStore or a string kvstore type: 'local', 'dist_sync', 'dis_async'.
           Defaults to 'local', often no need to change for single machine.
        logger : logging logger, optional
            When not specified, default logger will be used.
        work_load_list : list of float or int, optional
            The list of work load for different devices,
            in the same order as `ctx`.
        """
        model = FeedForward(symbol, ctx=ctx, num_epoch=num_epoch,
                            epoch_size=epoch_size,
                            optimizer=optimizer, initializer=initializer, **kwargs)
        model.fit(X, y, eval_data=eval_data, eval_metric=eval_metric,
                  epoch_end_callback=epoch_end_callback,
                  batch_end_callback=batch_end_callback,
                  kvstore=kvstore,
                  logger=logger,
                  work_load_list=work_load_list,
                  eval_end_callback=eval_end_callback,
                  eval_batch_end_callback=eval_batch_end_callback)
        return model
