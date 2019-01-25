#!/usr/bin/python
# -*- coding: utf-8 -*-
# convert_param.py

import os, sys
import argparse
import mxnet as mx
import mxnet.model as data_parallel
sys.path.append(os.path.join(os.path.dirname(__file__), 'model_parallel_softmax_mpi'))
import model_mpi_parallel_FF_splitnum as model_parallel

parser = argparse.ArgumentParser(description='convert param between dataparalle and modelparallel')
parser.add_argument('--mode', type=str, default='data_to_model')
parser.add_argument('--data-parallel-dir', type=str, default='./')
parser.add_argument('--model-parallel-dir', type=str, default='./')
parser.add_argument('--epoch', type=int)

args = parser.parse_args()

def convert_data_to_model(prefix, epoch, model_parallel_dir):
    symbol, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    mp_symbol = mx.symbol.load('%s-symbol.json' % model_parallel_dir)
    mp_arg_names = mp_symbol.list_arguments()
    mp_aux_names = mp_symbol.list_auxiliary_states()

    mp_arg_params = {}
    mp_aux_params = {}
    load_arg_names = []
    load_aux_names = []
    for k,v in arg_params.items():
        if k in mp_arg_names:
            mp_arg_params[k] = v
            load_arg_names.append(k)
            del arg_params[k]

    for k,v in aux_params.items():
        if k in mp_aux_names:
            mp_aux_params[k] = v
            load_aux_names.append(k)
            del aux_params[k]

    # check all names in modelparallel should be found in dataparallel
    assert len(load_arg_names) == len(mp_arg_names)-1
    assert len(load_aux_names) == len(mp_aux_names)

    fullyconnect_dict = {}
    for k,v in arg_params.items():
        if 'weight' in k:
            fullyconnect_dict[k] = v
        if 'bias' in k:
            fullyconnect_dict[k] = v
    assert len(fullyconnect_dict)

    model_parallel.save_checkpoint(model_parallel_dir, epoch, mp_symbol,
                                   mp_arg_params, mp_aux_params, fullyconnect_dict)


def convert_model_to_data(prefix, epoch, data_parallel_dir):
    symbol, arg_params, aux_params, fullyconnect_dict = model_parallel.load_checkpoint(prefix, epoch)
    dp_symbol = mx.symbol.load('%s-symbol.json' % data_parallel_dir)
    dp_arg_names = dp_symbol.list_arguments()
    dp_aux_names = dp_symbol.list_auxiliary_states()

    dp_arg_params = {}
    dp_aux_params = {}
    for k,v in arg_params.items():
        if k in dp_arg_names:
           dp_arg_params[k] = v
           del arg_params[k]
           dp_arg_names.remove(k)

    for k,v in aux_params.items():
        if k in dp_aux_names:
           dp_aux_params[k] = v
           del aux_params[k]
           dp_aux_names.remove(k)

    # check all names in dataparallel should be found in modelparallel
    assert not arg_params
    assert not aux_params

    for name in dp_arg_names:
        if 'weight' in name:
            dp_arg_params[name] = fullyconnect_dict['weight']
        if 'bias' in name:
            assert 'bias' in fullyconnect_dict.keys()
            dp_arg_params[name] = fullyconnect_dict['bias']

    data_parallel.save_checkpoint(data_parallel_dir, epoch, dp_symbol, dp_arg_params, dp_aux_params)


if __name__ == '__main__':
    if args.mode == 'data_to_model':
        convert_data_to_model(args.data_parallel_dir, args.epoch, args.model_parallel_dir)
    elif args.mode == 'model_to_data':
        convert_model_to_data(args.model_parallel_dir, args.epoch, args.data_parallel_dir)
    else:
        raise ValueError("only support convert mode in data_to_model or model_to_data")
