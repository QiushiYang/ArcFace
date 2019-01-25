from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import random
import logging
import pickle
import numpy as np
import time

# from image_iter_prefetch import FaceImageIter,PrefetchFaceIter
from image_iter_process_pk import FaceImageIter,PrefetchFaceIter
from image_iter import FaceImageIterList
import mxnet as mx
from mxnet import filestream as fs
from mxnet import ndarray as nd
import argparse
import mxnet.optimizer as optimizer
import learning_rate_decay
sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
import face_image
sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'symbols'))
import fresnet
import fresnet_fixbn
import fresnet_nobn
import fresnet_raw
import fresnet_raw_fixbn
import resnet_mx
import finception_resnet_v2
import fmobilenet
import fmobilenetv2
import fmobilefacenet
import fxception
import fdensenet
import fdpn
import fnasnet
import spherenet
import sphere64_hobot
import verification
from hobot_face_val_iter_splitnum import get_val_iter2
from metriclearn.metric import findMetricThreshold
import sklearn
from mxnet.device_sync import *
#sys.path.append(os.path.join(os.path.dirname(__file__), 'losses'))
#import center_loss
sys.path.append(os.path.join(os.path.dirname(__file__), 'model_parallel_softmax_mpi'))
import model_mpi_parallel_FF_splitnum
from callback import do_checkpoint

logger = logging.getLogger()
logger.setLevel(logging.INFO)

args = None

class LFWVeri(mx.metric.EvalMetric):
    def __init__(self, st, ed, steps):
        super(LFWVeri, self).__init__('veri')
        self.thres = np.linspace(st, ed, steps)
        print("veri_thres: ")
        #print(self.thres)

    def update(self, labels, preds):
        assert len(labels) == 1
        assert len(preds) == 1
        pred = preds[0].asnumpy()
        label = labels[0].asnumpy().astype('int32')
        self.qrys.append(pred)
        self.labels.append(label)

    def reset(self):
        self.labels = []
        self.qrys = []

    def calVeriPrecision(self, feat, label, thres):
        cor_veri = np.zeros((len(thres), 1))
        all_veri = int(len(label) / 2)
        print('lfw_veri: ', all_veri)
        bListIsCaled=np.zeros(all_veri*2)
        for i in range(all_veri*2):
            if bListIsCaled[i]==1:
                continue
            else:
                fea_0 = feat[i + 0]
                lab_0 = int(label[i])
                index_find=np.where(label==lab_0)
                index_1=15000
                if index_find[0][0]==i:
                    index_1=index_find[0][1]
                else:
                    index_1=index_find[0][0]
                #print '%d!!!!!!!'%(index_1)
                fea_1 = feat[index_1]
                dist = np.sqrt(((fea_0 - fea_1) ** 2).sum())
                if lab_0%2==0:     #positive pairs using even numbers as labels
                    for j in range(len(thres)):
                        if dist < thres[j]:
                            cor_veri[j] += 1
                else:
                    for j in range(len(thres)):
                        if dist >= thres[j]:
                            cor_veri[j] += 1
                bListIsCaled[i]=1
                bListIsCaled[index_1]=1

        max_prec = 0.0
        max_thres = 0.0
        print('lfw_veri_precision: ')
        for j in range(len(thres)):
            ostr = '%2d  %.4f  %.4f' % (j, thres[j], cor_veri[j] / all_veri)
            #print(ostr)
            if (cor_veri[j] / all_veri) > max_prec:
                max_prec = cor_veri[j] / all_veri
                max_thres = thres[j]

        return max_prec, max_thres

class NdcgEval(mx.metric.EvalMetric):
    """Calculate NDCG metrics, for metric learning"""

    def __init__(self, topk):
        super(NdcgEval, self).__init__('ndcg')
        self.topk = topk

    def update(self, labels, preds):
        assert len(labels) == 1
        assert len(preds) == 1      #there are two outputs on pred, embeding and softmax fc, and we select the 2nd output in executor
        pred = preds[0].asnumpy()
        label = labels[0].asnumpy().astype('int32')
        self.qrys.append(pred)
        self.labels.append(label)

    def reset(self):
        self.labels = []
        self.qrys = []

    def get(self):
        local_qry_feat = np.concatenate(self.qrys)
        local_qry_label = np.concatenate(self.labels)
        print(local_qry_feat.shape, local_qry_label.shape)
        assert local_qry_feat.shape[0] == local_qry_label.shape[0]
        ref_feat = local_qry_feat #my_Allgatherv(local_qry_feat)#local_qry_feat#my_Allgatherv(local_qry_feat)
        ref_label = local_qry_label #my_Allgatherv(local_qry_label)#local_qry_feat#my_Allgatherv(local_qry_label)
        print(ref_feat.shape, ref_label.shape)
        assert ref_feat.shape[0] == ref_label.shape[0]
        if ref_feat.shape[0] == 12000 or ref_feat.shape[0] == 11904 or \
            ref_feat.shape[0]/2 == 12000 or ref_feat.shape[0]/2 == 11904:
            return (self.name + "_lfw", 0.0)
        st = time.time()
        findMetricThreshold(
            local_qry_feat,
            local_qry_label,
            ref_feat,
            ref_label)
        print('time spent {} sec'.format(time.time() - st))
        #return (self.name, mean_nDCG)
        return (self.name, 1.0)


def parse_args():
  parser = argparse.ArgumentParser(description='Train face network')
  # general
  parser.add_argument('--data-rec-path', default='', help='training rec path')
  parser.add_argument('--val-data-dir', default='', help='validation set directory')
  parser.add_argument('--prefix', default='../model/model', help='directory to save model.')
  parser.add_argument('--pretrained', default='', help='pretrained model to load')
  parser.add_argument('--backbone_only', type=int, default=0, help='load backbone only')
  parser.add_argument('--freeze_backbone', type=int, default=0, help='freeze backbone in training')
  parser.add_argument('--ckpt', type=int, default=1, help='checkpoint saving option. 0: discard saving. 1: save when necessary. 2: always save')
  parser.add_argument('--loss-type', type=int, default=1,
            help='0 stands for vanilla softmax, 1 is arcface, defaults to 1')
  parser.add_argument('--max-steps', type=int, default=0, help='max training batches')
  parser.add_argument('--end-epoch', type=int, default=100000, help='training epoch size.')
  parser.add_argument('--network', default='r50', help='specify network')
  parser.add_argument('--version-se', type=int, default=0, help='whether to use se in network')
  parser.add_argument('--version-input', type=int, default=1, help='network input config')
  parser.add_argument('--version-output', type=str, default='E', help='network embedding output config')
  parser.add_argument('--version-unit', type=int, default=3, help='resnet unit config')
  parser.add_argument('--version-act', type=str, default='prelu', help='network activation config')
  parser.add_argument('--use-deformable', type=int, default=0, help='use deformable cnn in network')
  parser.add_argument('--lr', type=float, default=0.1, help='start learning rate')
  parser.add_argument('--lr-scheduler', type=str, default='Factor',
          help='Choose learning rate scheduler for different decaying policy')

  parser.add_argument('--pwconst-epoch-steps', type=str, default='10,14,16',
          help='When using PiecewiseConstant, epoch steps to change lr')
  parser.add_argument('--pwconst-lr-values', type=str, default='0.1,0.01,0.001,0.0001',
          help='When using PiecewiseConstant, lr values within each step')

  parser.add_argument('--lr_factor_epoch', type=int, default=10, help='When using Factor, lr changes each factor')
  parser.add_argument('--lr_factor',type=float,default=0.1,help='When using Factor, lr factor')

  parser.add_argument('--cos-lr-decay-epochs', type=int, default=100,
          help='When using CosineScheduler, number of epochs for learning rate reaches mimumim')
  parser.add_argument('--cos-lr-alpha', type=float, default=0.001,
          help='When using CosineScheduler, base lr multiplied by this factor is the minimum lr')
  parser.add_argument('--cos-lr-cycle', type=int, default=0,
          help='When using CosineScheduler, whether to recover to the base lr')

  parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
  parser.add_argument('--fc7-lr-mult', type=float, default=1.0, help='learning rate mult for fc7')
  parser.add_argument('--fc7-wd-mult', type=float, default=1.0, help='weight decay mult for fc7')
  parser.add_argument('--bn-mom', type=float, default=0.9, help='bn mom')
  parser.add_argument('--mom', type=float, default=0.9, help='momentum')
  parser.add_argument('--emb-size', type=int, default=512, help='embedding length')
  parser.add_argument('--per-batch-size', type=int, default=128, help='batch size in each context')
  parser.add_argument('--margin-m', type=float, default=0.5, help='margin for loss')
  parser.add_argument('--margin-s', type=float, default=64.0, help='scale for feature')
  parser.add_argument('--margin-a', type=float, default=1.0, help='')
  parser.add_argument('--margin-b', type=float, default=0.0, help='')
  parser.add_argument('--easy-margin', type=int, default=0, help='')
  parser.add_argument('--margin', type=int, default=4, help='margin for sphere')
  parser.add_argument('--beta', type=float, default=1000., help='param for sphere')
  parser.add_argument('--beta-min', type=float, default=5., help='param for sphere')
  parser.add_argument('--beta-freeze', type=int, default=0, help='param for sphere')
  parser.add_argument('--gamma', type=float, default=0.12, help='param for sphere')
  parser.add_argument('--power', type=float, default=1.0, help='param for sphere')
  parser.add_argument('--scale', type=float, default=0.9993, help='param for sphere')
  parser.add_argument('--rand-mirror', type=int, default=1, help='if do random mirror in training')
  parser.add_argument('--cutoff', type=int, default=0, help='cut off aug')
  parser.add_argument('--target', type=str, default='lfw,cfp_fp,agedb_30', help='verification targets')
  parser.add_argument('--kv_store', type=str, default='local', help='the kvstore type')
  parser.add_argument('--num_classes',type=int, default=0, help='number of ids')
  parser.add_argument('--epoch_size', type=int, default=-1,
            help='number of iteration steps within one epoch')
  parser.add_argument('--eval_first', type=int, default=0, help='whether to evaluate before training')
  parser.add_argument('--split-num', type=int, default=1, help='slice batchsize while training')
  parser.add_argument('--with-bn', type=int, default=1, help='0:withoutbn 1:withbn')
  parser.add_argument('--fix-bn', type=int, default=0, help='0:nofixbn 1:fixbn')
  parser.add_argument('--sync-bn', type=int, default=1, help='0: sync-bn turn off 1: sync-bn: turn on')
  parser.add_argument('--num-group', type=int, default=32, help='num group for convolution')
  parser.add_argument('--mirroring-level', type=int, default=2,
                      help='0: turn off 1: act turn on,2: bn and act turn on')
  parser.add_argument('--dtype', type=str, default='float32',
          help='Use float32 or float16 precision')
  parser.add_argument('--imgsize', type=str, default='112x96',
          help='Choose input image size from 112x96 and 112x112')
  parser.add_argument('--dataiter', type=str, default='FaceImageIter',
          help='Choose DataIter between FaceImageIter and ImageRecordIter')
  parser.add_argument('--pk', type=int, default=0,
          help='PK sampling, if 0 disabled, if > 0 number of samples per id is pk')
  args = parser.parse_args()
  return args


def get_symbol(args, arg_params, aux_params, fullyconnect_params):
    data_shape = (args.image_channel, args.image_h, args.image_w)
    image_shape = ",".join([str(x) for x in data_shape])
    print('init resnet', args.num_layers)
    if args.with_bn == 1:
        if args.fix_bn == 1:
            embedding = fresnet_fixbn.get_symbol(args.emb_size, args.num_layers,
                                       version_se=args.version_se, version_input=args.version_input,
                                       version_output=args.version_output, version_unit=args.version_unit,
                                       version_act=args.version_act,
                                       use_sync_bn=args.sync_bn,
                                       mirroring_level=args.mirroring_level,
                                       num_group = args.num_group,)
        else:
            embedding = fresnet.get_symbol(args.emb_size, args.num_layers,
                                       dtype=args.dtype,
                                       version_se=args.version_se, version_input=args.version_input,
                                       version_output=args.version_output, version_unit=args.version_unit,
                                       version_act=args.version_act,
                                       use_sync_bn=args.sync_bn,
                                       mirroring_level=args.mirroring_level,
                                       num_group = args.num_group,)
    else:
        embedding = sphere64_hobot.get_symbol(args)
        #embedding = spherenet.get_symbol(args.emb_size, 64)
        '''
        embedding = fresnet_nobn.get_symbol(args.emb_size, args.num_layers,
                                   version_se=args.version_se, version_input=args.version_input,
                                   version_output=args.version_output, version_unit=args.version_unit,
                                   version_act=args.version_act,
                                   use_sync_bn=args.sync_bn,
                                   mirroring_level=args.mirroring_level,
                                   num_group = args.num_group,)
        '''
    nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc5n') * args.margin_s
    return (nembedding, arg_params, aux_params, fullyconnect_params)


def train_net(args):
    ctx = []
    cvd = os.environ['DEVICES'].strip()
    if len(cvd)>0:
      for i in map(int,cvd.split(',')):
        ctx.append(mx.gpu(i))
    if len(ctx)==0:
      ctx = [mx.cpu()]
      print('use cpu')
    else:
      print('gpu num:', len(ctx))

    kv = mx.kvstore.create(args.kv_store)
    use_sync_bn = args.sync_bn
    if use_sync_bn:
       init_device_sync(ctx)

    prefix = args.prefix
    prefix_dir = os.path.dirname(prefix)
    if not os.path.exists(prefix_dir):
      os.makedirs(prefix_dir)
    end_epoch = args.end_epoch
    args.ctx_num = len(ctx)
    args.num_layers = int(args.network[1:])
    print('num_layers', args.num_layers)
    if args.per_batch_size == 0:
      args.per_batch_size = 128
    args.batch_size = args.per_batch_size * args.ctx_num
    args.rescale_threshold = 0
    args.image_channel = 3

    os.environ['BETA'] = str(args.beta)
    image_size = [int(val) for val in args.imgsize.split('x')]
    args.image_h = image_size[0]
    args.image_w = image_size[1]
    assert(args.num_classes > 0)
    print('num_classes', args.num_classes)
    path_imgrec = args.data_rec_path

    if args.loss_type == 1:
        args.loss_type = True
    else:
        if args.loss_type == 0:
            args.loss_type = False

    print('Called with argument:', args)
    data_shape = (args.image_channel, image_size[0], image_size[1])
    base_lr = args.lr
    base_wd = args.wd
    base_mom = args.mom
    # if len(args.pretrained) == 0:
    if args.pretrained == 'None':
      arg_params = None
      aux_params = None
      fullyconnect_params = None
      sym, arg_params, aux_params, fullyconnect_params = get_symbol(args, arg_params, aux_params, fullyconnect_params)
    else:
      vec = args.pretrained.split(',')
      print('==> loading', vec)
      if (args.backbone_only == 1):
          logging.info('Only backbone loaded')
          _, arg_params, aux_params = mx.model.load_checkpoint(vec[0], int(vec[1]))
          sym, arg_params, aux_params, fullyconnect_params = get_symbol(args, arg_params, aux_params, None)
      else:
          logging.info('Both backbone and softmax weight loaded')
          tmp = model_mpi_parallel_FF_splitnum.FeedForward.load(vec[0], int(vec[1]))
          sym, arg_params, aux_params, fullyconnect_params = get_symbol(args, tmp.arg_params, tmp.aux_params, tmp.fullyconnect_params)

    if args.dataiter == 'FaceImageIter':
        # train_dataiter = FaceImageIter(
        train_dataiter = PrefetchFaceIter(
            batch_size = args.batch_size,
            data_shape = data_shape,
            path_imgrec = path_imgrec,
            num_samples_per_class = args.pk,
            shuffle = True,
            rand_mirror = False,
            aug_list = None,
            # num_parts = kv.num_workers,
            # part_index = kv.rank
            )
        args.epoch_size = train_dataiter.epoch_size
        logging.info('==>1 epoch is {} batches'.format(args.epoch_size))

    elif args.dataiter == 'ImageRecordIter':
        train_dataiter = mx.io.ImageRecordIter(
            path_imgrec = path_imgrec,
            data_shape = data_shape,
            batch_size = args.batch_size,
            rand_mirror = False,
            num_parts = kv.num_workers,
            part_index = kv.rank,
            prefetch_buffer = 4,
            prefetch_buffer_keep = 2,
            label_width = 1,
            shuffle = True,
            shuffle_chunk_size = 128
            )

    initializer_bb = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2)
    initializer_fc = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2)

    # _rescale = 1.0 / args.ctx_num
    _rescale = 1.0 / (args.batch_size)
    if args.lr_scheduler == 'Factor':
        lr_scheduler = mx.lr_scheduler.FactorScheduler(step=max(int(args.epoch_size*args.lr_factor_epoch), 1),
                                                    factor=args.lr_factor, stop_factor_lr=5e-9)

    elif args.lr_scheduler == 'PWConst':
        steps = [args.epoch_size*int(epc) for epc in args.pwconst_epoch_steps.split(',')]
        lr_values = [float(val) for val in args.pwconst_lr_values.split(',')]
        base_lr = lr_values[0]
        lr_scheduler = learning_rate_decay.PiecewiseConstantScheduler(steps, lr_values)

    elif args.lr_scheduler == 'Cosine':
        lr_scheduler = learning_rate_decay.CosineScheduler(
                decay_steps=args.epoch_size*args.cos_lr_decay_epochs,
                alpha=args.cos_lr_alpha,
                cycle=True if args.cos_lr_cycle==1 else False)

    if args.dtype == 'float16':
        opt = optimizer.SGD(learning_rate=base_lr,
                    momentum=base_mom,
                    wd=base_wd,
                    rescale_grad=_rescale,
                    lr_scheduler=lr_scheduler,
                    multi_precision=True)
        opt_fc = optimizer.SGD(learning_rate=base_lr*args.fc7_lr_mult,
                    momentum=base_mom,
                    wd=base_wd*args.fc7_wd_mult,
                    rescale_grad=_rescale,
                    lr_scheduler=lr_scheduler,
                    multi_precision=True)
    else:
        opt = optimizer.SGD(learning_rate=base_lr,
                    momentum=base_mom,
                    wd=base_wd,
                    rescale_grad=_rescale,
                    lr_scheduler=lr_scheduler)

        opt_fc = optimizer.SGD(learning_rate=base_lr*args.fc7_lr_mult,
                    momentum=base_mom,
                    wd=base_wd*args.fc7_wd_mult,
                    rescale_grad=_rescale,
                    lr_scheduler=lr_scheduler)

    #opt = optimizer.Adam(learning_rate=base_lr, beta1=0.9, beta2=0.999, epsilon=1e-08, wd=base_wd, rescale_grad=_rescale,lr_scheduler=lr_scheduler)
    #opt_fc = optimizer.Adam(learning_rate=base_lr*args.fc7_lr_mult, beta1=0.9, beta2=0.999, epsilon=1e-08, wd=base_wd*args.fc7_wd_mult, rescale_grad=_rescale,lr_scheduler=lr_scheduler)

    model = model_mpi_parallel_FF_splitnum.FeedForward(
        symbol = sym,
        ctx = ctx,   # [mx.cpu()]*4,
        num_epoch = end_epoch,
        optimizer = opt,
        optimizer_fc = opt_fc,
        initializer_bb = initializer_bb,
        initializer_fc = initializer_fc,
        epoch_size = args.epoch_size if args.epoch_size!=-1 else None,
        arg_params = arg_params,
        aux_params = aux_params,
        fullyconnect_params= fullyconnect_params,
        begin_epoch = 0,
        #**model_args
        )

    epoch_cb = None
    val_dataiter = get_val_iter2(args, kv, None)
    batch_end_callback = []
    batch_end_callback.append(mx.callback.Speedometer(args.batch_size, 100))

    if prefix is not None:
        if kv.rank!=0:
            prefix += "-%d" % (kv.rank)
    checkpoint = None if args.prefix is None else do_checkpoint(prefix)

    if args.freeze_backbone == 1:  #freeze all params except last fc
        logging.info('freeze backbone')
        fixed_param_names = sym.list_arguments()
        # fixed_param_names.remove('fc7_weight')
        # fixed_param_names.remove('fc7_bias')
    else:
        fixed_param_names = []
        # for param_name in sym.list_arguments():
        #     if 'gamma' in param_name or 'beta' in param_name:
        #         fixed_param_names.append(param_name)

    model.fit(
        X = train_dataiter,
        eval_data = val_dataiter,
        embedding_size = args.emb_size,
        num_classes = args.num_classes,
        margin = args.loss_type,
        margin_s = args.margin_s,
        margin_m = args.margin_m,
        easy_margin = args.easy_margin,
        train_metric = 'acc',
        eval_metric = NdcgEval(10), #[ LFWVeri(0.0, 30.0, 300)],
        eval_first = args.eval_first,
        kvstore = kv,
        split_num = args.split_num,
        fixed_param_names  = fixed_param_names,
        batch_end_callback = batch_end_callback, #_batch_callback_hobot,  #mx.callback.Speedometer(args.batch_size, 100),
        epoch_end_callback = checkpoint
        )  #epoch_cb)       #checkpoint


def print_sym():
    global args
    args = parse_args()
    args.num_layers = int(args.network[1:])
    args.image_channel = 3
    image_size = [112, 96]
    args.image_h = image_size[0]
    args.image_w = image_size[1]
    _symbol = get_symbol(args, None, None)
    digraph = mx.viz.plot_network(_symbol[0],
            shape={"data":(1,3,112,96),"fc7_weight":(2155000,256), 'softmax_label':(1,)}, node_attrs={"shape":'oval',"fixedsize":'false'}, save_format = 'jpg')
    digraph.view()

def main():
    #time.sleep(3600*6.5)
    global args
    args = parse_args()
    #import pudb; pudb.set_trace()
    train_net(args)

if __name__ == '__main__':
    main()
    #print_sym()

