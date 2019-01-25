from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import math
import random
import logging
import pickle
import numpy as np
from image_iter_process_plus import FaceImageIter,PrefetchFaceIter
import mxnet as mx
from mxnet.device_sync import *
from mxnet import filestream as fs
from mxnet import ndarray as nd
import argparse
import mxnet.optimizer as optimizer
import learning_rate_decay
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
import face_image
sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'symbols'))
#import probe
import fresnet
import fresnet_fixbn
import sphere64_hobot
import finception_resnet_v2
import fmobilenet
import fmobilenetv2
import fmobilefacenet
import fxception
import fdensenet
import fdpn
import fnasnet
import fresatten
import spherenet
import verification
import sklearn
import verification_hobot
import numpy
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), 'losses'))
import center_loss

from hobot_face_val_iter import get_val_iter2 as get_val
from hobot_face_val_iter_splitnum import get_val_iter2 as get_val_spltn
from metriclearn.metric import findMetricThreshold

sys.path.append(os.path.join(os.path.dirname(__file__), 'model_parallel_softmax_mpi'))
import model_mpi_parallel_FF_splitnum
from callback import do_checkpoint

logger = logging.getLogger()
logger.setLevel(logging.INFO)

args = None


class AccMetric(mx.metric.EvalMetric):
    def __init__(self):
        self.axis = 1
        super(AccMetric, self).__init__(
            'acc', axis=self.axis, output_names=None, label_names=None)
        self.losses = []
        self.count = 0

    def update(self, labels, preds):
        self.count += 1
        preds = [preds[1]]  #use softmax output
        for label, pred_label in zip(labels, preds):
            if pred_label.shape != label.shape:
                pred_label = mx.ndarray.argmax(pred_label, axis=self.axis)
            pred_label = pred_label.asnumpy().astype('int32').flatten()
            label = label.asnumpy()
            if label.ndim == 2:
                label = label[:, 0]
            label = label.astype('int32').flatten()
            assert label.shape == pred_label.shape
            self.sum_metric += (pred_label.flat == label.flat).sum()
            self.num_inst += len(pred_label.flat)


class LossValueMetric(mx.metric.EvalMetric):
    def __init__(self):
        self.axis = 1
        super(LossValueMetric, self).__init__(
            'lossvalue', axis=self.axis, output_names=None, label_names=None)
        self.losses = []

    def update(self, labels, preds):
        loss = preds[-1].asnumpy()[0]
        self.sum_metric += loss
        self.num_inst += 1.0
        gt_label = preds[-2].asnumpy()
        #print(gt_label)

class LossMetric(mx.metric.EvalMetric):
  def __init__(self):
    self.axis = 1
    super(LossMetric, self).__init__(
        'loss', axis=self.axis,
        output_names=None, label_names=None)
    self.losses = []
    self.count = 0

  def update(self, labels, preds):
    preds = [preds[1]] #use softmax output
    self.reset()
    for label, pred in zip(labels, preds):
        label = label.asnumpy()
        pred = pred.asnumpy()
        if label.ndim==2:
          label = label[:,0]
        label = label.astype('int32').flatten()
        assert label.shape[0]==pred.shape[0]
        prob = pred[numpy.arange(label.shape[0]), numpy.int64(label)]
        self.sum_metric += (-numpy.log(numpy.maximum(prob, 1e-30))).sum()
        self.num_inst += label.shape[0]


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
        # _norm = np.linalg.norm(local_qry_feat)
        # local_qry_feat = local_qry_feat / _norm
        local_qry_label = np.concatenate(self.labels)
        print(local_qry_feat.shape, local_qry_label.shape)
        assert local_qry_feat.shape[0] == local_qry_label.shape[0]
        ref_feat = local_qry_feat
        ref_label = local_qry_label
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
    parser.add_argument(
            '--train-mode', default='data_parallel', help='training mode: "data_parallel" or "model_parallel"')
    parser.add_argument('--split-num', type=int, default=1, help='slice batchsize while training')
    parser.add_argument(
        '--data-rec-path', default='', help='training set rec path')
    parser.add_argument('--val-data-dir', default='', help='validation set directory')
    parser.add_argument('--dataiter', type=str, default='FaceImageIter',
            help='Choose DataIter between FaceImageIter and ImageRecordIter')
    parser.add_argument('--epoch-size', type=int, default=-1,
                help='number of iteration steps within one epoch')
    parser.add_argument('--kv-store', type=str, default='local', help='the kvstore type')
    parser.add_argument(
        '--prefix', default='../model/model', help='directory to save model.')
    parser.add_argument(
        '--pretrained', default='', help='pretrained model to load')
    parser.add_argument('--backbone-only', type=int, default=0, help='load backbone parameters only')
    parser.add_argument('--freeze-backbone', type=int, default=0, help='freeze backbone in training')
    parser.add_argument('--with-bn', type=int, default=1, help='0:withoutbn 1:withbn')
    parser.add_argument('--fix-bn', type=int, default=0, help='0:nofixbn 1:fixbn')
    parser.add_argument(
        '--ckpt',
        type=int,
        default=1,
        help=
        'checkpoint saving option. 0: discard saving. 1: save when necessary. 2: always save'
    )
    parser.add_argument('--loss-type', type=int, default=4, help='loss type')
    parser.add_argument(
        '--verbose',
        type=int,
        default=2000,
        help='do verification testing and model saving every verbose batches')
    parser.add_argument(
        '--max-steps', type=int, default=0, help='max training batches')
    parser.add_argument(
        '--end-epoch', type=int, default=100000, help='training epoch size.')
    parser.add_argument('--network', default='r50', help='specify network')
    parser.add_argument(
        '--version-se',
        type=int,
        default=0,
        help='whether to use se in network')
    parser.add_argument(
        '--version-input', type=int, default=1, help='network input config')
    parser.add_argument(
        '--version-output',
        type=str,
        default='E',
        help='network embedding output config')
    parser.add_argument(
        '--version-unit', type=int, default=3, help='resnet unit config')
    parser.add_argument(
        '--version-act',
        type=str,
        default='prelu',
        help='network activation config')
    parser.add_argument(
        '--use-deformable',
        type=int,
        default=0,
        help='use deformable cnn in network')
    parser.add_argument(
        '--lr', type=float, default=0.1, help='start learning rate')
    parser.add_argument(
        '--lr-steps', type=str, default='', help='steps of lr changing')
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

    parser.add_argument(
        '--wd', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--fc7-lr-mult', type=float, default=1.0, help='learning rate mult for fc7')
    parser.add_argument(
        '--fc7-wd-mult',
        type=float,
        default=1.0,
        help='weight decay mult for fc7')
    parser.add_argument('--bn-mom', type=float, default=0.9, help='bn mom')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument(
        '--emb-size', type=int, default=512, help='embedding length')
    parser.add_argument(
        '--per-batch-size',
        type=int,
        default=128,
        help='batch size in each context')
    parser.add_argument(
        '--margin-m', type=float, default=0.5, help='margin for loss')
    parser.add_argument(
        '--margin-s', type=float, default=64.0, help='scale for feature')
    parser.add_argument('--margin-a', type=float, default=1.0, help='')
    parser.add_argument('--margin-b', type=float, default=0.0, help='')
    parser.add_argument('--easy-margin', type=int, default=0, help='')
    parser.add_argument(
        '--margin', type=int, default=4, help='margin for sphere')
    parser.add_argument(
        '--beta', type=float, default=1000., help='param for sphere')
    parser.add_argument(
        '--beta-min', type=float, default=5., help='param for sphere')
    parser.add_argument(
        '--beta-freeze', type=int, default=0, help='param for sphere')
    parser.add_argument(
        '--gamma', type=float, default=0.12, help='param for sphere')
    parser.add_argument(
        '--power', type=float, default=1.0, help='param for sphere')
    parser.add_argument(
        '--scale', type=float, default=0.9993, help='param for sphere')
    parser.add_argument(
        '--rand-mirror',
        type=int,
        default=1,
        help='if do random mirror in training')
    parser.add_argument('--cutoff', type=int, default=0, help='cut off aug')
    parser.add_argument(
        '--target',
        type=str,
        default='lfw,cfp_fp,agedb_30',
        help='verification targets')

    parser.add_argument(
        '--sync-bn',
        type=int,
        default=1,
        help='0: sync-bn turn off 1: sync-bn: turn on')
    parser.add_argument(
        '--num-group', type=int, default=32, help='num group for convolution')
    parser.add_argument(
        '--mirroring-level',
        type=int,
        default=2,
        help='0: turn off 1: act turn on,2: bn and act turn on')

    parser.add_argument(
        '--num-samples-per-class',
        type=int,
        default=1,
        help='num samples per class')
    parser.add_argument(
        '--repet-flag', type=int, default=0, help='repetive flag')
    parser.add_argument(
        '--repet-num',
        type=int,
        default=1,
        help='repetive number of each batch data for training')

    parser.add_argument(
        '--warmup', type=int, default=0, help='use warm-up lr scheduler')
    parser.add_argument(
        '--step-sizes',
        type=int,
        default=50000,  #50000,
        help='warm-up param')
    parser.add_argument(
        '--max-iter-steps', type=int, default=150000, help='warm-up param')
    parser.add_argument(
        '--start-lr', type=float, default=0.001, help='warm-up param')
    parser.add_argument(
        '--max-lr', type=float, default=0.8, help='warm-up param')
    parser.add_argument(
        '--target-lr', type=float, default=0.0001, help='warm-up param')
    parser.add_argument(
        '--warmup-power', type=float, default=1.0, help='warm-up param')

    parser.add_argument('--use-center', type=int, default=0, help='')
    parser.add_argument('--center-alpha', type=float, default=0.5, help='')
    parser.add_argument('--center-scale', type=float, default=0.003, help='')
    parser.add_argument('--val_data_dir', type=str, default='', help='')

    parser.add_argument(
        '--num-classes', type=int, default=0, help='')
    parser.add_argument('--imgsize', type=str,default='112x96', help='choose input image size from 112x96 and 112x112')
    parser.add_argument('--eval-first', type=int, default=0, help='whether to evaluate before training')
    parser.add_argument('--sample-type', type=int, default=0,
          help='0: normal shuffle; 1: pk with partial data; 2: pk with all data; 3: pk with partial data (max_num_per_id); 4: id shuffle only')
    args = parser.parse_args()
    return args


def get_emb(args):
    data_shape = (args.image_channel, args.image_h, args.image_w)
    print("data shape: ",data_shape)
    image_shape = ",".join([str(x) for x in data_shape])
    if args.network[0] == 'd':
        embedding = fdensenet.get_symbol(
            args.emb_size,
            args.num_layers,
            version_se=args.version_se,
            version_input=args.version_input,
            version_output=args.version_output,
            version_unit=args.version_unit)
    elif args.network[0] == 'm':
        print('init mobilenet', args.num_layers)
        if args.num_layers == 1:
            embedding = fmobilenet.get_symbol(
                args.emb_size,
                version_se=args.version_se,
                version_input=args.version_input,
                version_output=args.version_output,
                version_unit=args.version_unit)
        else:
            embedding = fmobilenetv2.get_symbol(args.emb_size)
    elif args.network[0] == 'i':
        print('init inception-resnet-v2', args.num_layers)
        embedding = finception_resnet_v2.get_symbol(
            args.emb_size,
            version_se=args.version_se,
            version_input=args.version_input,
            version_output=args.version_output,
            version_unit=args.version_unit)
    elif args.network[0] == 'x':
        print('init xception', args.num_layers)
        embedding = fxception.get_symbol(
            args.emb_size,
            version_se=args.version_se,
            version_input=args.version_input,
            version_output=args.version_output,
            version_unit=args.version_unit)
    elif args.network[0] == 'p':
        print('init dpn', args.num_layers)
        embedding = fdpn.get_symbol(
            args.emb_size,
            args.num_layers,
            version_se=args.version_se,
            version_input=args.version_input,
            version_output=args.version_output,
            version_unit=args.version_unit)
    elif args.network[0] == 'n':
        print('init nasnet', args.num_layers)
        embedding = fnasnet.get_symbol(args.emb_size)
    elif args.network[0] == 's':
        print('init spherenet', args.num_layers)
        embedding = spherenet.get_symbol(args.emb_size, args.num_layers)
    elif args.network[0] == 'y':
        print('init mobilefacenet', args.num_layers)
        embedding = fmobilefacenet.get_symbol(
            args.emb_size, bn_mom=args.bn_mom, wd_mult=args.fc7_wd_mult)
    elif args.network[0] == 'a':
        print('init resnet attention net ', args.num_layers)
        embedding = fresatten.get_symbol(
            args.emb_size, args.num_layers, version_act=args.version_act)
    else:
        print('init resnet', args.num_layers)
        if args.with_bn == 1:
            if args.fix_bn == 1:
                embedding = fresnet_fixbn.get_symbol(
                    args.emb_size,
                    args.num_layers,
                    version_se=args.version_se,
                    version_input=args.version_input,
                    version_output=args.version_output,
                    version_unit=args.version_unit,
                    version_act=args.version_act,
                    use_sync_bn=args.sync_bn,
                    mirroring_level=args.mirroring_level,
                    num_group=args.num_group,
                    )
            else:
                embedding = fresnet.get_symbol(
                    args.emb_size,
                    args.num_layers,
                    version_se=args.version_se,
                    version_input=args.version_input,
                    version_output=args.version_output,
                    version_unit=args.version_unit,
                    version_act=args.version_act,
                    use_sync_bn=args.sync_bn,
                    mirroring_level=args.mirroring_level,
                    num_group=args.num_group,
                    )
        else:
            embedding = sphere64_hobot.get_symbol(args)

    if args.train_mode == 'model_parallel':
        nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc5n') * args.margin_s
        return nembedding
    else:
        return embedding


def get_sym_dp(args, arg_params, aux_params):
    embedding = get_emb(args)
    all_label = mx.symbol.Variable('softmax_label')
    gt_label = all_label
    extra_loss = None
    _weight = mx.symbol.Variable(
        "fc7_weight",
        shape=(args.num_classes, args.emb_size),
        lr_mult=1.0,
        wd_mult=args.fc7_wd_mult)
    if args.loss_type == 0:  #softmax
        _bias = mx.symbol.Variable('fc7_bias', lr_mult=2.0, wd_mult=0.0)
        fc7 = mx.sym.FullyConnected(
            data=embedding,
            weight=_weight,
            bias=_bias,
            num_hidden=args.num_classes,
            name='fc7')
    elif args.loss_type == 1:  #sphere
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        fc7 = mx.sym.LSoftmax(
            data=embedding,
            label=gt_label,
            num_hidden=args.num_classes,
            weight=_weight,
            beta=args.beta,
            margin=args.margin,
            scale=args.scale,
            beta_min=args.beta_min,
            verbose=1000,
            name='fc7')
    elif args.loss_type == 2:
        s = args.margin_s
        m = args.margin_m
        assert (s > 0.0)
        assert (m > 0.0)
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        nembedding = mx.symbol.L2Normalization(
            embedding, mode='instance', name='fc1n') * s
        fc7 = mx.sym.FullyConnected(
            data=nembedding,
            weight=_weight,
            no_bias=True,
            num_hidden=args.num_classes,
            name='fc7')
        s_m = s * m
        gt_one_hot = mx.sym.one_hot(
            gt_label, depth=args.num_classes, on_value=s_m, off_value=0.0)
        fc7 = fc7 - gt_one_hot
    elif args.loss_type == 4:
        s = args.margin_s
        m = args.margin_m
        assert s > 0.0
        assert m >= 0.0
        assert m < (math.pi / 2)
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        nembedding = mx.symbol.L2Normalization(
            embedding, mode='instance', name='fc1n') * s
        fc7 = mx.sym.FullyConnected(
            data=nembedding,
            weight=_weight,
            no_bias=True,
            num_hidden=args.num_classes,
            name='fc7')
        if args.use_center:
            print('center-loss', args.center_alpha, args.center_scale)
            extra_loss = mx.symbol.Custom(data=embedding, label=gt_label, name='center_loss', op_type='centerloss',\
                num_class=args.num_classes, alpha=args.center_alpha, scale=args.center_scale, batchsize=args.per_batch_size)
        #fc7 = mx.symbol.Custom(fc7, layername='after fc7', op_type='ProbeLayer')
        zy = mx.sym.pick(fc7, gt_label, axis=1)
        cos_t = zy / s
        cos_m = math.cos(m)
        sin_m = math.sin(m)
        mm = math.sin(math.pi - m) * m
        #threshold = 0.0
        threshold = math.cos(math.pi - m)
        if args.easy_margin:
            cond = mx.symbol.Activation(data=cos_t, act_type='relu')
        else:
            cond_v = cos_t - threshold
            cond = mx.symbol.Activation(data=cond_v, act_type='relu')
        body = cos_t * cos_t
        body = 1.0 - body
        sin_t = mx.sym.sqrt(body)
        new_zy = cos_t * cos_m
        b = sin_t * sin_m
        new_zy = new_zy - b
        new_zy = new_zy * s
        if args.easy_margin:
            zy_keep = zy
        else:
            zy_keep = zy - s * mm
        new_zy = mx.sym.where(cond, new_zy, zy_keep)

        diff = new_zy - zy
        diff = mx.sym.expand_dims(diff, 1)
        gt_one_hot = mx.sym.one_hot(
            gt_label, depth=args.num_classes, on_value=1.0, off_value=0.0)
        body = mx.sym.broadcast_mul(gt_one_hot, diff)
        fc7 = fc7 + body
    elif args.loss_type == 5:
        s = args.margin_s
        m = args.margin_m
        assert s > 0.0
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        nembedding = mx.symbol.L2Normalization(
            embedding, mode='instance', name='fc1n') * s
        fc7 = mx.sym.FullyConnected(
            data=nembedding,
            weight=_weight,
            no_bias=True,
            num_hidden=args.num_classes,
            name='fc7')
        if args.use_center:
            print('center-loss', args.center_alpha, args.center_scale)
            extra_loss = mx.symbol.Custom(data=embedding, label=gt_label, name='center_loss', op_type='centerloss',\
                num_class=args.num_classes, alpha=args.center_alpha, scale=args.center_scale, batchsize=args.per_batch_size)
        if args.margin_a != 1.0 or args.margin_m != 0.0 or args.margin_b != 0.0:
            if args.margin_a == 1.0 and args.margin_m == 0.0:
                s_m = s * args.margin_b
                gt_one_hot = mx.sym.one_hot(
                    gt_label,
                    depth=args.num_classes,
                    on_value=s_m,
                    off_value=0.0)
                fc7 = fc7 - gt_one_hot
            else:
                zy = mx.sym.pick(fc7, gt_label, axis=1)
                cos_t = zy / s
                t = mx.sym.arccos(cos_t)
                if args.margin_a != 1.0:
                    t = t * args.margin_a
                if args.margin_m > 0.0:
                    t = t + args.margin_m
                body = mx.sym.cos(t)
                if args.margin_b > 0.0:
                    body = body - args.margin_b
                new_zy = body * s
                diff = new_zy - zy
                diff = mx.sym.expand_dims(diff, 1)
                gt_one_hot = mx.sym.one_hot(
                    gt_label,
                    depth=args.num_classes,
                    on_value=1.0,
                    off_value=0.0)
                body = mx.sym.broadcast_mul(gt_one_hot, diff)
                fc7 = fc7 + body
    out_list = [mx.symbol.BlockGrad(embedding)]
    softmax = mx.symbol.SoftmaxOutput(data=fc7, label=gt_label, name='softmax')
    out_list.append(softmax)
    if extra_loss is not None:
        out_list.append(extra_loss)
    out = mx.symbol.Group(out_list)
    return (out, arg_params, aux_params)


def get_sym_mp(args, arg_params, aux_params, fullyconnect_params):
    nembedding = get_emb(args)
    return (nembedding, arg_params, aux_params, fullyconnect_params)


def train_net(args):
    ctx = []
    cvd = os.environ['DEVICES'].strip()
    if len(cvd) > 0:
        for i in map(int, cvd.split(',')):
            ctx.append(mx.gpu(i))
    if len(ctx) == 0:
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
    os.environ['BETA'] = str(args.beta)

    args.image_channel = 3
    image_size = [int(val) for val in args.imgsize.split('x')]
    args.image_h = image_size[0]
    args.image_w = image_size[1]
    print('image_size', image_size)

    path_imgrec = args.data_rec_path
    assert (args.num_classes > 0)
    print('num_classes', args.num_classes)

    print('Called with argument:', args)

    data_shape = (args.image_channel, image_size[0], image_size[1])
    mean = None
    begin_epoch = 0
    base_lr = args.lr
    base_wd = args.wd
    base_mom = args.mom

    if args.dataiter == 'FaceImageIter':
        train_dataiter = PrefetchFaceIter(
            batch_size = args.batch_size,
            data_shape = data_shape,
            path_imgrec = path_imgrec,
            shuffle = True,
            rand_mirror = False,
            # aug_list = ['color','gauss_blur','motion_blur'],
            num_parts = kv.num_workers,
            part_index = kv.rank,
            )
        args.epoch_size = train_dataiter.epoch_size
        logging.info('==> 1 epoch is {} batches'.format(args.epoch_size))
    elif args.dataiter == 'ImageRecordIter':
        train_dataiter = mx.io.ImageRecordIter(
            path_imgrec = path_imgrec,
            data_shape = data_shape,
            batch_size = args.batch_size,
            rand_mirror = False,
            prefetch_buffer = 4,
            prefetch_buffer_keep = 2,
            label_width = 1,
            shuffle = True,
            shuffle_chunk_size = 128
            )
        logging.info('==> 1 epoch is {} batches'.format(args.epoch_size))

    initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2)

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

    _rescale = 1.0 / args.batch_size / kv.num_workers
    # _rescale = 1.0 / args.per_batch_size
    opt = optimizer.SGD(
            learning_rate = base_lr,
            momentum = base_mom,
            wd = base_wd,
            rescale_grad = _rescale,
            lr_scheduler = lr_scheduler
            )
    opt_fc = optimizer.SGD(
            learning_rate = base_lr*args.fc7_lr_mult,
            momentum = base_mom,
            wd = base_wd,
            rescale_grad = _rescale,
            lr_scheduler = lr_scheduler
            )

    if args.train_mode == 'model_parallel':
        logging.info('==> MODEL PARALLEL TRAINING MODE')
        if args.pretrained == 'None':
            arg_params = None
            aux_params = None
            fullyconnect_params = None
            sym, arg_params, aux_params, fullyconnect_params = get_sym_mp(args, arg_params, aux_params, fullyconnect_params)
        else:
            vec = args.pretrained.split(',')
            print('==> loading', vec)
            if (args.backbone_only == 1):
                logging.info('==> only backbone loaded')
                _, arg_params, aux_params = mx.model.load_checkpoint(vec[0], int(vec[1]))
                sym, arg_params, aux_params, fullyconnect_params = get_sym_mp(args, arg_params, aux_params, None)
            else:
                logging.info('==> both backbone and fc weight loaded')
                tmp = model_mpi_parallel_FF_splitnum.FeedForward.load(vec[0], int(vec[1]))
                sym, arg_params, aux_params, fullyconnect_params = get_sym_mp(args, tmp.arg_params, tmp.aux_params, tmp.fullyconnect_params)

        model = model_mpi_parallel_FF_splitnum.FeedForward(
            symbol = sym,
            ctx = ctx,
            num_epoch = end_epoch,
            optimizer = opt,
            optimizer_fc = opt_fc,
            initializer_bb = initializer,
            initializer_fc = initializer,
            epoch_size = args.epoch_size if args.epoch_size!=-1 else None,
            arg_params = arg_params,
            aux_params = aux_params,
            fullyconnect_params= fullyconnect_params,
            begin_epoch = 0,
            #**model_args
            )

        val_dataiter = get_val_spltn(args, kv, None)
        batch_end_callback = []
        batch_end_callback.append(mx.callback.Speedometer(args.batch_size, 100))

        if prefix is not None:
            if kv.rank != 0:
                prefix += "-%d" % (kv.rank)
        checkpoint = None if args.prefix is None else do_checkpoint(prefix)

        if args.freeze_backbone == 1:
            logging.info('==> freeze backbone')
            fixed_param_names = sym.list_arguments()
        else:
            fixed_param_names = []

        if args.loss_type == 4:
            args.loss_type = True
        elif args.loss_type == 0:
            args.loss_type = False
        else:
            raise ValueError('Choose loss_type from 0 (vanilla softmax) and 4 (ArcFace)')

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
            eval_metric = NdcgEval(10),
            eval_first = args.eval_first,
            kvstore = kv,
            split_num = args.split_num,
            fixed_param_names  = fixed_param_names,
            batch_end_callback = batch_end_callback,
            epoch_end_callback = checkpoint
            )

    elif args.train_mode == 'data_parallel':
        logging.info('==> DATA PARALLEL TRAINING MODE')
        if args.loss_type == 1 and args.num_classes > 20000:
            args.beta_freeze = 5000
            args.gamma = 0.06

        if args.pretrained == 'None':
            arg_params = None
            aux_params = None
            sym, arg_params, aux_params = get_sym_dp(args, arg_params, aux_params)
        else:
            vec = args.pretrained.split(',')
            print('loading', vec)
            _, arg_params, aux_params = mx.model.load_checkpoint(vec[0], int(vec[1]))
            sym, arg_params, aux_params = get_sym_dp(args, arg_params, aux_params)
        if args.network[0] == 's':
            data_shape_dict = {'data': (args.per_batch_size, ) + data_shape}
            spherenet.init_weights(sym, data_shape_dict, args.num_layers)
        print('params: ', sym.list_arguments())

        model = mx.mod.Module(context=ctx, symbol=sym,)
        # Use model instead of module
        # model = mx.model.FeedForward(
        #     symbol = sym,
        #     ctx = ctx,
        #     num_epoch = end_epoch,
        #     optimizer = opt,
        #     initializer = initializer,
        #     epoch_size = args.epoch_size if args.epoch_size!=-1 else None,
        #     arg_params = arg_params,
        #     aux_params = aux_params,
        #     begin_epoch = 0,
        #     #**model_args
        #     )

        val_dataiter = None

        if args.loss_type < 10:
            _metric = AccMetric()
        else:
            _metric = LossValueMetric()
        eval_metrics = mx.metric.CompositeEvalMetric()
        eval_metrics.add(AccMetric())
        eval_metrics.add(LossMetric())

        som = 100
        _cb = mx.callback.Speedometer(args.batch_size, som)

        val_iters = get_val(args, kv, mean)
        def ver_test(nbatch):
            verification_hobot.eval_data(val_iters, args.batch_size, model, nbatch)

        global_step = [0]
        def _batch_callback(param):
            #global global_step
            global_step[0] += 1
            mbatch = global_step[0]

            _cb(param)
            if mbatch % args.epoch_size == 0:
                print('lr-batch-epoch:', opt.learning_rate, param.nbatch, param.epoch)

            if mbatch >= 0 and mbatch % args.epoch_size == 0:
                msave = mbatch // args.epoch_size
                ver_test(mbatch)
                do_save = True
                if do_save:
                    print('saving', msave)
                    arg, aux = model.get_params()
                    mx.model.save_checkpoint(prefix, msave, model.symbol, arg, aux)
            if args.max_steps > 0 and mbatch > args.max_steps:
                sys.exit(0)

        epoch_cb = None

        model.fit(
            train_dataiter,
            begin_epoch = begin_epoch,
            num_epoch = end_epoch,
            eval_data = val_dataiter,
            eval_metric = eval_metrics,
            kvstore = kv,
            optimizer = opt,
            #optimizer_params   = optimizer_params,
            initializer = initializer,
            arg_params = arg_params,
            aux_params = aux_params,
            allow_missing = True,
            batch_end_callback = _batch_callback,
            epoch_end_callback = epoch_cb
            )
        # Use model not module
        # model.fit(
        #     X = train_dataiter,
        #     eval_data = val_dataiter,
        #     eval_metric = eval_metrics,
        #     batch_end_callback = _batch_callback,
        #     epoch_end_callback = epoch_cb
        #     )

    else:
        raise ValueError('Choose train_mode from "data_parallel" and "model_parallel"')


def main():
    #time.sleep(3600*6.5)
    global args
    args = parse_args()
    #import pudb; pudb.set_trace()
    train_net(args)


if __name__ == '__main__':
    main()
