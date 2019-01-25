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
from image_iter_pk import FaceImageIter
from image_iter_pk import FaceImageIterList
import mxnet as mx
from mxnet import filestream as fs
from mxnet import ndarray as nd
import argparse
import mxnet.optimizer as optimizer
sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
import face_image
sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'symbols'))
#import probe
import fresnet
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
import rank1_metric
import numpy
import numpy as np
from learning_rate_decay import CLRScheduler
sys.path.append(os.path.join(os.path.dirname(__file__), 'losses'))
import center_loss
from hobot_face_val_iter import get_val_iter2
from metriclearn.metric import CalMeanAvgPrecNDCG_MPI, CalAccuracyTopN_MPI, CalKNNError_MPI, my_Allgatherv
from metriclearn.metric import findMetricThreshold, CalROC_GivenCoord_MPI, findMetricThreshold_MPI

logger = logging.getLogger()
logger.setLevel(logging.INFO)

args = None

class SliceByLabel(mx.operator.CustomOp):
  def __init__(self, axis=0):
    self.axis = axis

  def forward(self, is_train, req, in_data, out_data, aux):
    weight = in_data[0]
    labels = in_data[1]
    if self.axis == '0':
       temp = weight[labels]
    if self.axis == '1':
       temp = weight[:,labels]
    self.assign(out_data[0], req[0], temp)

  def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
    labels=in_data[1]
    temp = mx.nd.zeros(in_data[0].shape)
    if self.axis == '0':
      temp[labels] = out_grad[0]
    if self.axis == '1':
      temp[:,labels] = out_grad[0]
    self.assign(in_grad[0],req[0], temp)

@mx.operator.register("SliceByLabel")
class SliceByLabelProp(mx.operator.CustomOpProp):
    def __init__(self, axis=0):
        super(SliceByLabelProp, self).__init__(need_top_grad=True)
        self.axis = axis

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def list_auxiliary_states(self):
        return []

    def infer_shape(self, in_shape):
        out_data_shape = [in_shape[1][0],in_shape[0][1]]
        return in_shape, [out_data_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return SliceByLabel(self.axis)

class MheLossMetric(mx.metric.EvalMetric):
  def __init__(self):
    self.axis = 1
    super(MheLossMetric, self).__init__(
        'mhe_loss', axis=self.axis,
        output_names=None, label_names=None)
    self.losses = []
    self.count = 0

  def update(self, labels, preds):
    preds = preds[2] #use softmax output
    pred = preds.asnumpy()
    self.reset()
    self.sum_metric += pred.sum()
    self.num_inst += pred.shape[0] * pred.shape[1]

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


class OriginLossMetric(mx.metric.EvalMetric):
  def __init__(self):
    self.axis = 1
    super(OriginLossMetric, self).__init__(
        'origin_loss', axis=self.axis,
        output_names=None, label_names=None)
    self.losses = []
    self.count = 0

  def update(self, labels, preds):
    preds = [preds[2]] #use softmax output
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
        assert len(preds) == 1
        pred = preds[0].asnumpy()
        label = labels[0].asnumpy().astype('int32')
        self.qrys.append(pred)
        self.labels.append(label)

    def reset(self):
        self.labels = []
        self.qrys = []

    def get(self):
        local_qry_feat = np.concatenate(self.qrys)
        _norm = np.linalg.norm(local_qry_feat)
        local_qry_feat = local_qry_feat / _norm
        local_qry_label = np.concatenate(self.labels)
        print(local_qry_feat.shape, local_qry_label.shape)
        assert local_qry_feat.shape[0] == local_qry_label.shape[0]
        ref_feat = local_qry_feat  #my_Allgatherv(local_qry_feat)#local_qry_feat#my_Allgatherv(local_qry_feat)
        ref_label = local_qry_label  #my_Allgatherv(local_qry_label)#local_qry_feat#my_Allgatherv(local_qry_label)
        print(ref_feat.shape, ref_label.shape)
        assert ref_feat.shape[0] == ref_label.shape[0]
        if ref_feat.shape[0] == 12000 or ref_feat.shape[0] == 11904 or \
            ref_feat.shape[0]/2 == 12000 or ref_feat.shape[0]/2 == 11904:
            return (self.name + "_lfw", 0.0)
        '''
        st = time.time()
        CalKNNError_MPI(
            local_qry_feat=local_qry_feat,
            local_qry_label=local_qry_label,
            ref_feat=ref_feat,
            ref_label=ref_label,
            K=self.topk,
            leave_one_out=True)
        print('time spent {} sec'.format(time.time() - st))
     '''
        st = time.time()
        '''
        if ref_feat.shape[0] == 19913:
            CalAccuracyTopN_MPI(
                local_qry_coord=local_qry_feat,
                local_qry_label=local_qry_label,
                ref_coord=ref_feat,
                ref_label=ref_label,
                topN_l=[2, 4, 6, 10, 100],
                leave_one_out=True)
            print('time spent {} sec'.format(time.time() - st))
        '''
        st = time.time()
        '''
        mean_avg_prec, mean_nDCG = CalMeanAvgPrecNDCG_MPI(
            local_qry_feat=local_qry_feat,
            local_qry_label=local_qry_label,
            ref_feat=ref_feat,
            ref_label=ref_label,
        )
        print('time spent {} sec'.format(time.time() - st))
        st = time.time()
        roc_info = CalROC_GivenCoord_MPI(
            local_tst_coord=local_qry_feat,
            local_tst_label=local_qry_label)
        print('roc_info: ', roc_info)
        print('time spent {} sec'.format(time.time() - st))
    	'''
        st = time.time()
        findMetricThreshold(local_qry_feat, local_qry_label, ref_feat,
                            ref_label)
        print('time spent {} sec'.format(time.time() - st))
        #return (self.name, mean_nDCG)
        return (self.name, 1.0)


def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    # general
    parser.add_argument(
        '--data-dir', default='', help='training set directory')
    parser.add_argument(
        '--prefix', default='../model/model', help='directory to save model.')
    parser.add_argument(
        '--pretrained', default='', help='pretrained model to load')
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
    parser.add_argument(
        '--wd', type=float, default=0.0005, help='weight decay')
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
        '--num_classes', type=int, default=0, help='')
    parser.add_argument('--img_size', type=str,default='', help='Choose input image size from 112x96 and 112x112')
    args = parser.parse_args()
    return args


def get_symbol(args, arg_params, aux_params):
    data_shape = (args.image_channel, args.image_h, args.image_w)
    print("data shape: ",data_shape)
    image_shape = ",".join([str(x) for x in data_shape])
    margin_symbols = []
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
    #fc7 = mx.symbol.Custom(fc7, layername='after fc7', op_type='ProbeLayer')
    out_list = [mx.symbol.BlockGrad(embedding)]
    softmax = mx.symbol.SoftmaxOutput(
        data=fc7, label=gt_label, name='softmax', normalization='valid')
    out_list.append(softmax)
    if extra_loss is not None:
        out_list.append(extra_loss)
    out = mx.symbol.Group(out_list)
    return (out, arg_params, aux_params)


def train_net(args):
    ctx = []
    cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
    if len(cvd) > 0:
        for i in xrange(len(cvd.split(','))):
            ctx.append(mx.gpu(i))
    if len(ctx) == 0:
        ctx = [mx.cpu()]
        print('use cpu')
    else:
        print('gpu num:', len(ctx))

    #use_sync_bn=args.sync_bn
    #if use_sync_bn:
    #    from mxnet.device_sync import *
    #    init_device_sync(ctx)
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
    data_dir_list = args.data_dir.split(',')
    assert len(data_dir_list) == 1
    data_dir = data_dir_list[0]
    path_imgrec = None
    path_imglist = None
    assert (args.img_size is not '')
    if args.img_size == '112x96':
       args.image_h = 112
       args.image_w = 96
    elif args.img_size == '112x112':
       args.image_h = 112
       args.image_w = 112
    else:
       print("Input image size is not allowed.")
       sys.exit()
    image_size = [args.image_h,args.image_w]
    print('image_size', image_size)
    assert (args.num_classes > 0)
    print('num_classes', args.num_classes)
    path_imgrec = data_dir#os.path.join(data_dir, "train.rec")

    if args.loss_type == 1 and args.num_classes > 20000:
        args.beta_freeze = 5000
        args.gamma = 0.06

    print('Called with argument:', args)
    data_shape = (args.image_channel, image_size[0], image_size[1])
    mean = None
    begin_epoch = 0
    base_lr = args.lr
    base_wd = args.wd
    base_mom = args.mom
    if len(args.pretrained) == 0:
        arg_params = None
        aux_params = None
        sym, arg_params, aux_params = get_symbol(args, arg_params, aux_params)
    else:
        vec = args.pretrained.split(',')
        print('loading', vec)
        _, arg_params, aux_params = mx.model.load_checkpoint(
            vec[0], int(vec[1]))
        sym, arg_params, aux_params = get_symbol(args, arg_params, aux_params)
    if args.network[0] == 's':
        data_shape_dict = {'data': (args.per_batch_size, ) + data_shape}
        spherenet.init_weights(sym, data_shape_dict, args.num_layers)
    print('params: ', sym.list_arguments())
    #label_name = 'softmax_label'
    #label_shape = (args.batch_size,)
    model = mx.mod.Module(
        context=ctx,
        symbol=sym,
    )
    val_dataiter = None

    train_dataiter = FaceImageIter(
        batch_size=args.batch_size,
        data_shape=data_shape,
        path_imgrec=path_imgrec,
        shuffle=True,
        rand_mirror=args.rand_mirror,
        mean=mean,
        cutoff=args.cutoff,
        num_samples_per_class=args.num_samples_per_class,
        repet_flag=args.repet_flag,
        repet_num=args.repet_num,
    )

    if args.loss_type < 10:
        _metric = AccMetric()
    else:
        _metric = LossValueMetric()
    eval_metrics = mx.metric.CompositeEvalMetric()
    eval_metrics.add(AccMetric())
    eval_metrics.add(LossMetric())
    #eval_metrics.add(CenterLossMetric())

    #eval_metrics = [mx.metric.create(_metric)]

    if args.network[0] == 'r' or args.network[0] == 'y':
        initializer = mx.init.Xavier(
            rnd_type='gaussian', factor_type="out", magnitude=2)  #resnet style
    elif args.network[0] == 'i' or args.network[0] == 'x':
        initializer = mx.init.Xavier(
            rnd_type='gaussian', factor_type="in", magnitude=2)  #inception
    else:
        initializer = mx.init.Xavier(
            rnd_type='uniform', factor_type="in", magnitude=2)
    _rescale = 1.0 / args.ctx_num
    if args.warmup:
        lr_scheduler = CLRScheduler(
            step_sizes=args.step_sizes,
            max_iter_steps=args.max_iter_steps,
            start_lr=args.start_lr,
            max_lr=args.max_lr,
            target_lr=args.target_lr,
            power=args.power)
        opt = optimizer.SGD(
            learning_rate=base_lr,
            momentum=base_mom,
            wd=base_wd,
            rescale_grad=_rescale,
            lr_scheduler=lr_scheduler)
    else:
        opt = optimizer.SGD(
            learning_rate=base_lr,
            momentum=base_mom,
            wd=base_wd,
            rescale_grad=_rescale)
    som = 20
    _cb = mx.callback.Speedometer(args.batch_size, som)

    ver_list = []
    ver_name_list = []
    for name in args.target.split(','):
        path = os.path.join('./dataset/', name + ".bin")
        #if os.path.exists(path):
        if fs.exists(path):
            data_set = verification.load_bin(path, image_size)
            ver_list.append(data_set)
            ver_name_list.append(name)
            print('ver', name)

    data_shape = (args.image_channel, args.image_h, args.image_w)
    
    args.val_ ='hdfs://moscpu02.hogpu.cc/user/mengjia.yan/valSet/'
    kv = 'local'   
    val_iters = get_val_iter2(args,kv,mean) 
    
    probe_dataiter = FaceImageIter(
        batch_size=args.batch_size,
        data_shape=(3, 112, 112),
        path_imgrec = args.val_data_dir+'/facescrub_5perid.rec',
        #path_imgrec=args.HDFS +
        #'/user/mengjia.yan/HobotFace/facescrub_full_DeepInsight.rec',
        shuffle=True,
        rand_mirror=1,
        mean=None,
        cutoff=0,
        num_samples_per_class=0,
        repet_flag=0,
        repet_num=1,
    )
    gallery_dataiter = FaceImageIter(
        batch_size=args.batch_size,
        data_shape=(3, 112, 112),
        path_imgrec = args.val_data_dir+'/mega_10000.rec',
        #path_imgrec=args.HDFS +
        #'/user/mengjia.yan/HobotFace/MegaFace_Align.rec',
        shuffle=True,
        rand_mirror=1,
        mean=None,
        cutoff=0,
        num_samples_per_class=0,
        repet_flag=0,
        repet_num=1,
     ) 
    
   
    def ver_test(nbatch):
        #rank1_metric.rank_data(probe_dataiter,gallery_dataiter, args.batch_size, model, nbatch)
        verification_hobot.eval_data(val_iters, args.batch_size, model, nbatch)
        results = []
        for i in xrange(len(ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(
                ver_list[i], model, args.batch_size, 10, None, None)
            print('[%s][%d]XNorm: %f' % (ver_name_list[i], nbatch, xnorm))
            #print('[%s][%d]Accuracy: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc1, std1))
            print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i],
                                                           nbatch, acc2, std2))
            results.append(acc2)
        return results

    highest_acc = [0.0, 0.0]  #lfw and target
    #for i in xrange(len(ver_list)):
    #  highest_acc.append(0.0)
    global_step = [0]
    save_step = [0]
    if len(args.lr_steps) == 0:
        lr_steps = [40000, 60000, 80000]
        if args.loss_type >= 1 and args.loss_type <= 7:
            lr_steps = [100000, 140000, 160000]
        p = 512.0 / args.batch_size
        for l in xrange(len(lr_steps)):
            lr_steps[l] = int(lr_steps[l] * p)
    else:
        lr_steps = [int(x) for x in args.lr_steps.split(',')]
    print('lr_steps', lr_steps)

    def _batch_callback(param):
        #global global_step
        global_step[0] += 1
        mbatch = global_step[0]
        if args.warmup:
            if mbatch % 1000 == 0:
                print('learning rate is: ', opt.learning_rate)

        for _lr in lr_steps:
            if mbatch == args.beta_freeze + _lr:
                if not args.warmup:
                    opt.lr *= 0.1
                    print('lr change to', opt.lr)
                    break

        _cb(param)
        if mbatch % 1000 == 0:
            print('lr-batch-epoch:', opt.lr, param.nbatch, param.epoch)

        if mbatch >= 0 and mbatch % args.verbose == 0:
            acc_list = ver_test(mbatch)
            save_step[0] += 1
            msave = save_step[0]
            do_save = False
            if len(acc_list) > 0:
                lfw_score = acc_list[0]
                if lfw_score >= highest_acc[0]:
                    highest_acc[0] = lfw_score
                    if lfw_score >= 0.998:
                        do_save = True
                if acc_list[-1] >= highest_acc[-1]:
                    highest_acc[-1] = acc_list[-1]
                    if lfw_score >= 0.99:
                        do_save = True
                if lfw_score > highest_acc[0]:
                    do_save = True
            if args.ckpt == 0:
                do_save = False
            elif args.ckpt > 1:
                do_save = True
            if mbatch % 10000 == 0:
                do_save = True
            if do_save:
                print('saving', msave)
                arg, aux = model.get_params()
                mx.model.save_checkpoint(prefix, msave, model.symbol, arg, aux)
            print('[%d]Accuracy-Highest: %1.5f' % (mbatch, highest_acc[0]))
        if mbatch <= args.beta_freeze:
            _beta = args.beta
        else:
            move = max(0, mbatch - args.beta_freeze)
            _beta = max(
                args.beta_min,
                args.beta * math.pow(1 + args.gamma * move, -1.0 * args.power))
        #print('beta', _beta)
        os.environ['BETA'] = str(_beta)
        if args.max_steps > 0 and mbatch > args.max_steps:
            sys.exit(0)

    epoch_cb = None
    #import pudb;pudb.set_trace()
    model.fit(
        train_dataiter,
        begin_epoch=begin_epoch,
        num_epoch=end_epoch,
        eval_data=val_dataiter,
        eval_metric=eval_metrics,
        kvstore='local',  #'dist_sync',#'device',
        optimizer=opt,
        #optimizer_params   = optimizer_params,
        initializer=initializer,
        arg_params=arg_params,
        aux_params=aux_params,
        allow_missing=True,
        batch_end_callback=_batch_callback,
        epoch_end_callback=epoch_cb)


def main():
    #time.sleep(3600*6.5)
    global args
    args = parse_args()
    #import pudb; pudb.set_trace()
    train_net(args)


if __name__ == '__main__':
    main()
