__author__ = 'yu01.wang'
import mxnet as mx
#import mxnet.ndarray as nd
import numpy as np
import time
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))
from metriclearn.metric import findMetricThreshold
from hobot_face_val_iter import get_val_iter2


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

    def get(self):
        local_qry_feat = np.concatenate(self.qrys)
        local_qry_label = np.concatenate(self.labels)
        print(local_qry_feat.shape, local_qry_label.shape)
        assert local_qry_feat.shape[0] == local_qry_label.shape[0]
        ref_feat = local_qry_feat#my_Allgatherv(local_qry_feat)
        ref_label = local_qry_label#my_Allgatherv(local_qry_label)
        print(ref_feat.shape, ref_label.shape)
        assert ref_feat.shape[0] == ref_label.shape[0]
        if ref_feat.shape[0] != 12000 and ref_feat.shape[0] != 11904:
            return (self.name + "_test", 0.0)

        st = time.time()
        if self.thres is not None:
            max_prec, max_thres = self.calVeriPrecision(
                feat=ref_feat,
                label=ref_label,
                thres=self.thres)
            print('time spent {} sec'.format(time.time() - st))
            return (self.name + '_lfw(%.4f)' % (max_thres), max_prec)
        return (self.name + '_lfw', 0.0)


class NdcgEval(mx.metric.EvalMetric):
    """Calculate NDCG metrics, for metric learning"""

    def __init__(self, topk):
        super(NdcgEval, self).__init__('ndcg')
        self.topk = topk

    def update(self, labels, preds):
        assert len(labels) == 1
        assert len(preds) == 2      #there are two outputs on pred, embeding and softmax fc
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

def eval_data(eval_datas, part_size, mx_module, epoch):
    eval_metric = NdcgEval(10) #[NdcgEval(10), LFWVeri(0.0, 30.0, 300)]
    for k, _eval_data in enumerate(eval_datas):
        eval_metric.reset()
        _eval_data.reset()
        #preds = mx_module.predict(_eval_data,merge_batches=False)
        #_eval_data.reset()
        for i, eval_batch in enumerate(_eval_data):
            assert _eval_data.batch_size == part_size
            #out_shape = tuple([_eval_data.batch_size] + list(mx_module.output_shapes[0][1:]))
            #cpu_out_feature = nd.zeros(out_shape)
            mx_module.forward(eval_batch, is_train=False)
            mx_module.update_metric(eval_metric, eval_batch)
        name_value = eval_metric.get_name_value()
        for name, value in name_value:
            print('Batch[%d] eval data-%d Validation-%s=%f'%(epoch, k, name, value))

def test(args,kv,mean,mx_module,mbatch):
    val_iters = get_val_iter2(args,kv,mean)
    eval_data(val_iters, args.batch_size, mx_module, mbatch)    #_eval_data
